import os
import time
import torch
import torch.distributed as dist
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import RMSprop, Adam
from torch.amp import autocast, GradScaler
import wandb
import socket
from autoCell.data_loader import SingleCellDataset

from src.vae import CellVAE
import torch.nn.functional as F
from scipy.stats import pearsonr

import pandas as pd

def save_latent_scatter_plots(epoch, adata, label_df, labeled_latents):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 

    df_pca = pd.DataFrame(adata.obsm['X_pca'], columns=[f'PC{i+1}' for i in range(adata.obsm['X_pca'].shape[1])])
    labeled_pca = pd.concat([label_df.reset_index(drop=True), df_pca], axis=1)

    plot_latent_scatter(labeled_pca, x='PC1', y='PC2', label_col='tissue', ax=axs[0,0], title='Tissue by top 2 principal components', dot_size=1, legend_dot_size=None, ticks=False)
    plot_latent_scatter(labeled_pca, x='PC1', y='PC2', label_col='disease', ax=axs[1,0], title='Disease by top 2 principal components', dot_size=1, legend_dot_size=None, ticks=False)
    plot_latent_scatter(labeled_latents, x='z1', y='z2', label_col='tissue', ax=axs[0,1], title='Tissue by VAE latents', dot_size=1, legend_dot_size=20, ticks=True)
    plot_latent_scatter(labeled_latents, x='z1', y='z2', label_col='disease', ax=axs[1,1], title='Disease by VAE latents', dot_size=1, legend_dot_size=20, ticks=True)

    plt.tight_layout()

    filename = os.path.join(output_dir, f'latent_scatter_epoch_{epoch:03d}.png')
    plt.savefig(filename)
    plt.close(fig)
    print(f'Saved latent scatter plot for epoch {epoch} to {filename}')


def permute_dims(z):
    z_perm = []
    for i in range(z.size(1)):
        z_perm.append(z[:, i][torch.randperm(z.size(0))])
    return torch.stack(z_perm, dim=1)


def print_latent_statistics(model, dataloader, device, num_batches=1):
    model.eval()
    tau0 = torch.tensor([1.0], device=device)

    with torch.no_grad():
        all_mu = []
        all_var = []
        count = 0

        for batch in dataloader:
            data = batch[0].to(device) if isinstance(batch, (tuple, list)) else batch.to(device)

            recon_batch, mu, logvar, _, _ = model(data, tau0)

            all_mu.append(mu.cpu())
            all_var.append(logvar.exp().cpu())

            count += 1
            if count >= num_batches:
                break

    mu_all = torch.cat(all_mu, dim=0).numpy()
    var_all = torch.cat(all_var, dim=0).numpy()

    print(f"\nLatent Space Statistics (first {mu_all.shape[0]} samples):")
    print("Mean of latent means (mu):", mu_all.mean(axis=0))
    print("Std of latent means (mu):", mu_all.std(axis=0))
    print("Mean of latent variances (exp(logvar)):", var_all.mean(axis=0))
    print("Std of latent variances (exp(logvar)):", var_all.std(axis=0))

    print("\nPearson correlation between latent means and variances per dimension:")
    for i in range(mu_all.shape[1]):
        corr, p_value = pearsonr(mu_all[:, i], var_all[:, i])
        print(f"Dimension {i}: correlation={corr:.4f}, p-value={p_value:.4g}")

    print("\nPearson correlation between latent dimensions (mu[:, 0] vs mu[:, 1]):")
    corr, p_value = pearsonr(mu_all[:, 0], mu_all[:, 1])
    print(f"Correlation: {corr:.4f}, p-value: {p_value:.2e}")

def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

def evaluate_and_print_reconstructions(model, dataloader, device):
    model.eval()
    tau0 = torch.Tensor([1.0]).to(device)
    with torch.no_grad():
        data_iter = iter(dataloader)
        data = next(data_iter).to(device)
        recon_batch, mu, logvar, _, _ = model(data, tau0)

    print("\nTop 5 largest and smallest features (by original value) for first 10 samples:\n")
    for i in range(10):
        orig = data[i]
        recon = recon_batch[i]
        values = [(j, orig[j].item(), recon[j].item()) for j in range(orig.shape[0])]
        sorted_vals = sorted(values, key=lambda x: x[1])

        print(f"Sample {i}:")
        print("  Smallest 5 features:")
        for j, orig_val, recon_val in sorted_vals[:5]:
            print(f"    Feature {j}: orig={orig_val:.4f}, recon={recon_val:.4f}")

        print("  Largest 5 features:")
        for j, orig_val, recon_val in sorted_vals[-5:][::-1]:
            print(f"    Feature {j}: orig={orig_val:.4f}, recon={recon_val:.4f}")
        print("")

def kl_anneal_function(epoch, max_beta=1e-2, anneal_epochs=50):
    """Linearly increase beta from 0 to max_beta over `anneal_epochs` epochs."""
    if epoch >= anneal_epochs:
        return max_beta
    return max_beta * (epoch / anneal_epochs)


def train():
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    batch_size = 1000
    n_epochs = 750
    data_file_path = "data.h5ad"
    n_data_samples = 20000
    learning_rate = 1e-4
    scale_factor = 1.0
    latent_dim = 2
    number_of_features = 2000
    use_variance = True
    beta = 1e-4
    vae_processing = True

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    if rank == 0:
        wandb.init(
            entity="coyfelix7-universit-t-leipzig",
            project="big_data_vae",
            config={
                "learning_rate": learning_rate,
                "dataset": data_file_path,
                "n_data_samples": n_data_samples,
                "epochs": n_epochs,
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "vae-beta": beta,
            }
        )

    tau0 = torch.Tensor([1.0]).to(device)

    dataset_tmp = SingleCellDataset(
        file_path=data_file_path,
        cell_subset=list(range(n_data_samples)),
        log_transform=True,
        normalize=True,
        scale_factor=scale_factor,
        remove_outliers=[0.05, 0.95],
        select_n_genes=number_of_features,
        use_vae_preprocessing=vae_processing
    )
    sampler = DistributedSampler(dataset_tmp, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset_tmp,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=os.cpu_count() // world_size,
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize VAE and wrap in DDP
    vae_model = CellVAE(input_dim=dataset_tmp.n_genes, latent_dim=latent_dim, use_variance=use_variance).to(device)
    vae_model.discriminator = None  # just in case discriminator attribute exists
    vae_model = DDP(vae_model, device_ids=[device.index])

    optimizer = Adam(vae_model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    output_dir = 'training_plots'
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, n_epochs + 1):
        vae_model.train()
        sampler.set_epoch(epoch)

        current_beta = kl_anneal_function(epoch, max_beta=1e-3, anneal_epochs=200)

        train_loss = train_rmse = train_KLD = train_MI = train_TC = train_dim_wise_kl = 0
        t0 = time.perf_counter()

        for data in dataloader:
            data = data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                recon_batch, mu, logvar, z, _ = vae_model(data, tau0)
        
                # Use built-in loss function from your VAE model
                loss, rmse, MI, TC, dim_wise_kl = vae_model.module.loss_function(
                    recon_x=recon_batch,
                    x=data,
                    mu=mu,
                    logvar=logvar,
                    z=z,
                    beta=1e-4,
                    gamma=1e-4,
                    theta=1e-4,
                    sparsity_threshold=1e-4,
                    value_weight=1.0
                )
        
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
            train_loss += loss.item() * data.size(0)
            train_rmse += rmse.item() * data.size(0)
            # Optionally track MI, TC, dim_wise_kl if you want:
            train_MI += MI.item() * data.size(0)
            train_TC += TC.item() * data.size(0)
            train_dim_wise_kl += dim_wise_kl.item() * data.size(0)
        
        t1 = time.perf_counter()
        
        avg_loss = train_loss / len(dataloader.dataset)
        avg_rmse = train_rmse / len(dataloader.dataset)
        avg_MI = train_MI / len(dataloader.dataset)
        avg_TC = train_TC / len(dataloader.dataset)
        avg_dim_wise_kl = train_dim_wise_kl / len(dataloader.dataset)

        if rank == 0:
            print(f"[Epoch {epoch}] Time: {t1 - t0:.2f}s | Loss: {avg_loss:.4f} | RMSE: {avg_rmse:.4f}  | MI: {avg_MI:.4f} | TC: {avg_TC:.4f} | dKL: {avg_dim_wise_kl:.4f}")
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "rmse": avg_rmse,
                "MI": avg_MI,
                "TC": avg_TC,
                "dim_wise_KL": avg_dim_wise_kl,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

    

    if rank == 0:
        # Evaluate reconstructions and latent statistics (existing)
        evaluate_and_print_reconstructions(vae_model.module, dataloader, device)
        print_latent_statistics(vae_model.module, dataloader, device, num_batches=2)

        # Save model checkpoint
        torch.save({'vae_state_dict': vae_model.module.state_dict()}, "model.pth")
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    train()