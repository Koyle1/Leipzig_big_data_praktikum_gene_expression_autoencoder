import os
import time
import torch
import torch.distributed as dist
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import RMSprop
from torch.amp import autocast,GradScaler
import wandb
import os
import socket
import torch.distributed as dist

from src.vae import CellVAE
from autoCell.data_loader3 import SingleCellDataset

import torch.nn.functional as F


def print_latent_statistics(model, dataloader, device, num_batches=1):
    """
    Prints latent mean and variance statistics for the first `num_batches` of the dataloader.
    """
    model.eval()
    tau0 = torch.Tensor([1.0]).to(device)

    with torch.no_grad():
        all_mu = []
        all_var = []
        count = 0

        for data in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar, _, _ = model(data, tau0)

            all_mu.append(mu.cpu())
            all_var.append(logvar.exp().cpu())  # Variance = exp(logvar)

            count += 1
            if count >= num_batches:
                break

    mu_all = torch.cat(all_mu, dim=0)
    var_all = torch.cat(all_var, dim=0)

    print(f"\nLatent Space Statistics (first {mu_all.shape[0]} samples):")
    print("Mean of latent means (mu):", mu_all.mean(dim=0).numpy())
    print("Std of latent means (mu):", mu_all.std(dim=0).numpy())
    print("Mean of latent variances (exp(logvar)):", var_all.mean(dim=0).numpy())
    print("Std of latent variances (exp(logvar)):", var_all.std(dim=0).numpy())


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 0.0):
        """
        ELBO loss using RMSE + beta * KLD.
        """
        if logvar is None:
            logvar = torch.ones_like(mu)

        mse = F.mse_loss(recon_x, x, reduction='mean')
        rmse = torch.sqrt(mse)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rmse + beta * KLD, rmse, KLD

def evaluate_and_print_reconstructions(model, dataloader, device):
    model.eval()
    tau0 = torch.Tensor([1.0]).to(device)
    with torch.no_grad():
        data_iter = iter(dataloader)
        data = next(data_iter).to(device)
        recon_batch, mu, logvar, values, sparsity_logits = model(data, tau0)

    print("\nTop 5 largest and smallest features (by original value) for first 10 samples:\n")
    for i in range(10):
        orig = data[i]
        recon = recon_batch[i]
        values = [(j, orig[j].item(), recon[j].item()) for j in range(orig.shape[0])]

        # Sort by original value
        sorted_vals = sorted(values, key=lambda x: x[1])

        print(f"Sample {i}:")

        print("  Smallest 5 features:")
        for j, orig_val, recon_val in sorted_vals[:5]:
            print(f"    Feature {j}: orig={orig_val:.4f}, recon={recon_val:.4f}")

        print("  Largest 5 features:")
        for j, orig_val, recon_val in sorted_vals[-5:][::-1]:  # reversed to show largest first
            print(f"    Feature {j}: orig={orig_val:.4f}, recon={recon_val:.4f}")
        
        print("")

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
    
    # Hyperparams
    batch_size = 1_000
    n_epochs = 1_000
    data_file_path = "data.h5ad"
    n_data_samples = 20_000
    learning_rate = 2e-4
    scale_factor = 10_000
    latent_dim = 10
    number_of_features = 2_000
    use_variance = True
    beta = 10.0 / (number_of_features / latent_dim)
    vae_processing = True
    beta_annealing = False
    
    # New loss function hyperparameters
    sparsity_threshold = 1e-3
    sparsity_weight = 0.0
    value_weight = 1.0
    
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
                "n_features": number_of_features,
                "epochs": n_epochs,
                "latent_dim": latent_dim,
                "batch_size": batch_size,
                "vae-beta": beta,
                "sparsity_threshold": sparsity_threshold,
                "sparsity_weight": sparsity_weight,
                "value_weight": value_weight
            }
        )
    
    tau0 = torch.Tensor([1.0]).to(device)
    
    # Load and cache dataset into memory
    dataset_tmp = SingleCellDataset(
        file_path=data_file_path,
        cell_subset=list(range(n_data_samples)),
        log_transform=True, normalize=True,
        scale_factor=scale_factor,
        remove_outliers=[0.05, 0.95],
        select_n_genes=number_of_features,
        use_vae_preprocessing=vae_processing
    )
    genes = dataset_tmp.adata.var_names
    
    dataset = dataset_tmp
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                           num_workers=os.cpu_count() // dist.get_world_size(),
                           pin_memory=True, persistent_workers=True)
    
    model = CellVAE(input_dim=dataset.n_genes, latent_dim=latent_dim, use_variance=use_variance).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)

        # Beta annealing
        
        if beta_annealing:
            beta_a = 0.5 * beta + 0.5 * (beta / n_epochs) * epoch
        else:
            beta_a = beta
        
        # Updated metrics tracking
        train_loss = train_value_rmse = train_sparsity_loss = train_KLD = 0
        train_sparsity_acc = 0
        
        t0 = time.perf_counter()
        
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            if rank == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, batch size: {data.size(0)}")
            
            with autocast(device_type='cuda'):
                recon_batch, mu, logvar, values, sparsity_logits = model(data, tau0) 
                
                # Updated loss function call with new parameters
                loss, value_rmse, sparsity_loss, KLD, sparsity_accuracy = model.module.loss_function(
                    recon_batch,        # recon_x
                    data,               # x
                    mu,                 # mu
                    logvar,             # logvar
                    values,             # values
                    sparsity_logits,    # sparsity_logits
                    beta=beta_a,     # beta (optional, has default)
                    sparsity_threshold=0.0001,  # sparsity_threshold (optional, has default)
                    sparsity_weight=0.0,      # sparsity_weight (optional, has default)
                    value_weight=1.0          # value_weight (optional, has default)
                )
            
            print(f"[Rank {rank}] Model forward completed.")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Updated metrics accumulation
            train_loss += loss.item() * data.size(0)
            train_value_rmse += value_rmse.item() * data.size(0)
            train_sparsity_loss += sparsity_loss.item() * data.size(0)
            train_KLD += KLD.item() * data.size(0)
            #train_sparsity_acc += sparsity_accuracy.item() * data.size(0)
            train_sparsity_acc = 0
        
        t1 = time.perf_counter()
        
        # Calculate averages
        avg_loss = train_loss / len(dataloader.dataset)
        avg_value_rmse = train_value_rmse / len(dataloader.dataset)
        avg_sparsity_loss = train_sparsity_loss / len(dataloader.dataset)
        avg_KLD = train_KLD / len(dataloader.dataset)
        avg_sparsity_acc = train_sparsity_acc / len(dataloader.dataset)
        
        if rank == 0:
            print(f"[Epoch {epoch}] Time: {t1-t0:.2f}s | Loss: {avg_loss:.4f} | "
                  f"Value RMSE: {avg_value_rmse:.4f} | Sparsity Loss: {avg_sparsity_loss:.4f} | "
                  f"Sparsity Acc: {avg_sparsity_acc:.4f}  | KLD: {avg_KLD:.4f}")
            
            # Updated wandb logging
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "value_rmse": avg_value_rmse,
                # "sparsity_loss": avg_sparsity_loss,
                # "pr_auc": avg_sparsity_acc,
                "KLD": avg_KLD,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    
    if rank == 0:
        evaluate_and_print_reconstructions(model.module, dataloader, device)
        print_latent_statistics(model.module, dataloader, device, num_batches=2)
        torch.save(model.module.state_dict(), "model.pth")
        wandb.finish()
    
    cleanup_ddp()

if __name__ == "__main__":
    train()
