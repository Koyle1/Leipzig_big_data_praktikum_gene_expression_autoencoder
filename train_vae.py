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
import torch.nn.functional as F
import wandb
import os
import socket
import torch.distributed as dist

from src.vae import CellVAE
from src.dataloader import SingleCellDataset
from src.utils import print_latent_statistics, evaluate_and_print_reconstructions

def setup_ddp():
    # Initialize the default process group for distributed training.
    # 'nccl' is used as the backend, which is optimal for GPUs.
    # 'env://' means configuration is read from environment variables (e.g., MASTER_ADDR, RANK, WORLD_SIZE).
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    # Set the current process's GPU device.
    # This ensures each process uses a different GPU based on its rank.
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    # Cleanly shuts down the distributed process group.
    # This is important to release resources and avoid hangs at the end of training.
    dist.destroy_process_group()

def train():
    # Initialize distributed data parallel (DDP)
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Hyperparams
    batch_size = 1_000
    n_epochs = 500
    data_file_path = "data/data.h5ad"
    n_data_samples = 20_000
    learning_rate = 2e-4
    scale_factor = 10_000
    latent_dim = 2
    number_of_features = 2_000
    use_variance = True
    beta = 5.0 / (number_of_features / latent_dim)
    vae_processing = True
    beta_annealing = False

    # Set device for this process (one GPU per process)
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Initialize Weights & Biases (only from rank 0 process)
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
            }
        )

    # Initialize temperature
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

    # Create distributed sampler and DataLoader
    dataset = dataset_tmp
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                           num_workers=os.cpu_count() // dist.get_world_size(),
                           pin_memory=True, persistent_workers=True)

    # Initialize model, wrap in DistributedDataParallel
    model = CellVAE(input_dim=dataset.n_genes, latent_dim=latent_dim, use_variance=use_variance).to(device)
    model = DDP(model, device_ids=[device.index])

    # Set optimizer and gradient scaler for mixed-precision training
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Training loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        sampler.set_epoch(epoch)

        # (Optional) anneal beta
        if beta_annealing:
            beta_a = 0.5 * beta + 0.5 * (beta / n_epochs) * epoch
        else:
            beta_a = beta
        
        # Reset epoch metrics
        train_loss = train_value_rmse = train_KLD = 0  
        t0 = time.perf_counter()

        # Iterate over mini-batches
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            if rank == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, batch size: {data.size(0)}")

            # Forward pass under mixed precision
            with autocast(device_type='cuda'):
                recon_batch, mu, logvar = model(data, tau0) 
                
                # Updated loss function call with new parameters
                loss, RMSE, KLD = model.module.loss_function(
                    recon_batch,        # recon_x
                    data,               # x
                    mu,                 # mu
                    logvar,             # logvar
                    beta=beta_a,     # beta (optional, has default)
                )
            
            print(f"[Rank {rank}] Model forward completed.")

            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate metrics
            train_loss += loss.item() * data.size(0)
            train_value_rmse += RMSE.item() * data.size(0)
            train_KLD += KLD.item() * data.size(0)
        
        t1 = time.perf_counter()
        
        # Calculate averages
        avg_loss = train_loss / len(dataloader.dataset)
        avg_value_rmse = train_value_rmse / len(dataloader.dataset)
        avg_KLD = train_KLD / len(dataloader.dataset)

        # Log results (only rank 0)
        if rank == 0:
            print(f"[Epoch {epoch}] Time: {t1-t0:.2f}s | Loss: {avg_loss:.4f} | "
                  f"Value RMSE: {avg_value_rmse:.4f} | "
                  f"KLD: {avg_KLD:.4f}")
            
            # Updated wandb logging
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "value_rmse": avg_value_rmse,
                "KLD": avg_KLD,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

    # Final evaluation (rank 0 only)
    if rank == 0:
        evaluate_and_print_reconstructions(model.module, dataloader, device)
        print_latent_statistics(model.module, dataloader, device, num_batches=2)
        torch.save(model.module.state_dict(), "model.pth")
        wandb.finish()

     # Clean up distributed training
    cleanup_ddp()

if __name__ == "__main__":
    train()
