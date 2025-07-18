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
import argparse
from datetime import datetime

from src.vae import CellVAE
from src.dataloader import SingleCellDataset
from src.utils import print_latent_statistics, evaluate_and_print_reconstructions, load_model_config

def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

def train(model_config: dict):
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seed = model_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Hyperparams
    batch_size = model_config["batch_size"]
    n_epochs = model_config["n_epochs"]
    data_file_path = model_config["data_file_path"]
    n_data_samples = model_config["n_data_samples"]
    learning_rate = model_config["learning_rate"]
    scale_factor = model_config["scale_factor"]
    latent_dim = model_config["latent_dim"]
    number_of_features = model_config["number_of_features"]
    use_variance = model_config["use_variance"]
    beta = model_config["beta"]
    vae_processing = model_config["vae_preprocessing"]
    beta_annealing = model_config["beta_annealing"]
    
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
            }
        )
    
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

        # Beta annealing across all epochs
        if beta_annealing:
            beta_a = (beta / n_epochs) * epoch
        else:
            beta_a = beta
        
        # Updated metrics tracking
        train_loss = train_value_rmse = train_KLD = 0
        
        t0 = time.perf_counter()
        
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            if rank == 0 and epoch % 10 == 0:
                print(f"Epoch {epoch}, batch size: {data.size(0)}")
            
            with autocast(device_type='cuda'):
                recon_batch, mu, logvar = model(data) 
                
                # Updated loss function call with new parameters
                loss, RMSE, KLD = model.module.loss_function(
                    recon_batch,        # recon_x
                    data,               # x
                    mu,                 # mu
                    logvar,             # logvar
                    beta=beta_a,     # beta (optional, has default)
                )
            
            print(f"[Rank {rank}] Model forward completed.")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Updated metrics accumulation
            train_loss += loss.item() * data.size(0)
            train_value_rmse += RMSE.item() * data.size(0)
            train_KLD += KLD.item() * data.size(0)
        
        t1 = time.perf_counter()
        
        # Calculate averages
        avg_loss = train_loss / len(dataloader.dataset)
        avg_value_rmse = train_value_rmse / len(dataloader.dataset)
        avg_KLD = train_KLD / len(dataloader.dataset)
        
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
    
    if rank == 0:
        # Print information
        evaluate_and_print_reconstructions(model.module, dataloader, device)
        print_latent_statistics(model.module, dataloader, device, num_batches=2)

        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        model_save_path = os.path.join(
            model_config["model_save_path"],
            f"{timestamp}_{model_config['config_name']}.pth"
        )
        
        torch.save(model.module.state_dict(), model_save_path)
        
        wandb.finish()
    
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser(description="Train model with YAML config")
    parser.add_argument("--config", type=str, default="modelconfigs/example.yaml", help="Path to the YAML config file")
    args = parser.parse_args()

    config = load_model_config(args.config)

    print("Loaded hyperparameters:")
    for key, value in config.items():
        print(f"{key}: {value}")

    train(config)

if __name__ == "__main__":
    main()
