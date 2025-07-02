from src.vae import CellVAE, elbo_loss_function, elbo_loss_function_normalized
from autoCell.data_loader import SingleCellDataset
# from autoCell.data_loader_sequential import BatchedSingleCellDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import wandb

def main():
    n_epochs = 5000
    batch_size = 10
    data_file_path = "data.h5ad"
    n_data_samples = 100
    learning_rate = 0.001
    scale_factor = 1.0
    latent_dim = 2
    use_variance = True
    beta = 0.001
    tau0 = torch.Tensor([1.0])
    model_save_path = "model.pth"
    torch_device = "cpu"
    verbose = True
    log_interval = 5
    save_interval = 5
    # TODO add random seed

    # Initialize tracking
    run = wandb.init(
        entity="coyfelix7-universit-t-leipzig",
        project="big_data_vae",
        config={
            "learning_rate": learning_rate,
            "dataset": data_file_path,
            "n_data_samples": n_data_samples,
            "epochs": n_epochs,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "use_variance": use_variance,
            "vae-beta": beta
        }
    )

    # Set torch device
    device = torch.device(torch_device)
    tau0 = tau0.to(device)

    # Load Data
    dataset_tmp = SingleCellDataset(file_path=data_file_path, cell_subset=[i for i in range(n_data_samples)], log_transform=True, normalize=True, scale_factor=scale_factor)
    genes = dataset_tmp.adata.var_names
    gene_subset = genes[:50]
    dataset = SingleCellDataset(file_path=data_file_path, cell_subset=[i for i in range(n_data_samples)], gene_subset=gene_subset, log_transform=True, normalize=True, scale_factor=scale_factor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # dataset = BatchedSingleCellDataset(
    #     file_path=data_file_path,
    #     cell_subset=[i for i in range(n_data_samples)],
    #     batch_size=batch_size,
    #     cache_size=400,
    #     log_transform=True,
    #     normalize=True,
    #     scale_factor=scale_factor,
    # )
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,num_workers=4,
    #     pin_memory=torch.cuda.is_available()
    # )


    # Define Model
    model = CellVAE(input_dim=dataset.n_genes, latent_dim=latent_dim, use_variance=use_variance)
    model = model.to(device)
    optimizer = RMSprop(model.parameters(), lr=learning_rate)


    if verbose: print("Starting training...")
    for epoch in range(1, n_epochs + 1):
        # simple beta annealing
        # beta_anneal = min(beta, epoch / 10000)
        beta_anneal = beta


        model.train()
        train_loss = 0
        train_BCE = 0
        train_KLD = 0
        for batch_idx, data, in enumerate(dataloader):
            data = data.to(device)
            (recon_batch, dropout_probs), mu, logvar = model(data, tau0)
            loss, BCE, KLD = elbo_loss_function(recon_batch, data, mu, logvar, beta=beta_anneal)
            loss.backward()
            train_loss += loss.item()
            train_BCE += BCE.item()
            train_KLD += KLD.item()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item() / len(data)))
                # print(mu.detach().numpy())
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         grad_mean = param.grad.mean().item()
                #         grad_max = param.grad.abs().max().item()
                #         print(f"{name}: grad mean = {grad_mean:.5f}, max = {grad_max:.5f}")

        avg_loss = train_loss / len(dataloader.dataset)
        if verbose: print(f"Train Epoch: {epoch} Average Loss: {avg_loss}")
        run.log({
                "step_avg_loss": avg_loss,
                "step_avg_binary_cross_entropy": train_BCE / len(dataloader.dataset),
                "step_avg_KL_divergence": train_KLD / len(dataloader.dataset),
                # "last_batch_mean_dropout_probability": dropout_probs.mean().item()
            })

        if epoch % save_interval == 0:
            print(mu.detach().numpy())
            for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.mean().item()
                        grad_max = param.grad.abs().max().item()
                        print(f"{name}: grad mean = {grad_mean:.5f}, max = {grad_max:.5f}")
            
            torch.save(model.state_dict(), model_save_path)

    run.finish()

    # Save final model
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()