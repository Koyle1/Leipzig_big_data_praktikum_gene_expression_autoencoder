from src.vae import CellVAE, elbo_loss_function
from autoCell.data_loader import SingleCellDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import wandb

def main():
    n_epochs = 5000
    batch_size = 16
    data_file_path = "data.h5ad"
    learning_rate = 0.001
    scale_factor = 1.0
    latent_dim = 10
    use_variance = False
    beta = 1.0
    model_save_path = "model.pth"
    torch_device = "cpu"
    verbose = True
    log_interval = 5
    # TODO add random seed

    # Initialize tracking
    run = wandb.init(
        entity="coyfelix7-universit-t-leipzig",
        project="big_data_vae",
        config={
            "learning_rate": learning_rate,
            "dataset": data_file_path,
            "epochs": n_epochs,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "use_variance": use_variance,
            "vae-beta": beta

        }
    )

    # Load Data
    dataset = SingleCellDataset(file_path=data_file_path, cell_subset=[i for i in range(1000)], log_transform=True, normalize=True, scale_factor=scale_factor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define Model
    model = CellVAE(input_dim=dataset.n_genes, latent_dim=10, use_variance=use_variance)
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    device = torch.device(torch_device)


    if verbose: print("Starting training...")
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, data, in enumerate(dataloader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = elbo_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item() / len(data)))

        avg_loss = train_loss / len(dataloader.dataset)
        if verbose: print(f"Train Epoch: {epoch} Average Loss: {avg_loss}")
        run.log({"step_avg_loss": avg_loss})

    run.finish()

    # Save final model
    torch.save(model.policy.state_dict(), model_save_path)
    

if __name__ == "__main__":
    main()