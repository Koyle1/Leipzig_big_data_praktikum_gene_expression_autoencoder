import torch
from torch.utils.data import DataLoader
from autoCell.data_loader import SingleCellDataset
from src.vae import CellVAE, elbo_loss_function

import pandas as pd

def test_vae(model, dataloader, device):
    model.eval()
    test_loss = 0
    test_BCE = 0
    test_KLD = 0

    tau0 = torch.Tensor([1.0]).to(device)  # same as in training
    
    all_originals = []
    all_reconstructions = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            (recon_batch, dropout_probs), mu, logvar = model(data, tau0)
            loss, BCE, KLD = elbo_loss_function(recon_batch, data, mu, logvar, beta=1.0)
            test_loss += loss.item()
            test_BCE += BCE.item()
            test_KLD += KLD.item()
            
            all_originals.append(data.cpu())
            all_reconstructions.append(recon_batch.cpu())

    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)

    # Select first 10 samples
    originals_10 = all_originals[:10].numpy()
    reconstructions_10 = all_reconstructions[:10].numpy()

    n_features = originals_10.shape[1]

    # Prepare columns in the form: feature_1, reconstructed_feature_1, feature_2, reconstructed_feature_2, ...
    columns = []
    for i in range(n_features):
        columns.append(f"feature_{i+1}")
        columns.append(f"reconstructed_feature_{i+1}")

    # Create a DataFrame with double the number of columns interleaving originals and reconstructions
    data = []
    for orig_row, recon_row in zip(originals_10, reconstructions_10):
        interleaved = []
        for i in range(n_features):
            interleaved.append(orig_row[i])
            interleaved.append(recon_row[i])
        data.append(interleaved)

    df_compare = pd.DataFrame(data, columns=columns)

    df_compare.to_csv("vae_reconstruction_comparison.csv", index=False)

    n_samples = len(dataloader.dataset)
    print(f"\nTest Loss (ELBO): {test_loss / n_samples:.6f}")
    print(f"Test Reconstruction Loss (BCE): {test_BCE / n_samples:.6f}")
    print(f"Test KL Divergence: {test_KLD / n_samples:.6f}")


if __name__ == "__main__":
    device = torch.device("cpu")  # change to "cuda" if available

    # Parameters (make sure they match your training setup)
    data_file_path = "data.h5ad"
    n_data_samples = 100

    # Load a temporary dataset to get gene names for feature selection
    dataset_tmp = SingleCellDataset(
        file_path=data_file_path,
        cell_subset=[i for i in range(n_data_samples)],
        log_transform=True,
        normalize=True,
        scale_factor=1.0
    )

    genes = dataset_tmp.adata.var_names
    gene_subset = genes[:50]  # same subset as in training

    # Load the test dataset with the gene subset
    test_dataset = SingleCellDataset(
        file_path=data_file_path,
        cell_subset=[i for i in range(n_data_samples)],
        gene_subset=gene_subset,
        log_transform=True,
        normalize=True,
        scale_factor=1.0
    )
    
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Load model and weights
    model = CellVAE(input_dim=test_dataset.n_genes, latent_dim=2, use_variance=False)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    
    # Run the test
    test_vae(model, test_loader, device)
