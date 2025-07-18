import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
import yaml
import argparse
from pathlib import Path

def plot_latent_scatter(
    df, x='z1', y='z2', label_col='label', ax=None, title=None, 
    dot_size=30, legend_dot_size=20, colormap='tab20', ticks=True
):
    """
    Plots a scatterplot colored by label_col. Allows separate control of dot size
    in plot and in legend, and retains color in the legend.
    """
    if ax is None:
        ax = plt.gca()

    # Create a colormap
    unique_labels = df[label_col].unique()
    colors = plt.get_cmap(colormap)(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))

    # Custom legend handles
    handles = []

    for label in unique_labels:
        group = df[df[label_col] == label]
        color = color_map[label]

        # Plot scatter points
        ax.scatter(group[x], group[y], label=label, alpha=0.7, s=dot_size, color=color)

        # Create legend handle with color
        if legend_dot_size:
            handle = mlines.Line2D(
                [], [], marker='o', linestyle='None',
                markersize=np.sqrt(legend_dot_size), color=color, label=label
            )
            handles.append(handle)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)

    # Place legend in right margin with correct colors and sizes
    if legend_dot_size:
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    return ax

def print_latent_statistics(model, dataloader, device, num_batches=1):
    """
    Prints latent mean and variance statistics for the first `num_batches` of the dataloader.
    """
    model.eval()

    with torch.no_grad():
        all_mu = []
        all_var = []
        count = 0

        for data in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)

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

def evaluate_and_print_reconstructions(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        data = next(data_iter).to(device)
        recon_batch, mu, logvar = model(data)

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


def load_model_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add file name to config
    config["config_name"] = Path(config_path).stem

    # Derived hyperparameter: beta
    config["beta"] = config["beta_normalized"] / (config["number_of_features"] / config["latent_dim"])
    return config