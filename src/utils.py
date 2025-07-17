import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

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