import numpy as np
import matplotlib.pyplot as plt


MODEL2PATH = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "/home/yuli/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1",

}


def plot_attention(avg_attn_weight, ax, fig, max_length=None, tick_interval=None):
    max_length = max_length or avg_attn_weight.shape[0]
    tick_interval = tick_interval or max_length // 8
    Z = avg_attn_weight[:max_length, :max_length]

    # Mask the upper triangle
    mask = np.triu(np.ones_like(Z, dtype=bool), k=1)  # Upper triangle mask
    Z = np.ma.array(Z, mask=mask)  # Mask the upper triangle in the data array

    x_unique = np.arange(Z.shape[1] + 1)  # +1 because pcolormesh needs grid edges
    y_unique = np.arange(Z.shape[0] + 1)
    X, Y = np.meshgrid(x_unique, y_unique)

    # Set the colormap and specify gray for masked values
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='gray')  # Set color for masked values (upper triangle)

    # Plot the heatmap with masked values
    heatmap = ax.pcolormesh(X, Y, Z, cmap=cmap, edgecolors='none', linewidth=0, vmin=Z.min(), vmax=Z.max())

    # Invert y-axis for attention visualization
    ax.invert_yaxis()

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical', location='right', pad=0.03)
    cbar.ax.tick_params(axis='both', which='both', length=0, labelsize=9)
    
    # Set consistent tick intervals for both axes, shifted by 0.5 to center on cells
    ax.set_xticks(np.arange(0.5, max_length, tick_interval))  # Shifted by 0.5
    ax.set_yticks(np.arange(0.5, max_length, tick_interval))  # Shifted by 0.5

    # Set tick labels
    ax.set_xticklabels(np.arange(0, max_length, tick_interval))
    ax.set_yticklabels(np.arange(0, max_length, tick_interval))

    # Remove tick lines
    ax.tick_params(axis='both', which='both', length=0)



def plot_distributions(distribution, ax, fig, xrange=None, yrange=None, Zmin=None, Zmax=None):
    Z = distribution
    x_unique = np.arange(Z.shape[1] + 1) if xrange is None else xrange
    y_unique = np.arange(Z.shape[0] + 1) if yrange is None else yrange
    X, Y = np.meshgrid(x_unique, y_unique)

    # Set the colormap and specify gray for masked values
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='gray')  # Set color for masked values (upper triangle)

    # Plot the heatmap with masked values
    zmin = Z.min() if Zmin is None else Zmin
    zmax = Z.max() if Zmax is None else Zmax
    heatmap = ax.pcolormesh(X, Y, Z, cmap=cmap, edgecolors='none', linewidth=0, vmin=zmin, vmax=zmax)

    # Invert y-axis for attention visualization
    ax.invert_yaxis()

    # Add colorbar
    cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical', location='right', pad=0.03)
    cbar.ax.tick_params(axis='both', which='both', length=0, labelsize=9)

    # Remove tick lines
    ax.tick_params(axis='both', which='both', length=0)