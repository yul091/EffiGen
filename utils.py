
from typing import Any, Dict, Optional, List
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json


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



class Task:
    def __init__(
        self, 
        task_id: int, 
        query: Dict[str, Any], 
        rate_lambda: float,
        feedback: Optional[Any] = None,  
        require_training: Optional[bool] = None,
    ):
        self.task_id = task_id
        self.query = query
        self.rate_lambda = rate_lambda
        self.feedback = feedback
        self.require_training = False if require_training is None else require_training
        # self.hybrid_batch = None
        # Define do_backward for selective training: initially set to require_training
        self.do_backward = False if require_training is None else require_training
        # self.start = start
        self.decode_step = 0
        # self.batch_decode_steps = []


def record_time(
    device: int, 
    event_type: str, 
    opt_type: str, 
    taskID: int,
    timing_info: Dict[str, List[float]], 
    verbose: bool = False,
) -> float:
    # event_type can be 'start' or 'end'
    timestamp = time.time()
    timing_info[f"{device}_{event_type}"].append((timestamp, opt_type, taskID))
    if verbose:
        print(f"\t[CUDA {device}] Task {event_type} at time {timestamp}")
    return timestamp


def save_metrics_with_order(metrics: dict, filepath: str):
    # Extract keys with float or int values
    prioritized_keys = [k for k, v in metrics.items() if isinstance(v, (float, int))]
    # Create a reordered dictionary
    reordered_metrics = OrderedDict()
    for key in prioritized_keys:
        reordered_metrics[key] = metrics[key]  # Add prioritized keys first
    for key, value in metrics.items():
        if key not in prioritized_keys:
            reordered_metrics[key] = value  # Add remaining keys
    
    # Save the reordered dictionary as JSON
    with open(filepath, 'w') as f:
        json.dump(reordered_metrics, f, indent=4)
    print(f"Metrics saved with reordered keys to {filepath}")

