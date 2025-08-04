
from typing import Any, Dict, Optional, List, Union
from collections.abc import Mapping
from transformers import LlamaTokenizer
from transformers.cache_utils import DynamicCache
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



LEGENDS2UTIL = {
    'sync': {'name': 'Continuous', 'color': 'tab:blue', 'linestyle': '-', 'marker': 'o'},
    'async': {'name': 'Hybrid', 'color': 'tab:purple', 'linestyle': '--', 'marker': '^'},
    'train-first': {'name': 'Train-Test', 'color': 'tab:green', 'linestyle': '-.', 'marker': '*'},
    'test-first': {'name': 'Test-Train', 'color': 'tab:purple', 'linestyle': ':', 'marker': 's'},
    'train-middle': {'name': 'Test-Train-Test', 'color': 'tab:orange', 'linestyle': '-', 'marker': 'x'},
    'period-10': {'name': 'Periodic-10', 'color': 'tab:green', 'linestyle': '--', 'marker': 'D'},
    'period-15': {'name': 'Periodic-10', 'color': 'tab:green', 'linestyle': '--', 'marker': 'D'},
    'period-20': {'name': 'Periodic-20', 'color': 'tab:red', 'linestyle': '-.', 'marker': 's'},
    'period-30': {'name': 'Periodic-50', 'color': 'tab:orange', 'linestyle': '-.', 'marker': 'x'},
    'period-50': {'name': 'Periodic-50', 'color': 'tab:orange', 'linestyle': '-.', 'marker': 'x'},
    'period': {'name': 'Periodic', 'color': 'tab:orange', 'linestyle': '-.', 'marker': 'x'},
}
VARS2UTIL = {
    'retrain_rate': {'name': 'Retrain rate (%)', 'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    'arrival_rate': {'name': 'Arrival rate (rps)', 'color': 'red', 'linestyle': '--', 'marker': '^'},
    'taskID': {'name': 'Time [s]', 'color': 'green', 'linestyle': '-.', 'marker': '*'},
}
SCORES2UTIL = {
    'total_time': {'name': 'Total runtime (s)', 'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    'throughput': {'name': 'Throughput (tokens/s)', 'color': 'red', 'linestyle': '--', 'marker': '^'},
    'contrastive log probability difference (CLPD)': {'name': 'CLPD', 'color': 'green', 'linestyle': '-.', 'marker': '*'},
    'preference accuracy': {'name': 'Preference accuracy', 'color': 'purple', 'linestyle': ':', 'marker': 's'},
    'loss': {'name': 'Loss', 'color': 'orange', 'linestyle': '-', 'marker': 'x'},
    'perplexity': {'name': 'Perplexity', 'color': 'brown', 'linestyle': '--', 'marker': 'D'},
    'log_prob_diff': {'name': 'CLPD', 'color': 'green', 'linestyle': '-.', 'marker': '*'},
    'correct_preds': {'name': 'Preference accuracy', 'color': 'purple', 'linestyle': ':', 'marker': 's'},
}


# Smoothing function (Exponential Moving Average)
def smooth_curve(values, weight=0.6):
    smoothed_values = []
    last = values[0]  # First value remains the same
    for val in values:
        last = last * weight + (1 - weight) * val  # Exponential smoothing formula
        smoothed_values.append(last)
    return smoothed_values



def read_json(file_path):
    data = []
    buffer = ""
    with open(file_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            buffer += line + "\n"
            if line.endswith("}"):  # Detect end of a JSON object
                try:
                    data.append(json.loads(buffer))
                    buffer = ""  # Reset buffer after successful parsing
                except json.JSONDecodeError:
                    pass  # Keep reading until a full JSON object is accumulated
    return data



def _prepare_input(
    data: Union[torch.Tensor, Any],
    device: torch.device = 'cuda',
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, DynamicCache):
        data.key_cache = _prepare_input(data.key_cache, device)
        data.value_cache = _prepare_input(data.value_cache, device)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data
    

def prepare_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
    device: torch.device = 'cuda',
) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    new_inputs = _prepare_input(inputs, device=device)
    if new_inputs is None or len(new_inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it."
        )
    return new_inputs


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


def compute_generation_metrics(hypothesis: str, reference: str, tokenizer: Optional[LlamaTokenizer] = None) -> dict:
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True, tokenizer=tokenizer)
    rouge_l = scorer.score(reference, hypothesis)["rougeL"].fmeasure
    # BLEU
    if tokenizer is not None:
        # Tokenize using the provided tokenizer
        ref_tokens = tokenizer.tokenize(reference, add_special_tokens=False)
        hyp_tokens = tokenizer.tokenize(hypothesis, add_special_tokens=False)
    else:
        # Fallback to simple whitespace tokenization
        ref_tokens = reference.strip().split()
        hyp_tokens = hypothesis.strip().split()
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

    return {
        "rougeL": rouge_l,
        "bleu": bleu_score,
    }



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



def plot_performance_over_time(
    file_path: str, 
    x: str = "time", 
    y: str = "loss", 
    interval: int = 50, 
    set_grid: bool = False,
    set_title: bool = False,
    label_fontsize: int = 18,
    legend_fontsize: int = 14,
    tick_fontsize: int = 12,
    sort_y_values: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    method: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    min_y: Optional[float] = None,
):
    eval_metrics = json.load(open(file_path))
    method = file_path.split("/")[-1].split("_")[0] if method is None else method
    generation_res = eval_metrics["generation_results"]
    color = 'tab:purple' if method not in LEGENDS2UTIL else LEGENDS2UTIL[method]["color"]
    marker = '^' if method not in LEGENDS2UTIL else LEGENDS2UTIL[method]["marker"]
    label = LEGENDS2UTIL[method]["name"] if method in LEGENDS2UTIL else method
    res = []
    for task_dict in generation_res:
        if task_dict["workload"] == "train":
            continue 
        res.append({
            "taskID": task_dict["taskID"],
            "time": task_dict["execution_time"],
            "loss": task_dict["metrics"]["loss"],
            "correct": task_dict["metrics"]["correct_preds"],
            "CLPD": task_dict["metrics"]["log_prob_diff"],
        })
    res_df = pd.DataFrame(res)
    # res_df["throughput_inference"] = eval_metrics["throughput_inference"]
    
    # 添加 bin 列（按 interval 取整）-> 采样
    min_x = res_df[x].min()
    res_df["bin"] = ((res_df[x] - min_x) // interval).astype(int)
    # 对每个 bin 求平均 metric
    sampled_df = res_df.groupby("bin").agg({x: "mean", y: "mean"}).reset_index(drop=True)
    # If y == 'correct' and sampled_df[y] < 1, we need to scale it to [0, 100] (percentage)
    if y == "correct" and sampled_df[y].max() <= 1:
        sampled_df[y] = sampled_df[y] * 100
    if min_y is not None:
        sampled_df[y] = sampled_df[y].clip(lower=min_y)

    # if y not in sampled_df.columns:
    #     # Create y as the average of all metrics (CLPD, loss, correct)
    #     # Scale the CLP and loss to be in the same range as correct_preds (0-1) using min-max scaling
    #     sampled_df["CLPD"] = (sampled_df["CLPD"] - sampled_df["CLPD"].min()) / (sampled_df["CLPD"].max() - sampled_df["CLPD"].min())
    #     sampled_df["loss"] = (sampled_df["loss"] - sampled_df["loss"].min()) / (sampled_df["loss"].max() - sampled_df["loss"].min())
    #     sampled_df[y] = (sampled_df["CLPD"] + sampled_df["loss"] + sampled_df["correct"]) / 3

    # 可视化
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    if sort_y_values:
        ax.plot(sampled_df[x], sampled_df[y].sort_values(), label=label, color=color, marker=marker)
    else:
        ax.plot(sampled_df[x], sampled_df[y], label=label, color=color, marker=marker)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    else:
        ax.set_xlabel(x, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    else:
        ax.set_ylabel(y, fontsize=label_fontsize)
    if set_title:
        ax.set_title(f"{y} vs {x} without retraining", fontsize=label_fontsize)
    # Set y grid with certain interval
    if set_grid:
        ax.grid(set_grid, color='gray', linewidth=0.5, axis='y')
        # 分成 5 份
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)



def load_dataset_v1(retrain_rates, arrival_rates, datasets, models, methods, num_samples: int = 1000) -> pd.DataFrame:
    data = []
    for model in models:
        for method in methods:
            for dataset in datasets:
                for retrain_rate in retrain_rates:
                    for arrival_rate in arrival_rates:
                        try:
                            if method == "async":
                                res = json.load(open(f"profile_main/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json"))
                            else:
                                res = json.load(open(f"profile_new/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json"))
                            # res = json.load(open(f"profile_main/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_{exp}.json"))
                        except FileNotFoundError:
                            print(f"File not found: profile_main/1000/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json")
                            continue
                        data.append({
                            "retrain_rate": retrain_rate,
                            "arrival_rate": arrival_rate,
                            "TTFT": res["TTFT"],
                            "TBT": res["TBT"],
                            "total_time": res['total_time'],
                            "throughput_inference": res['throughput_inference'],
                            "Win rate": res["eval_metrics"]["preference accuracy"],
                            "PPL": res["eval_metrics"]["PPL"],
                            "CLPD": res["eval_metrics"]["CLPD"],
                            "NLL": res["eval_metrics"]["loss"],
                            "ROUGE": res["eval_metrics"]["rougeL"],
                            "BLEU": res["eval_metrics"]["bleu"],
                            "dataset": dataset,
                            "method": method,
                            "model": model,
                        })

    return pd.DataFrame(data)


def load_dataset_v2(retrain_rates, arrival_rates, datasets, models, methods, base_dir: str = "profile_main", num_samples: int = 1000) -> pd.DataFrame:
    data = []
    for model in models:
        for method in methods:
            for dataset in datasets:
                for retrain_rate in retrain_rates:
                    for arrival_rate in arrival_rates:
                        try:
                            if method == "async":
                                res = json.load(open(f"{base_dir}/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json"))
                            else:
                                res = json.load(open(f"{base_dir}/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json"))
                            # res = json.load(open(f"{base_dir}/{num_samples}/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_{exp}.json"))
                        except FileNotFoundError:
                            print(f"File not found: profile_main/1000/{dataset}/{model}/{method}_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json")
                            continue
                        TTFTs = [task["decode_times"][0] - task["execution_time"] for task in res["generation_results"] if task["workload"] == "decode" and task["decode_times"]]
                        data.append({
                            "retrain_rate": retrain_rate,
                            "arrival_rate": arrival_rate,
                            "TTFT": np.mean(TTFTs) if TTFTs else 0.0,  # res["TTFT"]
                            "TBT": res["TBT"],
                            "FT": res["FT"],
                            "total_time": res['total_time'],
                            "throughput_inference": res['throughput_inference'],
                            "Win rate": res["eval_metrics"]["preference accuracy"],
                            "PPL": res["eval_metrics"]["PPL"],
                            "CLPD": res["eval_metrics"]["CLPD"],
                            "NLL": res["eval_metrics"]["loss"],
                            "ROUGE": res["eval_metrics"]["rougeL"],
                            "BLEU": res["eval_metrics"]["bleu"],
                            "dataset": dataset,
                            "method": method,
                            "model": model,
                        })

    return pd.DataFrame(data)


def generate_alignment_latex_table(df, metrics, metric_display_names, retrain_map, datasets):
    """
    df: DataFrame with columns: ["dataset", "method", metric1, metric2, ...]
    metrics: List of column names to be included as metrics
    metric_display_names: Dict mapping from metric column name to LaTeX display name
    retrain_map: Dict like {"periodic": "P.", "sync": "sync"} to map method names
    datasets: List of dataset names to include in order
    """

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Alignment scores of periodic (P.) vs. continual (sync) retraining across two benchmark datasets.}")
    print("\\label{tab:alignment_metrics}")
    print("\\small")

    # Column spec: 1 for dataset name + 2 * metrics
    col_spec = "l" + "c" * (2 * len(metrics))
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    # Header row 1
    header_1 = ["\\textbf{Dataset}"]
    for metric in metrics:
        header_1 += [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{metric_display_names[metric]}}}}}"]
    print(" & ".join(header_1) + " \\\\")

    # Header row 2
    header_2 = [""] + ["P." if i % 2 == 0 else "sync" for i in range(2 * len(metrics))]
    for i in range(len(metrics)):
        start = 2 * i + 2
        end = start + 1
        print(f"\\cmidrule(lr){{{start}-{end}}}", end=" ")
    print()
    print(" & ".join(header_2) + " \\\\")
    print("\\midrule")

    for dataset in datasets:
        row = [dataset]
        for metric in metrics:
            for method_key in ["periodic-50", "sync"]:
                val = df[
                    (df["dataset"] == dataset) &
                    (df["method"] == method_key)
                ][metric].mean()
                row.append(f"{val:.2f}" if not pd.isna(val) else "--")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")



def plot_inference_throughput_over_time(
    file_path, 
    window=10.0, 
    label_fontsize: int = 18,
    legend_fontsize: int = 14,
    tick_fontsize: int = 12,
    ax: Optional[plt.Axes] = None, 
    method: Optional[str] = None,
):
    # Load JSON
    method = file_path.split("/")[-1].split("_")[0] if method is None else method
    color = 'tab:purple' if method not in LEGENDS2UTIL else LEGENDS2UTIL[method]["color"]
    marker = '^' if method not in LEGENDS2UTIL else LEGENDS2UTIL[method]["marker"]
    label = LEGENDS2UTIL[method]["name"] if method in LEGENDS2UTIL else method
    with open(file_path, "r") as f:
        data = json.load(f)

    decode_timestamps = []
    # Collect all decode timestamps
    for task in data["generation_results"]:
        if task["workload"] == "decode" and "decode_times" in task:
            decode_timestamps.extend(task["decode_times"])

    if not decode_timestamps:
        print("No decode steps found.")
        return

    # Convert to DataFrame
    decode_timestamps = sorted(decode_timestamps)
    df = pd.DataFrame({"timestamp": decode_timestamps})
    df["count"] = 1

    # Set timeline from beginning
    t0 = df["timestamp"].min()
    df["time_elapsed"] = df["timestamp"] - t0

    # Sliding window throughput calculation
    time_range = df["time_elapsed"].max()
    step = 1.0  # slide step in seconds
    times = []
    throughputs = []

    current = 0.0
    while current <= time_range:
        start = current
        end = current + window
        in_window = df[(df["time_elapsed"] >= start) & (df["time_elapsed"] < end)]
        throughput = len(in_window) / window
        times.append(current + window / 2)  # center point
        throughputs.append(throughput)
        current += step

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times, throughputs, color=color, label=label)
    ax.set_xlabel("Time [s]", fontsize=label_fontsize)
    ax.set_ylabel("Throughput [tok/s]", fontsize=label_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # plt.title(f"Inference Throughput Over Time (window={window}s)")
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.show()