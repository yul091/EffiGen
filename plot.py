import os
import math
import random
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from typing import Union, List, Dict, Optional, Callable
from token_prioritization import compute_token_scores, create_prompt, preprocess_and_tokenize
from utils import smooth_curve, plot_performance_over_time, load_dataset_v1, load_dataset_v2, generate_alignment_latex_table, read_json, LEGENDS2UTIL, VARS2UTIL, SCORES2UTIL


# Acc vs Time / Retrain rate / #GPUs
fig, axes = plt.subplots(1, 3, figsize=(15, 3.3))
dataset, model, retrain_rate, arrival_rate = "StanfordNLP", "Mistral-7B-Instruct-v0.2", 0.5, 10.0
plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-50_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="correct", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
# plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-20_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="correct", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
# plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-10_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="correct", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/sync_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="correct", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
plot_performance_over_time(f"profile_main/1000/{dataset}/{model}/async_retrain-{retrain_rate}_lambda-{arrival_rate}.json", x="taskID", y="correct", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
# plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/async_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="correct", interval=10, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)


# plot_performance_over_time("mock_profile/periodic.json", x="taskID", y="score", interval=50, ax=axes[0], method="period", xlabel="Time [s]", ylabel="Score (%)")
# plot_performance_over_time("mock_profile/sync.json", x="taskID", y="score", interval=50, ax=axes[0], method="sync", xlabel="Time [s]", ylabel="Score (%)")
retrain_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
arrival_rates = [5.0, 10.0, 15.0, 20.0,]
datasets = ["StanfordNLP", "Anthropic"]
methods = ["period-50", "period-20", "period-10", "sync", "async"]
models = ["Mistral-7B-Instruct-v0.2",] # "Meta-Llama-3-8B-Instruct",]
df = load_dataset_v2(retrain_rates, arrival_rates, datasets, models, methods, num_samples=1000)

# Throughput vs Time / Retrain rate / #GPUs
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
dataset, model, retrain_rate, arrival_rate = "StanfordNLP", "Mistral-7B-Instruct-v0.2", 0.5, 10.0
plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-50_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="throughput_inference", interval=50, method='period', ax=axes[0], xlabel="Time [s]", ylabel="Throughput [tok/s]")
# plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-20_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
# plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/period-10_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", set_grid=True)
plot_performance_over_time(f"profile_new/1000/{dataset}/{model}/sync_retrain-{retrain_rate}_lambda-{arrival_rate}_0.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Throughput [tok/s]")
plot_performance_over_time(f"profile_main/1000/{dataset}/{model}/sync_retrain-{retrain_rate}_lambda-{arrival_rate}.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Throughput [tok/s]")
# plot_performance_over_time("mock_profile/periodic.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], method="period", xlabel="Time [s]", ylabel="Win rate (%)")
# plot_performance_over_time("mock_profile/sync.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], method="sync", xlabel="Time [s]", ylabel="Win rate (%)")
# plot_performance_over_time(f"mock_profile/async.json", x="taskID", y="throughput_inference", interval=50, ax=axes[0], xlabel="Time [s]", ylabel="Score (%)", method="async", sort_y_values=True)

# Retrain rate vs. metric for different methods
metric = "throughput_inference"
m2m = {'period': 'sync', 'sync': 'period-50', 'async': 'async'}
for method in m2m:
    # method_data = df[(df["method"] == method) & (df["dataset"] == "StanfordNLP")]
    # Retrain rate * 100 & Win rate * 100
    method_data = df[(df["method"] == m2m[method])].groupby("retrain_rate")[metric].mean().reset_index()
    axes[1].plot(
        method_data["retrain_rate"] * 100, 
        method_data[metric].sort_values() * 100, 
        label=LEGENDS2UTIL[method]['name'], 
        color=LEGENDS2UTIL[method]['color'], 
        # linestyle=LEGENDS2UTIL[method]['linestyle'], 
        marker=LEGENDS2UTIL[method]['marker']
    )
        
axes[1].set_xlabel("Retrain rate (%)", fontsize=18)
axes[1].set_ylabel("Avg. throughput [tok/s]", fontsize=18)
axes[1].legend(fontsize=14)
# axes[1].sharey(axes[0])
# axes[1].grid(True, color='gray', linewidth=0.5, axis='y')
# Only integer y-ticks
axes[1].yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
axes[1].tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
fig.subplots_adjust(wspace=0.3)
# plt.savefig("figures/main_throughput.pdf", bbox_inches="tight")
plt.show()