import os
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset
import torch
import tqdm
import numpy as np
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import LlamaForCausalLMWithOutput
from tasks import compute_perplexity



# Load LLaMA model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # "facebook/blenderbot-3B", "gpt2", "microsoft/DialoGPT-large", "meta-llama/Llama-2-7b-chat-hf"
access_token = "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"
batch_size = 3
num_samples = 100
device = 'cuda:2'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=access_token)
if 'gpt' in model_name or 'llama' in model_name:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    if 'llama' in model_name:
        model = LlamaForCausalLMWithOutput.from_pretrained(model_name, token=access_token)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token)
model.eval().to(device)  # Move model to GPU for faster evaluation
max_length = model.config.max_position_embeddings

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Pick a subset and filter those 'text' strip are empty
num_samples = min(num_samples, len(dataset))
dataset = dataset.select(range(num_samples)).filter(lambda x: x["text"].strip())

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Prepare data loader
def data_collator(batch):
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
    }

dataloader = torch.utils.data.DataLoader(
    tokenized_dataset, batch_size=batch_size, collate_fn=data_collator,
)

# Collect layer-wise attention and MLP output norms
layerwise_attention_norms = defaultdict(list)
layerwise_mlp_output_norms = defaultdict(list)

steps = len(dataloader)
for batch in tqdm.tqdm(dataloader, total=steps):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)

    for layer, (attn_norm, mlp_norm) in enumerate(zip(outputs.attention_norms, outputs.mlp_norms)):
        # offload to CPU and store
        layerwise_attention_norms[layer].append(attn_norm.cpu())  # shape: (B, num_heads)
        layerwise_mlp_output_norms[layer].append(mlp_norm.cpu())  # shape: (B, hidden_dim)

# Compute mean and std of norms
mname = model_name.split("/")[-1]
output_dir = f"profile_results/{mname}"
os.makedirs(output_dir, exist_ok=True)
results = {}
for layer in layerwise_attention_norms:
    attn_output_norms = torch.cat(layerwise_attention_norms[layer], dim=0)  # shape: (N, num_heads)
    mlp_output_norms = torch.cat(layerwise_mlp_output_norms[layer], dim=0)  # shape: (N, hidden_dim)
    results[layer] = {"attn_sparsity": {}, "mlp_sparsity": {}}
    for sparsity_threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
        attn_sparsity = (attn_output_norms <= sparsity_threshold).float().mean().item()
        mlp_sparsity = (mlp_output_norms <= sparsity_threshold).float().mean().item()
        results[layer]["attn_sparsity"][sparsity_threshold] = attn_sparsity
        results[layer]["mlp_sparsity"][sparsity_threshold] = mlp_sparsity

    # Save numpy arrays to disk
    attn_output_norms = attn_output_norms.numpy()
    mlp_output_norms = mlp_output_norms.numpy()
    np.save(f"{output_dir}/layer-{layer}_norms-attn.npy", attn_output_norms)
    np.save(f"{output_dir}/layer-{layer}_norms-mlp.npy", mlp_output_norms)

# Save results to json
with open(f"{output_dir}/layerwise_norms.json", "w") as f:
    json.dump(results, f, indent=4)