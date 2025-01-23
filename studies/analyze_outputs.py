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
from models import LlamaForCausalLM
from tasks import compute_perplexity



# Load LLaMA model and tokenizer
def main(args):
    model_name = args.model_name_or_path
    access_token = args.access_token
    batch_size = args.batch_size
    num_samples = args.num_samples
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=access_token)
    if 'gpt' in model_name or 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        if 'llama' in model_name:
            model = LlamaForCausalLM.from_pretrained(model_name, token=access_token, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token).to(device)
    model.eval()  # Move model to GPU for faster evaluation
    max_length = model.config.max_position_embeddings

    # Load dataset
    if args.task == "language_modeling":
        # WikiText dataset
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Pick a subset and filter those 'text' strip are empty
        wiki_samples = min(num_samples, len(wiki))
        wiki = wiki.filter(lambda x: x["text"].strip()).select(range(wiki_samples))
        dataset = wiki
    elif args.task == "QA":
        # OpenBookQA dataset
        openbookqa = load_dataset("openbookqa", split="test")
        openbookqa_samples = min(num_samples, len(openbookqa))
        openbookqa = openbookqa.select(range(openbookqa_samples)).filter(lambda x: x["question"].strip())
        dataset = openbookqa
    else:
        raise ValueError(f"Invalid task: {args.task}")

    # # C4 dataset
    # c4 = load_dataset("c4", "en", split="test")
    # c4_samples = min(num_samples, len(c4))
    # c4 = c4.select(range(num_samples)).filter(lambda x: x["text"].strip())

    # # Concatenate datasets
    # dataset = wiki.concatenate(c4)

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
    layerwise_attn_output_norms = defaultdict(list)
    layerwise_mlp_output_norms = defaultdict(list)
    layerwise_attn_weights = defaultdict(list)
    min_seq_len = max_length

    steps = len(dataloader)
    for batch in tqdm.tqdm(dataloader, total=steps):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, output_attentions=True)

        for layer, (attn_norm, mlp_norm, attn_weight) in enumerate(zip(outputs.attention_norms, outputs.mlp_norms, outputs.attentions)):
            # offload to CPU and store
            layerwise_attn_output_norms[layer].append(attn_norm.cpu())  # shape: (B, num_heads)
            layerwise_mlp_output_norms[layer].append(mlp_norm.cpu())  # shape: (B, hidden_dim)
            layerwise_attn_weights[layer].append(attn_weight.cpu())  # shape: (B, num_heads, T, T)
            assert attn_weight.shape[-1] == batch["input_ids"].shape[1]
        min_seq_len = min(min_seq_len, batch["input_ids"].shape[1])

    print(f"Min sequence length (for attentiom maps): {min_seq_len}")
    # Compute mean and std of norms
    mname = model_name.split("/")[-1]
    output_dir = f"profile_results/{args.task}/{mname}"
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    # Instantiate a average attention output norm matrix (num_layers, num_heads)
    avg_attn_norms, avg_mlp_norms = [], []
    for layer in layerwise_attn_output_norms:
        attn_output_norms = torch.cat(layerwise_attn_output_norms[layer], dim=0)  # shape: (N, num_heads)
        mlp_output_norms = torch.cat(layerwise_mlp_output_norms[layer], dim=0)  # shape: (N, hidden_dim)
        attn_weights = torch.cat([attn_weight[:, :, :min_seq_len, :min_seq_len] for attn_weight in layerwise_attn_weights[layer]], dim=0)  # shape: (N, num_heads, T, T)
        print(f"Layer {layer}: attn_output_norms {attn_output_norms.shape}, mlp_output_norm {mlp_output_norms.shape}, attn_weights {attn_weights.shape}")
        results[layer] = {"attn_sparsity": {}, "mlp_sparsity": {}}
        for sparsity_threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
            attn_sparsity = (attn_output_norms <= sparsity_threshold).float().mean().item()
            mlp_sparsity = (mlp_output_norms <= sparsity_threshold).float().mean().item()
            results[layer]["attn_sparsity"][sparsity_threshold] = attn_sparsity
            results[layer]["mlp_sparsity"][sparsity_threshold] = mlp_sparsity

        # Save numpy arrays to disk
        attn_output_norms = attn_output_norms.numpy()  # shape: (N, num_heads)
        mlp_output_norms = mlp_output_norms.numpy()  # shape: (N, hidden_dim)
        avg_attn_weight = attn_weights.numpy()  # shape: (N, num_heads, T, T)
        np.save(f"{output_dir}/layer-{layer}_norms-attn.npy", attn_output_norms)
        np.save(f"{output_dir}/layer-{layer}_norms-mlp.npy", mlp_output_norms)
        np.save(f"{output_dir}/layer-{layer}_attn-weights.npy", avg_attn_weight)

        avg_attn_norms.append(attn_output_norms.mean(axis=0))  # shape: (num_heads)
        avg_mlp_norms.append(mlp_output_norms.mean(axis=0))

    # Save average norms to disk
    avg_attn_norms = np.stack(avg_attn_norms, axis=0)  # shape: (num_layers, num_heads)
    avg_mlp_norms = np.stack(avg_mlp_norms, axis=0)  # shape: (num_layers, hidden_dim)
    np.save(f"{output_dir}/avg_norms-attn.npy", avg_attn_norms)
    np.save(f"{output_dir}/avg_norms-mlp.npy", avg_mlp_norms)

    # Save results to json
    with open(f"{output_dir}/layerwise_norms.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--task", type=str, default="language_modeling", choices=["language_modeling", "QA"])

    args = parser.parse_args()

    main(args)