import os
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset
import torch
import tqdm
import json
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import LlamaForCausalLM
from tasks import compute_perplexity


def get_sparse_tensor(tensor: torch.Tensor, sparsity: float, min_elements_per_row: int = 1) -> torch.Tensor:
    """Keep the top (1 - sparsity) elements globally. """
    # Flatten the tensor and sort globally
    _, flat_indices = torch.sort(tensor.flatten(), descending=True)
    # print("Flattened indices:\n", flat_indices)
    # Determine the number of elements to keep globally
    total_elements = tensor.numel()
    keep_count = round((1 - sparsity) * total_elements)
    # print(f"Number of elements to keep globally: {keep_count}")
    # Get the global top indices
    global_top_indices = flat_indices[:keep_count]
    # Map global indices back to row and column
    rows, cols = torch.div(global_top_indices, tensor.shape[1], rounding_mode='floor'), global_top_indices % tensor.shape[1]
    # Create a mask to apply the sparsity to the original tensor
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask[rows, cols] = True
    # Ensure each row has at least `s` elements
    for row in range(tensor.shape[0]):
        _, row_indices = torch.sort(tensor[row], descending=True)
        row_top_indices = row_indices[:min_elements_per_row]  # Top `s` indices in the row
        mask[row, row_top_indices] = True
    # Sparse tensor with only the top elements retained
    sparse_tensor = torch.where(mask, tensor, torch.tensor(0.0))
    return sparse_tensor


# Load LLaMA model and tokenizer
def main(args):
    model_name = args.model_name_or_path
    access_token = args.access_token
    batch_size = args.batch_size
    num_samples = args.num_samples
    # sparsity = args.sparsity
    save_intermediate_outputs = args.save_intermediate_outputs
    pruning_target = args.pruning_target
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

    results = {}
    # Load dataset
    if args.task == "language_modeling":
        # WikiText dataset
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Pick a subset and filter those 'text' strip are empty
        wiki_samples = min(num_samples, len(wiki))
        wiki = wiki.filter(lambda x: x["text"].strip()).select(range(wiki_samples))
        dataset = wiki

        # Evaluate perplexity
        predictions = dataset["text"]
        metrics = compute_perplexity(predictions, tokenizer, model, batch_size=5, max_length=max_length)
        print("[Dense model] perplexities: ", [round(x, 2) for x in metrics["perplexities"]])
        print("[Dense model] mean_perplexity: ", round(metrics["mean_perplexity"], 2)) 
        results['perplexity (dense)'] = metrics["mean_perplexity"]

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

    # Instantiate a average attention output norm matrix (num_layers, num_heads)
    avg_attn_norms, avg_mlp_norms = [], []
    for layer in tqdm.tqdm(layerwise_attn_output_norms):
        attn_output_norms = torch.cat(layerwise_attn_output_norms[layer], dim=0)  # shape: (N, num_heads)
        mlp_output_norms = torch.cat(layerwise_mlp_output_norms[layer], dim=0)  # shape: (N, hidden_dim)
        attn_weights = torch.cat([attn_weight[:, :, :min_seq_len, :min_seq_len] for attn_weight in layerwise_attn_weights[layer]], dim=0)  # shape: (N, num_heads, T, T)
        # print(f"Layer {layer}: attn_output_norms {attn_output_norms.shape}, mlp_output_norm {mlp_output_norms.shape}, attn_weights {attn_weights.shape}")
        # results[layer] = {"attn_sparsity": {}, "mlp_sparsity": {}}
        # for sparsity_threshold in [0.1, 0.5, 1.0, 2.0, 5.0]:
        #     attn_sparsity = (attn_output_norms <= sparsity_threshold).float().mean().item()
        #     mlp_sparsity = (mlp_output_norms <= sparsity_threshold).float().mean().item()
        #     results[layer]["attn_sparsity"][sparsity_threshold] = attn_sparsity
        #     results[layer]["mlp_sparsity"][sparsity_threshold] = mlp_sparsity

        # Save tensor to disk
        torch.save(attn_weights, f"{output_dir}/layer-{layer}_attn-weights.pt")
        torch.save(mlp_output_norms, f"{output_dir}/layer-{layer}_norms-mlp.pt")
        torch.save(attn_output_norms, f"{output_dir}/layer-{layer}_norms-attn.pt")
        avg_attn_norms.append(attn_output_norms.mean(dim=0))  # shape: (num_heads)
        avg_mlp_norms.append(mlp_output_norms.mean(dim=0))  # shape: (hidden_dim)

    # Save average norms
    avg_attn_norms = torch.stack(avg_attn_norms, dim=0)  # shape: (num_layers, num_heads)
    avg_mlp_norms = torch.stack(avg_mlp_norms, dim=0)  # shape: (num_layers, hidden_dim)
    torch.save(avg_attn_norms, f"{output_dir}/avg_norms-attn.pt")
    torch.save(avg_mlp_norms, f"{output_dir}/avg_norms-mlp.pt")

    for sparsity in np.arange(0.1, 1, 0.1):  # sparsity in [0.1, 0.9]
        # [Global] Rank reverse and keep (1 - sparsity) proportion of heads and neurons
        # sparse_attn_norms = get_sparse_tensor(avg_attn_norms, sparsity)  # shape: (num_layers, num_heads)
        # sparse_mlp_norms = get_sparse_tensor(avg_mlp_norms, sparsity)  # shape: (num_layers, hidden_dim)

        # # [Layer-wise] Rank reverse and keep (1 - sparsity) proportion of heads and neurons
        # _, attn_ranks = torch.sort(attn_output_norms.mean(dim=0), descending=True)  # attn_ranks (num_heads)
        # _, mlp_ranks = torch.sort(mlp_output_norms.mean(dim=0), descending=True)  # mlp_ranks (hidden_size)
        # active_attn_heads = attn_ranks[: round((1 - sparsity) * attn_ranks.shape[0])]  
        # active_mlp_neurons = mlp_ranks[: round((1 - sparsity) * mlp_ranks.shape[0])]

        # for layer, (layer_attn, layer_mlp) in enumerate(zip(sparse_attn_norms, sparse_mlp_norms)):
        for layer, (attn_norm, mlp_norm) in enumerate(zip(avg_attn_norms, avg_mlp_norms)):
            if pruning_target == "attn" or pruning_target == "all":
                # active_attn_heads = torch.nonzero(layer_attn, as_tuple=True)[0]
                _, attn_ranks = torch.sort(attn_norm, descending=True)  # attn_ranks (num_heads)
                active_attn_heads = attn_ranks[: round((1 - sparsity) * attn_ranks.shape[0])]
                model.model.layers[layer].active_heads = active_attn_heads
                print(f"[After pruning] Layer {layer} active_attn_heads: {active_attn_heads.shape[0]}")
            if pruning_target == "mlp" or pruning_target == "all":
                # active_mlp_neurons = torch.nonzero(layer_mlp, as_tuple=True)[0]
                _, mlp_ranks = torch.sort(mlp_norm, descending=True)  # mlp_ranks (hidden_size)
                active_mlp_neurons = mlp_ranks[: round((1 - sparsity) * mlp_ranks.shape[0])]
                model.model.layers[layer].active_neurons = active_mlp_neurons
                print(f"[After pruning] Layer {layer} active_mlp_neurons: {active_mlp_neurons.shape[0]}")

        # Evaluate perplexity
        predictions = dataset["text"]
        metrics = compute_perplexity(predictions, tokenizer, model, batch_size=5, max_length=max_length)
        model.model.reset_sparsity()  # Reset sparsity
        if pruning_target == "mlp":
            print(f'perplexities (MLP-sparse-{sparsity}): ', [round(x, 2) for x in metrics["perplexities"]])
            print(f'mean perplexity (MLP-sparse-{sparsity}): ', round(metrics["mean_perplexity"], 2)) 
            results[f'perplexity (MLP-sparse-{sparsity})'] = metrics["mean_perplexity"]
        elif pruning_target == "attn":
            print(f'perplexities (ATTN-sparse-{sparsity}): ', [round(x, 2) for x in metrics["perplexities"]])
            print(f'mean perplexity (ATTN-sparse-{sparsity}): ', round(metrics["mean_perplexity"], 2)) 
            results[f'perplexity (ATTN-sparse-{sparsity})'] = metrics["mean_perplexity"]
        else:
            print(f'perplexities (sparse-{sparsity}): ', [round(x, 2) for x in metrics["perplexities"]])
            print(f'mean perplexity (sparse-{sparsity}): ', round(metrics["mean_perplexity"], 2)) 
            results[f'perplexity (sparse-{sparsity})'] = metrics["mean_perplexity"]

    # Save results to json
    # with open(f"{output_dir}/eval_metrics.json", "w") as f:
    #     json.dump(results, f, indent=4)
    json_file_path = f"{output_dir}/eval_metrics.json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    existing_data.update(results)
    with open(json_file_path, "w") as f:
        json.dump(existing_data, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--access_token", type=str, default="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")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--task", type=str, default="language_modeling", choices=["language_modeling", "QA"])
    # parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--save_intermediate_outputs", action="store_true")
    parser.add_argument("--pruning_target", type=str, default="all", choices=["attn", "mlp", "all"])

    args = parser.parse_args()

    main(args)