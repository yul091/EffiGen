import os
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset, Dataset
import torch
import tqdm
import json
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, MixtralForCausalLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tasks import compute_perplexity, compute_QA_accuracy
from models import patch_mixtral_model_forward_with_outputs, modify_indexed_weights


def eval_metrics(task: str, model: MixtralForCausalLM, dataset: Dataset, tokenizer: AutoTokenizer, max_length=None, batch_size=5):
    if task == "language_modeling":
        predictions = dataset["text"]
        metrics = compute_perplexity(predictions, tokenizer, model, batch_size=batch_size, max_length=max_length)
        # print(f'perplexities: ', [round(x, 2) for x in metrics["perplexities"]])
        print(f'Perplexity: ', round(metrics["mean_perplexity"], 2)) 
        # results['perplexity'] = metrics["mean_perplexity"]
    elif task == "QA":
        metrics = compute_QA_accuracy(model, dataset, tokenizer)
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        # results["accuracy"] = metrics["accuracy"]
    return metrics



def get_dataset(task, tokenizer, batch_size=1, max_length=None, num_samples=100):
    # Load dataset
    if task == "language_modeling":
        # WikiText dataset
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Pick a subset and filter those 'text' strip are empty
        if num_samples > 0:
            wiki_samples = min(num_samples, len(wiki))
            wiki = wiki.filter(lambda x: x["text"].strip()).select(range(wiki_samples))
        dataset = wiki

        def tokenize_function(example):
            return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    elif task == "QA":
        # OpenBookQA dataset
        openbookqa = load_dataset("allenai/openbookqa", split="test")
        if num_samples > 0:
            openbookqa_samples = min(num_samples, len(openbookqa))
            openbookqa = openbookqa.filter(lambda x: x["question_stem"].strip()).select(range(openbookqa_samples))
        dataset = openbookqa

        def tokenize_function(example):
            question = example["question_stem"]
            choices = example["choices"]["text"]
            labels = example["choices"]["label"]
            # Format the input prompt
            formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
            prompt = f"Question: {question}\nChoices:\n{formatted_choices}\nAnswer:"
            # Tokenize the prompt
            tokenized = tokenizer(prompt, truncation=True, padding=True, max_length=max_length)
            return tokenized
        tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)

    else:
        raise ValueError(f"Invalid task: {args.task}")
    
    # def data_collator(batch):
    #     return {
    #     "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
    #     "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
    # }
    # Use padding side left
    def data_collator(batch):
        return tokenizer.pad(batch, max_length=max_length, return_tensors="pt", padding=True,)

    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=data_collator,
    )
    return dataloader, dataset



# Load LLaMA model and tokenizer
def prune_model(model: MixtralForCausalLM, dataloader: DataLoader, sparsity: float=0.1, pruning_target: str = "mlp", device: str = "cuda"):

    layerwise_expert_output_norms = {}
    for step, batch in enumerate(tqdm.tqdm(dataloader, total=len(dataloader), desc="Computing expert norms")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        for layer, expert_norm_dict in enumerate(outputs.expert_norms):
            if step == 0:  # Initialize the dict for the first batch
                layerwise_expert_output_norms[layer] = defaultdict(list)
            for expert_id, expert_norm in expert_norm_dict.items():
                layerwise_expert_output_norms[layer][expert_id].append(expert_norm)

    # Cat and average norm
    for layer, expert_norm_dict in layerwise_expert_output_norms.items():
        for expert_id, expert_norms in expert_norm_dict.items():
            expert_norm_dict[expert_id] = torch.stack(expert_norms).mean(dim=0)  # (hidden_dim,)

    # Compute average expert norms and prune
    for layer, expert_norm_dict in tqdm.tqdm(layerwise_expert_output_norms.items(), desc=f"Pruning {pruning_target} with sparsity {sparsity}", total=len(layerwise_expert_output_norms)):
        for expert_id, expert_norm in expert_norm_dict.items():
            neuron_mask = expert_norm > expert_norm.topk(int(sparsity * expert_norm.shape[0]), largest=False).values[-1]
            # Prune mlp neurons
            modify_indexed_weights(model.model.layers[layer].block_sparse_moe.experts[expert_id], neuron_mask)

    


if __name__ == "__main__":
    import argparse
    from utils import MODEL2PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--task", type=str, default="language_modeling", choices=["language_modeling", "QA"])
    parser.add_argument("--sparsity", type=float, default=0.2)
    # parser.add_argument("--save_intermediate_outputs", action="store_true")
    parser.add_argument("--pruning_target", type=str, default="mlp", choices=["attn", "mlp", "all"])
    args = parser.parse_args()

    patch_mixtral_model_forward_with_outputs()

    model_name = args.model_name_or_path
    model_path = MODEL2PATH[model_name]
    batch_size = args.batch_size
    num_samples = args.num_samples
    sparsity = args.sparsity
    pruning_target = args.pruning_target
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        padding_side="left"
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()  # Move model to GPU for faster evaluation

    try:  # 8 experts, 2 experts per token
        print(f"Number of experts: {model.config.num_local_experts}, max experts per token: {model.config.num_experts_per_tok}")
    except AttributeError:
        pass
    max_length = model.config.max_position_embeddings

    dataloader, dataset = get_dataset(args.task, tokenizer, batch_size=batch_size, max_length=max_length, num_samples=num_samples)
    # Evaluate the model
    eval_metrics(args.task, model, dataset, tokenizer, max_length=max_length, batch_size=batch_size)
    # Prune the model
    prune_model(model, dataloader, sparsity=sparsity, pruning_target=pruning_target, device=device)
    # Evaluate the pruned model
    eval_metrics(args.task, model, dataset, tokenizer, max_length=max_length, batch_size=batch_size)