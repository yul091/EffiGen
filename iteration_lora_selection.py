# This script is responsible for selecting LORA layers for update based on specific criteria.

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


def get_lora_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return a list of (layer_name, module) for LoRA-affected layers only.
    """
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            layers.append((name, module))
    return layers

def compute_rgn(layers: List[Tuple[str, nn.Module]]) -> Dict[str, float]:
    layer_values = {}
    for name, layer in layers:
        total_grad_norm = 0
        total_param_norm = 0
        for param_name, param in layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                total_grad_norm += grad_norm
                total_param_norm += param_norm
        rgn = total_grad_norm / (total_param_norm + 1e-8)
        layer_values[name] = rgn
    return layer_values

def compute_snr(layers: List[Tuple[str, nn.Module]]) -> Dict[str, float]:
    layer_values = {}
    for name, layer in layers:
        grads = []
        for param_name, param in layer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.view(-1))
        if grads:
            all_grads = torch.cat(grads)
            mean = all_grads.mean()
            var = all_grads.var()
            snr = (mean ** 2 / (var + 1e-8)).item()
            layer_values[name] = snr
        else:
            layer_values[name] = 0.0
    return layer_values

def normalize(values: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize the values to the range [0, 1].
    """
    v = np.array(list(values.values()))
    min_v, max_v = v.min(), v.max()
    return {k: (v_ - min_v) / (max_v - min_v + 1e-8) for k, v_ in values.items()}

def select_top_k_layers(layer_values: Dict[str, float], top_ratio: float = 0.3) -> List[str]:
    """
    Select top K layers based on the normalized values.
    """
    sorted_items = sorted(layer_values.items(), key=lambda x: x[1], reverse=True)
    top_k = int(np.ceil(len(sorted_items) * top_ratio))
    return [name for name, _ in sorted_items[:top_k]]

def freeze_unselected_layers(layers: List[Tuple[str, torch.nn.Module]], selected_names: List[str]):
    """
    Freeze layers that are not selected for training.
    """
    for name, layer in layers:
        requires_grad = name in selected_names
        for param in layer.parameters():
            param.requires_grad = requires_grad

def selective_training(
    model: torch.nn.Module,
    losses: torch.Tensor,
    loss_threshold: Optional[float] = None,
    layer_selection: Optional[str] = None,  # "RGN" or "SNR" or None
    layer_threshold: float = 0.3,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Optional[torch.Tensor]:
    """
    Perform selective training: sample selection + gradient-based layer selection.
    """
    if loss_threshold is not None:
        selected_idx = (losses > loss_threshold).nonzero(as_tuple=True)[0]
        if len(selected_idx) == 0:
            return None  # skip step
        losses = losses[selected_idx]
    
    losses.mean().backward()

    if layer_selection:
        layers = get_lora_layers(model)
        if layer_selection == 'RGN':
            layer_values = compute_rgn(layers)
        elif layer_selection == 'SNR':
            layer_values = compute_snr(layers)
        else:
            raise ValueError(f"Unknown layer_selection: {layer_selection}")

        layer_values = normalize(layer_values)
        selected_layers = select_top_k_layers(layer_values, top_ratio=layer_threshold)
        freeze_unselected_layers(layers, selected_layers)

        # Optional: log
        print(f"Selected {len(selected_layers)} / {len(layer_values)} LoRA layers")
        print(f"Top layers: {selected_layers[:5]}")

    optimizer.step()
    return losses.detach()




if __name__ == "__main__":
    import sys 
    sys.dont_write_bytecode = True
    import time
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from iteration_producer import Producer
    from iteration_bin import Bin
    from alignment_study import dpo_loss

    arrival_rate = 5
    retrain_rate = 1.0
    n_test_samples = 20
    arrival_pattern = "poisson"
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    max_context_length = 1024
    strategy = "sync"
    data_path = "data/Anthropic"
    device = 0
    lr = 5e-5
    attn_implementation = "flash_attention_2"

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": device},
        use_cache=True,
        attn_implementation=attn_implementation,
    )

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        task_type="CAUSAL_LM", 
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    max_context_length = model.config.max_position_embeddings

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Get producer (loading dataset)
    producer = Producer(
        arrival_rate=arrival_rate, 
        retrain_rate=retrain_rate, 
        n_test_samples=n_test_samples,
        strategy=strategy,
        arrival_pattern=arrival_pattern, 
    )
    preloaded_tasks = producer.load_dataset(
        tokenizer=tokenizer,
        max_length=max_context_length,
        dataset_name=data_path,
    )

    # Test a single batch
    bin = Bin(
        strategy,
        device=device,
    )
    for task in preloaded_tasks[:3]:
        bin.add_task(task, model, attn_implementation)
    inputs = bin._create_batch(bin.train_batch, tokenizer)

    execution_time = time.time()
    model.train()
    optimizer.zero_grad()
    losses = dpo_loss(model, inputs, return_average=False)
    # losses.mean().backward()
    # optimizer.step()
    losses = selective_training(
        model=model,
        losses=losses,
        loss_threshold=None,
        layer_selection="RGN",
        layer_threshold=0.3,
        optimizer=optimizer,
    )

    # Update task status
    for i, task in enumerate(bin.train_batch):
        task.metrics["loss"] = losses[i].item()
        task.execution_time = execution_time
        print(f"Task {task.taskID} ({task.workload}) finished with loss {task.metrics['loss']}")