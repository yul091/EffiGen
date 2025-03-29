# A dummy test for the iteration system.
import os
from typing import List
import time
from queue import PriorityQueue
import torch
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys 
sys.dont_write_bytecode = True
from iteration_task import Task  
from iteration_bin import Bin
from iteration_scheduler import Scheduler
from utils import save_metrics_with_order



# Define batch tokenization
def tokenize_and_align_labels(examples):
    """Tokenizes inputs in batch and masks out context for chosen labels."""
    contexts = examples["context"]
    chosens = examples["chosen_response"]
    rejecteds = examples["rejected_response"]

    # Tokenize all inputs in batch mode
    chosen_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, chosens)],
                                 truncation=True, padding=False,  # Use dynamic padding later during collation
                                 max_length=max_length)
    
    rejected_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, rejecteds)],
                                   truncation=True, padding=False,  # Use dynamic padding later during collation
                                   max_length=max_length)

    # Tokenize context separately (to get its length)
    context_encodings = tokenizer(contexts, truncation=True, padding=False, max_length=max_length)
    context_lengths = [len(enc) for enc in context_encodings["input_ids"]]

    # Create labels: Mask out context tokens by setting them to `-100`
    chosen_labels = [
        [-100] * ctx_len + chosen_encodings["input_ids"][i][ctx_len:]
        for i, ctx_len in enumerate(context_lengths)
    ]

    return {
        "context_input_ids": context_encodings["input_ids"],
        "context_attention_mask": context_encodings["attention_mask"],
        "chosen_input_ids": chosen_encodings["input_ids"],
        "chosen_attention_mask": chosen_encodings["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
    }



if __name__ == "__main__":
    
    # Verify the number of available CUDA devices
    strategy = "async"  # or "sync"
    attn_implementation = "flash_attention_2"  # or "triton"
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map="auto",
        use_cache=True,
        attn_implementation=attn_implementation,
    ).to("cuda:0")
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should list LoRA parameters as trainable
    max_length = model.config.max_position_embeddings
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # # Get batch collator
    # batch_collator = DPOCollator(
    #     tokenizer, 
    #     inference_input_feature="input_ids", 
    #     inference_mask_feature="attention_mask",
    # )
    
    # Load dataset 
    rlhf_data = load_dataset("data/Anthropic")
    test_dataset = rlhf_data["test"].select(range(5))
    train_dataset = rlhf_data["train"].select(range(5))
    processed_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=False,
    ).remove_columns(test_dataset.column_names)
    processed_train_dataset = train_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        load_from_cache_file=False,
    ).remove_columns(train_dataset.column_names)  
  
    # Preloaded tasks
    preloaded_tasks: List[Task] = []
    for i, test_example in enumerate(processed_test_dataset):
        task = Task(
            taskID=i,
            workload="prefill", 
            rate_lambda=10, 
            prompt=test_dataset[i]["context"],
            input_kwargs=test_example,
        )
        preloaded_tasks.append(task)
        
    accum = i+1
    for j, train_example in enumerate(processed_train_dataset):
        task = Task(
            taskID=accum+j,
            workload="train", 
            rate_lambda=10, 
            prompt=train_dataset[j]["context"],
            input_kwargs=train_example,
        )
        preloaded_tasks.append(task)

    # Simulate global scheduler
    task_queue = PriorityQueue()
    for taskID, task in enumerate(preloaded_tasks):
        task_queue.put((task.get_priority(initial=True), taskID))

    # Create iteration bin and execute tasks
    start = time.time()
    # bin = Bin(eval_metrics=True)
    scheduler = Scheduler(lambda1=0.5, lambda2=0.5)
    iteration = 0
    tokens = 0
    max_workers=3 if strategy == "async" else 2
    bin_kwargs = {
        "attn_implementation": attn_implementation,
        "max_workers": max_workers,
        "max_length": 1024,
        "strategy": strategy,
        "optimizer": optimizer,
        "eval_metrics": True,
    }
    while task_queue.qsize() > 0:
        print(f"  **  Iteration {iteration} (queue size {task_queue.qsize()})  **  ")
        tokens += task_queue.qsize()
        # while task_queue.qsize() > 0:
        #     _, taskID = task_queue.get(timeout=0.5)
        #     bin.add_task(preloaded_tasks[taskID])
        bin = scheduler.best_fit_allocate(task_queue, preloaded_tasks, model, **bin_kwargs)
        print(f"\t\tBin allocation (prefill_batch {len(bin.prefill_batch)}, decode_batch {len(bin.decode_batch)}, train_batch {len(bin.train_batch)})")
        bin.concurrent_execute(model, tokenizer, task_queue, **bin_kwargs)
        iteration += 1
        
    end = time.time()

    print(f"Completed in {iteration} iterations! Total time: {end - start}, throughput: {tokens / (end - start)} tokens/sec")

    # Save the task's prompt and response to a file
    output_dir = "profile_main/dummy/Mistral-7B-Instruct-v0.2"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_cache_{strategy}_binpacking.json")
    metrics = {
        "iteration": iteration,
        "total_time": end - start,
        "throughput": tokens / (end - start),
        "generation_results": [
            {
                "taskID": task.taskID,
                "prompt": task.prompt,
                "response": task.response,
                "workload": task.workload,
                "step": task.step,
                # "rate_lambda": task.rate_lambda,
                "metrics": task.metrics,
            }
            for task in preloaded_tasks
        ],
    }
    save_metrics_with_order(metrics, output_file)
    print(f"Metrics saved to {output_file}")