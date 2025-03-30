# A dummy test for the iteration system.
import os
from typing import List
import time
from queue import PriorityQueue
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys 
sys.dont_write_bytecode = True
from iteration_task import Task  
from iteration_producer import Producer
from iteration_scheduler import Scheduler
from utils import save_metrics_with_order




if __name__ == "__main__":    
    
    # Verify the number of available CUDA devices
    device = 1
    strategy = "async"  # or "sync"
    attn_implementation = "flash_attention_2"  # or "triton"
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    arrival_rate = 10
    n_test_samples = 100
    retrain_rate = 0.2
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
    ).to(device)
    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should list LoRA parameters as trainable
    max_length = model.config.max_position_embeddings
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Get producer (loading dataset and pushing tasks to queue)
    producer = Producer(
        arrival_rate=arrival_rate, 
        retrain_rate=retrain_rate, 
        arrival_pattern="poisson", 
        n_test_samples=n_test_samples,
    )
    preloaded_tasks = producer.load_dataset(
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_name="data/Anthropic",
    )

    # Create iteration bin and execute tasks
    start = time.time()
    # bin = Bin(eval_metrics=True)
    task_queue = PriorityQueue()
    # Produce tasks
    producer.produce(task_queue, preloaded_tasks)
    end = time.time()


    # scheduler = Scheduler(lambda1=0.5, lambda2=0.5)
    # max_workers=3 if strategy == "async" else 2
    # bin_kwargs = {
    #     "attn_implementation": attn_implementation,
    #     "max_workers": max_workers,
    #     "max_length": 1024,
    #     "strategy": strategy,
    #     "optimizer": optimizer,
    #     "eval_metrics": True,
    # }
    # iteration = 0
    # tokens = 0
    # while task_queue.qsize() > 0:
    #     print(f"  **  Iteration {iteration} (queue size {task_queue.qsize()})  **  ")
    #     tokens += task_queue.qsize()
    #     # while task_queue.qsize() > 0:
    #     #     _, taskID = task_queue.get(timeout=0.5)
    #     #     bin.add_task(preloaded_tasks[taskID])
    #     bin = scheduler.best_fit_allocate(task_queue, preloaded_tasks, model, **bin_kwargs)
    #     print(f"\tBin allocation (prefill_batch {len(bin.prefill_batch)}, decode_batch {len(bin.decode_batch)}, train_batch {len(bin.train_batch)})")
    #     bin.concurrent_execute(model, tokenizer, task_queue, **bin_kwargs)
    #     iteration += 1
    #     break
        
    # end = time.time()
    # print(f"Completed in {iteration} iterations! Total time: {end - start}, throughput: {tokens / (end - start)} tokens/sec")

    # # Save the task's prompt and response to a file
    # output_dir = "profile_main/dummy/Mistral-7B-Instruct-v0.2"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"output_cache_{strategy}_binpacking.json")
    # metrics = {
    #     "iteration": iteration,
    #     "total_time": end - start,
    #     "throughput": tokens / (end - start),
    #     "generation_results": [
    #         {
    #             "taskID": task.taskID,
    #             "workload": task.workload,
    #             "prompt": task.prompt,
    #             "response": task.response,
    #             "step": task.step,
    #             "metrics": task.metrics,
    #         }
    #         for task in preloaded_tasks
    #     ],
    # }
    # save_metrics_with_order(metrics, output_file)
    # print(f"Metrics saved to {output_file}")