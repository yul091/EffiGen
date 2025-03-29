# A definition of the bin packing class and supported functions.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Must be before torch is imported
import time
import sys 
sys.dont_write_bytecode = True
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
import torch
import pdb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    LogitsProcessorList, 
    MinLengthLogitsProcessor,
)
from transformers.cache_utils import DynamicCache, Cache
from peft import get_peft_model, LoraConfig
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
# import multiprocessing as mp
from iteration_task import Task
from alignment_study import DPOCollator, dpo_loss, compute_batch_metrics
from utils import prepare_inputs, save_metrics_with_order


class IterationBin:

    def __init__(
        self,
        tokenizer: LlamaTokenizer,
        inference_input_feature: str = "input_ids",
        inference_mask_feature: str = "attention_mask",
        eval_metrics: bool = False,
        batch_collator: Optional[DPOCollator] = None,
    ):
        self.inference_input_feature = inference_input_feature
        self.inference_mask_feature = inference_mask_feature
        self.prefill_batch: List[Task] = []
        self.decode_batch: List[Task] = []
        self.train_batch: List[Task] = []
        self.tokenizer = tokenizer
        self.batch_collator = batch_collator if batch_collator is not None else \
            DPOCollator(
                tokenizer, 
                inference_input_feature=inference_input_feature, 
                inference_mask_feature=inference_mask_feature,
            )
        self.eval_metrics = eval_metrics
        

    def add_task(self, task: Task):
        if task.workload == "prefill":
            self.prefill_batch.append(task)
        elif task.workload == "decode":
            self.decode_batch.append(task)
        elif task.workload == "train":
            self.train_batch.append(task)
        else:
            # self.test_batch.append(task)
            raise ValueError(f"Invalid workload: {task.workload}")
        

    def _create_batch(
        self,
        batch: List[Task],
        device: Optional[str] = "cuda",
    ):  
        if not batch:
            return None
        inputs = [task.input_kwargs for task in batch]
        inputs = self.batch_collator(inputs)
        inputs["past_key_values"] = None

        # For inference-only: pad key values with varient sequence lengths
        split_caches = [task.past_key_values for task in batch]
        if all(c is None for c in split_caches):  # Handle case where all caches are None
            return prepare_inputs(inputs, device=device)
       
        ref_cache = next(c for c in split_caches if c is not None)
        if ref_cache is not None:
            attention_mask = inputs[self.inference_mask_feature][:, :-1]  # Remove the newly decoded token
            batch_size, max_seq_len = attention_mask.shape
            num_heads, _, head_dim  = ref_cache.key_cache[0].shape
            inputs["past_key_values"] = DynamicCache()

            for layer_idx in range(len(ref_cache.key_cache)):
                key_tensor = torch.zeros(
                    (batch_size, num_heads, max_seq_len, head_dim),
                    dtype=ref_cache.key_cache[layer_idx].dtype,
                    device=ref_cache.key_cache[layer_idx].device,
                )
                value_tensor = torch.zeros_like(key_tensor)

                for i in range(batch_size):
                    if split_caches[i] is None:
                        continue  # Skip samples without KV cache (zero)
                    # print(f"Task {i} (output) cache size: {split_caches[i].key_cache[layer_idx].shape}")
                    mask = attention_mask[i] == 1
                    key_tensor[i, :, mask, :] = split_caches[i].key_cache[layer_idx]  # [H, valid_len, D]
                    value_tensor[i, :, mask, :] = split_caches[i].value_cache[layer_idx]
                inputs["past_key_values"].update(key_tensor, value_tensor, layer_idx)

        return prepare_inputs(inputs, device=device)
    

    def _batch_decoding(
        self,
        batch: List[Task],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lm_logits: torch.Tensor,
        task_queue: PriorityQueue,
        logits_processor: Callable,
        max_length: Optional[int] = None,
        past_key_values: Optional[DynamicCache] = None,
    ):
        max_length = max_length if max_length is not None else 1024
        # Finished sentences should have their next token be a padding token
        batch_size = lm_logits.shape[0]  # shape: [B, S, V]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=lm_logits.device)
        next_token_logits = lm_logits[:, -1, :]  # B X V
        # Pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # Greedy decoding (B)
        # Update unfinished sequences
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.eos_token_id).long())
        # stoppings = stopping_criteria(input_ids, next_tokens_scores)
    
        # Update task status
        for i, task in enumerate(batch):
            task.update_decoding(next_tokens[i].item())
            # if unfinished_sequences[i] == 1 and stoppings[i] == True:  # continue decoding
            if unfinished_sequences[i] == 1 and task.step < max_length: # continue decoding
                # Update past key values [batch_size, num_heads, seq_len, head_dim]
                if past_key_values is None:  
                    continue
                # Split and update individual task's KV cache
                mask = attention_mask[i] == 1
                task.past_key_values = DynamicCache()
                for layer_idx in range(len(past_key_values.key_cache)):
                    task.past_key_values.update(
                        past_key_values.key_cache[layer_idx][i, :, mask, :],  # [H, S_valid, D]
                        past_key_values.value_cache[layer_idx][i, :, mask, :],
                        layer_idx=layer_idx,
                    )
                # print(f"Cache size (new): {task.past_key_values.key_cache[0].shape}")
                # Add task back to queue
                task_queue.put((task.get_priority(initial=False), task.taskID))
            else:  # stop decoding
                # Update response (text) with next token
                task.get_response(self.tokenizer)
                


    def execute(
        self,
        batch: List[Task],
        model: LlamaForCausalLM,
        workload: str,
        task_queue: PriorityQueue,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not batch:
            return 
        
        inputs = self._create_batch(batch, device=model.device)
        if workload == "train":
            model.train()
            optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=5e-5)
            optimizer.zero_grad()
            losses = dpo_loss(model, inputs, return_average=False)
            losses.mean().backward()
            optimizer.step()
            # Update task status
            for i, task in enumerate(batch):
                task.metrics["loss"] = losses[i].item()
        else:
            model.eval()
            generation_config, model_kwargs = model._prepare_generation_config(generation_config, **inputs)
            with torch.no_grad():
                model_kwargs = model._get_initial_cache_position(model_kwargs[self.inference_input_feature], model_kwargs)
                # Handle decoding forward pass
                model_forward = model.__call__
                if isinstance(model_kwargs.get("past_key_values"), Cache):
                    is_compileable = model_kwargs["past_key_values"].is_compileable and model._supports_static_cache
                    is_compileable = is_compileable and not model.generation_config.disable_compile
                    if is_compileable and (
                        model.device.type == "cuda" or generation_config.compile_config._compile_all_devices
                    ):
                        os.environ["TOKENIZERS_PARALLELISM"] = "0"
                        model_forward = model.get_compiled_call(generation_config.compile_config)
                # Slicing the inputs based on cache positions
                model_inputs = model.prepare_inputs_for_generation(**model_kwargs)
                # Forward pass
                if workload == "prefill":
                    outputs = model(
                        **model_inputs,
                        return_dict=True,
                    )  # [loss, logits, past_key_values, hidden_states, attentions]
                    if self.eval_metrics:
                        eval_outputs = compute_batch_metrics(model, inputs, compute_average=False)
                        for i, task in enumerate(batch):
                            for key, value in eval_outputs.items():
                                # print(f"idx {i}, key {key}, value {value.shape}")
                                task.metrics[key] = value[i].item()
                            # task.metrics.update(eval_outputs)
                else:
                    outputs = model_forward(
                        **model_inputs,
                        return_dict=True,
                    )

            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
                [MinLengthLogitsProcessor(1, eos_token_id=self.tokenizer.eos_token_id, device=model.device),]
            )
            # Decode the next tokens and update task status
            self._batch_decoding(
                batch=batch,
                input_ids=model_kwargs[self.inference_input_feature],
                attention_mask=model_kwargs[self.inference_mask_feature],
                lm_logits=outputs.logits,
                task_queue=task_queue,
                logits_processor=logits_processor,
                past_key_values=outputs.past_key_values,
            ) 

        # Clear the batch
        batch.clear()


    def concurrent_execute(
        self,
        model: LlamaForCausalLM,
        task_queue: PriorityQueue,
        strategy: str = "sync",
        max_workers: int = 3,
        **kwargs,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if strategy == "sync":
                # Prioritize the training workload
                if self.train_batch:
                    self.execute(self.train_batch, model, "train", task_queue, **kwargs)
                for workload, batch in [
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, workload, task_queue, **kwargs)

            elif strategy == "async":
                for workload, batch in [
                    ("train", self.train_batch),
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, workload, task_queue, **kwargs)
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Choose 'sync' or 'async'.")

    

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
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
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
    # print(f"task queue size: {task_queue.qsize()}")

    # Create iteration bin and execute tasks
    start = time.time()
    bin = IterationBin(tokenizer, eval_metrics=True)
    iteration = 0
    tokens = 0
    while task_queue.qsize() > 0:
        print(f"  **  Iteration {iteration} (queue size {task_queue.qsize()})  **  ")
        while task_queue.qsize() > 0:
            _, taskID = task_queue.get(timeout=0.5)
            bin.add_task(preloaded_tasks[taskID])
        # print(f"\t\tPrefill {[task.taskID for task in bin.prefill_batch]}, Decode {[task.taskID for task in bin.decode_batch]}, Train {[task.taskID for task in bin.train_batch]}")
        # bin.execute(bin.prefill_batch, model, workload='prefill', task_queue=task_queue, optimizer=optimizer)
        # bin.execute(bin.decode_batch, model, workload='decode', task_queue=task_queue, optimizer=optimizer)
        # bin.execute(bin.train_batch, model, workload='train', task_queue=task_queue, optimizer=optimizer)
        bin.concurrent_execute(
            model,
            task_queue,
            strategy=strategy,
            max_workers=3,
            optimizer=optimizer,
        )
        iteration += 1
        tokens += task_queue.qsize()
    end = time.time()

    print(f"Completed in {iteration} iterations! Total time: {end - start}, throughput: {tokens / (end - start)} tokens/sec")

    # Save the task's prompt and response to a file
    output_dir = "profile_main/dummy/Mistral-7B-Instruct-v0.2"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"output_cache_{strategy}.json")
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

    