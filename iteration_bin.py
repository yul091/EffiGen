# A definition of the bin packing class and supported functions.
import os
import gc
import logging
import sys 
sys.dont_write_bytecode = True
from typing import List, Optional, Dict, Any, Callable
import time
import torch
import torch.nn as nn
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    LogitsProcessorList, 
    MinLengthLogitsProcessor,
)
from transformers.cache_utils import DynamicCache, Cache
from concurrent.futures import ThreadPoolExecutor
from iteration_task import Task
from iteration_queue import IterQueue
from alignment_study import DPOCollator, dpo_loss, compute_batch_metrics
from utils import prepare_inputs


class Bin:

    def __init__(
        self,
        device: int = 0,
        inference_input_feature: str = "input_ids",
        inference_mask_feature: str = "attention_mask",
        eval_metrics: bool = True,
    ):
        self.inference_input_feature = inference_input_feature
        self.inference_mask_feature = inference_mask_feature
        self.prefill_batch: List[Task] = []
        self.decode_batch: List[Task] = []
        self.train_batch: List[Task] = []
        self.eval_metrics = eval_metrics
        # Initialize memory and latency
        self.memory_capacity = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
        self.total_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
        self.max_latency = 0
        self.free_memory = self.memory_capacity
        

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
        

    def get_num_tasks(self) -> int:
        return len(self.prefill_batch) + len(self.decode_batch) + len(self.train_batch)
        

    def update_workload(self, operation: str = "add", memory: float = 0, latency: float = 0):
        """
        Estimate the resource consumption of all workloads in the bin. Update the free memory and max latency.
        """
        if operation == "add":
            # Calculate the total memory used by the tasks in the bin
            # memory, latency = task.get_workload()
            self.total_memory += memory
            self.max_latency = max(self.max_latency, latency)
            if self.free_memory < memory:
                logging.warning(f"Memory overflow! Preallocate: {self.total_memory} MB, capacity: {self.memory_capacity} MB")
            self.free_memory = max(self.memory_capacity - self.total_memory, 0)
        elif operation == "clear":
            self.total_memory = 0
            self.max_latency = 0
            self.free_memory = self.memory_capacity
        else:
            raise ValueError(f"Invalid operation: {operation}. Choose 'add' or 'clear'.")


    def _create_batch(
        self,
        batch: List[Task],
        tokenizer: LlamaTokenizer,
        batch_collator: Optional[DPOCollator] = None,
        device: Optional[str] = "cuda",
    ):  
        if not batch:
            return None
        batch_collator = batch_collator if batch_collator is not None else DPOCollator(
            tokenizer, 
            inference_input_feature=self.inference_input_feature, 
            inference_mask_feature=self.inference_mask_feature,
        )
        inputs = [task.input_kwargs for task in batch]
        inputs = batch_collator(inputs)
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
        task_queue: IterQueue,
        tokenizer: LlamaTokenizer,
        do_sample: bool = False,
        logits_processor: Optional[Callable] = None,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[DynamicCache] = None,
    ):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
            [MinLengthLogitsProcessor(1, eos_token_id=tokenizer.eos_token_id, device=input_ids.device),]
        )
        max_new_tokens = max_new_tokens if max_new_tokens is not None else 1024
        # Finished sentences should have their next token be a padding token
        batch_size = lm_logits.shape[0]  # shape: [B, S, V]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=lm_logits.device)
        next_token_logits = lm_logits[:, -1, :]  # B X V
        # Pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        # Token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)  # Greedy decoding (B)
        # Update unfinished sequences
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(tokenizer.eos_token_id).long())
        # stoppings = stopping_criteria(input_ids, next_tokens_scores)
    
        # Update task status
        for i, task in enumerate(batch):
            # print(f"Task {task.taskID} (step {task.step}, workload {task.workload}), current task queue {task_queue.queue}")
            task.update_decoding(next_tokens[i].item())
            # if unfinished_sequences[i] == 1 and stoppings[i] == True:  # continue decoding
            if unfinished_sequences[i] == 1 and task.step < max_new_tokens: # continue decoding
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
                task_queue.put((task.get_priority(initial=False), task.workload, task.taskID))
            else:  # stop decoding
                # Update response (text) with next token
                task.get_response(tokenizer)
                # Empty cache
                task.past_key_values = None
                


    def execute(
        self,
        batch: List[Task],
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        workload: str,
        task_queue: IterQueue,
        optimizer: torch.optim.Optimizer,
        batch_collator: Optional[DPOCollator] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        max_new_tokens: Optional[int] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        if not batch:
            return 
        
        inputs = self._create_batch(batch, tokenizer, batch_collator=batch_collator, device=model.device)
        execution_time = time.time()
        try:
            if workload == "train":
                model.train()
                optimizer.zero_grad()
                losses = dpo_loss(model, inputs, return_average=False)
                losses.mean().backward()
                optimizer.step()
                # Update task status
                for i, task in enumerate(batch):
                    task.metrics["loss"] = losses[i].item()
                    task.execution_time = execution_time
                    print(f"Task {task.taskID} ({task.workload}) finished with loss {task.metrics['loss']}")
            else:
                model.eval()
                with torch.no_grad():
                    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **inputs)
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
                            eval_outputs = compute_batch_metrics(model, inputs, compute_average=False)  # return batch results
                            for i, task in enumerate(batch):
                                for key, value in eval_outputs.items():
                                    task.metrics[key] = value[i].item()
                                task.execution_time = execution_time
                    else:
                        # Decode the next token
                        outputs = model_forward(
                            **model_inputs,
                            return_dict=True,
                        )

                # Decode the next tokens and update task status
                self._batch_decoding(
                    batch=batch,
                    input_ids=model_kwargs[self.inference_input_feature],
                    attention_mask=model_kwargs[self.inference_mask_feature],
                    lm_logits=outputs.logits,
                    task_queue=task_queue,
                    tokenizer=tokenizer,
                    do_sample=generation_config.do_sample,
                    logits_processor=logits_processor,
                    max_new_tokens=max_new_tokens,
                    past_key_values=outputs.past_key_values,
                ) 
                
        except Exception as e:
            logging.error(f"Error during {workload} execution: {e}")
            for task in batch:
                task.execution_time = execution_time
                task.response = "Error occurred during execution."
                task.metrics["error"] = str(e)

        # Clear the batch
        batch.clear()
        gc.collect()
        torch.cuda.empty_cache()


    def concurrent_execute(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        task_queue: IterQueue,
        strategy: str,
        optimizer: torch.optim.Optimizer,
        **kwargs,
    ):
        if strategy == "sync":
            # print(f"Sequantial execution (prioritize training)!")
            # Prioritize the training workload
            if self.train_batch:
                self.execute(self.train_batch, model, tokenizer, "train", task_queue, optimizer, **kwargs)
            with ThreadPoolExecutor(max_workers=2) as executor:
                for workload, batch in [
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, **kwargs)
        elif strategy == "async":
            with ThreadPoolExecutor(max_workers=3) as executor:
                # print(f"Concurrent execution (prioritize inference)!")
                for workload, batch in [
                    ("train", self.train_batch),
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose 'sync' or 'async'.")

    



    