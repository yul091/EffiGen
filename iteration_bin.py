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
from line_profiler import profile
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    LogitsProcessorList, 
    MinLengthLogitsProcessor,
    GenerationConfig,
)
from transformers.cache_utils import DynamicCache, Cache
from concurrent.futures import ThreadPoolExecutor
from iteration_task import Task
from iteration_queue import IterQueue
from iteration_lora_selection import selective_training
from alignment_study import DPOCollator, dpo_loss, compute_batch_metrics
from utils import prepare_inputs, _prepare_input


class Bin:

    def __init__(
        self,
        strategy: str,
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
        self.device = device
        self.strategy = strategy
        # Initialize memory and latency
        self.memory_capacity = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
        self.base_memory = torch.cuda.memory_allocated(device) / (1024**2)  # MB
        self.max_latency = 0
        self.workload_stats = {
            "train": {"max_length": 1, "memory": 0, "batch_size": 0, "latency": 0,},
            "prefill": {"max_length": 1, "memory": 0, "batch_size": 0, "latency": 0,},
            "decode": {"max_length": 1, "memory": 0, "batch_size": 0, "latency": 0,},
        }
        

    def add_task(
        self, 
        task: Task,
        model: LlamaForCausalLM,
        attn_implementation: str = "flash_attention_2", 
        new_seq_length: Optional[int] = None,
    ):
        if task.workload == "prefill":
            self.prefill_batch.append(task)
        elif task.workload == "decode":
            self.decode_batch.append(task)
        elif task.workload == "train":
            self.train_batch.append(task)
        else:
            raise ValueError(f"Invalid workload: {task.workload}")
        
        # Update the workload stats
        batch_memory, batch_latency, batch_length, _, _ = self.get_workload(
            task, model, attn_implementation, new_seq_length=new_seq_length,
        )
        self.workload_stats[task.workload]["batch_size"] += 1
        self.workload_stats[task.workload]["max_length"] = batch_length
        self.workload_stats[task.workload]["memory"] = batch_memory
        self.workload_stats[task.workload]["latency"] = batch_latency
        self.max_latency = max(self.max_latency, batch_latency)
        

    def get_num_tasks(self, target: str = "all") -> int:
        if target == "all":
            return len(self.prefill_batch) + len(self.decode_batch) + len(self.train_batch)
        elif target == "inference":
            return len(self.prefill_batch) + len(self.decode_batch)
        elif target == "train":
            return len(self.train_batch)
        else:
            raise ValueError(f"Invalid target: {target}. Supported targets: all, inference, train.")
        

    def get_workload(
        self, 
        task: Task, 
        model: LlamaForCausalLM,
        attn_implementation: str = "flash_attention_2", 
        new_seq_length: Optional[int] = None,
    ):
        """
        Estimate the resource consumption of all workloads in the bin. 
        Return the maximum new length, batch memory and latency.
        """
        # Calculate the total memory used by the tasks in the bin
        basic_factor = model.model.config.num_hidden_layers * model.model.config.hidden_size  # num_layers * hidden_dim
        if new_seq_length is None:
            new_seq_length = task.get_input_length()
        batch_length = max(self.workload_stats[task.workload]["max_length"], new_seq_length)
        # Calculate the memory and latency for the task
        length_multiplier = batch_length if attn_implementation == "flash_attention_2" else batch_length**2
        new_batch_size = self.workload_stats[task.workload]["batch_size"] + 1
        batch_memory = basic_factor * new_batch_size * length_multiplier * task.coefficients[task.workload]["memory_coeff"]
        batch_latency = basic_factor * new_batch_size * length_multiplier * task.coefficients[task.workload]["latency_coeff"]
        # accumulated_memory = sum(self.workload_stats[w]['memory'] for w in self.workload_stats if w != task.workload) + batch_memory
        accumulated_memory = batch_memory
        if self.strategy == "async":
            # Asynchronous execution, so we count all workloads
            accumulated_memory += sum(self.workload_stats[w]['memory'] for w in self.workload_stats if w != task.workload)
        else:
            # Training workload is independently executed, so we only count the batch memory
            # Prefill workload is co-executed with decode workload
            if task.workload == "prefill":  
                accumulated_memory += self.workload_stats["decode"]["memory"]
            elif task.workload == "decode":  
                accumulated_memory += self.workload_stats["prefill"]["memory"]
            
        memory_fit = max(self.memory_capacity - accumulated_memory - self.base_memory, 0)
        max_latency = max(self.max_latency, batch_latency)
        latency_fit = max_latency - batch_latency
        return batch_memory, batch_latency, batch_length, memory_fit, latency_fit
        

    @profile
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

        # Handle case where all caches are None
        split_caches = [task.past_key_values for task in batch]
        if all(c is None for c in split_caches):  
            return prepare_inputs(inputs, device=device)

        # For decode-only: pad key values with varient sequence lengths
        ref_cache = next(c for c in split_caches if c is not None)
        attention_mask = inputs[self.inference_mask_feature][:, :-1]  # Remove decode step
        batch_size, max_seq_len = attention_mask.shape
        num_heads, _, head_dim  = ref_cache.key_cache[0].shape
        inputs["past_key_values"] = DynamicCache()

        valid_lens = attention_mask.sum(dim=1).tolist()
        pad_left = tokenizer.padding_side == "left"
        non_empty = [(i, c) for i, c in enumerate(split_caches) if c is not None]

        for layer_idx in range(len(ref_cache.key_cache)):
            # key_tensor = torch.zeros(
            #     (batch_size, num_heads, max_seq_len, head_dim),
            #     dtype=ref_cache.key_cache[layer_idx].dtype,
            #     device=ref_cache.key_cache[layer_idx].device,
            # )
            # value_tensor = torch.zeros_like(key_tensor)
            # for i in range(batch_size):
            #     if split_caches[i] is None:
            #         continue  # Skip samples without KV cache (zero)
            #     # print(f"Task {i} (output) cache size: {split_caches[i].key_cache[layer_idx].shape}")
            #     mask = attention_mask[i] == 1
            #     key_tensor[i, :, mask, :] = split_caches[i].key_cache[layer_idx]  # [H, valid_len, D]
            #     value_tensor[i, :, mask, :] = split_caches[i].value_cache[layer_idx]
            # inputs["past_key_values"].update(key_tensor, value_tensor, layer_idx)

            k_buf = torch.empty(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=ref_cache.key_cache[layer_idx].dtype,
                device=ref_cache.key_cache[layer_idx].device,
            )
            v_buf = torch.empty_like(k_buf)
            for i, c in non_empty:
                T_i = valid_lens[i]
                if pad_left:
                    target_slice = slice(max_seq_len - T_i, max_seq_len)
                else:
                    target_slice = slice(0, T_i)
                k_buf[i, :, target_slice, :] = c.key_cache[layer_idx]
                v_buf[i, :, target_slice, :] = c.value_cache[layer_idx]
            inputs["past_key_values"].update(k_buf, v_buf, layer_idx)
            
        return prepare_inputs(inputs, device=device)
    
    @profile
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
        batch_size, max_seq_len = attention_mask.shape  # shape: [B, S]
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
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=lm_logits.device)
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(tokenizer.eos_token_id).long())
        valid_lengths = attention_mask.sum(dim=1).tolist()
        pad_left = tokenizer.padding_side == "left"
    
        # Update task status
        for i, task in enumerate(batch):
            task.update_decoding(next_tokens[i].item())
            # if unfinished_sequences[i] == 1 and stoppings[i] == True:  # continue decoding
            if unfinished_sequences[i] == 1 and task.step < max_new_tokens: # continue decoding
                # Update past key values [batch_size, num_heads, seq_len, head_dim]
                if past_key_values is None:  
                    continue
                # Split and update individual task's KV cache
                task.past_key_values = DynamicCache()
                # mask = attention_mask[i] == 1
                T_i = valid_lengths[i]
                if pad_left:
                    target_slice = slice(max_seq_len - T_i, max_seq_len)
                else:
                    target_slice = slice(0, T_i)
                
                for layer_idx in range(len(past_key_values.key_cache)):
                    # task.past_key_values.update(
                    #     past_key_values.key_cache[layer_idx][i, :, mask, :],  # [H, S_valid, D]
                    #     past_key_values.value_cache[layer_idx][i, :, mask, :],  # [H, S_valid, D]
                    #     layer_idx=layer_idx,
                    # )
                    task.past_key_values.update(
                        past_key_values.key_cache[layer_idx][i, :, target_slice, :],  # [H, S_valid, D]
                        past_key_values.value_cache[layer_idx][i, :, target_slice, :],  # [H, S_valid, D]
                        layer_idx=layer_idx,
                    )
                # Add task back to queue
                task_queue.put((task.get_priority(self.strategy, initial=False), task.workload, task.taskID))
            else:  # stop decoding
                # Update response (text) with next token
                task.get_response(tokenizer)
                # Empty cache
                task.past_key_values = None
                

    @profile
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
        generation_config: Optional[GenerationConfig] = None,
        memory_threshold: Optional[float] = None,
        loss_threshold: Optional[float] = None,
        layer_selection: Optional[str] = None,  # "RGN" or "SNR" or None
        layer_threshold: Optional[float] = None,
    ):
        if not batch:
            return 
        
        inputs = self._create_batch(batch, tokenizer, batch_collator=batch_collator, device=model.device)
        memory_threshold = memory_threshold if memory_threshold is not None else 0.95
        
        # Check memory threshold
        preallocated_memory = torch.cuda.memory_allocated(model.device) / (1024**2)  # MB
        if preallocated_memory > memory_threshold * self.memory_capacity:
            logging.warning(f"Memory overflow! Preallocate: {preallocated_memory:.2f} MB, capacity: {self.memory_capacity:.2f} MB")
            for task in batch:
                # Offload the cache to CPU
                task.past_key_values = _prepare_input(task.past_key_values, device="cpu")
                # Update task status and put it back to the queue
                task_queue.put((task.get_priority(self.strategy, initial=False), task.workload, task.taskID))
            # Clear the batch
            batch.clear()
            return
        
        execution_time = time.time()
        try:
            if workload == "train":
                model.train()
                optimizer.zero_grad()
                losses = dpo_loss(model, inputs, return_average=False)
                
                # Handle selective training
                if layer_selection is not None:
                    losses = selective_training(
                        model=model,
                        losses=losses,
                        loss_threshold=loss_threshold,
                        layer_selection=layer_selection,
                        layer_threshold=layer_threshold,
                        optimizer=optimizer,
                    )
                else:
                    losses.mean().backward()
                    optimizer.step()

                # Update task status
                for i, task in enumerate(batch):
                    task.metrics["loss"] = losses[i].item()
                    if task.execution_time is None:
                        task.execution_time = execution_time
                    task.step += 1
                    # If loss is larger than a threshold, put it back to queue
                    if task.metrics["loss"] > 1.0:
                        task_queue.put((task.get_priority(self.strategy, initial=True), task.workload, task.taskID))
                        print(f"Task {task.taskID} ({task.workload}) with loss {task.metrics['loss']} (put back for retraining)")
                    else:
                        print(f"Task {task.taskID} ({task.workload}) with loss {task.metrics['loss']} (finished)") 
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
                                if task.execution_time is None:
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

                del outputs, inputs, model_kwargs, model_inputs

        except Exception as e:
            logging.error(f"Error during {workload} execution (stats {self.workload_stats[workload]}): {e}")
            for task in batch:
                if task.execution_time is None:
                    task.execution_time = execution_time
                task.response = "Error occurred during execution."
                task.metrics["error"] = str(e)

        # Clear the batch
        batch.clear()
        # gc.collect()
        # torch.cuda.empty_cache()


    def concurrent_execute(
        self,
        model: LlamaForCausalLM,
        tokenizer: LlamaTokenizer,
        task_queue: IterQueue,
        optimizer: torch.optim.Optimizer,
        memory_threshold: Optional[float] = None,
        **kwargs,
    ):
        if self.strategy == "async":
            with ThreadPoolExecutor(max_workers=3) as executor:
                # print(f"Concurrent execution (prioritize inference)!")
                for workload, batch in [
                    ("train", self.train_batch),
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)

        else:
            # print(f"Sequantial execution (prioritize training)!")
            # Prioritize the training workload
            self.execute(self.train_batch, model, tokenizer, "train", task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)
            with ThreadPoolExecutor(max_workers=2) as executor:
                for workload, batch in [
                    ("prefill", self.prefill_batch),
                    ("decode", self.decode_batch),
                ]:
                    executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)

            # if self.strategy == "train-first" or self.strategy == "sync":
            #     self.execute(self.train_batch, model, tokenizer, "train", task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)
            #     with ThreadPoolExecutor(max_workers=2) as executor:
            #         for workload, batch in [
            #             ("prefill", self.prefill_batch),
            #             ("decode", self.decode_batch),
            #         ]:
            #             executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)
            # elif self.strategy == "test-first":
            #     with ThreadPoolExecutor(max_workers=2) as executor:
            #         for workload, batch in [
            #             ("prefill", self.prefill_batch),
            #             ("decode", self.decode_batch),
            #         ]:
            #             executor.submit(self.execute, batch, model, tokenizer, workload, task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)
            #     self.execute(self.train_batch, model, tokenizer, "train", task_queue, optimizer, memory_threshold=memory_threshold, **kwargs)
            # else:
            #     raise ValueError(f"Invalid strategy: {self.strategy}. Supported strategies: train-first, test-first, async, sync.")


    