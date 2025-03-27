# A definition of the bin packing class and supported functions.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Must be before torch is imported
import time
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    LlamaTokenizer,
    LogitsProcessorList, 
    StoppingCriteriaList, 
    MaxLengthCriteria, 
    MinLengthLogitsProcessor,
)
from transformers.cache_utils import DynamicCache
from peft import get_peft_model, LoraConfig
from queue import PriorityQueue
import sys 
sys.dont_write_bytecode = True
from iteration_task import Task
from alignment_study import DPOCollator, dpo_loss
from utils import prepare_inputs


class IterationBin:

    def __init__(
        self,
        tokenizer: LlamaTokenizer,
        batch_collator: Optional[DPOCollator] = None,
    ):
        self.prefill_batch: List[Task] = []
        self.decode_batch: List[Task] = []
        self.train_batch: List[Task] = []
        self.tokenizer = tokenizer
        self.batch_collator = batch_collator if batch_collator is not None else DPOCollator(tokenizer)
        

    def add_task(self, task: Task):
        if task.workload == "prefill":
            self.prefill_batch.append(task)
        elif task.workload == "decode":
            self.decode_batch.append(task)
        elif task.workload == "train":
            self.train_batch.append(task)
        else:
            raise ValueError(f"Invalid workload: {task.workload}")
        

    def _create_batch(
        self,
        batch: List[Task],
        device: Optional[str] = "cuda",
    ):  
        if not batch:
            return None
        input_kwargs = [task.input_kwargs for task in batch]
        input_kwargs = self.batch_collator(input_kwargs)
        # TO-DO: pad key values with varient sequence lengths
        input_kwargs["past_key_values"] = None
        return prepare_inputs(input_kwargs, device=device)
    

    def _batch_decoding(
        self,
        input_ids: torch.Tensor,
        outputs: Union[Tuple[Any], Dict[str, Any]],
        logits_processor: Callable,
        stopping_criteria: Callable,
    ):
        # Finished sentences should have their next token be a padding token
        lm_logits = outputs.logits
        batch_size = lm_logits.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=lm_logits.device)
        next_token_logits = lm_logits[:, -1, :]  # B X V
        # Pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # Greedy decoding (B)
        # Update unfinished sequences
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.eos_token_id).long())
        return next_tokens, unfinished_sequences, stopping_criteria(input_ids, next_tokens_scores)
    

    def _update(
        self,
        batch: List[Task],
        next_tokens: torch.Tensor,
        task_queue: PriorityQueue,
        unfinished_sequences: torch.Tensor,
        stoppings: torch.BoolTensor,
        attention_mask: torch.Tensor,
        output_cache: Optional[DynamicCache] = None,
    ):
        # Update
        # print(f"Model output cache size: {output_cache.key_cache[0].shape}")
        for i, task in enumerate(batch):
            # if unfinished_sequences[i] == 1 and stoppings[i] == 0:  # continue decoding
            if unfinished_sequences[i] == 1 and stoppings[i] == True:  # continue decoding
                task.update_status(workload="decode")
                task.input_kwargs["context_input_ids"].append(next_tokens[i].item())
                task.input_kwargs["context_attention_mask"].append(1)
                task.step += 1
                # Update response (text) with next token
                task.response += self.tokenizer.decode(next_tokens[i].item())
                # print(f"[Task {task.taskID} ({task.workload})] \nprompt: {task.prompt} \nresponse: {task.response}")
                # Update past key values (cache)
                if output_cache is None:  # [batch_size, num_heads, seq_len, head_dim]
                    continue
                # Split and update individual task's KV cache
                mask = attention_mask[i] == 1
                task.past_key_values = DynamicCache()
                for layer_idx in range(len(output_cache.key_cache)):
                    task.past_key_values.update(
                        output_cache.key_cache[layer_idx][i, :, mask, :],  # [H, S_valid, D]
                        output_cache.value_cache[layer_idx][i, :, mask, :],
                        layer_idx=layer_idx,
                    )
                print(f"Cache size (new): {task.past_key_values.key_cache[0].shape}")
                # Add task back to queue
                task_queue.put((task.get_priority(initial=False), task.taskID))


    def execute(
        self,
        batch: List[Task],
        model: AutoModelForCausalLM,
        workload: str,
        task_queue: PriorityQueue,
        max_length: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        
    ):
        if not batch:
            return None
        input_kwargs = self._create_batch(batch, device=model.device)
        if workload == "train":
            model.train()
            optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=5e-5)
            optimizer.zero_grad()
            # outputs = model(**input_kwargs)
            loss = dpo_loss(model, input_kwargs)
            loss.backward()
            optimizer.step()

            # Clear the batch
            batch.clear()
            return loss

        else:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_kwargs["context_input_ids"],
                    attention_mask=input_kwargs["context_attention_mask"],
                    past_key_values=input_kwargs["past_key_values"],
                    use_cache=True,
                    return_dict=True,
                )  # <loss, logits, past_key_values, hidden_states, attentions>

            max_length = max_length if max_length is not None else 128
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
                [MinLengthLogitsProcessor(1, eos_token_id=self.tokenizer.eos_token_id, device=model.device),]
            )
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
                [MaxLengthCriteria(max_length=max_length, max_position_embeddings=model.config.max_position_embeddings),]
            )

            next_tokens, unfinished_sequences, stoppings = self._batch_decoding(
                input_ids=input_kwargs["context_input_ids"],
                outputs=outputs,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )  # (B, ), (B, ), (B, )
            # print(f"next tokens: {next_tokens}")
            # print(f"unfinished sequences: {unfinished_sequences}")
            # print(f"stoppings: {stoppings}")

            # Update task status
            self._update(
                batch=batch,
                next_tokens=next_tokens,
                unfinished_sequences=unfinished_sequences,
                task_queue=task_queue,
                stoppings=stoppings,
                attention_mask=input_kwargs["context_attention_mask"],
                output_cache=outputs.past_key_values,
            )
                
            # Clear the batch
            batch.clear()

        return outputs
    

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
        
    accum = i
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
    print(f"task queue size: {task_queue.qsize()}")

    # Create iteration bin and execute tasks
    bin = IterationBin(tokenizer)
    # print(f"prefill batch: {bin.prefill_batch}")
    for iteration in range(5):
        print(f"  **  Iteration {iteration}  **  ")
        while task_queue.qsize() > 0:
            _, taskID = task_queue.get(timeout=0.5)
            bin.add_task(preloaded_tasks[taskID])

        # print(f"prefill batch: {bin.prefill_batch}")
        outputs = bin.execute(bin.prefill_batch, model, workload='prefill', task_queue=task_queue)
        print(f"execution results (prefill): {outputs.keys() if outputs else None}")
        outputs = bin.execute(bin.decode_batch, model, workload='decode', task_queue=task_queue)
        print(f"execution results (decode): {outputs.keys() if outputs else None}")
        # print(f"decode batch: {bin.decode_batch}")
        # print(f"train batch: {bin.train_batch}")
        print(f"execution results (train): {bin.execute(bin.train_batch, model, workload='train', task_queue=task_queue)})")
