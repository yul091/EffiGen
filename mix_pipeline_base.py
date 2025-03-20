import os
import sys
sys.dont_write_bytecode = True
import queue
import time
import json
import random
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Callable, Optional, Tuple, Any
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, LlamaForCausalLM, LlamaTokenizer, get_scheduler
from concurrent.futures import ThreadPoolExecutor
# from models import prepare_decoding_inputs, prepare_inputs, pad_batch
from utils import Task, record_time, save_metrics_with_order

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BasicPipeline:

    def __init__(self, args, model: LlamaForCausalLM, tokenizer: LlamaTokenizer, device: torch.device, experimentID: int = 0):

        self.args = args
        self.n_samples = args.n_samples
        self.model = model
        self.model_n = args.model_name_or_path.split("/")[-1]
        self.tokenizer = tokenizer
        self.device = device
        self.experimentID = experimentID
        self.lr = args.lr
        self.rate_lambda = args.rate_lambda
        self.dataset_name_or_path = args.dataset_name_or_path
        self.retraining_rate = args.retraining_rate 
        self.serving_batch_size = args.serving_batch_size
        self.training_batch_size = args.training_batch_size
        self.max_wait = args.max_wait
        self.run_mode = args.run_mode
        self.output_dir = args.output_dir
        self.RECORD_MODE = False if self.run_mode == 'online' else True

        self.memory_threshold = args.memory_threshold
        self.device_total_memory = torch.cuda.get_device_properties(self.device).total_memory
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Load datasets and dataloaders
        datasets = load_dataset(self.dataset_name_or_path)
        dataset = datasets['test']

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples['query'], 
                padding=False, 
                truncation=True,
            )
            labels = self.tokenizer(
                examples['reference'], 
                padding=False, 
                truncation=True,
            )
            tokenized_inputs['labels'] = labels['input_ids']
            return tokenized_inputs
        
        dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns(dataset.column_names)

        # Sample a subset of the dataset
        if self.n_samples > 0:
            n_samples = min(self.n_samples, len(dataset))
            indices = random.sample(range(len(dataset)), n_samples)
            dataset = dataset.select(indices)
            subset_lengths = [len(x) for x in dataset['input_ids']]
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length \
                = np.mean(subset_lengths), np.std(subset_lengths), np.median(subset_lengths), min(subset_lengths), max(subset_lengths)
            print(f" ** Sampled {len(subset_lengths)} data points: mean={self.mean_length}, std={self.std_length}, medium={self.medium_length}, min={self.min_length}, max={self.max_length} **")
        

        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )

        # Preloaded datasets
        self.dataset = dataset
        self.dataset = self.dataset.shuffle(seed=42)
        # self.dataloader = DataLoader(
        #     self.dataset,
        #     batch_size=1,  # Online training or serving
        #     collate_fn=self.data_collator,
        # )
        self.preloaded_tasks, self.total_tasks, self.num_training_samples, self.training_taskIDs = self.get_preloaded_dataset(
            # dataloader=self.dataloader, 
            dataset=self.dataset,
            retraining_rate=self.retraining_rate,
        )
        self.inference_taskIDs = [i for i, t in enumerate(self.preloaded_tasks) if not t.require_training]
        print(f"Total tasks: {self.total_tasks} | Training tasks: {self.num_training_samples}")

        # Initialize continuous batching
        self.serving_batch: List[Task] = []
        self.isBatchServing: bool = False
        self.training_batch: List[Task] = []
        self.isBatchTraining: bool = False
        self.task_arrival = defaultdict(dict)
        self.task_trace = {}
        self.train_trace = defaultdict(dict)
        self.user_task_record = defaultdict(dict)
        self.all_trace = defaultdict(dict)
        self.tokens_generated = 0
        self.prefills = 0
        self.decodes = 0
        self.metrics = defaultdict(list)
        self.timing_info = defaultdict(list)
    

    def get_preloaded_dataset(
        self,
        # dataloader: Optional[DataLoader] = None, 
        dataset: Optional[torch.utils.data.Dataset] = None,
        retraining_rate: Optional[float] = None,
    ) -> Tuple[List[Task], int, int, List[int]]:
        
        print("Using preloaded data ...")
        # dataloader = dataloader if dataloader is not None else self.dataloader
        dataset = dataset if dataset is not None else self.dataset
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        total_tasks, retraining_taskIDs, preloaded_tasks = 0, [], []
        
        selected_data = []
        for i, instance in enumerate(dataset):
            # seq_length = batch['input_ids'].shape[1]
            seq_length = len(instance['input_ids'])
            selected_data.append((seq_length, instance))
        
        # If workload is 'alternate', we need to create a list of varying lambda values 
        # (e.g., 5, 5, ..., 10, 10, ..., 30, 30, ..., 50, 50, ...), each for 20 consecutive tasks
        if self.rate_lambda == -1:
            print("\n ** Using alternate workload **\n")
            lambda_values = [5, 10, 20, 30]
            current_length, idx, tasks_per_lambda = 0, 0, 20
            task_lambdas = []
            while current_length < len(selected_data):
                if current_length + tasks_per_lambda > len(selected_data):
                    # Add fewer elements if adding 20 would exceed the total length
                    repetitions = len(selected_data) - current_length
                else:
                    repetitions = tasks_per_lambda
                    
                task_lambdas.extend([lambda_values[idx]] * repetitions)
                current_length += repetitions
                idx = (idx + 1) % len(lambda_values)
            
        # Create preloaded tasks with each one on a specific CUDA device
        # for taskID, (_, batch) in enumerate(selected_data):
        for taskID, (seq_length, instance) in enumerate(selected_data):
            lamda = self.rate_lambda if self.rate_lambda != -1 else task_lambdas[taskID]
            # 10% of the time, produce a task with feedback
            require_training = random.random() < retraining_rate
            total_tasks += 1
            if require_training: 
                retraining_taskIDs.append(i)
                # batch = prepare_decoding_inputs(batch)
            
            task = Task(
                task_id=taskID,
                rate_lambda=lamda,
                # query=prepare_inputs(batch, device=self.device),
                input_ids=instance['input_ids'],
                attention_mask=instance['attention_mask'],
                labels=instance['labels'],
                require_training=require_training,
            )
                
            preloaded_tasks.append(task)
        
        return preloaded_tasks, total_tasks, len(retraining_taskIDs), retraining_taskIDs

    

    def producer(self, taskQueue: queue.Queue) -> None:
        # Produce using the dataset
        for taskID, task in enumerate(self.preloaded_tasks):
            # print(f"Producing task {taskID} with input length {task.query['input_ids'].shape[1]}")
            time.sleep(random.expovariate(task.rate_lambda))
            # 10% of the time, produce a task with feedback
            # print("Producing task {} with input length {}".format(taskID, task.query['input_ids'].shape[1]))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(taskID)
            # We record and calculate the response time for each user task (no retraining)
            self.task_arrival[taskID]['release'] = time.time()
            if not task.require_training:
                self.user_task_record[taskID]['release'] = self.task_arrival[taskID]['release']
            
        taskQueue.put(None)  # Signal the end of the dataset
        logging.info("Producer finished producing tasks")
    

    def continuous_serving_batching(
        self,
        taskID: int,
        priority: Union[int, float],
        task: Task,
        deviceQueue: Union[queue.Queue, queue.PriorityQueue], 
        preloaded_tasks: List[Task], 
        max_wait_time: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        while True:
            if not self.isBatchServing:  # If the current batch is done, exit the loop
                break
            time.sleep(0.01)
        
        if len(self.serving_batch) < self.serving_batch_size: 
            self.serving_batch.append(task)
        else:  # if the batch is full, put the new task back and execute the batch
            deviceQueue.put((priority, taskID))

        # Append the remaining tasks (in the queue) to the current batch if it doesn't exceed batch size or max wait time
        start = time.time()
        while (len(self.serving_batch) < self.serving_batch_size) and (time.time() - start < max_wait_time):
            try:
                tp, td = deviceQueue.get(timeout=max_wait_time)
                if td == float('inf'):
                    deviceQueue.put((tp, td))
                    break
                tt: Task = preloaded_tasks[td]
                if tt.require_training:
                    deviceQueue.put((tp, td))
                    break
                self.serving_batch.append(tt)
            except queue.Empty:
                break

        # Use the data collator to pad and combine the batch
        # input_ids_list = [t.query['input_ids'].squeeze(0) for t in self.serving_batch]
        # attention_mask_list = [t.query['attention_mask'].squeeze(0) for t in self.serving_batch]
        context_features = [
            {
                "input_ids": task.input_ids,
                "attention_mask": task.attention_mask,
            }
            for task in self.serving_batch
        ]
        # print(f"Context features: {context_features}")
        padded_context = self.tokenizer.pad(
            context_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        # print(f"Padded context: {padded_context}")
        # input_ids = pad_batch(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        # attention_mask = pad_batch(attention_mask_list, pad_value=0)  


        # Update the current task to use the collated batch
        self.isBatchServing = True
        return {
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            "input_ids": padded_context['input_ids'],
            "attention_mask": padded_context['attention_mask'],
            "labels": padded_context['input_ids'].clone(),
        }


    def continuous_training_batching(
        self,
        taskID: int,
        priority: Union[int, float],
        task: Task,
        deviceQueue: Union[queue.Queue, queue.PriorityQueue], 
        preloaded_tasks: List[Task], 
        max_wait_time: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        while True:
            if not self.isBatchTraining:  # If the current batch is done, exit the loop
                break
            time.sleep(0.01)
        
        if len(self.training_batch) < self.training_batch_size: 
            self.training_batch.append(task)
        else:  # if the batch is full, put the new task back and execute the batch
            deviceQueue.put((priority, taskID))

        # Append the remaining tasks (in the queue) to the current batch if it doesn't exceed batch size or max wait time
        start = time.time()
        while (len(self.training_batch) < self.training_batch_size) and (time.time() - start < max_wait_time):
            try:
                tp, td = deviceQueue.get(timeout=max_wait_time)
                if td == float('inf'):
                    deviceQueue.put((tp, td))
                    break
                tt: Task = preloaded_tasks[td]
                if not tt.require_training:
                    deviceQueue.put((tp, td))
                    break
                self.training_batch.append(tt)
            except queue.Empty:
                break

        # Use the data collator to pad and combine the batch
        # input_ids_list = [t.query['input_ids'].squeeze(0) for t in self.training_batch]
        # attention_mask_list = [t.query['attention_mask'].squeeze(0) for t in self.training_batch]
        # label_list = [t.feedback.squeeze(0) for t in self.training_batch]
        context_features = [
            {
                "input_ids": t['input_ids'],
                "attention_mask": t['attention_mask'],
            }
            for t in self.training_batch
        ]
        padded_context = self.tokenizer.pad(
            context_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        # input_ids = pad_batch(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        # attention_mask = pad_batch(attention_mask_list, pad_value=0)  
        # labels = pad_batch(label_list, pad_value=-100)

        # Update the current task to use the collated batch
        self.isBatchTraining = True
        return {
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "labels": labels,
            "input_ids": padded_context['input_ids'],
            "attention_mask": padded_context['attention_mask'],
            "labels": padded_context['input_ids'].clone(),
        }


    def autoregressive_decoding(
        self, 
        hybrid_batch: Dict[str, torch.Tensor],
        logits_processor: Callable,
        stopping_criteria: Callable,
        device: torch.device,
        batch_size: int,
        lm_logits: torch.Tensor,
    ):
        # Finished sentences should have their next token be a padding token
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        next_token_logits = lm_logits[:, -1, :].to(device)  # B X V
        # Pre-process distribution
        next_tokens_scores = logits_processor(hybrid_batch["input_ids"], next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # Greedy decoding (B)
        # next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
        # Update unfinished sequences
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.eos_token_id).long())
        # Stop when each sentence is finished or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(hybrid_batch["input_ids"], next_tokens_scores).all():
            self.serving_batch.clear()  # Clear the batch
        else:
            # print(f"Continuing decoding for batch {len(self.serving_batch)}")
            finished_indices = (unfinished_sequences == 0).nonzero(as_tuple=True)[0]
            # print(f"Finished indices: {finished_indices}")
            # Append generated tokens to the sequence for all tasks in the batch
            for i, task in enumerate(self.serving_batch):
                # task.query['input_ids'] = torch.cat([ttask.query['input_ids'], next_tokens[i].view(1, 1)], dim=1)
                # task.query['attention_mask'] = torch.cat(
                #     [task.query['attention_mask'], torch.ones((1, 1), device=device)], dim=1
                # )
                task.input_ids.append(next_tokens[i].item())
                task.attention_mask.append(1)
                # print(f"[Current batch  {len(self.serving_batch)} | finished {finished_indices}] task {task.task_id} (index {i}) -> next token {next_tokens[i]}")
            # Remove finished sequences from the batch
            if len(finished_indices) > 0:
                # print(f"Removing finished sequences: {finished_indices}")
                for i, task in enumerate(self.serving_batch):
                    if i in finished_indices:
                        # # Record the decoding in the user task record
                        # self.user_task_record[task.task_id]['generation_ids'] = task.input_ids
                        # Remove the task from the batch
                        del self.serving_batch[i]

                # self.serving_batch = [
                #     t for i, t in enumerate(self.serving_batch) if i not in finished_indices
                # ]


            # print(f"Remaining tasks: {len(self.node_batches[nodeID])}")
        self.isBatchServing = False
        hybrid_batch = None  # Clear the input no longer needed


    def device_inference(
        self,
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        timing_info: Optional[dict] = None, 
        device: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("stage_inference method must be implemented") 


    def forward(
        self, 
        taskID: int,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        device: int, 
        timing_info: Dict[str, List[float]],
        require_training: bool,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # Memory check and scale down if necessary
        # self.wait_for_device_availability(device)
        try:
            if require_training: # this is a training task
                self.model.train()
                fb = record_time(device, 'start', 'forward', taskID, timing_info)
                self.task_trace['fb'] = fb
                if taskID in self.train_trace: # this training task has been completely recorded
                    self.train_trace[taskID]['fb'] = fb # update the already recorded task
                    self.all_trace[taskID]['fb'] = fb # update the already recorded task

                with autocast("cuda"):
                    outputs = self.model(**inputs, labels=labels)
                fe = record_time(device, 'end', 'forward', taskID, timing_info)
                
                self.task_trace['fe'] = fe
                if taskID in self.train_trace: # this training task has been completely recorded
                    self.train_trace[taskID]['fe'] = fe # update the already recorded task
                    self.all_trace[taskID]['fe'] = fe # update the already recorded task
           
            else: # this is a user (test) task
                self.model.eval()
                fb = time.time()
                self.task_trace['fb'] = fb
                if taskID in self.all_trace: # this task has been completely recorded
                    self.all_trace[taskID]['fb'] = fb # update the already recorded task            
                    
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=labels)

                fe = time.time()
                self.task_trace['fe'] = fe
                if taskID in self.all_trace: # this task has been completely recorded
                    self.all_trace[taskID]['fe'] = fe # update the already recorded task
                
                for task in self.serving_batch:
                    # seq_length = task.query['input_ids'].shape[1]
                    if task.decode_step == 0:
                        self.user_task_record[task.task_id]['start'] = fb
                        self.user_task_record[task.task_id]['prefill_end'] = fe
                        self.user_task_record[task.task_id]['loss'] = outputs.loss.item()
                        self.prefills += 1
                        timing_info[f"{device}_start"].append((fb, 'prefill', task.task_id))
                        timing_info[f"{device}_end"].append((fe, 'prefill', task.task_id))
                        # logging.info(f"[Task {task.task_id} (length {seq_length}) finished prefill pass!")
                    else:
                        self.user_task_record[task.task_id]['decode_end'] = fe
                        self.decodes += 1
                        timing_info[f"{device}_start"].append((fb, 'decode', task.task_id))
                        timing_info[f"{device}_end"].append((fe, 'decode', task.task_id))
                        # logging.info(f"[Task {task.task_id} (length {seq_length}) finished decode step {task.decode_step}!")

                    self.tokens_generated += 1
                    self.user_task_record[task.task_id]['decode_step'] = task.decode_step
                    task.decode_step += 1
                
        except Exception as e:
            logging.error(f"[device {device}] Forward error occurred: {e}")
            outputs = None
        
        return outputs 
    

    def globalScheduler(self, taskQueue: queue.Queue, deviceQueue: queue.PriorityQueue) -> None:
        raise NotImplementedError("globalScheduler method must be implemented")
    

    def check_device_availability(self, device: int):
        """
        Check if the device has enough available memory.
        Args:
        - device: The device to check.
        Returns:
        - is_available: Boolean indicating if the device is available.
        """
        # Get device memory status
        allocated_memory = torch.cuda.memory_allocated(device)
        return allocated_memory / self.device_total_memory <= self.memory_threshold

    
    
    def wait_for_device_availability(
        self, 
        device: int, 
        check_interval: float = 0.1, 
        force_check: bool = True,
    ):
        """
        Wait until the device is available based on memory usage.
        Args:
        - device: The device to wait for.
        - check_interval: How often to check the device status (in seconds).
        """
        if not force_check:
            return True
        start_time = time.time()
        while not self.check_device_availability(device):
            # print(f"Waiting for device {device} to become available...")
            time.sleep(check_interval)
            if time.time() - start_time > self.max_wait:
                print(f"Exceeded max wait time for device {device}. Exit forward waiting loop.")
                return False
        return True


    def run(self):
        
        task_queue = queue.Queue()
        device_queue = queue.PriorityQueue()

        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.submit(
                self.producer,
                task_queue, 
            )
            executor.submit(
                self.globalScheduler,
                task_queue,
                device_queue,
            )
            executor.submit(
                self.device_inference,
                self.preloaded_tasks,
                device_queue,
            )
        
        # Save timing info
        self.save_timing_info()
        
        # Calculate metrics
        self.calculate_metrics()


    def save_timing_info(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timing_file = os.path.join(self.output_dir, f"timing_info_{self.model_n}_{self.retraining_rate}_{self.experimentID}.json")
        with open(timing_file, 'w') as f:
            json.dump(self.timing_info, f, indent=4)


    def calculate_metrics(self, metrics: Optional[Dict[str, Union[float, int]]] = None):
        metrics = metrics if metrics is not None else self.metrics
        metrics['inference_losses'] = [t['loss'] for t in self.user_task_record.values()]

        # Calculate metrics
        start, end = float('inf'), float('-inf')
        for device_op, timelines in self.timing_info.items():
            # device_op: "0_start", "0_end"
            # timelines: [(time, 'prefill', taskID), (time, 'decode', taskID), (time, 'backward', taskID), ...]
            if 'start' in device_op:
                first_start = timelines[0][0]
                start = min(start, first_start)
            elif 'end' in device_op:
                last_end = timelines[-1][0]
                end = max(end, last_end)
            else:
                raise ValueError(f"Invalid device operation type: {device_op}")
        
        total_runtime = end - start
        for key, value in metrics.items():
            if key == 'train_loss':
                train_losses = value
            metrics[key] = sum(value) / len(value) if value else 0
        if self.training_taskIDs:
            metrics['train_losses'] = train_losses # list 

        
        # Calculate response times
        metrics['num_tasks'] = self.total_tasks
        metrics['retrain_tasks'] = self.num_training_samples
        metrics['num_tokens_generated'] = self.tokens_generated
        metrics['user_tasks'] = len(self.user_task_record)
        metrics['E2E_latency'] = total_runtime
        metrics['throughput (tasks)'] = self.total_tasks / total_runtime
        metrics['throughput (tokens)'] = metrics['num_tokens_generated'] / total_runtime
        metrics['throughput (prefill)'] = self.prefills / total_runtime
        metrics['throughput (decode)'] = self.decodes / total_runtime
        if self.inference_taskIDs:
            decode_steps = []
            task2decode_steps = {}
            for taskID in self.user_task_record:
                task: Task = self.preloaded_tasks[taskID]
                assert task.require_training == False
                decode_steps.append(task.decode_step)
                task2decode_steps[taskID] = task.decode_step
            averge_decode_steps = np.mean(decode_steps)
            metrics['decoding_length'] = averge_decode_steps
        metrics['input_length_stats'] = {
            'mean': self.mean_length,
            'std': self.std_length,
            'medium': self.medium_length,
            'min': self.min_length,
            'max': self.max_length,
        }
        
        if self.user_task_record:
            response_times, wait_times, latencies, E2E_decoding_times, token_decoding_times = [], [], [], [], []
            user_global_min_time, user_global_max_time = float('inf'), float('-inf')
            for taskID, record_dict in self.user_task_record.items():
                if 'start' not in record_dict or 'prefill_end' not in record_dict:
                    # print(f"Unrecorded user request (ID={taskID})! \tOnly have keys: {record_dict.keys()}")
                    continue
                user_global_min_time = min(user_global_min_time, record_dict['start'])
                user_global_max_time = max(user_global_max_time, record_dict['prefill_end'])
                record_dict['response_time'] = record_dict['prefill_end'] - record_dict['release']
                response_times.append(record_dict['response_time'])
 
                if 'decode_end' in record_dict:
                    E2E_decoding_time = record_dict['decode_end'] - record_dict['prefill_end']
                    E2E_decoding_times.append(E2E_decoding_time)
                    try:
                        decode_step = task2decode_steps[taskID]
                        token_decoding_times.append(E2E_decoding_time / decode_step if decode_step > 0 else 0)
                    except KeyError:
                        print(f"Decode step not found for task {taskID}")

                record_dict['generation'] = self.tokenizer.decode(self.preloaded_tasks[taskID].input_ids, skip_special_tokens=True)
                
                wait_times.append(record_dict['start'] - record_dict['release'])
                latencies.append(record_dict['prefill_end'] - record_dict['start'])
                
            metrics['wait_time'] = sum(wait_times) / len(wait_times) if wait_times else 0
            metrics['TTFT'] = sum(latencies) / len(latencies) if latencies else 0
            metrics['TBT'] = sum(token_decoding_times) / len(token_decoding_times) if token_decoding_times else 0
            metrics['response_time'] = sum(response_times) / len(self.user_task_record)
            metrics['E2E_latency (inference)'] = user_global_max_time - user_global_min_time
            metrics['throughput (inference)'] = len(self.user_task_record) / (user_global_max_time - user_global_min_time)
            metrics['inference_record'] = self.user_task_record
        
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        stats_f = f'{self.output_dir}/metrics_{self.model_n}_{self.retraining_rate}_ID={self.experimentID}.json'

        save_metrics_with_order(metrics, stats_f)