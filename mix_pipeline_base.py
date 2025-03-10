
import sys
sys.dont_write_bytecode = True
import queue
import time
import random
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union, Callable, Optional, Tuple, Any
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from concurrent.futures import ThreadPoolExecutor
from models import prepare_decoding_inputs, prepare_inputs, pad_batch
from utils import Task, record_time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BasicPipeline:

    def __init__(self, args, model, tokenizer, device, experimentID: int = 0):

        self.args = args
        self.n_samples = args.n_samples
        self.model = model
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
        self.RECORD_MODE = False if self.run_mode == 'online' else True

        self.memory_threshold = args.memory_threshold
        self.device_total_memory = torch.cuda.get_device_properties(0).total_memory
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
        self.preloaded_tasks: List[Task] = []
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # Online training or serving
            collate_fn=self.data_collator,
        )
        self.total_tasks, self.num_training_samples, self.training_taskIDs = self.get_preloaded_dataset(
            self.preloaded_tasks, dataloader=self.dataloader, retraining_rate=self.retraining_rate,
        )
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
    

    def get_preloaded_dataset(
        self,
        preloaded_tasks: List[Task],
        dataloader: Optional[DataLoader] = None, 
        retraining_rate: Optional[float] = None,
    ) -> Tuple[List[Task], int, int]:
        
        print("Using preloaded data ...")
        dataloader = dataloader if dataloader is not None else self.dataloader
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        total_tasks, retraining_taskIDs = 0, []
        
        selected_data = []
        for i, batch in enumerate(dataloader):
            seq_length = batch['input_ids'].shape[1]
            selected_data.append((seq_length, batch))
        
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
        for taskID, (_, batch) in enumerate(selected_data):
            # 10% of the time, produce a task with feedback
            require_training = random.random() < retraining_rate
            total_tasks += 1
            if require_training: 
                retraining_taskIDs.append(i)
                batch = prepare_decoding_inputs(batch)
            
            task = Task(
                task_id=taskID,
                rate_lambda=self.rate_lambda,
                query=prepare_inputs(batch, device=self.device),
                feedback=prepare_inputs(batch['labels'], device=self.device),
                require_training=True,
            )
                
            preloaded_tasks.append(task)
        
        return total_tasks, len(retraining_taskIDs), retraining_taskIDs

    

    def producer(self, taskQueue: queue.Queue) -> None:
        # Produce using the dataset
        for taskID, task in enumerate(self.preloaded_tasks):
            if self.workload == 'all':
                time.sleep(0)
            else:
                time.sleep(random.expovariate(task.rate_lambda))
            # 10% of the time, produce a task with feedback
            # print("Producing task {} with input length {}".format(taskID, task.query['input_ids'].shape[1]))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(taskID)
            # We record and calculate the response time for each user task (no retraining)
            release = time.time()
            self.task_arrival[taskID]['release'] = release
            
        taskQueue.put(None)  # Signal the end of the dataset
        print("Producer finished producing tasks")
    

    def continuous_batching(
        self,
        taskID: int,
        priority: Union[int, float],
        task: Task,
        deviceQueue: Union[queue.Queue, queue.PriorityQueue], 
        preloaded_tasks: List[Task], 
        max_wait_time: float = 0.1,
    ):
        while True:
            if not self.isBatchServing:  # If the current batch is done, exit the loop
                break
            time.sleep(0.01)

        if not self.serving_batch:  # If the batch is empty, start a new batch
            task.start = time.time()
        
        if len(self.serving_batch) < self.serving_batch_size: 
            self.serving_batch.append(task)
        else:  # if the batch is full, put the new task back and execute the batch
            deviceQueue.put((priority, taskID))

        # Append the remaining tasks (in the queue) to the current batch if it doesn't exceed batch size or max wait time
        while len(self.serving_batch) < self.serving_batch_size:
            # print(f"Current batch size: {len(self.node_batches[nodeID])}, remaining tasks: {deviceQueue.queue}")
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
        input_ids_list = [t.query['input_ids'].squeeze(0) for t in self.serving_batch]
        attention_mask_list = [t.query['attention_mask'].squeeze(0) for t in self.serving_batch]
        input_ids = pad_batch(input_ids_list, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_batch(attention_mask_list, pad_value=0)  

        # Update the current task to use the collated batch
        task.hybrid_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        self.isBatchServing = True



    def autoregressive_decoding(
        self, 
        task: Task,
        logits_processor: Callable,
        stopping_criteria: Callable,
        init_device: torch.device,
        batch_size: int,
        lm_logits: torch.Tensor,
    ):
        # Finished sentences should have their next token be a padding token
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=init_device)
        next_token_logits = lm_logits[:, -1, :].to(init_device)  # B X V
        # Pre-process distribution
        next_tokens_scores = logits_processor(task.hybrid_batch["input_ids"], next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)  # Greedy decoding (B)
        # next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
        # Update unfinished sequences
        unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(self.tokenizer.eos_token_id).long())
        # print(f"Unfinished sequences: {unfinished_sequences}")
        # Stop when each sentence is finished or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(task.hybrid_batch["input_ids"], next_tokens_scores):
            self.serving_batch.clear()  # Clear the batch
        else:
            finished_indices = (unfinished_sequences == 0).nonzero(as_tuple=True)[0]
            # Append generated tokens to the sequence for all tasks in the batch
            for i, t in enumerate(self.serving_batch):
                t.query['input_ids'] = torch.cat([t.query['input_ids'], next_tokens[i].view(1, 1)], dim=1)
                # print(f"\tUpdated input_ids: {t.query['input_ids'].shape}")
                t.query['attention_mask'] = torch.cat(
                    [t.query['attention_mask'], torch.ones((1, 1), device=init_device)], dim=1
                )
                # print(f"[Current batch  {len(self.serving_batch)} | finished {finished_indices}] task {t.task_id} (index {i}) -> next token {next_tokens[i]}")
            # Remove finished sequences from the batch
            if len(finished_indices) > 0:
                # print(f"Removing finished sequences: {finished_indices}")
                self.serving_batch = [
                    t for i, t in enumerate(self.serving_batch) if i not in finished_indices
                ]
            # print(f"Remaining tasks: {len(self.node_batches[nodeID])}")
        self.isBatchServing = False
        task.hybrid_batch = None  # Clear the input no longer needed


    def device_inference(
        self,
        timing_info: Dict[str, List[float]],
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        device: Optional[int] = None,
    ):
        raise NotImplementedError("stage_inference method must be implemented") 


    def forward(
        self, 
        taskID: int,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        input_length: int,
        device: int, 
        timing_info: Dict[str, List[float]],
        require_training: bool,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # Memory check and scale down if necessary
        # self.wait_for_device_availability(device)
        try:
            if require_training: # this is a training task
                fb = record_time(device, 'start', 'forward_grad', taskID, timing_info)
                self.task_trace['fb'] = fb
                if taskID in self.train_trace: # this training task has been completely recorded
                    self.train_trace[taskID]['fb'] = fb # update the already recorded task
                    self.all_trace[taskID]['fb'] = fb # update the already recorded task

                # with autocast():
                tuple_outputs = self.model(**inputs, labels=labels)
                fe = record_time(device, 'end', 'forward_grad', taskID, timing_info)
                
                self.task_trace['fe'] = fe
                if taskID in self.train_trace: # this training task has been completely recorded
                    self.train_trace[taskID]['fe'] = fe # update the already recorded task
                    self.all_trace[taskID]['fe'] = fe # update the already recorded task
           
            else: # this is a user (test) task
                fb = record_time(device, 'start', 'forward', taskID, timing_info)
                self.task_trace['fb'] = fb
                if taskID in self.all_trace: # this task has been completely recorded
                    self.all_trace[taskID]['fb'] = fb # update the already recorded task
                    
                for task in self.serving_batch:
                    if task.decode_step == 0:
                        self.user_task_record[task.task_id]['start'] = fb
                    
                with torch.no_grad():
                    # with autocast():
                    tuple_outputs = self.model(**inputs, labels=labels)
                fe = record_time(device, 'end', 'forward', taskID, timing_info)
                
                self.task_trace['fe'] = fe
                if taskID in self.all_trace: # this task has been completely recorded
                    self.all_trace[taskID]['fe'] = fe # update the already recorded task
                
                for task in self.serving_batch:
                    seq_length = task.query['input_ids'].shape[1]
                    if task.decode_step == 0:
                        self.user_task_record[task.task_id]['prefill_end'] = fe
                        self.prefills += 1
                        logging.info(f"[Task {task.task_id} (length {seq_length}) finished prefill pass!")
                    else:
                        self.user_task_record[task.task_id]['decode_end'] = fe
                        self.decodes += 1
                        logging.info(f"[Task {task.task_id} (length {seq_length}) finished decode step {task.decode_step}!")
                    self.tokens_generated += 1
                    task.decode_step += 1
                    
            # if self.froward_eta is None:
            #     self.froward_eta = (fe - fb) / (len(self.serving_batch) * input_length ** 2)
                    
            # if self.RECORD_MODE:
            #     # Profile forward time (seconds) per stage
            #     self.record_dict['forward_etas'].append(
            #         ((fe - fb) / (len(self.node_batches[nodeID]) * input_length ** 2), taskID)
            #     )
            #     self.record_dict['FTs'].append((fe - fb, taskID))
                
        except Exception as e:
            logging.error(f"[device {device}] Forward error occurred: {e}")
            tuple_outputs = None
        
        return tuple_outputs 
    

    def globalScheduler(self, taskQueue: queue.Queue, deviceQueue: queue.PriorityQueue) -> None:
        # Global scheduler
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
        # available_memory = self.device_total_memory - allocated_memory
        # # Calculate the available memory ratio
        # available_ratio = available_memory / self.device_total_memory
        # # Check if the available memory ratio is above the threshold
        # return available_ratio > (1 - threshold)
    
    
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


    def run(self):
        
        task_queue = queue.Queue()
        device_queue = queue.PriorityQueue()

        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with ThreadPoolExecutor(max_workers=1) as producer_executor, \
             ThreadPoolExecutor(max_workers=1) as scheduler_executor, \
             ThreadPoolExecutor(max_workers=1) as execution_executor:

            # Submit producers
            producer_executor.submit(self.producer, task_queue)
            
            # Submit global scheduler
            scheduler_executor.submit(
                self.globalScheduler,
                task_queue,
                device_queue,
            )
            # Submit execution tasks
            execution_executor.submit(
                self.device_inference,
                {}, 
                self.preloaded_tasks, 
                device_queue,
            )
        
        # # Save timing info
        # self.save_timing_info()
        
        # # Calculate metrics
        # self.calculate_metrics()