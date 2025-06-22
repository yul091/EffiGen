# A dummy test for the iteration system.
import os
from typing import List, Optional, Dict, Any
import time
import logging
import traceback
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
import gc
import sys 
sys.dont_write_bytecode = True
import argparse 
from tqdm import tqdm
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from iteration_task import Task  
from iteration_producer import Producer
from iteration_scheduler import Scheduler
from iteration_queue import IterQueue, heapify
from iteration_prefix import PrefixManager
from alignment_study import DPOCollator
from utils import save_metrics_with_order, compute_generation_metrics
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams



def run_experiment(args: argparse.Namespace, experimentID: int):

    # Initialize and run the distributed model
    system = EffiGenTune(args, experimentID=experimentID)
    system.run()
    
    # Clean up resources explicitly
    del system
    torch.cuda.empty_cache()
    gc.collect()




class EffiGenTune:
    """
    A system for efficiently scheduling concurrent serving & retraining tasks for LLMs on a single GPU.
    """
    def __init__(self, args: argparse.Namespace, experimentID: Optional[int] = None):
        """
        Initialize the EffiGenTune system.
        - args (argparse.Namespace): Command line arguments for configuration.
        - experimentID (int, optional): Identifier for the experiment.
        """
        self.experimentID = experimentID
        self.device = args.device
        self.cache_optimization = args.cache_optimization
        self.record_lock = threading.Lock()
        if self.cache_optimization:
            self.prefix_manager = PrefixManager()
        self.max_train_batch_size = args.max_train_batch_size
        self.max_inference_batch_size = args.max_inference_batch_size
        self.inference_engine = args.inference_engine

        # Handle retraining strategy (steps_per_train)
        self.n_test_samples = args.n_test_samples
        self.strategy = args.strategy
        self.retrain_rate = args.retrain_rate
        # Periodic settings
        if self.strategy in {"sync", "async"}:
            steps_per_train = 1
        else:
            # For periodic retraining, we set the steps per train based on the strategy
            steps_per_train = 1 if len(self.strategy.split('-')) == 1 else int(self.strategy.split('-')[1]) 
        
        inference_step_size = int(1 / self.retrain_rate) if self.retrain_rate > 0 else 1
        self.inferece_step_size = min(max(inference_step_size, steps_per_train), self.n_test_samples // 2) if self.n_test_samples > 0 else 1
        # self.inferece_step_size = 1 if len(self.strategy.split('-')) == 1 else int(self.strategy.split('-')[1]) 

        # Handle model paths and configurations
        self.attn_implementation = args.attn_implementation
        self.model_path = args.model_path
        self.loss_threshold = args.loss_threshold
        self.layer_selection = args.layer_selection
        self.layer_threshold = args.layer_threshold
        self.max_new_tokens = args.max_new_tokens

        # Handle data paths and scheduling parameters
        self.data_path = args.data_path if args.data_path != "data/mixed" else [
            "data/Anthropic", 
            "data/StanfordNLP", 
            # "data/OpenAI",
        ]
        self.arrival_rate = args.arrival_rate
        self.arrival_pattern = args.arrival_pattern
        self.output_dir = args.output_dir
        self.memory_threshold = args.memory_threshold
        self.task_limit = args.task_limit
        self.decode_only_threshold = args.decode_only_threshold
        if self.decode_only_threshold < 1:
            raise ValueError("decode_only_threshold must be at least 1, got {self.decode_only_threshold}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map={"": self.device} if self.inference_engine == "HF" else {"": 'cpu'},
            use_cache=True,
            attn_implementation=self.attn_implementation if self.inference_engine == "HF" else "eager",
        )

        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=64,  # Scaling factor, 16
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # ["q_proj", "v_proj"]
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters() 
        self.max_context_length = self.model.config.max_position_embeddings

        # Get producer (loading dataset)
        self.producer = Producer(
            arrival_rate=self.arrival_rate, 
            retrain_rate=self.retrain_rate, 
            n_test_samples=self.n_test_samples,
            strategy=self.strategy,
            arrival_pattern=self.arrival_pattern, 
        )

        # Create scheduler
        self.scheduler = Scheduler(
            self.strategy, 
            lambda1=args.latency_weight, 
            lambda2=args.memory_weight,
        )

        # Get arguments
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
 
        self.kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "logits_processor": LogitsProcessorList([
                MinLengthLogitsProcessor(1, eos_token_id=self.tokenizer.eos_token_id, device=self.device),
            ]),
            "batch_collator": DPOCollator(
                self.tokenizer, 
                inference_input_feature="input_ids", 
                inference_mask_feature="attention_mask",
            ),
            "generation_config": None,
            "loss_threshold": self.loss_threshold,
            "layer_selection": self.layer_selection,
            "layer_threshold": self.layer_threshold,
        }

        self.start_time = time.time()  # Record the start time for logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s] %(message)s',  # 不用 asctime，提高性能
        )
        self.log_with_time(f"\n ** Experiment {self.experimentID+1} **\n")
        self.log_with_time(f"[Strategy {self.strategy}] - inference sample {self.n_test_samples}, retrain rate {self.retrain_rate}, inference step size {self.inferece_step_size}\n")


    def log_with_time(self, msg: str):
        elapsed = time.time() - self.start_time
        logging.info(f"[+{elapsed:.2f}s] {msg}")


    def check_termination(self, task_queue: IterQueue):
        """
        Check if the task queue has reached the end of the preloaded tasks.
        """
        # Check if the last task in the queue is the end signal
        return task_queue.qsize() == 1 and task_queue.queue[-1][-1] is None

    
    def sync_iteration(
        self,
        task_queues: List[IterQueue],
        preloaded_tasks: List[Task],
        records: Dict[str, Any],
    ):
        """
        Periodic iteration for retraining tasks.
        """
        # print(f"[Iteration {records['iteration']}] current_prefills: {records['current_prefills']}, total_prefills: {records['total_prefills']}, train_queue: {task_queues[0].qsize()}, inference_queue: {task_queues[1].qsize()}")
        if (
            ((records["current_prefills"] >= self.inferece_step_size and records["total_trains"] < records["total_prefills"] * self.retrain_rate) or (records["total_prefills"] == self.n_test_samples)) 
            and not self.check_termination(task_queues[0])
        ):
            # For periodic retraining, we run retraining if the fixed interval is reached
            reach_end = True
            self.log_with_time(f"Start training (current prefills: {records['current_prefills']}, total prefills: {records['total_prefills']}, total retrains: {records['total_trains']})")
            while True:
                if task_queues[0].qsize() == 0:
                    # If there is no retraining task, we can break the loop
                    break
                bin, reach_end = self.scheduler.fifo_allocate(
                    task_queues[0], 
                    preloaded_tasks, 
                    iteration=records["iteration"],
                    model=self.model, 
                    max_batch_size=self.max_train_batch_size,
                    workload="train",
                    attn_implementation=self.attn_implementation,
                    eval_metrics=True,
                    logger=self.log_with_time,
                )
                if bin is None:
                    # No suitable bin found, continue to the next iteration
                    break
                training_tasks = bin.get_num_tasks(target="train")

                with self.record_lock:
                    records["tokens"] += training_tasks
                    records["iteration"] += 1

                # Execute the training tasks in the bin
                bin.execute(
                    "train", 
                    self.model, 
                    self.tokenizer, 
                    task_queues[0], 
                    self.optimizer, 
                    memory_threshold=self.memory_threshold, 
                    logger=self.log_with_time,
                    **self.kwargs,
                )
                # print(f"[Iteration {records['iteration']}] scheduled training tasks: {training_tasks}, finished training tasks {bin.finished_training_tasks}")

                with self.record_lock:
                    records["total_trains"] += bin.finished_training_tasks

                    if records["total_trains"] >= records["total_prefills"] * self.retrain_rate:
                        # If the total retraining tasks reach the expected number, we can break the loop
                        # self.log_with_time(f"Reached expected retraining tasks: {records['total_trains']} >= {records['total_prefills']} * {self.retrain_rate}")
                        break

                # # If the bin has finished all training tasks, we can break the loop
                # if bin.finished_training_tasks == training_tasks:
                #     break
            
            # Reset the inference steps count after retraining as the remaining beyond the fixed interval is not counted
            with self.record_lock:
                records["current_prefills"] -= min(self.inferece_step_size, records["current_prefills"])

        else: 
            # Otherwise, we run inference tasks
            bin, reach_end = self.scheduler.fifo_allocate(
                task_queues[1], 
                preloaded_tasks, 
                iteration=records["iteration"],
                model=self.model, 
                max_batch_size=self.max_inference_batch_size,
                workload="inference",
                attn_implementation=self.attn_implementation,
                eval_metrics=True,
                logger=self.log_with_time,
            )
            if bin is None:
                # No suitable bin found, continue to the next iteration
                return
            
            if self.record_lock:
                records["tokens"] += bin.get_num_tasks()
                records["inference_tokens"] += bin.get_num_tasks(target="inference")
                records["current_prefills"] += bin.get_num_tasks(target="prefill")
                records["total_prefills"] += bin.get_num_tasks(target="prefill")
                records["iteration"] += 1

            # Execute the inference tasks in the bin
            bin.concurrent_execute(
                model=self.model, 
                tokenizer=self.tokenizer, 
                task_queue=task_queues[1], 
                optimizer=self.optimizer,
                memory_threshold=self.memory_threshold, 
                logger=self.log_with_time,
                **self.kwargs,
            )
        
        return reach_end
    


    def async_iteration(
        self,
        task_queues: List[IterQueue],
        preloaded_tasks: List[Task],
        records: Dict[str, Any],
    ):
        """
        Continuous iteration for retraining tasks.
        """
        # For async strategy, we use best-fit allocation to find the best bin for retraining tasks
        # print(f"\nHybrid batching (total prefills: {records['total_prefills']}, total retrains: {records['total_trains']})")
        self.log_with_time(f"Hybrid batching (total prefills: {records['total_prefills']}, total retrains: {records['total_trains']})")
        # decode_only_mode = task_queues[0].decode_size >= self.decode_only_threshold  # priority decode batches
        bin, reach_end = self.scheduler.best_fit_allocate(
            task_queues[0], 
            preloaded_tasks, 
            iteration=records["iteration"],
            model=self.model, 
            attn_implementation=self.attn_implementation,
            eval_metrics=True,
            memory_threshold=self.memory_threshold,
            task_limit=self.task_limit,
            max_train_batch_size=self.max_train_batch_size,
            max_inference_batch_size=self.max_inference_batch_size,
            logger=self.log_with_time,
            # decode_only=decode_only_mode,  # ✅ 传入 decode-only flag
        )
        
        if bin is None:
            # No suitable bin found, continue to the next iteration
            return
        
        with self.record_lock:
            records["tokens"] += bin.get_num_tasks()
            records["iteration"] += 1
            records["inference_tokens"] += bin.get_num_tasks(target="inference")
            records["total_prefills"] += bin.get_num_tasks(target="prefill")

        bin.concurrent_execute(
            model=self.model, 
            tokenizer=self.tokenizer, 
            task_queue=task_queues[0], 
            optimizer=self.optimizer,
            memory_threshold=self.memory_threshold, 
            logger=self.log_with_time,
            **self.kwargs,
        )

        with self.record_lock:
            records["total_trains"] += bin.finished_training_tasks
        
        return reach_end
        
        

    def executor(
        self, 
        task_queues: List[IterQueue], 
        preloaded_tasks: List[Task], 
        record_metrics: Dict[str, Any],
    ):
        """
        Executor function to run the tasks in the queue.
        - task_queue (IterQueue): Queue of tasks to be executed.
        - preloaded_tasks (List[Task]): List of preloaded tasks.
        - record_metrics (Dict[str, Any]): Dictionary to record metrics during execution.
        """
        while True:
            qsize = sum(task_queue.qsize() for task_queue in task_queues)
            if qsize == 0:
                # print("No tasks in the queue. Waiting for tasks...")
                time.sleep(0.01)
                continue
            try:
                # if self.strategy in {"sync", "async"}:
                if self.strategy == "async":
                    reach_end = self.async_iteration(task_queues, preloaded_tasks, record_metrics)
                else:
                    reach_end = self.sync_iteration(task_queues, preloaded_tasks, record_metrics)
            except Exception as e:
                # print(f"Error during execution: {e}")
                logging.error(f"Error during execution: {e}")
                traceback.print_exc()
                # If an error occurs, we can break the loop or continue based on the strategy
                break

            if reach_end and all(self.check_termination(task_queue) for task_queue in task_queues): 
                # Because we always put back the end signal for each queue
                # print("Executor reached the end of the preloaded tasks.")
                self.log_with_time("Executor reached the end of the preloaded tasks.")
                break

            # for i, q in enumerate(task_queues):
            #     print(f"[Queue {i}] Size: {q.qsize()}, Content: {list(q.queue)}")
        # print(f"Execution completed in {record_metrics['iteration']} iterations!")
        self.log_with_time(f"Execution completed in {record_metrics['iteration']} iterations!")
        
    
    async def vllm_executor(
        self, 
        task_queue: asyncio.Queue, 
        preloaded_tasks: List[Task], 
        record_metrics: Dict[str, Any],
        engine: AsyncLLMEngine, 
        params: SamplingParams,
    ):
        while True:
            _, _, taskID = await task_queue.get()

            if taskID is None:
                await task_queue.put((float('inf'), None, None))
                break

            task: Task = preloaded_tasks[taskID]
            if task.workload == "train":
                continue

            async for output in engine.generate(
                prompt=task.prompt,
                params=params,
                request_id=str(taskID),
            ):
                self.log_with_time(f"[GEN] Task {taskID} output: {output}")
                if output.finished:
                    task.response = output.outputs[0].text
                    self.log_with_time(f"[DONE] Task {taskID}: {task.response}")
                    break


    async def async_run(self, preloaded_tasks: List[Task]):

        task_queues = [asyncio.Queue()] if self.strategy == "async" else [asyncio.Queue(), asyncio.Queue()]
        record_metrics = {
            "iteration": 0,
            "tokens": 0,
            "inference_tokens": 0,
            "current_prefills": 0,
            "total_prefills": 0,
            "total_trains": 0,
        }

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tokenizer=None,
            dtype="float16",
            tensor_parallel_size=1,
            device=self.device,
            enforce_eager=True,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

        # loop = asyncio.get_event_loop()
        # executor = ThreadPoolExecutor(max_workers=1)
        # loop.run_in_executor(executor, self.producer.produce_async, task_queues, preloaded_tasks)

        # await self.vllm_executor(
        #     task_queue=task_queues[0],
        #     preloaded_tasks=preloaded_tasks,
        #     record_metrics=record_metrics,
        #     engine=engine,
        #     params=sampling_params,
        # )
        await asyncio.gather(
            self.producer.produce_async(task_queues, preloaded_tasks),
            self.vllm_executor(
                task_queue=task_queues[0],
                preloaded_tasks=preloaded_tasks,
                record_metrics=record_metrics,
                engine=engine,
                params=sampling_params,
            )
        )


    # def vllm_executor(
    #     self, 
    #     task_queues: List[IterQueue], 
    #     preloaded_tasks: List[Task], 
    #     record_metrics: Dict[str, Any],
    #     engine: AsyncLLMEngine,
    #     params: SamplingParams,
    # ):
    #     task_queue = task_queues[0]  # or both, if dual queues
    #     active_requests = set()

    #     while True:
    #         try:
    #             _, _, taskID = task_queue.get(timeout=0.01)
    #         except Exception:
    #             taskID = None

    #         task: Task = preloaded_tasks[taskID]
    #         if task.workload == "train":    # Skip training tasks in async executor
    #             continue

    #         # 🔸 Submit 到 engine（并行等待处理）
    #         engine.add_request(
    #             request_id=str(taskID),
    #             inputs=task.prompt,  # one or more prompts
    #             params=params,  
    #             arrival_time=task.release_time,  # 任务的释放时间
    #             priority=task.get_priority(self.strategy, initial=True),  # 初始优先级
    #         )
    #         active_requests.add(str(taskID))

    #         # 🔸 获取完成的结果（非阻塞模式）
    #         results = engine.engine_step()
    #         for r in results:
    #             if r.finished and r.request_id in active_requests:
    #                 task_id = r.request_id
    #                 output = r.outputs[0].text
    #                 task = preloaded_tasks[int(task_id)]
    #                 task.response = output
    #                 self.log_with_time(f"[DONE] Task {task_id}: {output}")
    #                 active_requests.remove(task_id)
                
    #         # ✅ 退出条件：no new task AND nothing left
    #         if taskID is None and not active_requests:
    #             break


    def run(self, preloaded_tasks: Optional[List[Task]] = None):
        """
        Run the system with the given tasks.
        - preloaded_tasks (List[Task]): List of preloaded tasks.
        """
        # def start_priority_refresher(task_queue: IterQueue, preloaded_tasks: List[Task], interval: float = 1.0):
        #     def refresher_loop():
        #         while True:
        #             with task_queue.mutex:
        #                 for i, (priority, workload, taskID) in enumerate(task_queue.queue):
        #                     if taskID is not None:
        #                         task = preloaded_tasks[taskID]
        #                         new_priority = task.get_priority(self.strategy, initial=False)
        #                         task_queue.queue[i] = (new_priority, workload, taskID)
        #                 heapify(task_queue.queue)
        #             # Print the task queue for debugging
        #             # print(f"  **  Task queue: {[(priority, workload, taskID) for priority, workload, taskID in task_queue.queue]} ** \n")
        #             time.sleep(interval)

        #     thread = threading.Thread(target=refresher_loop, daemon=True)
        #     thread.start()

        # Get preloaded tasks
        if preloaded_tasks is None:
            preloaded_tasks = self.producer.load_dataset(
                tokenizer=self.tokenizer,
                max_length=self.max_context_length,
                dataset_name=self.data_path,
            )
  
        start = time.time()
        # # Start background refresher (if strategy is async or sync)
        # if self.strategy == "async":
        #     for task_queue in task_queues:
        #         start_priority_refresher(task_queue, preloaded_tasks, interval=1.0)
        if self.inference_engine == 'vllm':
            asyncio.run(self.async_run(preloaded_tasks))
        else:
            # Create iteration bin and execute tasks
            if self.strategy == "async":
                task_queues = [IterQueue()]
            else:
                # Maintain two queues, first is for retraining, second is for inference
                task_queues = [IterQueue(), IterQueue()]

            # Initialize the record metrics
            record_metrics = {}
            record_metrics["iteration"] = 0
            record_metrics["tokens"] = 0
            record_metrics["inference_tokens"] = 0
            record_metrics["current_prefills"] = 0
            record_metrics["total_prefills"] = 0
            record_metrics["total_trains"] = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Start the producer in a separate thread
                executor.submit(self.producer.produce, task_queues, preloaded_tasks)

                # Start the executor in main thread
                executor.submit(self.executor, task_queues, preloaded_tasks, record_metrics=record_metrics)

        end = time.time()
       
        # Save the task's prompt and response to a file
        output_dir = os.path.join(self.output_dir, self.model_path.split("/")[-1])
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.strategy}_retrain-{self.retrain_rate}_lambda-{self.arrival_rate}.json")
        # if self.layer_selection is not None:
        #     output_file = os.path.join(output_dir, f"{self.strategy}_retrain-{self.retrain_rate}_lambda-{self.arrival_rate}_{self.layer_selection}-{self.layer_threshold}.json")
        # else:
        #     output_file = os.path.join(output_dir, f"{self.strategy}_retrain-{self.retrain_rate}_lambda-{self.arrival_rate}.json")
        if self.layer_selection is not None:
            output_file = output_file.replace(".json", f"_{self.layer_selection}-{self.layer_threshold}.json")
        if self.experimentID is not None:
            self.log_with_time(f"Output file {output_file} (before adding experiment ID)")
            output_file = output_file.replace(".json", f"_{self.experimentID}.json")
            self.log_with_time(f"Output file: {output_file}")


        inference_tasks = [task for task in preloaded_tasks if task.workload != 'train']
        eval_metrics = self.compute_metrics(inference_tasks)
        train_losses = [task.metrics["loss"] for task in preloaded_tasks if task.workload == 'train' and "loss" in task.metrics]
        TTFTs = [task.decode_times[0] - task.release_time for task in preloaded_tasks if task.workload == "decode" and task.decode_times]
        TBTs = [(task.decode_times[-1] - task.execution_time) / len(task.decode_times) for task in preloaded_tasks if task.workload == "decode" and task.decode_times]
        metrics = {
            "strategy": self.strategy,
            "arrival_rate": self.arrival_rate,
            "retrain_rate": self.retrain_rate,
            "num_test_samples": self.n_test_samples,
            "executed_samples": {
                "train": record_metrics["total_trains"], 
                "inference": record_metrics["total_prefills"],
            },
            "iteration": record_metrics["iteration"],
            "tokens": record_metrics["tokens"],
            "inference_tokens": record_metrics["inference_tokens"],
            "total_time": end - start,
            "TTFT": np.mean(TTFTs) if TTFTs else 0,
            "TBT": np.mean(TBTs) if TBTs else 0,
            "throughput": record_metrics["tokens"] / (end - start),
            "throughput_inference": record_metrics["inference_tokens"] / (end - start),
            "decoding_steps": np.mean([task.step for task in preloaded_tasks if task.workload != 'train']) if any(task.workload != 'train' for task in preloaded_tasks) else 0,
            "training_steps": np.mean([task.step for task in preloaded_tasks if task.workload == 'train']) if any(task.workload == 'train' for task in preloaded_tasks) else 0,
            "train_loss": np.mean(train_losses) if train_losses else 0,
            "eval_metrics": eval_metrics,
            "generation_results": [
                {
                    "taskID": task.taskID,
                    "workload": task.workload,
                    "prompt": task.prompt,
                    "prompt_length": task.prompt_length,
                    "response": task.response,
                    "step": task.step,
                    "metrics": task.metrics,
                    "release_time": task.release_time,
                    "execution_time": task.execution_time,
                    "decode_times": task.decode_times,
                    "priority": task.get_priority(self.strategy, initial=False),
                    "source_dataset": task.source_dataset,
                }
                for task in preloaded_tasks
            ],
        }
        print(f"Total time: {end - start}, throughput: {record_metrics['tokens'] / (end - start)} tokens/sec, eval metrics: {eval_metrics}")
        save_metrics_with_order(metrics, output_file)


    def compute_metrics(self, preloaded_tasks: List[Task]) -> Dict[str, float]:
        total_correct = 0
        total_samples = 0
        total_log_prob_diff = 0.0
        total_perplexity = 0.0
        total_loss = 0.0
        total_rougeL = 0.0
        total_bleu = 0.0

        for task in tqdm(preloaded_tasks, desc="Averaging metrics", total=len(preloaded_tasks)):
            if task.workload == "train":
                continue
            eval_outputs = task.metrics
            try:
                total_samples += 1
                total_loss += eval_outputs["loss"]
                total_correct += eval_outputs["correct_preds"]
                total_log_prob_diff += eval_outputs["log_prob_diff"]
                # Generation metrics
                ppl = eval_outputs["nll_sum"] / task.step if task.step > 0 else 0
                total_perplexity += np.exp(ppl)
                gen_metrics = compute_generation_metrics(
                    hypothesis=task.response, 
                    reference=task.reference,
                    tokenizer=self.tokenizer,
                )
                total_rougeL += gen_metrics["rougeL"]
                total_bleu += gen_metrics["bleu"]

            except KeyError as e:
                # print(f"KeyError: {e} in task {task.taskID}. Skipping this task.")
                self.log_with_time(f"KeyError: {e} in task {task.taskID}. Skipping this task.")
                continue

        # Compute final averages
        preference_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_clpd = total_log_prob_diff / total_samples if total_samples > 0 else 0
        avg_perplexity = total_perplexity / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_rougeL = total_rougeL / total_samples if total_samples > 0 else 0
        avg_bleu = total_bleu / total_samples if total_samples > 0 else 0

        return {
            "loss": avg_loss,
            "preference accuracy": preference_accuracy,
            "CLPD": avg_clpd,
            "PPL": avg_perplexity,
            "rougeL": avg_rougeL,
            "bleu": avg_bleu,
        }
        


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU device number")
    parser.add_argument("--strategy", type=str, default="sync", help="Scheduling strategy")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", choices=["flash_attention_2", "eager"], help="Attention implementation")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Path to the model")
    parser.add_argument("--data_path", type=str, default="data/Anthropic", help="Path to the dataset")
    parser.add_argument("--arrival_rate", type=float, default=10, help="Arrival rate of tasks")
    parser.add_argument("--arrival_pattern", type=str, default="poisson", help="Arrival pattern of tasks")
    parser.add_argument("--n_test_samples", type=int, default=25, help="Number of test samples")
    parser.add_argument("--retrain_rate", type=float, default=0.4, help="Ratio of retraining tasks over inference tasks")
    parser.add_argument("--latency_weight", type=float, default=0.5, help="Weight for latency in scheduling")
    parser.add_argument("--memory_weight", type=float, default=0.5, help="Weight for memory in scheduling")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--output_dir", type=str, default="profile_main/dummy", help="Output directory for saving metrics")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--memory_threshold", type=float, default=0.95, help="Memory threshold for bin packing")
    parser.add_argument("--task_limit", type=int, default=50, help="Task limit for bin packing")
    parser.add_argument("--loss_threshold", type=float, default=0.7, help="Loss threshold for selective training")
    parser.add_argument("--cache_optimization", action="store_true", help="Use prefix sharing for prefilling acceleration")
    parser.add_argument("--layer_selection", type=str, default=None, choices=["RGN", "SNR"], help="Layer selection method for selective training")
    parser.add_argument("--layer_threshold", type=float, default=0.5, help="Layer threshold for selective training")
    parser.add_argument("--max_train_batch_size", type=int, default=5, help="Maximum training batch size for LoRA training")
    parser.add_argument("--max_inference_batch_size", type=int, default=50, help="Maximum inference batch size for inference")
    parser.add_argument("--decode_only_threshold", type=int, default=100, help="Threshold for decode-only mode in async strategy")
    parser.add_argument("--inference_engine", type=str, default="HF", choices=["HF", "FT", "vllm"], help="Inference engine to use")
    parser.add_argument('--experiments', type=int, default=1, help='number of experiments')
    args = parser.parse_args()
    
    
    # system = EffiGenTune(args)
    # system.run()
    if args.inference_engine == 'vllm':
        mp.set_start_method("spawn", force=True)

    for i in range(args.experiments):
        run_experiment(args, i)

    

    