# A definition of the Scheduler class and supported functions.
import logging
from typing import List, Tuple, Optional, Callable
from transformers import AutoModelForCausalLM
import sys 
sys.dont_write_bytecode = True
import torch
from iteration_task import Task  
from iteration_bin import Bin
from iteration_queue import IterQueue


class Scheduler:
    def __init__(self, strategy: str, lambda1: float = 0.5, lambda2: float = 0.5):
        self.strategy = strategy
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    # def look_ahead(self, task: Task) -> Tuple[float, float]:
    #     """
    #     Memory & latency estimator.
    #     """
    #     return task.get_workload()

    def memory_saturation(self, bin: Bin, memory_threshold: float) -> bool:
        """
        Check if the bin is saturated with memory.
        """
        return torch.cuda.memory_allocated(bin.device) / (1024**2) / bin.memory_capacity > memory_threshold

    def best_fit_allocate(
        self, 
        task_queue: IterQueue, 
        preloaded_tasks: List[Task],
        iteration: int, 
        model: AutoModelForCausalLM,
        attn_implementation: str = "flash_attention_2",
        eval_metrics: bool = False,
        memory_threshold: float = 0.95,
        task_limit: int = 50,
        max_train_batch_size: int = 2,
        max_inference_batch_size: int = 50,
        logger: Optional[Callable] = None,
        decode_only: bool = False,
    ) -> Tuple[Bin, bool]:
        """
        Priority-aware best-fit bin packing considering both memory & latency.
        """
        # print(f"  **  [Iteration {iteration}] Start scheduling tasks with memory threshold {memory_threshold:.2f}...  **")
        bins: List[Bin] = []
        reach_end = False
        initial_qsize, itrain, iprefill, idecode = task_queue.qsize(), task_queue.train_size, task_queue.prefill_size, task_queue.decode_size
        task_count, strain, sprefill, sdecode = 0, 0, 0, 0
        # skipped_tasks = []  
        while task_queue.qsize() > 0:
            if task_count >= task_limit or (bins and self.memory_saturation(bins[0], memory_threshold)):
                break
            _, _, taskID = task_queue.get()  # priority, workload, taskID
            if taskID is None:
                # Reach the end of the preloaded tasks
                task_queue.put((float('inf'), None, None))  # Put back the end signal since we may have unfinished decoding tasks
                reach_end = True
                break
            
            task: Task = preloaded_tasks[taskID]
            # if decode_only and task.workload != "decode":
            #     # task_queue.put((task.get_priority(self.strategy, initial=False), task.workload, task.taskID))
            #     skipped_tasks.append(task)
            #     continue

            task_count += 1
            if task.workload == "train":
                strain += 1
            elif task.workload == "prefill":
                sprefill += 1
            elif task.workload == "decode":
                sdecode += 1

            
            best_bin, best_score = None, float('inf')
            for idx, bin in enumerate(bins):
                # Check if the bin can accommodate the task
                if idx == 0 and task.workload == "train" and bin.get_num_tasks("train") >= max_train_batch_size:
                    continue
                if idx == 0 and task.workload != "train" and bin.get_num_tasks("inference") >= max_inference_batch_size:
                    continue

                _, _, _, memory_fit, latency_fit = bin.get_workload(task, model, attn_implementation=attn_implementation)
                # print(f"  **  [Iteration {iteration} - Scheduling task {taskID} ({task.workload})] - (seq_len {batch_length}, memory {batch_memory:.2f}, latency {batch_latency:.2f}) - bin {idx} (base memory {bin.base_memory:.2f}, memory capacity {bin.memory_capacity}, max latency {bin.max_latency:.2f}, workload stats {bin.workload_stats}) - memory fit {memory_fit:.2f} / latency fit {latency_fit:.2f}  **")
                if memory_fit > 0:
                    score = self.lambda1 * memory_fit + self.lambda2 * latency_fit
                    if score < best_score:
                        best_bin = bin
                        best_score = score

            if best_bin is not None:
                best_bin.add_task(task, model, attn_implementation=attn_implementation)
                # print(f" - Task {taskID} ({task.workload}) is assigned to bin {bins.index(best_bin)} (with accum memory {best_bin.total_memory} and max latency {best_bin.max_latency})")
            else:  
                # current bins are exhausted
                new_bin = Bin(self.strategy, device=model.device, eval_metrics=eval_metrics)
                new_bin.add_task(task, model, attn_implementation=attn_implementation)
                bins.append(new_bin)
                # print(f" - Task {taskID} ({task.workload}) is assigned to bin {bins.index(new_bin)} (with accum memory {new_bin.total_memory} and max latency {new_bin.max_latency})")

            # if decode_only:
            #     if bins and bins[0].get_num_tasks("inference") >= max_inference_batch_size:
            #         break

        # Put the remaining tasks (from remaining bin (if exists)) back into the queue
        if bins:
            # print(f"  **  [Iteration {iteration}] queue size {initial_qsize} (prefill {iprefill}, decode {idecode}, train {itrain}) --- involve {task_count} tasks (prefill {sprefill}, decode {sdecode}, train {strain}) --- schedule {bins[0].get_num_tasks()} tasks (prefill {len(bins[0].prefill_batch)}, decode {len(bins[0].decode_batch)}, train {len(bins[0].train_batch)}) --- {len(bins)} bins created  **  ")
            if logger is not None:
                # mode = "decode-only" if decode_only else "hybrid"
                logger(f"  **  [Iteration {iteration}] queue size {initial_qsize} (prefill {iprefill}, decode {idecode}, train {itrain}) --- involve {task_count} tasks (prefill {sprefill}, decode {sdecode}, train {strain}) --- schedule {bins[0].get_num_tasks()} tasks (prefill {len(bins[0].prefill_batch)}, decode {len(bins[0].decode_batch)}, train {len(bins[0].train_batch)}) --- {len(bins)} bins created  **  ")
        
        # End of loop
        # for task in skipped_tasks:
        #     task_queue.put((task.get_priority(self.strategy, initial=False), task.workload, task.taskID))
        if len(bins) > 1:
            for i in range(1, len(bins)):
                for task in bins[i].prefill_batch + bins[i].decode_batch + bins[i].train_batch:
                    task_queue.put((task.get_priority(self.strategy, initial=False), task.workload, task.taskID))

        # Return the next bin for execution
        return bins[0] if bins else None, reach_end



    def fifo_allocate(
        self, 
        task_queue: IterQueue, 
        preloaded_tasks: List[Task],
        iteration: int, 
        model: AutoModelForCausalLM,
        max_batch_size: int,
        workload: str,
        attn_implementation: str = "flash_attention_2",
        eval_metrics: bool = False,
        logger: Optional[Callable] = None,
    ) -> Tuple[Bin, bool]:
        """
        FIFO with predefined maximum batch sizes for a specific workload ('inference' or 'train').
        We ensure that task_queue only contains one type of workload.
        """
        bin = None
        reach_end = False
        initial_qsize = task_queue.qsize()
        while task_queue.qsize() > 0:
            if bin is None:
                # Create a new bin for the current workload
                bin = Bin(self.strategy, device=model.device, eval_metrics=eval_metrics)
            if bin.get_num_tasks(workload) == max_batch_size:
                # Reached the maximum number of tasks for inference or training
                break
            _, _, taskID = task_queue.get()
            if taskID is None:
                # Reach the end of the preloaded tasks
                task_queue.put((float('inf'), None, None))
                reach_end = True
                break
            task: Task = preloaded_tasks[taskID]
            # Add task to the bin
            bin.add_task(task, model, attn_implementation=attn_implementation)

        if bin is not None:
            # print(f"  **  [Iteration {iteration} | {workload}] queue size {initial_qsize} -- schedule {bin.get_num_tasks()} tasks (prefill {bin.get_num_tasks('prefill')}, decode {bin.get_num_tasks('decode')}, train {bin.get_num_tasks('train')})  **  ")
            if logger is not None:
                logger(f"  **  [Iteration {iteration} | {workload}] queue size {initial_qsize} -- schedule {bin.get_num_tasks()} tasks (prefill {bin.get_num_tasks('prefill')}, decode {bin.get_num_tasks('decode')}, train {bin.get_num_tasks('train')})  **  ")
        return bin, reach_end