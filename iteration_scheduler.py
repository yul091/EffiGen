# A definition of the Scheduler class and supported functions.
from typing import List
from queue import PriorityQueue
from transformers import AutoModelForCausalLM
import sys 
sys.dont_write_bytecode = True
from iteration_task import Task  
from iteration_bin import Bin


class Scheduler:
    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    # def look_ahead(self, task: Task) -> Tuple[float, float]:
    #     """
    #     Memory & latency estimator.
    #     """
    #     return task.get_workload()

    def best_fit_allocate(
        self, 
        task_queue: PriorityQueue, 
        preloaded_tasks: List[Task], 
        model: AutoModelForCausalLM,
        **bin_kwargs,
    ) -> Bin:
        """
        Priority-aware best-fit bin packing considering both memory & latency.
        """
        bins: List[Bin] = []

        while task_queue.qsize() > 0:
            _, taskID = task_queue.get()
            task: Task = preloaded_tasks[taskID]
            best_bin, best_score = None, float('inf')
            # Get task workload anticipation
            memory, latency = task.get_workload(model, bin_kwargs.get("attn_implementation", "flash_attention_2"))

            for bin in bins:
                # Check if the bin can accommodate the task
                if bin.free_memory >= memory:
                    memory_fit = abs(bin.free_memory - memory)
                    latency_fit = abs(bin.max_latency - latency)
                    score = self.lambda1 * memory_fit + self.lambda2 * latency_fit
                    if score < best_score:
                        best_bin = bin
                        best_score = score

            if best_bin is not None:
                best_bin.add_task(task)
                best_bin.update_workload(operation="add", memory=memory, latency=latency)
            else:  
                # current bins are exhausted
                new_bin = Bin(eval_metrics=bin_kwargs.get("eval_metrics", False))
                new_bin.add_task(task)
                new_bin.update_workload(operation="add", memory=memory, latency=latency)
                bins.append(new_bin)

        # Put the remaining tasks (from remaining bin (if exists)) back into the queue
        # print(f"  **  Current bins {bins}  **  ")
        if len(bins) > 1:
            for i in range(1, len(bins)):
                for task in bins[i].prefill_batch + bins[i].decode_batch + bins[i].train_batch:
                    task_queue.put((task.get_priority(initial=False), task.taskID))

        # Return the next bin for execution
        return bins[0]  



