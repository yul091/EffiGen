# A definition of the Scheduler class and supported functions.
from typing import List, Tuple, Optional
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
        iteration: int, 
        model: AutoModelForCausalLM,
        attn_implementation: str = "flash_attention_2",
        eval_metrics: bool = False,
        memory_threshold: Optional[float] = 0.95,
    ) -> Tuple[Bin, bool]:
        """
        Priority-aware best-fit bin packing considering both memory & latency.
        """
        bins: List[Bin] = []
        reach_end = False
        initial_qsize = task_queue.qsize()

        while task_queue.qsize() > 0:
            if bins and bins[0].total_memory >= memory_threshold * bins[0].memory_capacity:
                break
            _, taskID = task_queue.get()
            if taskID is None:
                # Reach the end of the preloaded tasks
                task_queue.put((float('inf'), None))  # Put back the end signal since we may have unfinished decoding tasks
                reach_end = True
                break

            task: Task = preloaded_tasks[taskID]
            best_bin, best_score = None, float('inf')
            # Get task workload anticipation
            memory, latency = task.get_workload(model, attn_implementation=attn_implementation)

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
                # print(f" - Task {taskID} ({task.workload}) is assigned to bin {bins.index(best_bin)} (with accum memory {best_bin.total_memory} and max latency {best_bin.max_latency})")
            else:  
                # current bins are exhausted
                new_bin = Bin(device=model.device, eval_metrics=eval_metrics)
                new_bin.add_task(task)
                new_bin.update_workload(operation="add", memory=memory, latency=latency)
                bins.append(new_bin)
                # print(f" - Task {taskID} ({task.workload}) is assigned to bin {bins.index(new_bin)} (with accum memory {new_bin.total_memory} and max latency {new_bin.max_latency})")

        # Put the remaining tasks (from remaining bin (if exists)) back into the queue
        # print(f" - Current bins's anticipation: {[(bin.total_memory, bin.max_latency) for bin in bins]}")
        if bins:
            print(f" ** [Iteration {iteration}] queue size {initial_qsize}, {initial_qsize - task_queue.qsize()} tasks participated, {bins[0].get_num_tasks()} tasks scheduled (prefill {len(bins[0].prefill_batch)}, decode {len(bins[0].decode_batch)}, train {len(bins[0].train_batch)}), {len(bins)} bins created")
        if len(bins) > 1:
            for i in range(1, len(bins)):
                for task in bins[i].prefill_batch + bins[i].decode_batch + bins[i].train_batch:
                    task_queue.put((task.get_priority(initial=False), task.taskID))

        # Return the next bin for execution
        return bins[0] if bins else None, reach_end



