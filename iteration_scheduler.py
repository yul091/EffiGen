# A definition of the Scheduler class and supported functions.
from typing import List, Tuple
import numpy as np
import sys 
sys.dont_write_bytecode = True
from iteration_task import Task  


class Scheduler:
    def __init__(self, lambda1: float = 0.5, lambda2: float = 0.5, K: int = 3):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.K = K  # number of bins to search for best-fit
    
    def look_ahead(self, task: Task) -> Tuple[float, float]:
        """
        Dummy memory & latency estimator (to be replaced with real profiler).
        """
        return task.memory_estimate, task.latency_estimate

    def bin_allocator(self, task_queue: List[Task], bin_capacity: float) -> List[List[Task]]:
        """
        Priority-aware best-fit bin packing considering both memory & latency.
        """
        bins: List[List[Task]] = []

        # Sort by priority (higher priority goes first)
        sorted_tasks = sorted(task_queue, key=lambda t: -t.priority)

        for task in sorted_tasks:
            task_mem, task_lat = self.look_ahead(task)
            best_bin_index = None
            best_score = float("inf")

            # Search top-K best bins
            for i in range(min(self.K, len(bins))):
                current_bin = bins[i]
                mem_used = sum(t.memory_estimate for t in current_bin)
                lat_max = max((t.latency_estimate for t in current_bin), default=0)

                if mem_used + task_mem > bin_capacity:
                    continue  # skip bin if memory exceeds

                mem_fit = abs((bin_capacity - mem_used) - task_mem)
                lat_fit = abs(lat_max - task_lat)
                score = self.lambda1 * mem_fit + self.lambda2 * lat_fit

                if score < best_score:
                    best_score = score
                    best_bin_index = i

            if best_bin_index is not None:
                bins[best_bin_index].append(task)
            else:
                bins.append([task])  # new bin

        return bins