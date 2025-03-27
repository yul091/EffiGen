# A definition of the Task class and supported functions.
from typing import List, Optional, Dict, Any
import time
from transformers.cache_utils import DynamicCache


class Task:

    base_priority = {
        "prefill": 5,
        "decode": 10,
        "train": 1,
    }
    base_factor = {
        "prefill": 5e-2,
        "decode": 1e-2,
        "train": 1e-1,
    }

    def __init__(
        self,
        taskID: int,
        workload: str,
        rate_lambda: float,
        prompt: Optional[str] = None,
        input_kwargs: Optional[Dict[str, Any]] = None,
        past_key_values: Optional[DynamicCache] = None,
    ):
        """
        Initialize a Task object.
        taskID (int): unique identifier for the task.
        workload (str): prefill, decode, or train.
        rate_lambda (float): expected request rate of a Poisson process.
        prompt (str): text prompt for the task.
        response (str): generated response for the task.
        input_kwargs (Dict[str, Any]): tokenized input for the task.
        past_key_values (DynamicCache): cached key-values for the task.
        """
        # Fixed attributes
        self.taskID = taskID
        if workload not in self.base_priority:
            raise ValueError(f"Invalid workload: {workload}")
        self.workload = workload
        self.rate_lambda = rate_lambda
        self.prompt = prompt
        # Changable attributes
        self.input_kwargs = input_kwargs
        self.past_key_values = past_key_values if past_key_values is not None else DynamicCache()
        self.step = 0
        self.release_time = self.get_release_time()
        self.response = ""
        self.priority = self.get_priority(initial=True)


    def get_release_time(self) -> float:
        return time.time()
    
    def get_priority(self, initial: bool = False) -> float:
        if initial:
            return self.base_priority[self.workload]
        else:
            return self.base_priority[self.workload] + self.base_factor[self.workload] * (time.time() - self.release_time)
    
    def update_status(self, workload: str) -> None:
        if workload not in self.base_priority:
            raise ValueError(f"Invalid workload: {workload}")
        self.workload = workload





if __name__ == "__main__":
    task = Task(taskID=1, workload="train", rate_lambda=10)
    print(task.get_priority(initial=True))
    task.update_cache(past_key_values=DynamicCache())
    print(f"step: {task.step}")
    print(task.past_key_values)
    print(task.prompt)
    print(task.input_kwargs)
    print(task.response)
    print(task.get_release_time())
    time.sleep(10)
    print(task.get_priority())
