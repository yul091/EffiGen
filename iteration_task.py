# A definition of the Task class and supported functions.
from typing import List, Optional, Dict, Any
import time
from transformers import GPT2Tokenizer
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

    CONTEXT_FEATURE = "context_input_ids"
    CONTEXT_MASK = "context_attention_mask"

    def __init__(
        self,
        taskID: int,
        workload: str,
        rate_lambda: float,
        prompt: Optional[str] = None,
        input_kwargs: Optional[Dict[str, List[Any]]] = None,
        past_key_values: Optional[DynamicCache] = None,
    ):
        """
        Initialize a Task object.
        taskID (int): unique identifier for the task.
        workload (str): prefill, decode, or train.
        rate_lambda (float): expected request rate of a Poisson process.
        prompt (str): text prompt for the task.
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
        self.input_kwargs = input_kwargs if input_kwargs is not None else {}
        # self.past_key_values = past_key_values if past_key_values is not None else DynamicCache()
        self.past_key_values = past_key_values
        self.step = 0
        self.release_time = self.get_release_time()
        self.response = None
        self.priority = self.get_priority(initial=True)
        self.prompt_length = len(self.input_kwargs.get(self.CONTEXT_FEATURE, []))
        self.next_token = None


    def get_release_time(self) -> float:
        return time.time()
    
    def get_priority(self, initial: bool = False) -> float:
        if initial:
            return self.base_priority[self.workload]
        else:
            return self.base_priority[self.workload] + self.base_factor[self.workload] * (time.time() - self.release_time)
    
    def update_decoding(self, next_token: int) -> None:
        self.workload = "decode"
        self.step += 1
        # Update the input_kwargs
        self.input_kwargs[self.CONTEXT_FEATURE].append(next_token)
        self.input_kwargs[self.CONTEXT_MASK].append(1)
        

    def get_response(
        self, 
        output_tokens: List[int],
        tokenizer: GPT2Tokenizer, 
    ) -> None:
        # print(f"Output length: {len(output_tokens)}, context_length: {self.prompt_length}")
        assert len(output_tokens) >= self.prompt_length, "Output length is less than context length"
        self.response = tokenizer.decode(output_tokens[self.prompt_length:], skip_special_tokens=True)




if __name__ == "__main__":
    task = Task(taskID=1, workload="train", rate_lambda=10)
    print(task.get_priority(initial=True))
    print(f"step: {task.step}")
    print(task.past_key_values)
    print(task.prompt)
    print(task.input_kwargs)
    print(task.response)
    print(task.get_release_time())
    time.sleep(10)
    print(task.get_priority())
