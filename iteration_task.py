# A definition of the Task class and supported functions.
from typing import List, Optional, Dict, Any, Tuple
import time
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


class Task:

    coefficients = {
        "prefill": {"base_priority": -5, "priority_factor": -5e-2, "latency_coeff": 3.9e-6, "memory_coeff": 5e-5},
        "decode": {"base_priority": -10, "priority_factor": -1e-2, "latency_coeff": 9.1e-7, "memory_coeff": 4.9e-5},
        "train": {"base_priority": -1, "priority_factor": -1e-1, "latency_coeff": 9.9e-6, "memory_coeff": 1.5e-4},
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
        if workload not in self.coefficients:
            raise ValueError(f"Invalid workload: {workload}")
        self.workload = workload
        self.rate_lambda = rate_lambda
        self.prompt = prompt
        # Changable attributes
        self.input_kwargs = input_kwargs if input_kwargs is not None else {}
        self.prompt_length = len(self.input_kwargs.get(self.CONTEXT_FEATURE, []))
        self.past_key_values = past_key_values
        self.step = 0
        self.release_time = None
        self.priority = self.get_priority(initial=True)
        self.response = ""
        self.metrics = {}


    def get_release_time(self) -> float:
        return time.time()
    
    def get_priority(self, initial: bool = False) -> float:
        if initial:
            self.release_time = self.get_release_time()
            return self.coefficients[self.workload]["base_priority"]
        else:
            return self.coefficients[self.workload]["base_priority"] + self.coefficients[self.workload]["priority_factor"] * (time.time() - self.release_time)
    
    def update_decoding(self, next_token: int):
        self.workload = "decode" if self.workload == "prefill" else self.workload  # prefill -> decode
        self.step += 1
        # Update the input_kwargs
        self.input_kwargs[self.CONTEXT_FEATURE].append(next_token)
        self.input_kwargs[self.CONTEXT_MASK].append(1)
        
    def get_response(
        self, 
        tokenizer: LlamaTokenizer, 
        output_tokens: Optional[List[int]] = None,
    ):
        output_tokens = self.input_kwargs[self.CONTEXT_FEATURE] if output_tokens is None else output_tokens
        self.response = tokenizer.decode(output_tokens[self.prompt_length:], skip_special_tokens=True)

    def get_workload(
        self,
        model: LlamaForCausalLM,
        attn_implementation: str = "flash_attention_2",
        new_seq_length: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Estimate the workload based on the model and input.
        This is a placeholder function and should be replaced with actual profiling logic.
        """
        basic_factor = model.model.config.num_hidden_layers * model.model.config.hidden_size  # num_layers * hidden_dim
        if new_seq_length is None:
            new_seq_length = len(self.input_kwargs[self.CONTEXT_FEATURE]) if self.workload != "decode" else 1
        length_multiplier = new_seq_length if attn_implementation == "flash_attention_2" else new_seq_length**2
        memory_delta = basic_factor * length_multiplier * self.coefficients[self.workload]["memory_coeff"]
        latency = basic_factor * length_multiplier * self.coefficients[self.workload]["latency_coeff"]
        # print(f"\t[Task {self.taskID} ({self.workload}) with {new_seq_length} new tokens] memory: {memory_delta}, latency: {latency}")
        return memory_delta, latency
        
        




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
