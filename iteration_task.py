# A definition of the Task class and supported functions.
from typing import List, Optional, Dict, Any, Tuple
import time
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


class Task:

    CONTEXT_FEATURE = "context_input_ids"
    CONTEXT_MASK = "context_attention_mask"

    def __init__(
        self,
        taskID: int,
        workload: str,
        rate_lambda: float,
        prompt: Optional[str] = None,
        reference: Optional[str] = None,
        input_kwargs: Optional[Dict[str, List[Any]]] = None,
        past_key_values: Optional[DynamicCache] = None,
        coefficients: Optional[Dict[str, Dict[str, float]]] = None,
        source_dataset: Optional[str] = None,
    ):
        """
        Initialize a Task object.
        taskID (int): unique identifier for the task.
        workload (str): prefill, decode, or train.
        rate_lambda (float): expected request rate of a Poisson process.
        prompt (str): text prompt for the task.
        reference (str): chosen response or response for the task.
        input_kwargs (Dict[str, Any]): tokenized input for the task.
        past_key_values (DynamicCache): cached key-values for the task.
        """
        # Fixed attributes
        self.taskID = taskID
        self.coefficients = coefficients if coefficients is not None else {
            "prefill": {"base_priority": -5, "priority_factor": -6e-2, "latency_coeff": 3.9e-6, "memory_coeff": 1.5e-4},
            "decode": {"base_priority": -7, "priority_factor": -2e-2, "latency_coeff": 9.1e-7, "memory_coeff": 1e-4},
            "train": {"base_priority": -2, "priority_factor": -2e-1, "latency_coeff": 9.9e-6, "memory_coeff": 3e-4},
        }
        if self.coefficients is not None and workload not in self.coefficients:
            raise ValueError(f"Invalid workload: {workload}, supported workloads: {list(self.coefficients.keys())}")
        self.workload = workload
        self.rate_lambda = rate_lambda
        self.prompt = prompt
        self.reference = reference
        self.source_dataset = source_dataset
        # Changable attributes
        self.input_kwargs = input_kwargs if input_kwargs is not None else {}
        self.prompt_length = len(self.input_kwargs.get(self.CONTEXT_FEATURE, []))
        self.past_key_values = past_key_values
        self.step = 0
        self.release_time = None
        self.execution_time = None
        self.finish_time = None    # for training tasks
        self.decode_times = []
        self.response = ""
        self.metrics = {}

    
    def get_priority(self, strategy: str, initial: bool = False) -> float:
        if initial:
            self.release_time = time.time()

        return self.taskID  # For FIFO, we use taskID as the priority
        # if strategy != "async":
        #     priority = self.taskID
        # else:
        #     if initial:
        #         priority = self.coefficients[self.workload]["base_priority"]   
        #     else:
        #         priority = self.coefficients[self.workload]["base_priority"] + self.coefficients[self.workload]["priority_factor"] * (time.time() - self.release_time)
        # return priority
    
    def update_decoding(self, next_token: int, nll: Optional[float] = None):
        self.workload = "decode" if self.workload == "prefill" else self.workload  # prefill -> decode
        self.step += 1
        self.decode_times.append(time.time())
        # Update the input_kwargs
        self.input_kwargs[self.CONTEXT_FEATURE].append(next_token)  # input_ids
        self.input_kwargs[self.CONTEXT_MASK].append(1)  # attention_mask
        # Update NLL sum if provided
        if nll is not None:
            if "nll_sum" not in self.metrics:
                self.metrics["nll_sum"] = 0.0
            self.metrics["nll_sum"] += nll
        
    def get_response(
        self, 
        tokenizer: LlamaTokenizer, 
        output_tokens: Optional[List[int]] = None,
    ):
        output_tokens = self.input_kwargs[self.CONTEXT_FEATURE] if output_tokens is None else output_tokens
        self.response = tokenizer.decode(output_tokens[self.prompt_length:], skip_special_tokens=True)

    def get_input_length(self) -> int:
        return len(self.input_kwargs[self.CONTEXT_FEATURE]) if self.workload != "decode" else 1

    # def get_workload(
    #     self,
    #     model: LlamaForCausalLM,
    #     attn_implementation: str = "flash_attention_2",
    #     new_seq_length: Optional[int] = None,
    # ) -> Tuple[float, float]:
    #     """
    #     Estimate the workload based on the model and input.
    #     This is a placeholder function and should be replaced with actual profiling logic.
    #     """
    #     basic_factor = model.model.config.num_hidden_layers * model.model.config.hidden_size  # num_layers * hidden_dim
    #     if new_seq_length is None:
    #         new_seq_length = self.get_input_length()
    #     length_multiplier = new_seq_length if attn_implementation == "flash_attention_2" else new_seq_length**2
    #     memory_delta = basic_factor * length_multiplier * self.coefficients[self.workload]["memory_coeff"]
    #     latency = basic_factor * length_multiplier * self.coefficients[self.workload]["latency_coeff"]
    #     # print(f"\t[Task {self.taskID} ({self.workload}) with {new_seq_length} new tokens] memory: {memory_delta}, latency: {latency}")
    #     return memory_delta, latency
        
        




if __name__ == "__main__":
    task = Task(taskID=1, workload="train", rate_lambda=10)
    strategy = "async"
    print(task.get_priority(strategy, initial=True))
    print(f"step: {task.step}")
    print(task.past_key_values)
    print(task.prompt)
    print(task.input_kwargs)
    print(task.response)
    time.sleep(10)
    print(task.get_priority(strategy))
