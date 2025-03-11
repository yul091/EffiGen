

import sys
sys.dont_write_bytecode = True
from typing import Optional, List
import logging
import torch
import time
import argparse
import queue
from peft import LoraConfig, get_peft_model
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria, MinLengthLogitsProcessor
import gc
from mix_pipeline_base import BasicPipeline
from utils import Task, record_time
scaler = torch.amp.GradScaler("cuda")  # Add this during initialization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_experiment(args, model, tokenizer, device, experimentID: int):
    print(f"\n ** Experiment {experimentID+1} **\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize and run the distributed model
    llm_pipeline = SequentialPipeline(args, model, tokenizer, device, experimentID=experimentID)
    llm_pipeline.run()
    record_mode = llm_pipeline.RECORD_MODE
    
    # Clean up resources explicitly
    del llm_pipeline
    torch.cuda.empty_cache()
    gc.collect()

    # Rerun if necessary based on specific conditions
    if record_mode and args.run_mode == 'online':  # Assuming record_mode is a valid arg
        new_run = SequentialPipeline(args, model, tokenizer, device, experimentID=experimentID)
        new_run.run()
        
        # Final clean up
        del new_run
        torch.cuda.empty_cache()
        gc.collect()



class SequentialPipeline(BasicPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = []


    def globalScheduler(self, taskQueue, deviceQueue):
        while True:
            taskID: int = taskQueue.get() # ID
            if taskID is None:
                deviceQueue.put((float('inf'), float('inf'))) # for priority_queue, use a large number to signal the end
                print("Global scheduler finished scheduling tasks")
                break
            
            # Calculate priority
            deviceQueue.put((taskID, taskID)) # ID
            # print("Global scheduler scheduled task {} (requre_training={})".format(taskID, self.distributed_preloaded_tasks[0][taskID].require_training))

    
    
    def device_inference(
        self, 
        preloaded_tasks: List[Task], 
        deviceQueue: queue.PriorityQueue, 
        timing_info: Optional[dict] = None, 
        device: Optional[int] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
    ):
        timing_info = timing_info if timing_info is not None else self.timing_info
        device = device if device is not None else self.device
        max_length = max_length if max_length is not None else 128
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList([MinLengthLogitsProcessor(10, eos_token_id=self.tokenizer.eos_token_id),])
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
        
       
        while True:
            # If some tasks are currently being trained, wait for them to finish
            while self.isBatchTraining:
                time.sleep(0.1)

            priority, taskID = deviceQueue.get()
            print(f"Retrieved task {taskID} from deviceQueue")

            if taskID == float('inf'):
                # Signal that this thread is done
                print(f"Received termination signal, ending inference.")
                break
            
            task: Task = preloaded_tasks[taskID]
            assert task.task_id == taskID

            if not task.require_training:
                hybrid_batch = self.continuous_serving_batching(
                    taskID=taskID, 
                    priority=priority, 
                    task=task, 
                    deviceQueue=deviceQueue, 
                    preloaded_tasks=preloaded_tasks, 
                    max_wait_time=0.1,
                )  
            else:
                hybrid_batch = self.continuous_training_batching(
                    taskID=taskID, 
                    priority=priority, 
                    task=task, 
                    deviceQueue=deviceQueue, 
                    preloaded_tasks=preloaded_tasks, 
                    max_wait_time=0.1,
                )  
            
            if hybrid_batch is None:
                print(f"Waiting for inputs for task {taskID}")
                continue   
            batch_size, input_length = hybrid_batch['input_ids'].shape
                
            # prepare inputs
            task.feedback = hybrid_batch.pop('labels', None)

            # Memory check and scale down if necessary
            if self.wait_for_device_availability(device):
                outputs = self.forward(
                    taskID=taskID, 
                    inputs=hybrid_batch, 
                    device=device, 
                    timing_info=timing_info, 
                    require_training=task.require_training, 
                    labels=task.feedback,
                )
            else:
                outputs = None
                print(f"Failed waiting for device {device} to be available, dropping task {taskID}")
            # tuple_outputs = self.forward(task, inputs, stageID, nodeID, device, timing_info)
            # task.hiddens[stageID] = None # clear the input that is no longer needed

            if outputs is None:  # Error occurred
                continue
                
            loss = outputs.loss
            # print(f"Finished task {taskID} with loss {loss}")
            # self.metrics["loss"].append(loss.item())
            if task.require_training:
                self.metrics["train_loss"].append(loss.item())
            else:
                self.autoregressive_decoding(
                    hybrid_batch, logits_processor, stopping_criteria, device, batch_size, lm_logits=outputs[1],
                )
            
            if task.do_backward and self.wait_for_device_availability(device, force_check=True):
                # Backprop on the last stage
                bb = time.time()
                self.task_trace['bb'] = bb
                if taskID in self.train_trace:
                    self.train_trace[taskID]['bb'] = bb # update the recorded time
                    self.all_trace[taskID]['bb'] = bb # update the recorded time
                try:
                    # loss.backward()
                    scaler.scale(loss).backward()
                    be = record_time(device, 'end', 'backward', taskID, timing_info)
                    self.task_trace['be'] = be
                    if taskID in self.train_trace:
                        self.train_trace[taskID]['be'] = be   
                        self.all_trace[taskID]['be'] = be
                    
                    print("Stage {} finish backward propagation for task {} !".format(device, taskID))
                except Exception as e:
                    logging.error(f"[task {taskID}] Backward error occurred: {e}")
                    pass

                # Optimization
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # for stability
                # Ensure scaler is initialized before calling step()
                try:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                except Exception as e:
                    # logging.error(f"[task {taskID}] Optimization error occurred: {e}")  
                    pass
                
                self.optimizer.zero_grad() # clear gradients
                
                # self._training_step += 1

            if task.require_training:
                self.training_batch.clear()
                self.isBatchTraining = False
 



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama//Llama-2-7b-chat-hf', help='model name or path')
    parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--memory_threshold', type=float, default=0.8, help='threshold for maximum memory allocation in each GPU device')
    parser.add_argument('--device', type=int, default=0, help='device ID')
    parser.add_argument('--max_wait', type=float, default=10, help='maximum time to wait from available memory')
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_length', action='store_true', help='save the length of each task')
    parser.add_argument('--serving_batch_size', type=int, default=3)
    parser.add_argument('--training_batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=int, default=10, help='Average number of tasks produced per second')
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--profile_dir', type=str, default='profile', help='directory to save profiling results')
    parser.add_argument('--experiments', type=int, default=1, help='number of experiments')
    parser.add_argument('--run_mode', type=str, default='online', choices=['online', 'offline'], help='Whether to use RECORD MODEL for offline profiling')
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    args = parser.parse_args()

    model_path = args.model_name_or_path
    if model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
        )
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=args.use_fast_tokenizer,
                padding_side="left",
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                padding_side="left"
            )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
    ).to(args.device)

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )

    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    
    for i in range(args.experiments):
        run_experiment(args, model, tokenizer, args.device, i)