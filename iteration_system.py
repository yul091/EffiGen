# A dummy test for the iteration system.
import os
from typing import List, Optional, Dict, Any
import time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
import sys 
sys.dont_write_bytecode = True
import argparse 
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
from iteration_task import Task  
from iteration_producer import Producer
from iteration_scheduler import Scheduler
from iteration_queue import IterQueue, heapify
from alignment_study import DPOCollator
from utils import save_metrics_with_order



class EffiGenTune:
    """
    A system for efficiently scheduling concurrent serving & retraining tasks for LLMs on a single GPU.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the EffiGenTune system.
        - device (int): GPU device number.
        - strategy (str): Scheduling strategy ("async" or "sync").
        - attn_implementation (str): Attention implementation ("flash_attention_2" or "eager").
        """
        self.device = args.device
        self.strategy = args.strategy
        self.attn_implementation = args.attn_implementation
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.arrival_rate = args.arrival_rate
        self.arrival_pattern = args.arrival_pattern
        self.n_test_samples = args.n_test_samples
        self.retrain_rate = args.retrain_rate
        self.output_dir = args.output_dir
        self.max_new_tokens = args.max_new_tokens
        self.memory_threshold = args.memory_threshold
        self.task_limit = args.task_limit

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
            device_map={"": self.device},
            use_cache=True,
            attn_implementation=self.attn_implementation,
        )

        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=16, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05, 
            task_type="CAUSAL_LM", 
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.max_context_length = self.model.config.max_position_embeddings

        # Get producer (loading dataset)
        self.producer = Producer(
            arrival_rate=self.arrival_rate, 
            retrain_rate=self.retrain_rate, 
            n_test_samples=self.n_test_samples,
            arrival_pattern=self.arrival_pattern, 
        )

        # Create scheduler
        self.scheduler = Scheduler(lambda1=args.latency_weight, lambda2=args.memory_weight)

        # Get arguments
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
 
        self.bin_kwargs = {
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
        }


    def executor(
        self, 
        task_queue: IterQueue, 
        preloaded_tasks: List[Task], 
        record_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Executor function to run the tasks in the queue.
        - task_queue (IterQueue): Queue of tasks to be executed.
        - preloaded_tasks (List[Task]): List of preloaded tasks.
        """
        # iteration = 0
        # tokens = 0
        record_metrics = record_metrics if record_metrics is not None else {}
        record_metrics["iteration"] = 0
        record_metrics["tokens"] = 0
        
        while True:
        # while task_queue.qsize() > 0:
            qsize = task_queue.qsize()
            if qsize == 0:
                # print("No tasks in the queue. Waiting for tasks...")
                time.sleep(0.01)
                continue
            
            # tokens += task_queue.qsize()
            record_metrics["tokens"] += qsize
            bin, reach_end = self.scheduler.best_fit_allocate(
                task_queue, 
                preloaded_tasks, 
                iteration=record_metrics["iteration"],
                model=self.model, 
                attn_implementation=self.attn_implementation,
                eval_metrics=True,
                memory_threshold=self.memory_threshold,
                task_limit=self.task_limit,
            )
            
            bin.concurrent_execute(
                model=self.model, 
                tokenizer=self.tokenizer, 
                task_queue=task_queue, 
                strategy=self.strategy,
                optimizer=self.optimizer,
                **self.bin_kwargs,
            )
            # iteration += 1
            record_metrics["iteration"] += 1
            if reach_end and task_queue.qsize() == 1:  # 1 because we always put back the end signal
                print("Executor reached the end of the preloaded tasks.")
                break

        print(f"Execution completed in {record_metrics['iteration']} iterations!")
        
        


    def run(self, preloaded_tasks: Optional[List[Task]] = None):
        """
        Run the system with the given tasks.
        - preloaded_tasks (List[Task]): List of preloaded tasks.
        """

        def start_priority_refresher(task_queue: IterQueue, preloaded_tasks: List[Task], interval: float = 1.0):
            def refresher_loop():
                while True:
                    with task_queue.mutex:
                        for i, (priority, workload, taskID) in enumerate(task_queue.queue):
                            if taskID is not None:
                                task = preloaded_tasks[taskID]
                                new_priority = task.get_priority(initial=False)
                                task_queue.queue[i] = (new_priority, workload, taskID)
                        heapify(task_queue.queue)
                    # Print the task queue for debugging
                    # print(f"  **  Task queue: {[(priority, workload, taskID) for priority, workload, taskID in task_queue.queue]} ** \n")
                    time.sleep(interval)

            thread = threading.Thread(target=refresher_loop, daemon=True)
            thread.start()
            return thread


        # Get preloaded tasks
        if preloaded_tasks is None:
            preloaded_tasks = self.producer.load_dataset(
                tokenizer=self.tokenizer,
                max_length=self.max_context_length,
                strategy=self.strategy,
                dataset_name=self.data_path,
            )

        # Create iteration bin and execute tasks
        task_queue = IterQueue()
        record_metrics = {}
        start = time.time()

        # ✅ Start background refresher
        self.refresher_thread = start_priority_refresher(task_queue, preloaded_tasks, interval=1.0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start the producer in a separate thread
            executor.submit(self.producer.produce, task_queue, preloaded_tasks)

            # Start the executor in main thread
            # self.executor(task_queue, preloaded_tasks, record_metrics=record_metrics)
            executor.submit(self.executor, task_queue, preloaded_tasks, record_metrics=record_metrics)

        end = time.time()
        print(f"Total time: {end - start}, throughput: {record_metrics['tokens'] / (end - start)} tokens/sec")


        # Save the task's prompt and response to a file
        output_dir = os.path.join(self.output_dir, self.model_path.split("/")[-1])
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.strategy}_retrain-{self.retrain_rate}_lambda-{self.arrival_rate}.json")
        eval_metrics = self.compute_metrics(preloaded_tasks)
        metrics = {
            "arrival_rate": self.arrival_rate,
            "retrain_rate": self.retrain_rate,
            "strategy": self.strategy,
            "num_test_samples": self.n_test_samples,
            "iteration": record_metrics["iteration"],
            "total_time": end - start,
            "throughput": record_metrics["tokens"] / (end - start),
            "metrics": eval_metrics,
            "generation_results": [
                {
                    "taskID": task.taskID,
                    "workload": task.workload,
                    "prompt": task.prompt,
                    "response": task.response,
                    "step": task.step,
                    "metrics": task.metrics,
                    "release_time": task.release_time,
                    "execution_time": task.execution_time,
                    "priority": task.priority,
                }
                for task in preloaded_tasks
            ],
        }
        save_metrics_with_order(metrics, output_file)


    def compute_metrics(self, preloaded_tasks: List[Task]) -> Dict[str, float]:
        total_correct = 0
        total_samples = 0
        total_log_prob_diff = 0.0
        total_perplexity = 0.0
        total_loss = 0.0

        for task in tqdm(preloaded_tasks, desc="Averaging metrics", total=len(preloaded_tasks)):
            if task.workload == "train":
                continue
            eval_outputs = task.metrics
            try:
                total_loss += eval_outputs["loss"]
                total_perplexity += eval_outputs["perplexity"]
                total_correct += eval_outputs["correct_preds"]
                total_samples += 1
                total_log_prob_diff += eval_outputs["log_prob_diff"]
            except KeyError as e:
                print(f"KeyError: {e} in task {task.taskID}. Skipping this task.")
                continue

        # Compute final averages
        preference_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_clpd = total_log_prob_diff / total_samples if total_samples > 0 else 0
        avg_perplexity = total_perplexity / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        return {
            "preference accuracy": preference_accuracy,
            "contrastive log probability difference (CLPD)": avg_clpd,
            "perplexity": avg_perplexity,
            "loss": avg_loss,
        }
        


if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU device number")
    parser.add_argument("--strategy", type=str, default="sync", choices=["async", "sync"], help="Scheduling strategy")
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
    args = parser.parse_args()
    
    
    system = EffiGenTune(args)
    system.run()

    
    # random.seed(42)
    # # Verify the number of available CUDA devices
    # device = args.device
    # strategy = args.strategy  # "async" or "sync"
    # attn_implementation = args.attn_implementation  # "flash_attention_2" or "eager"
    # model_path = args.model_path
    # n_test_samples = args.n_test_samples
    # n_train_samples = int(n_test_samples * args.retrain_rate)
    # lr = args.lr
    # data_path = args.data_path
    # arrival_rate = args.arrival_rate

    # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     device_map={"": device},
    #     use_cache=True,
    #     attn_implementation=attn_implementation,
    # )
    # # Apply LoRA configuration
    # lora_config = LoraConfig(
    #     r=16,  # LoRA rank
    #     lora_alpha=16,  # Scaling factor
    #     target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
    #     lora_dropout=0.05,
    #     task_type="CAUSAL_LM",
    #     bias="none",
    # )
    # # Wrap model with LoRA
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()  # Should list LoRA parameters as trainable
    # max_length = model.config.max_position_embeddings
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    # # Get optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # # Define batch tokenization
    # def tokenize_and_align_labels(examples):
    #     """Tokenizes inputs in batch and masks out context for chosen labels."""
    #     contexts = examples["context"]
    #     chosens = examples["chosen_response"]
    #     rejecteds = examples["rejected_response"]

    #     # Tokenize all inputs in batch mode
    #     chosen_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, chosens)],
    #                                 truncation=True, padding=False,  # Use dynamic padding later during collation
    #                                 max_length=max_length)
        
    #     rejected_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, rejecteds)],
    #                                 truncation=True, padding=False,  # Use dynamic padding later during collation
    #                                 max_length=max_length)

    #     # Tokenize context separately (to get its length)
    #     context_encodings = tokenizer(contexts, truncation=True, padding=False, max_length=max_length)
    #     context_lengths = [len(enc) for enc in context_encodings["input_ids"]]

    #     # Create labels: Mask out context tokens by setting them to `-100`
    #     chosen_labels = [
    #         [-100] * ctx_len + chosen_encodings["input_ids"][i][ctx_len:]
    #         for i, ctx_len in enumerate(context_lengths)
    #     ]

    #     return {
    #         "context_input_ids": context_encodings["input_ids"],
    #         "context_attention_mask": context_encodings["attention_mask"],
    #         "chosen_input_ids": chosen_encodings["input_ids"],
    #         "chosen_attention_mask": chosen_encodings["attention_mask"],
    #         "chosen_labels": chosen_labels,
    #         "rejected_input_ids": rejected_encodings["input_ids"],
    #         "rejected_attention_mask": rejected_encodings["attention_mask"],
    #     }

    # # Load dataset 
    # rlhf_data = load_dataset(data_path)
    # test_indices = random.sample(range(len(rlhf_data["test"])), n_test_samples)
    # test_dataset = rlhf_data["test"].select(test_indices)
    # train_indices = random.sample(range(len(rlhf_data["train"])), n_train_samples)
    # train_dataset = rlhf_data["train"].select(train_indices)
    # processed_test_dataset = test_dataset.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     load_from_cache_file=False,
    # ).remove_columns(test_dataset.column_names)
    # processed_train_dataset = train_dataset.map(
    #     tokenize_and_align_labels, 
    #     batched=True, 
    #     load_from_cache_file=False,
    # ).remove_columns(train_dataset.column_names)  
  
    # # Preloaded tasks
    # preloaded_tasks: List[Task] = []
    # for i, test_example in enumerate(processed_test_dataset):
    #     task = Task(
    #         taskID=i,
    #         workload="prefill", 
    #         rate_lambda=arrival_rate, 
    #         prompt=test_dataset[i]["context"],
    #         input_kwargs=test_example,
    #     )
    #     preloaded_tasks.append(task)
        
    # accum = i+1
    # for j, train_example in enumerate(processed_train_dataset):
    #     task = Task(
    #         taskID=accum+j,
    #         workload="train", 
    #         rate_lambda=arrival_rate, 
    #         prompt=train_dataset[j]["context"],
    #         input_kwargs=train_example,
    #     )
    #     preloaded_tasks.append(task)

    # # Simulate global scheduler
    # task_queue = PriorityQueue()
    # for taskID, task in enumerate(preloaded_tasks):
    #     task_queue.put((task.get_priority(initial=True), taskID))

    # # Create iteration bin and execute tasks
    # start = time.time()
    # scheduler = Scheduler(lambda1=0.5, lambda2=0.5)
    # iteration = 0
    # tokens = 0
    # while task_queue.qsize() > 0:
    #     print(f"  **  Iteration {iteration} (queue size {task_queue.qsize()})  **  ")
    #     tokens += task_queue.qsize()
    #     # while task_queue.qsize() > 0:
    #     #     _, taskID = task_queue.get(timeout=0.5)
    #     #     bin.add_task(preloaded_tasks[taskID])
    #     bin = scheduler.best_fit_allocate(task_queue, preloaded_tasks, model, attn_implementation=attn_implementation, eval_metrics=True)
    #     print(f"\tBin allocation (prefill {len(bin.prefill_batch)}, decode {len(bin.decode_batch)}, train {len(bin.train_batch)})")
    #     bin.concurrent_execute(model, tokenizer, task_queue, strategy=strategy, optimizer=optimizer)
    #     iteration += 1
    #     # break
        
    # end = time.time()

    # print(f"Completed in {iteration} iterations! Total time: {end - start}, throughput: {tokens / (end - start)} tokens/sec")

    # # Save the task's prompt and response to a file
    # output_dir = "profile_main/dummy/Mistral-7B-Instruct-v0.2"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, f"output_cache_{strategy}_system.json")
    # metrics = {
    #     "iteration": iteration,
    #     "total_time": end - start,
    #     "throughput": tokens / (end - start),
    #     "generation_results": [
    #         {
    #             "taskID": task.taskID,
    #             "workload": task.workload,
    #             "prompt": task.prompt,
    #             "response": task.response,
    #             "step": task.step,
    #             "metrics": task.metrics,
    #         }
    #         for task in preloaded_tasks
    #     ],
    # }
    # save_metrics_with_order(metrics, output_file)
    # print(f"Metrics saved to {output_file}")

    