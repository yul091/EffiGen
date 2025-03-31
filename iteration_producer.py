# A definition of the Producer class and supported functions.
from typing import List, Dict, Any, Optional
import time
import random
import pdb
from queue import PriorityQueue
import sys 
sys.dont_write_bytecode = True
from datasets import load_dataset
from transformers import AutoTokenizer
from iteration_task import Task  



class Producer:

    def __init__(
        self,
        arrival_rate: float,
        retrain_rate: float,
        n_test_samples: int,
        arrival_pattern: str = "poisson",
    ):
        self.arrival_rate = arrival_rate
        self.retrain_rate = retrain_rate
        self.arrival_pattern = arrival_pattern
        self.n_test_samples = n_test_samples 
        self.n_train_samples = int(self.n_test_samples * retrain_rate)
        

    def load_dataset(
        self, 
        tokenizer: AutoTokenizer,
        max_length: int,
        dataset_name: str = "data/Anthropic",
    ) -> List[Task]:
        """
        Load dataset and tokenize inputs. Create a preloaded dataset of Task objects.
        """
        
        random.seed(42)
        # Load dataset 
        rlhf_data = load_dataset(dataset_name)
        test_indices = random.sample(range(len(rlhf_data["test"])), self.n_test_samples)
        test_dataset = rlhf_data["test"].select(test_indices)
        train_indices = random.sample(range(len(rlhf_data["train"])), self.n_train_samples)
        train_dataset = rlhf_data["train"].select(train_indices)

        # Define batch tokenization
        def tokenize_and_align_labels(examples):
            """Tokenizes inputs in batch and masks out context for chosen labels."""
            contexts = examples["context"]
            chosens = examples["chosen_response"]
            rejecteds = examples["rejected_response"]

            # Tokenize all inputs in batch mode
            chosen_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, chosens)],
                                        truncation=True, padding=False,  # Use dynamic padding later during collation
                                        max_length=max_length)
            
            rejected_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, rejecteds)],
                                        truncation=True, padding=False,  # Use dynamic padding later during collation
                                        max_length=max_length)

            # Tokenize context separately (to get its length)
            context_encodings = tokenizer(contexts, truncation=True, padding=False, max_length=max_length)
            context_lengths = [len(enc) for enc in context_encodings["input_ids"]]

            # Create labels: Mask out context tokens by setting them to `-100`
            chosen_labels = [
                [-100] * ctx_len + chosen_encodings["input_ids"][i][ctx_len:]
                for i, ctx_len in enumerate(context_lengths)
            ]

            return {
                "context_input_ids": context_encodings["input_ids"],
                "context_attention_mask": context_encodings["attention_mask"],
                "chosen_input_ids": chosen_encodings["input_ids"],
                "chosen_attention_mask": chosen_encodings["attention_mask"],
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_encodings["input_ids"],
                "rejected_attention_mask": rejected_encodings["attention_mask"],
            }


        processed_test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns(test_dataset.column_names)
        processed_train_dataset = train_dataset.map(
            tokenize_and_align_labels, 
            batched=True, 
            load_from_cache_file=False,
        ).remove_columns(train_dataset.column_names)

        # Preloaded tasks
        preloaded_tasks: List[Task] = []
        train_prob = self.retrain_rate / (self.retrain_rate + 1)
        train_idx, test_idx = 0, 0
        for taskID in range(self.n_train_samples + self.n_test_samples):
            if (random.random() < train_prob and train_idx < self.n_train_samples) or (test_idx == self.n_test_samples):
                # Add a train task
                task = Task(
                    taskID=taskID,
                    workload="train", 
                    rate_lambda=self.arrival_rate, 
                    prompt=train_dataset[train_idx]["context"],
                    input_kwargs=processed_train_dataset[train_idx],
                )
                train_idx += 1
            else:
                # Add a test task
                task = Task(
                    taskID=taskID,
                    workload="prefill", 
                    rate_lambda=self.arrival_rate, 
                    prompt=test_dataset[test_idx]["context"],
                    input_kwargs=processed_test_dataset[test_idx],
                )
                test_idx += 1

            preloaded_tasks.append(task)

        # print(f"Actually tasks {[task.workload for task in preloaded_tasks]}, train {sum([task.workload == 'train' for task in preloaded_tasks])}, test {sum([task.workload == 'prefill' for task in preloaded_tasks])}")
        # pdb.set_trace()
        print(f"  **  Loaded {len(preloaded_tasks)} tasks - {len(processed_test_dataset)} test ({len(processed_test_dataset) * 100 / len(preloaded_tasks):.2f}%) and {len(processed_train_dataset)} train ({len(processed_train_dataset) * 100 / len(preloaded_tasks):.2f}%)  **")
        return preloaded_tasks
    


    def produce(self, task_queue: PriorityQueue, preloaded_tasks: List[Task]):

        # Produce using the dataset
        for taskID, task in enumerate(preloaded_tasks):
            print(f"  **  Producing task {taskID} ({task.workload})  **  ")
            time.sleep(random.expovariate(task.rate_lambda))
            # Essentially, we are using preloaded data (task ID)
            task_queue.put((task.get_priority(initial=True), taskID))
            
        task_queue.put((float('inf'), None))  # Signal the end of the dataset
        print("Producer finished producing tasks")

