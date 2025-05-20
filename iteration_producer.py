# A definition of the Producer class and supported functions.
from typing import List, Dict, Any, Optional, Union
import time
import random
import pdb
import sys 
sys.dont_write_bytecode = True
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from iteration_task import Task  
from iteration_queue import IterQueue



class Producer:

    def __init__(
        self,
        arrival_rate: float,
        retrain_rate: float,
        n_test_samples: int,
        strategy: str,
        arrival_pattern: str = "poisson",
    ):
        self.arrival_rate = arrival_rate
        self.retrain_rate = retrain_rate
        self.arrival_pattern = arrival_pattern
        self.n_test_samples = n_test_samples 
        self.strategy = strategy
        self.n_train_samples = int(self.n_test_samples * retrain_rate)

    
    @staticmethod
    def load_and_prepare_datasets(dataset_names, n_train_samples, n_test_samples):
        """
        Load multiple datasets, tag each example with its dataset name, and combine their train/test splits.

        Args:
            dataset_names (str or list of str): Path(s) to dataset(s).
            n_train_samples (int): Number of training samples to select.
            n_test_samples (int): Number of test samples to select.

        Returns:
            train_dataset, test_dataset: Datasets ready for training and evaluation.
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        all_train = []
        all_test = []

        for name in dataset_names:
            data = load_dataset(name)

            # Add source_dataset field
            train_indices = random.sample(range(len(data['train'])), n_train_samples)
            train_data = data['train'].select(train_indices).map(lambda ex: {"source_dataset": name})
            test_indices = random.sample(range(len(data['test'])), n_test_samples)
            test_data = data['test'].select(test_indices).map(lambda ex: {"source_dataset": name})

            all_train.append(train_data)
            all_test.append(test_data)

        # Merge all train and all test
        train_dataset = concatenate_datasets(all_train)
        test_dataset = concatenate_datasets(all_test)

        # Optionally, you can shuffle the datasets
        train_dataset = train_dataset.shuffle(seed=42)
        test_dataset = test_dataset.shuffle(seed=42)

        # # Sample train/test subsets
        # train_indices = random.sample(range(len(merged_train)), min(n_train_samples, len(merged_train)))
        # test_indices = random.sample(range(len(merged_test)), min(n_test_samples, len(merged_test)))

        # train_dataset = merged_train.select(train_indices)
        # test_dataset = merged_test.select(test_indices)

        return train_dataset, test_dataset

        

    def load_dataset(
        self, 
        tokenizer: AutoTokenizer,
        max_length: int,
        dataset_name: Union[str, List[str]] = "data/Anthropic",
    ) -> List[Task]:
        """
        Load dataset and tokenize inputs. Create a preloaded dataset of Task objects.
        """
        random.seed(42)
        # Load dataset 
        if isinstance(dataset_name, str):
            rlhf_data = load_dataset(dataset_name)
            test_indices = random.sample(range(len(rlhf_data["test"])), self.n_test_samples)
            test_dataset = rlhf_data["test"].select(test_indices).map(lambda ex: {"source_dataset": dataset_name})
            train_indices = random.sample(range(len(rlhf_data["train"])), self.n_train_samples)
            train_dataset = rlhf_data["train"].select(train_indices).map(lambda ex: {"source_dataset": dataset_name})
        elif isinstance(dataset_name, list):
            train_dataset, test_dataset = self.load_and_prepare_datasets(
                dataset_names=dataset_name,
                n_train_samples=self.n_train_samples,
                n_test_samples=self.n_test_samples,
            )
            self.n_train_samples = self.n_train_samples * len(dataset_name)
            self.n_test_samples = self.n_test_samples * len(dataset_name)
        else:
            raise ValueError("dataset_name must be a string or a list of strings.")

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

        # We use enqueue time as priority
        if self.strategy == "train-first":  
            # Train tasks first, then test tasks
            for train_idx in range(self.n_train_samples):
                task = Task(
                    taskID=train_idx,
                    workload="train", 
                    rate_lambda=self.arrival_rate, 
                    prompt=train_dataset[train_idx]["context"],
                    input_kwargs=processed_train_dataset[train_idx],
                    source_dataset=train_dataset[train_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)
            for test_idx in range(self.n_test_samples):
                task = Task(
                    taskID=test_idx + self.n_train_samples,
                    workload="prefill", 
                    rate_lambda=self.arrival_rate, 
                    prompt=test_dataset[test_idx]["context"],
                    input_kwargs=processed_test_dataset[test_idx],
                    source_dataset=test_dataset[test_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)

        # We use enqueue time as priority
        elif self.strategy == "test-first":  
            # Test tasks first, then train tasks
            for test_idx in range(self.n_test_samples):
                task = Task(
                    taskID=test_idx,
                    workload="prefill", 
                    rate_lambda=self.arrival_rate, 
                    prompt=test_dataset[test_idx]["context"],
                    input_kwargs=processed_test_dataset[test_idx],
                    source_dataset=test_dataset[test_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)
            for train_idx in range(self.n_train_samples):
                task = Task(
                    taskID=train_idx + self.n_test_samples,
                    workload="train", 
                    rate_lambda=self.arrival_rate, 
                    prompt=train_dataset[train_idx]["context"],
                    input_kwargs=processed_train_dataset[train_idx],
                    source_dataset=train_dataset[train_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)

        # We use enqueue time as priority
        elif self.strategy == "train-middle":  # split the test into halves, test - train - test
            # Train tasks in the middle
            for test_idx in range(self.n_test_samples // 2):
                task = Task(
                    taskID=test_idx,
                    workload="prefill", 
                    rate_lambda=self.arrival_rate, 
                    prompt=test_dataset[test_idx]["context"],
                    input_kwargs=processed_test_dataset[test_idx],
                    source_dataset=test_dataset[test_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)
            for train_idx in range(self.n_train_samples):
                task = Task(
                    taskID=train_idx + test_idx + 1,
                    workload="train", 
                    rate_lambda=self.arrival_rate, 
                    prompt=train_dataset[train_idx]["context"],
                    input_kwargs=processed_train_dataset[train_idx],
                    source_dataset=train_dataset[train_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)
            for second_test_idx in range(test_idx + 1, self.n_test_samples):
                task = Task(
                    taskID=second_test_idx + train_idx + 1,
                    workload="prefill", 
                    rate_lambda=self.arrival_rate, 
                    prompt=test_dataset[second_test_idx]["context"],
                    input_kwargs=processed_test_dataset[second_test_idx],
                    source_dataset=test_dataset[second_test_idx]["source_dataset"],
                )
                preloaded_tasks.append(task)

        # We use pre-defined coefficients as priority
        else:  
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
                        source_dataset=train_dataset[train_idx]["source_dataset"],
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
                        source_dataset=test_dataset[test_idx]["source_dataset"],
                    )
                    test_idx += 1

                preloaded_tasks.append(task)

        # pdb.set_trace()
        print(f"  **  Loaded {len(preloaded_tasks)} tasks - {len(processed_test_dataset)} test ({len(processed_test_dataset) * 100 / len(preloaded_tasks):.2f}%) and {len(processed_train_dataset)} train ({len(processed_train_dataset) * 100 / len(preloaded_tasks):.2f}%)  **")
        return preloaded_tasks
    


    def produce(self, task_queues: List[IterQueue], preloaded_tasks: List[Task]):

        # Produce using the dataset
        for taskID, task in enumerate(preloaded_tasks):
            time.sleep(random.expovariate(task.rate_lambda))
            if task.workload == "train":
                # The first queue is always for retraining tasks
                print(f"  **  Producing task {taskID} ({task.workload}) **  ")
                task_queues[0].put((task.get_priority(self.strategy, initial=True), task.workload, taskID))
            else:
                # The last queue is always for inference tasks
                task_queues[-1].put((task.get_priority(self.strategy, initial=True), task.workload, taskID))
        
        for task_queue in task_queues:
            # Signal the end of the dataset
            task_queue.put((float('inf'), None, None))  # Signal the end of the dataset
        print("Producer finished producing tasks")

