# A definition of the IterQueue class and supported functions.
from queue import Queue
from heapq import heappush, heappop, heapify
from typing import List, Tuple, Optional
import sys
sys.dont_write_bytecode = True



class IterQueue(Queue):
    '''Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).
    '''

    def _init(self, maxsize):
        self.queue: List[Tuple[float, str, int]] = []
        self.train_size = 0
        self.prefill_size = 0
        self.decode_size = 0
        # self.workload_count = {"train": 0, "prefill": 0, "decode": 0}


    def _qsize(self):
        return len(self.queue)

    def _put(self, item: Tuple[float, str, int]):
        """Item is a tuple of the form (priority, workload, taskID)."""
        heappush(self.queue, item)
        if item[1] == "train":
            self.train_size += 1
        elif item[1] == "prefill":
            self.prefill_size += 1
        elif item[1] == "decode":
            self.decode_size += 1

    def _get(self):
        # return heappop(self.queue)
        item = heappop(self.queue)
        # if item[1] is not None and item[1] == "train":
        #     self.train_size -= 1
        if item[1] == "prefill":
            self.prefill_size -= 1
        elif item[1] == "decode":
            self.decode_size -= 1
        elif item[1] == "train":
            self.train_size -= 1
        return item
        



# if __name__ == "__main__":
#     import time
#     import threading
#     from transformers import AutoTokenizer
#     from iteration_producer import Producer
#     from iteration_task import Task

#     arrival_rate =5
#     retrain_rate = 0.4
#     n_test_samples = 20
#     arrival_pattern = "poisson"
#     model_path = "mistralai/Mistral-7B-Instruct-v0.2"
#     max_context_length = 1024
#     strategy = "async"
#     data_path = "data/Anthropic"

#     # Get tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     # Get producer (loading dataset)
#     producer = Producer(
#         arrival_rate=arrival_rate, 
#         retrain_rate=retrain_rate, 
#         n_test_samples=n_test_samples,
#         arrival_pattern=arrival_pattern, 
#     )
#     preloaded_tasks = producer.load_dataset(
#         tokenizer=tokenizer,
#         max_length=max_context_length,
#         strategy=strategy,
#         dataset_name=data_path,
#     )

#     def start_priority_refresher(task_queue: IterQueue, preloaded_tasks: List[Task], interval: float = 1.0):
#         def refresher_loop():
#             while True:
#                 with task_queue.mutex:
#                     for i, (priority, workload, taskID) in enumerate(task_queue.queue):
#                         if taskID is not None:
#                             task = preloaded_tasks[taskID]
#                             new_priority = task.get_priority(strategy, initial=False)
#                             task_queue.queue[i] = (new_priority, workload, taskID)
#                     heapify(task_queue.queue)
#                 # Print the task queue for debugging
#                 print(f"  **  Task queue: {[(priority, workload, taskID) for priority, workload, taskID in task_queue.queue]} ** \n")
#                 time.sleep(interval)

#         thread = threading.Thread(target=refresher_loop, daemon=False)
#         thread.start()
#         return thread

#     task_queue = IterQueue()

#     # ✅ Preload tasks into the queue
#     producer.produce(task_queue, preloaded_tasks)

#     # ✅ Start background refresher
#     refresher_thread = start_priority_refresher(task_queue, preloaded_tasks, interval=1.0)
