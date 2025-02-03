import os
import sys
sys.dont_write_bytecode = True
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
from datasets import Dataset
from tqdm.auto import tqdm
import time
import json
# import seaborn as sns
# import matplotlib
# import matplotlib.pylab as plt
# from matplotlib.colors import LinearSegmentedColormap
import ipdb
import pandas as pd
import numpy as np
from pathlib import Path

from reason_needle.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from reason_needle.metrics import compare_answers, TASK_LABELS

from datasets import load_from_disk
from reason_needle.reason_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset

import random
import argparse
from tqdm import tqdm

datasets = [
    'qa1',
]


datasets2name = {
    'qa1': 'single-supporting-fact',
    'qa2': 'two-supporting-facts',
    'qa3': 'three-supporting-facts',
    'qa4': 'two-arg-relations',
    'qa5': 'three-arg-relations',
}

model2maxlen = {
    "mistral": 2560000
}


def get_max_memory_allocated(devices):
    return sum(
        torch.cuda.max_memory_allocated(device) for device in devices
    )


def get_max_memory_reserved(devices):
    return sum(
        torch.cuda.max_memory_reserved(device) for device in devices
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main(args):
    

    print("Loading data...")

    test_data = []

    input_max_len = 0
    
    model_path = args.model_path.lower()


    if args.method.lower() == 'fullkv':
        num_haystack_parts = [i * 1000 for i in [1, 4, 8, 16, 32, 64, 128]]
        # num_haystack_parts = [i * 1000 for i in [1, 4, 8, 16]]
        # num_haystack_parts = [i * 1000 for i in [32]]
        num_insert_parts = 1
    else:
        num_haystack_parts = [i * 1000 for i in [1, 4, 8, 16, 32, 64, 128, 256]]
        # num_haystack_parts = [i * 1000 for i in [32]]
        num_insert_parts = 1

    num_test_examples = 1
    output_max_len = args.length
    # model_max_len = model_max_len - 100 # save for prompt.

    noise_dataset = load_from_disk('/mnt/users/t-yufu/HeadAllocation_share_2/reason_needle/noise_data')
    task_dataset = TaskDataset(args.data_path, max_n_facts=3)       
    noise_sampler = SentenceSampler(noise_dataset, tokenizer=tokenizer)

    pbar = tqdm(total = len(num_haystack_parts) * num_insert_parts * num_test_examples)
    haystack_parts = num_haystack_parts

    insert_parts = np.linspace(0, 1, num_insert_parts + 1)

    model_name = model_path.split("/")[-1]
    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}.json"), "w")
    
    memory_fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}_memory.json"), "w")

    for haystack_index in range(len(num_haystack_parts)):
        haystack_size = haystack_parts[haystack_index]
        for insert_index in range(num_insert_parts):
            task_start_pct = insert_parts[insert_index]
            task_end_pct = insert_parts[insert_index + 1]

            reason_dataset = NoiseInjectionDataset(task_dataset=task_dataset,
                                                    noise_sampler=noise_sampler,
                                                    tokenizer=tokenizer,
                                                    sample_size=haystack_size,
                                                    task_start_pct=task_start_pct,
                                                    task_end_pct=task_end_pct)

            num_test_examples = min(num_test_examples, len(reason_dataset))

            for ex_index in range(num_test_examples):
                example = reason_dataset[ex_index]
                context = tokenizer.decode(example['input_tokens'], skip_special_tokens=False)
                question = tokenizer.decode(example['question_tokens'], skip_special_tokens=True)
                target = tokenizer.decode(example['target_tokens'], skip_special_tokens=True)
                default_prompt = DEFAULT_PROMPTS[args.dataset]
                # ipdb.set_trace()

                # prompt = get_formatted_input(
                #     context=context, 
                #     question=question, 
                #     examples='',
                #     instruction=default_prompt['instruction'], 
                #     post_prompt=default_prompt['post_prompt'], 
                #     template=DEFAULT_TEMPLATE
                # )
                prompt = context
                tokenized_prompts = tokenizer([prompt], padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')

                batch_input_ids = tokenized_prompts.input_ids
                attention_mask = tokenized_prompts.attention_mask

                max_memory_allocated_after_input_to_cuda = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
                args.result_data[f"{haystack_size}:after_input_to_cuda"] = f"{max_memory_allocated_after_input_to_cuda} GB"

                # max_memory_reserved_after_input_to_cuda = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
                # args.result_data[f"{haystack_size}:max_memory_reserved_after_input_to_cuda"] = f"{max_memory_reserved_after_input_to_cuda} GB"
                
                model.model.config.window_size = 8
                model.model.config.base_capacity = args.max_capacity_prompts
                model.model.config.aug_capacity = args.aug_capacity
                model.model.config.head_choice = args.head_choice
                model.model.config.top_num = args.top_num
                model.model.config.beta = args.beta
                model.model.config.temp = args.temp
                model.model.config.alpha = args.alpha
                model.model.config.kernel_size = 7
                model.model.config.skip = 0
                model.model.config.normalize = True
                model.model.config.pooling = "maxpool"
                model.model.config.floor = 0.2
                model.model.config.pyram_beta = args.pyram_beta

                context_length = batch_input_ids.shape[-1]
                start_time = time.perf_counter()
                output = model.generate(
                    **tokenized_prompts,
                    output_attentions = args.output_attentions,
                    max_new_tokens=output_max_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+output_max_len,
                    eos_token_id=[tokenizer.eos_token_id]
                )
                end_time = time.perf_counter()

                max_memory_allocated_after_generate = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
                args.result_data[f"{haystack_size}:after_generate"] = f"{max_memory_allocated_after_generate} GB"
                args.result_data[f'{haystack_size}:time'] = end_time - start_time
                # import gc
                # gpu_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda]
                
                # print(f"Number of tensors on GPU: {len(gpu_tensors)}")
                # for i, tensor in enumerate(gpu_tensors):
                #     print(f"Tensor {i}: shape {tensor.shape}, dtype {tensor.dtype}")
                
                # 通过 get_referrers() 追踪变量名
                # for tensor in gpu_tensors:
                #     referrers = gc.get_referrers(tensor)
                #     for ref in referrers:
                #         if isinstance(ref, dict):
                #             for var_name, var_value in ref.items():
                #                 if var_value is tensor:
                #                     print(f"Tensor name: {var_name}, Tensor shape: {tensor.shape}")
                # ipdb.set_trace()
                # max_memory_reserved_after_generate = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
                # args.result_data[f"{haystack_size}:max_memory_reserved_after_generate"] = f"{max_memory_reserved_after_generate} GB"

                batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
                torch.cuda.empty_cache()

                # max_memory_allocated_after_empty = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
                # args.result_data[f"{haystack_size}:max_memory_allocated_after_empt"] = f"{max_memory_allocated_after_empty} GB"

                example = {}
                example["prompt"] = prompt
                example["input"] = question
                example["answers"] = target
                example['tokens'] = len(batch_input_ids)
                example["pred"] = batch_outputs[0]
                example['setting'] = f'{haystack_index}: haystack:{haystack_size} -- insert:{round(task_start_pct, 2)}~{round(task_end_pct, 2)}'
                example["dataset"] = args.dataset

                fout.write(json.dumps(example) + "\n")
                pbar.update(1)

    memory_fout.write(json.dumps(args.result_data) + "\n")
    pbar.close()
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")

    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")
    parser.add_argument("--output_attentions", type=bool, default=False, help="")
        
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")
    parser.add_argument("--aug_capacity", type=int, default=-1, help='-1 means use the full kv cache')
    parser.add_argument("--head_choice", type=str, default='random', choices=['random', 'copy', 'musique', 'reason', 'mix', 'mix_top1', 'musique_top1', 'mix_top3', 'musique_top3', 'merge', 'final', 'final_copy', 'final_reason'])
    parser.add_argument("--top_num", type=int, default=10)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1, help='should be [0, 1]. 1-- total copy.')
    parser.add_argument('--pyram_beta', type=float, default=20)
    parser.add_argument('--length', type=int, default=1)

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.model_path == 'mistralai/Mistral-7B-Instruct-v0.2':
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left",
            revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            padding_side="left"
        )

    if args.method.lower() != 'fullkv':
        from adaptive_snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_qwen2 
        replace_llama(args.method)
        replace_mistral(args.method)
        replace_qwen2(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    

    args.result_data = {}

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_devices = list(range(torch.cuda.device_count()))

    max_memory_allocated_after_load_model = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
    args.result_data["after_load_model"] = f"{max_memory_allocated_after_load_model} GB"

    # max_memory_reserved_after_load_model = get_max_memory_reserved(all_devices) / (1024 * 1024 * 1024)
    # args.result_data["max_memory_reserved_after_load_model"] = f"{max_memory_reserved_after_load_model} GB"

    model.eval()
    
    save_dir = args.save_dir

    max_capacity_prompts = args.max_capacity_prompts
    
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        print(f'base capacity: {args.max_capacity_prompts}\thead_choice:{args.head_choice}\tbeta:{args.beta}\ttemp:{args.temp}\talpha:{args.alpha}')

        args.dataset = dataset
        
        # args.data_file = f"/mnt/users/t-yufu/HeadAllocation_share_2/data/LongBench/{args.dataset}.jsonl"
        name = datasets2name[args.dataset]
        args.data_path = f'/mnt/users/t-yufu/HeadAllocation_share_2/reason_needle/tasks_1-20_v1-2/en-10k/{args.dataset}_{name}_train.txt'
        main(args)










