
from typing import List, Optional, Dict
import sys 
sys.dont_write_bytecode = True
import copy
import torch
import graphviz
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from utils import _prepare_input
from run_memory import generate, get_max_memory_allocated, CustomJSONEncoder


def clone(cache: DynamicCache) -> DynamicCache:
    """Clone a DynamicCache object."""
    cloned_cache = DynamicCache()
    cloned_cache.key_cache = [k.clone() for k in cache.key_cache]
    cloned_cache.value_cache = [v.clone() for v in cache.value_cache]
    return cloned_cache


class TrieNode:
    """A node in the prefix-sharing trie that stores KV cache tensors."""
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.kv_cache: DynamicCache = None  # Stores KV cache if computed
        self.is_end_of_query: bool = False

class PrefixSharingTrie:
    """Trie structure to manage shared prefixes and their KV caches."""
    def __init__(self, max_depth: Optional[int] = None):
        self.root = TrieNode()
        self.max_depth = max_depth

    def insert(self, tokens: List[int], model: GPT2LMHeadModel):
        """Insert a tokenized query into the trie and compute only necessary KV cache."""
        node = self.root
        cached_tokens = []  # Tokens that already have KV cache
        computed_tokens = []  # Tokens that need to be computed

        # Step 1: Traverse the trie to find the longest cached prefix
        prefix_ending_node: TrieNode = None
        for depth, token in enumerate(tokens):
            if self.max_depth is not None and depth >= self.max_depth:
                # print(f"⚠️ Max depth {self.max_depth} reached, stopping insertion for this query.")
                break  # Stop inserting beyond max depth
            
            if token in node.children and node.children[token].kv_cache is not None:
                node = node.children[token]
                cached_tokens.append(token)
            else:
                if prefix_ending_node is None and node.kv_cache is not None:
                    prefix_ending_node = node
                # First time seeing this token -> Needs computation
                if token not in node.children:
                    node.children[token] = TrieNode()
                computed_tokens.append(token)  
                node = node.children[token]
            # print(f"\ttoken {token}, node children {children}, cached {cached}")

        # If everything is already cached, no need to compute anything
        if not computed_tokens:
            return

        # Step 2: Compute KV cache only for new tokens
        real_cached_prefix_len = 0 if prefix_ending_node is None else prefix_ending_node.kv_cache.key_cache[0].shape[2]
        prefix_len = len(cached_tokens)
        assert prefix_len == real_cached_prefix_len, "KV cache shape mismatch!"
        
        # new_input_ids = torch.tensor([tokens]).cuda()
        new_input_ids = torch.tensor([cached_tokens + computed_tokens]).cuda()
        with torch.no_grad():
            # print(f"prefix kv cache size {real_cached_prefix_len}, new first token {computed_tokens[0]}")
            outputs = model(new_input_ids)
            # real_cached_prefix_len = 0 if prefix_ending_node is None else prefix_ending_node.kv_cache.key_cache[0].shape[2]
            # print(f"prefix kv cache size (after model forward) {real_cached_prefix_len}, new first token {computed_tokens[0]}")

        # Step 3: Store **incremental** KV caches for each new token
        past_kv_cache: DynamicCache = outputs.past_key_values
        # print(f"cached tokens: {prefix_len}, computed tokens: {len(computed_tokens)}, output cache size {past_kv_cache.key_cache[0].shape[2]}")
  
        # Now store KV cache progressively
        step_node = prefix_ending_node or self.root
        for i, token in enumerate(computed_tokens):
            step_node = step_node.children[token]
            # Store KV cache only up to this token [batch_size, num_heads, seq_len, head_dim]
            if step_node.kv_cache is not None:
                raise ValueError("KV cache already exists!")
            step_node.kv_cache = DynamicCache()
            for layer_idx in range(len(past_kv_cache.key_cache)):
                # step_node.kv_cache.key_cache[j] = step_node.kv_cache.key_cache[j][:, :, :prefix_len+i+1, :]
                # step_node.kv_cache.value_cache[j] = step_node.kv_cache.value_cache[j][:, :, :prefix_len+i+1, :]
                step_node.kv_cache.update(
                    past_kv_cache.key_cache[layer_idx][:, :, :prefix_len + i + 1, :],
                    past_kv_cache.value_cache[layer_idx][:, :, :prefix_len + i + 1, :],
                    layer_idx,
                )
            # print(f"  **  Storing KV cache for token {token} with shape {step_node.kv_cache.key_cache[0].shape}.  **\n")
        step_node.is_end_of_query = True  # Mark as a full query endpoint


    def find_longest_prefix(self, tokens: List[int]):
        """Find the longest matching prefix in the trie and return the node + remaining tokens."""
        node = self.root
        longest_prefix = []
        remaining_tokens = []
        for token in tokens:
            if token in node.children and node.children[token].kv_cache is not None:
                longest_prefix.append(token)
                node = node.children[token]
            else:
                remaining_tokens.append(token)

        return node, longest_prefix, remaining_tokens
    

    def generate_with_prefix_sharing(
        self, 
        model: GPT2LMHeadModel, 
        input_tokens: List[int], 
        # attn_mask: Optional[List[int]] = None,
        use_prefix: Optional[bool] = None, 
        max_new_tokens: Optional[int] = None,
    ):
        """Generate text using the longest shared prefix KV cache."""
        
        if use_prefix:
            # Tokenize input
            prefix_node, prefix_tokens, remaining_tokens = self.find_longest_prefix(input_tokens)
            # print(f"  ** Input {len(input_tokens)} matched prefix {len(prefix_tokens)} with remaining {len(remaining_tokens)}  **\n")

            # ✅ Fix: If no new tokens to process, provide a dummy token to `.generate()`
            if not remaining_tokens:
                # print(f"⚠️ Query fully matches a cached prefix! Generating output from cache.")
                input_tokens += [tokenizer.pad_token_id]  # Provide dummy token

            # Use stored KV cache if available
            # past_key_values = copy.deepcopy(prefix_node.kv_cache)
            past_key_values = clone(prefix_node.kv_cache)
            # if prefix_node.kv_cache:
                # input_ids = torch.tensor([remaining_tokens]).cuda()
                # input_text = tokenizer.decode(input_ids[0])
                # past_key_values = prefix_node.kv_cache
                # Prefix is the input minus the remaining tokens
                # print(f"  ** Input IDs {input_ids.shape} ({input_text}), prefix key {past_key_values.key_cache[0].shape}, prefix value {past_key_values.value_cache[0].shape}.  **\n")
                # with torch.no_grad():
                #     outputs = self.model(input_ids=input_ids, past_key_values=prefix_node.kv_cache, use_cache=True)
            # else:
                # Compute from scratch and store KV cache
                # input_ids = torch.tensor([input_tokens]).cuda()
                # input_text = tokenizer.decode(input_ids[0])
                # past_key_values = None
                # print(f"  **  No KV cache found. Input text ({input_text}). **\n")
                # with torch.no_grad():
                #     outputs = self.model(input_ids=input_ids, use_cache=True)
        else:
            past_key_values = None
            prefix_tokens = []
            remaining_tokens = input_tokens

        
        input_ids = torch.tensor([input_tokens]).cuda()
        # attention_mask = torch.tensor([attn_mask]).cuda() if attn_mask is not None else None
        # max_memory_allocated_after_input_to_cuda = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)

        outputs = model.generate(
            inputs=input_ids,
            # attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            num_beams=1,
            temperature=1.0,
            use_cache=True,
            return_dict_in_generate=True,
        )
        # with torch.no_grad():
        #     outputs = self.model(
        #         input_ids, 
        #         past_key_values=past_key_values, 
        #         use_cache=True,
        #     )

        # # Store the new KV cache at the prefix node
        # prefix_node.kv_cache = outputs.past_key_values
        return outputs, prefix_tokens, remaining_tokens
    


class TrieVisualizer:
    """Generates a compact visual representation of the prefix-sharing trie."""
    def __init__(self, trie, tokenizer):
        self.trie = trie
        self.tokenizer = tokenizer

    def build_graph(self, node=None, graph=None, token_id="ROOT", show_kv_cache=True):
        """Recursively builds a Graphviz graph of the trie."""
        if graph is None:
            graph = graphviz.Digraph(format='png')  # Change format here if needed
            graph.attr(dpi='300')
            node = self.trie.root  # Start from the root

        # Convert token ID to token string
        token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0] if token_id != "ROOT" else "ROOT"
        node_label = f"{token_str}\n(ID={token_id}, KV {'✓' if node.kv_cache else '✗'})" if show_kv_cache else token_str

        # Set node color if KV cache exists
        graph.node(str(token_id), node_label, shape='ellipse', style='filled',
                   fillcolor="lightblue" if node.kv_cache else "white")

        # Recursively add children
        for token, child in node.children.items():
            # child_token_str = self.tokenizer.convert_ids_to_tokens([token])[0]
            # graph.edge(token_str, child_token_str)
            graph.edge(str(token_id), str(token))
            self.build_graph(child, graph, token, show_kv_cache)

        return graph

    def save(self, filename="trie_visualization", file_format="png", show_kv_cache=True):
        """Save the trie visualization as an image file."""
        graph = self.build_graph(show_kv_cache=show_kv_cache)
        filepath = f"{filename}.{file_format}"
        graph.render(filename, format=file_format, cleanup=True)  # Save without opening
        print(f"Trie visualization saved as {filepath}")


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



if __name__ == "__main__":
    import os
    import time
    import json
    import pickle
    from tqdm import tqdm
    import argparse
    from datasets import load_dataset
    import transformers

    # Patch generation method
    # transformers.generation.utils.GenerationMixin.generate = generate

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_prefix", action="store_true", help="Use prefix sharing")
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument("--dataset", type=str, default="rlhf")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--experiment", type=str, default="prefill", choices=["prefill", "decode", "fine-tune"])
    parser.add_argument("--method", type=str,  default="fullkv")
    parser.add_argument("--max_capacity_prompts", type=int, default=None, help="")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--output_max_len", type=int, default=1, help="The maximum length of the output sequence.")
    parser.add_argument("--train_samples", type=int, default=None, help="The number of training samples to use.")
    parser.add_argument("--test_samples", type=int, default=None, help="The number of test samples to use.")
    parser.add_argument("--max_depth", type=int, default=256, help="The maximum depth of the trie.")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()
    
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # model_path = "facebook/opt-125m"
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    max_length = model.config.max_position_embeddings
    # Explicitly set pad_token_id to avoid warnings
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    prefix_trie = PrefixSharingTrie(max_depth=args.max_depth)

    args.result_data = {}
    all_devices = list(range(torch.cuda.device_count()))

    max_memory_allocated_after_load_model = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
    args.result_data["after_load_model"] = f"{max_memory_allocated_after_load_model} GB"

    rlhf_data = load_dataset("data/Anthropic")
    max_capacity_prompts = args.max_capacity_prompts if args.max_capacity_prompts is not None else "inf"

    # Insert some queries
    if args.use_prefix:
        # queries = [
        #     "What is AI?", 
        #     "What is AI used for?", 
        #     "What is AI's impact on society?",
        # ]
        # for q in queries:
        #     print(f"Inserting query: {q}")
        #     tokenized = tokenizer.encode(q)  # list of token IDs
        #     prefix_trie.insert(tokenized, model)
        
        if args.train_samples is not None:
            train_samples = min(args.train_samples, len(rlhf_data["train"]))
            train_dataset = rlhf_data["train"].select(range(train_samples))
        else:
            train_dataset = rlhf_data["train"]
        processed_train_dataset = train_dataset.map(
            tokenize_and_align_labels, 
            batched=True, 
            load_from_cache_file=False,
        ).remove_columns(train_dataset.column_names)  

        file_path = f"{args.save_dir}/prefix_trie.pkl"
        if not os.path.exists(file_path):
            for example in tqdm(processed_train_dataset, desc="Inserting queries", total=len(processed_train_dataset)):
                prefix_trie.insert(example["context_input_ids"], model)
            # Also save the prefixsharingtrie class (as pt file) for later use
            # pickle.dump(prefix_trie, open(file_path, "wb"))
        else:
            prefix_trie = pickle.load(open(file_path, "rb"))

        # # Visualize the trie structure
        # visualizer = TrieVisualizer(prefix_trie, tokenizer)
        # os.makedirs("figures", exist_ok=True)
        # visualizer.save("figures/trie_graph", file_format="png")  # Saves as trie_graph.png

    # Process new query using prefix sharing
    # text = "What is AI used in medicine?"
    # print(f"Processing query: {text}")
    # input_tokens = tokenizer.encode(text)
    # context_length = len(input_tokens)

    if args.test_samples is not None:
        test_samples = min(args.test_samples, len(rlhf_data["test"]))
        test_dataset = rlhf_data["test"].select(range(test_samples))
    else:
        test_dataset = rlhf_data["test"]
    processed_test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=False,
    ).remove_columns(test_dataset.column_names)

    optimization = "(prefix_cache)" if args.use_prefix else "(no_cache)"
    model_name = model_path.split("/")[-1]
    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}_{optimization}_{args.experiment}.json"), "w")
    batch_sizes = [1]
    for batch_size in batch_sizes:
        # for example in tqdm(processed_test_dataset, desc="Processing queries", total=len(processed_test_dataset)):
        for idx, example in tqdm(enumerate(processed_test_dataset), desc="Processing queries", total=len(processed_test_dataset)):
            input_tokens = example["context_input_ids"].copy()
            # attention_mask = example["context_attention_mask"].copy()
            context_length = len(input_tokens)

            metric = {}
            metric["prompt"] = test_dataset[idx]["context"]
            metric["prompt_length"] = context_length
            metric["batch_size"] = batch_size

            # GPU Memory profiling
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output, prefix_tokens, remaining_tokens = prefix_trie.generate_with_prefix_sharing(
                model, 
                input_tokens, 
                use_prefix=args.use_prefix, 
                max_new_tokens=args.output_max_len,
            )
            end_event.record()
            torch.cuda.synchronize()  # Ensure all CUDA kernels finish
            latency = start_event.elapsed_time(end_event)  # Time in milliseconds
            memory_used = torch.cuda.max_memory_allocated(model.device) / (1024**2)  # MB
            # memory_post = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
            batch_outputs = tokenizer.batch_decode(output.sequences[:, context_length:], skip_special_tokens=True)

            if args.use_prefix:
                metric["prefix_length"] = len(prefix_tokens)
                metric["remaining_tokens"] = len(remaining_tokens)
            metric["pred"] = batch_outputs[0]
            metric["pred_length"] = len(tokenizer.encode(metric["pred"]))
            metric["latency"] = latency  # end_time - start_time
            metric["memory"] = memory_used # memory_post - memory_pre
            metric["output_max_len"] = args.output_max_len
            metric["max_capacity_prompts"] = max_capacity_prompts
            # metric["generation_profile"] = output.profile_res

            # Dump with indent for better readability
            fout.write(json.dumps(metric, indent=4, cls=CustomJSONEncoder) + "\n")
            fout.flush()

    # # GPU Memory profiling
    # torch.cuda.reset_peak_memory_stats()
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    # output, _, _ = prefix_trie.generate_with_prefix_sharing(model, input_tokens, args.use_prefix, max_new_tokens=1)
    # end_event.record()
    # torch.cuda.synchronize()  # Ensure all CUDA kernels finish
    # latency = start_event.elapsed_time(end_event)  # Time in milliseconds
    # memory_used = torch.cuda.max_memory_allocated(model.device) / (1024**2)  # MB
    # print(f"Latency: {latency:.2f} ms, Memory used: {memory_used:.2f} MB")
    # batch_outputs = tokenizer.batch_decode(output.sequences[:, context_length:], skip_special_tokens=True)
    # print(f"Generated text: {batch_outputs[0]}")

