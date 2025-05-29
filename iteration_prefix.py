"""This module define the prefix node for prefilling (sharing KV cache)."""

import sys 
sys.dont_write_bytecode = True
import pickle
import torch
from typing import List, Tuple, Dict, Optional, Callable
from transformers import GenerationConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, Cache
from utils import prepare_inputs



class TrieNode:
    def __init__(self, token: Optional[int] = None, parent: Optional['TrieNode'] = None): 
        self.token = token
        self.parent = parent
        self.children: Dict[int, 'TrieNode'] = {}
        self.ref_count = 0      # Reference count for the number of tasks using this node
        self.kv_cache: DynamicCache = None    # Placeholder for KV cache
        self.materialized = False
        # self.path = parent.path + [token] if parent else [token]

    # def get_path(self) -> List[int]:
    #     """
    #     Get the path of tokens from the root to this node.
    #     """
    #     return self.path
    

class PrefixTree:
    def __init__(self):
        self.root = TrieNode() 
    
    def insert(self, tokens: List[int]) -> TrieNode:
        """
        Insert a sequence of tokens into the prefix tree.
        Returns the node corresponding to the last token in the sequence.
        """
        node = self.root 
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode(token=token, parent=node)
            node = node.children[token]
            node.ref_count += 1
        return node
    
    def get_shared_prefix_node(self, tokens: List[int]) -> Tuple[TrieNode, int]:
        """
        Get the longest shared prefix node for a sequence of tokens.
        Returns the node and the length of the shared prefix.
        """
        node = self.root
        shared_length = 0
        # print(f"[DEBUG] Searching for shared prefix for tokens: {tokens}")
        for token in tokens:
            if token in node.children and node.children[token].materialized:
                node = node.children[token]
                shared_length += 1
            else:
                break 
            # print(f"[DEBUG] Found shared prefix token: {token}, current shared length: {shared_length}")
        
        return node, shared_length
    

# class KVCachePool:
#     def __init__(self):
#         self.pool: Dict[str, Dict] = {}  # Dictionary to hold KV caches by task ID
    

#     def save(self, node_path: List[int], kv_cache: Cache):
#         key = self._path_to_key(node_path)
#         self.pool[key] = {
#             "kv": prepare_inputs(kv_cache, "cpu"),  # Convert to CPU for storage
#             "ref_count": 1  # Initialize reference count
#         }


#     def get(self, node_path: List[int]) -> Optional[Cache]:
#         key = self._path_to_key(node_path)
#         if key in self.pool:
#             self.pool[key]["ref_count"] += 1
#             return self.pool[key]["kv"]
#         return None
    

#     def release(self, node_path: List[int]):
#         key = self._path_to_key(node_path)
#         if key in self.pool:
#             self.pool[key]["ref_count"] -= 1
#             if self.pool[key]["ref_count"] <= 0:
#                 del self.pool[key]

#     def _path_to_key(self, path: List[int]) -> str:
#         """
#         Convert a list of tokens to a string key for the pool.
#         """
#         return '-'.join(map(str, path))
    


class PrefixManager:

    def __init__(self):
        self.tree = PrefixTree()
        # self.cache_pool = KVCachePool()


    def register_request(self, tokens: List[int]) -> Tuple[TrieNode, int]:
        """
        Register a request with a sequence of tokens.
        Returns the node corresponding to the last token and the length of the shared prefix.
        """
        return self.tree.get_shared_prefix_node(tokens)


    def get_kv_for_prefix(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",
        **kwargs,
    ) -> Tuple[List[Optional[DynamicCache]], List[int]]:
        """
        Get the KV cache for a batch of token sequences.
        Returns the KV cache if it exists, otherwise None, and shared prefix length.
        """
        batch_kvs, shared_lengths = [], []
        valid_lengths = attention_mask.sum(dim=1).tolist()
        for i in range(input_ids.shape[0]):
            # tokens = input_ids[i][attention_mask[i] == 1].tolist()  # Get the prefix tokens
            length = valid_lengths[i]
            if padding_side == "left":
                tokens = input_ids[i][-length:].tolist()  # Get the last `length` tokens
            elif padding_side == "right":
                tokens = input_ids[i][:length].tolist()
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}. Use 'left' or 'right'.")
            node, shared_len = self.register_request(tokens)
            shared_lengths.append(shared_len)
            if node.materialized:
                # If the node is materialized, return the cached KV cache
                batch_kvs.append(node.kv_cache)
            else:
                # If not materialized, return None
                batch_kvs.append(None)

        return batch_kvs, shared_lengths
        # return node.kv_cache, shared_len
        # cache = self.cache_pool.get(node.get_path())
        # return cache, shared_len


    def _materialize_prefix(self, tokens: List[int], new_kv_cache: List[DynamicCache]):
        """
        Materialize the prefix for a sequence of tokens.
        If the prefix already exists, it will be updated with the new KV cache.
        """
        node = self.tree.root
        for i, token in enumerate(tokens):
            if token not in node.children:
                node.children[token] = TrieNode(token=token, parent=node)
            node = node.children[token]
            node.ref_count += 1

            if not node.materialized:
                # Save the KV cache for the prefix
                # self.cache_pool.save(node.get_path(), new_kv_cache[i - 1])
                node.kv_cache = new_kv_cache[i]
                node.materialized = True


    def materialize_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        full_cache: DynamicCache,
        padding_side: str = "left",
        **kwargs,
    ):
        """
        Materialize the prefix for a batch of token sequences.
        This will update the prefix tree with the new KV cache.
        """
        batch_size = input_ids.shape[0]
        valid_lengths = attention_mask.sum(dim=1).tolist()
        for i in range(batch_size):
            length = valid_lengths[i]
            if padding_side == "left":
                tokens = input_ids[i][-length:].tolist()  # Get the last `length` tokens
            elif padding_side == "right":
                tokens = input_ids[i][:length].tolist()
            else:
                raise ValueError(f"Invalid padding_side: {padding_side}. Use 'left' or 'right'.")
            # tokens = input_ids[i][attention_mask[i] == 1].tolist()
            kv_cache_list = []
            for step in range(1, length + 1):
                sliced_cache = self.slice_cache_at_step(full_cache, step, i, valid_length=length, padding_side=padding_side)
                kv_cache_list.append(sliced_cache)
                
            # Materialize the prefix with the new KV cache
            self._materialize_prefix(tokens, kv_cache_list)


    def slice_cache_at_step(
        self, 
        full_cache: DynamicCache, 
        step: int, 
        sample_idx: int,
        valid_length: int,
        padding_side: str = "left",
    ) -> DynamicCache:
        """Slice full KV cache to only include prefix up to `step` tokens."""
        sliced_cache = DynamicCache()
        if padding_side == "left":
            start = valid_length - step
            end = valid_length
        elif padding_side == "right":
            start = 0
            end = step
        else:
            raise ValueError(f"Invalid padding_side: {padding_side}. Use 'left' or 'right'.")

        for layer_idx in range(len(full_cache.key_cache)):
            # [B, H, S, D] -> [H, S', D]
            sliced_key = full_cache.key_cache[layer_idx][sample_idx, :, start:end, :].detach().cpu()
            sliced_value = full_cache.value_cache[layer_idx][sample_idx, :, start:end, :].detach().cpu()
            sliced_cache.update(sliced_key, sliced_value, layer_idx)
        return sliced_cache
    

    def save(self, save_dir: str = "prefix_cache"):
        """
        Save the prefix tree and KV cache pool to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save trie structure
        with open(os.path.join(save_dir, "prefix_tree.pkl"), "wb") as f:
            pickle.dump(self.tree, f)


    @staticmethod
    def load(load_dir: str) -> "PrefixManager":
        with open(os.path.join(load_dir, "prefix_tree.pkl"), "rb") as f:
            tree = pickle.load(f)
        manager = PrefixManager()
        manager.tree = tree
        return manager

def patch_position_ids(inputs, prefix_sizes, tokenizer):
    # inputs: already padded
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    B, T = input_ids.shape
    pad_left = tokenizer.padding_side == "left"

    position_ids = torch.zeros_like(input_ids)
    for i in range(B):
        valid_len = attention_mask[i].sum().item()
        prefix_len = prefix_sizes[i]
        if pad_left:
            start = T - valid_len
            position_ids[i, start:] = torch.arange(prefix_len, prefix_len + valid_len, device=input_ids.device)
        else:
            position_ids[i, :valid_len] = torch.arange(prefix_len, prefix_len + valid_len, device=input_ids.device)
    inputs["position_ids"] = position_ids

def prepare_generation_inputs_with_cache(
    model: LlamaForCausalLM, 
    inputs: Dict[str, torch.Tensor], 
    generation_config: GenerationConfig,
    tokenizer,
):
    """
    Prepare inputs for generation by moving them to the specified device.
    """
    # Prepare generation config and model kwargs
    generation_config, model_kwargs = model._prepare_generation_config(generation_config, **inputs)
    # print(f"input_ids shape: {model_kwargs['input_ids'].shape}")
    # print(f"cache position before cache pos: {model_kwargs.get('cache_position', None)}")
    # model_kwargs = model._get_initial_cache_position(model_kwargs["input_ids"], model_kwargs)
    # print(f"cache position after cache pos: {model_kwargs.get('cache_position', None)}")

    # 🛠️ Patch 1: Add position_ids to support cache continuation
    patch_position_ids(model_kwargs, model_kwargs["prefix_sizes"], tokenizer)
    print(f"position_ids shape: {model_kwargs['position_ids'].shape}")

    # 🛠️ Patch 2: Set cache_position manually if prefix cache exists
    if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
        prefix_sizes = model_kwargs["prefix_sizes"]
        device = model_kwargs["input_ids"].device
        model_kwargs["cache_position"] = torch.tensor(prefix_sizes, device=device, dtype=torch.long)

    # Compile-aware forward path
    model_forward = model.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and model._supports_static_cache
        is_compileable = is_compileable and not model.generation_config.disable_compile
        if is_compileable and (
            model.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = model.get_compiled_call(generation_config.compile_config)
    # Slicing the inputs based on cache positions
    model_inputs = model.prepare_inputs_for_generation(**model_kwargs)
    return model_inputs, model_forward



if __name__ == "__main__":
    import os
    from iteration_producer import Producer
    from alignment_study import DPOCollator
    from iteration_bin import Bin
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    device = 0
    attn_implementation = "flash_attention_2"  # or "eager" for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": device},
        use_cache=True,
        attn_implementation=attn_implementation,
    )  # Load fine-tuned model
    generation_config = model.generation_config

    dataset_name = "data/Anthropic"
    producer = Producer(
        arrival_rate=5, 
        retrain_rate=1.0, 
        n_test_samples=100,
        strategy="sync",
    )
    preloaded_tasks = producer.load_dataset(
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
        dataset_name=dataset_name,
    )
    collator = DPOCollator(
        tokenizer, 
        inference_input_feature="input_ids", 
        inference_mask_feature="attention_mask",
    )
    
    # Example usage
    prefix_manager = PrefixManager()
    batch = preloaded_tasks[:3]
    inputs = Bin.create_batch(
        "prefill", batch, tokenizer,
        batch_collator=collator,
        device=device,
        get_prefix_cache=prefix_manager.get_kv_for_prefix,
    )

    # Register a sequence of tokens
    # requests = [
    #     "What is the best way to solve a Sudoku?",
    #     "Is it possible to solve a Rubik's Cube in under 10 seconds?",
    #     "How can I improve my chess skills?",
    # ]
    # inputs = tokenizer(requests, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Use shared prefix cache
    # shared_cache1, shared_len1 = prefix_manager.get_kv_for_prefix(**inputs)
    # print(f"[INFO] Shared length for request #1: {shared_len1}, KV Cache: {shared_cache1}")
    # for i, task in enumerate(batch):
    #     task.past_key_values = shared_cache1[i] if shared_cache1[i] is not None else None
        
    model_inputs, model_forward = prepare_generation_inputs_with_cache(
        model, inputs, generation_config, tokenizer=tokenizer,
    )
    # Get cache by forwarding the model 
    with torch.no_grad():
        outputs = model_forward(**model_inputs, return_dict=True)
    print(f"[INFO] KV Cache's size: {outputs.past_key_values.key_cache[0].shape[2]}")
    
    prefix_manager.materialize_prefix(**inputs, full_cache=outputs.past_key_values)
    # prefix_manager.save("prefix_cache")  # Save the prefix tree and cache pool 
    # del prefix_manager  # Clear the manager to test loading
    # prefix_manager = PrefixManager.load("prefix_cache")
    
    # Register another sequence of tokens
    # requests = ["What is the best way to transport goods?"]
    # inputs = tokenizer(requests, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # shared_cache2, shared_len2 = prefix_manager.get_kv_for_prefix(**inputs)
    # print(f"[INFO] Shared length for request #2: {shared_len2}, KV Cache: {shared_cache2}")
    batch = preloaded_tasks[3:6]
    inputs = Bin.create_batch(
        "prefill", batch, tokenizer,
        batch_collator=collator,
        device=device,
        get_prefix_cache=prefix_manager.get_kv_for_prefix,
    )

    model_inputs, model_forward = prepare_generation_inputs_with_cache(
        model, inputs, generation_config, tokenizer=tokenizer,
    )
    # Get cache by forwarding the model
    with torch.no_grad():
        outputs = model_forward(**model_inputs, return_dict=True)
    print(f"[INFO] KV Cache's size: {outputs.past_key_values.key_cache[0].shape[2]}")
    prefix_manager.materialize_prefix(**inputs, full_cache=outputs.past_key_values)


    # Visualize the prefix tree structure (with indent and / | \ for branches)
    def print_tree(node: TrieNode, indent: str = ""):
        print(f"{indent}{node.token} ({node.ref_count}, {node.materialized})")
        for child in node.children.values():
            print_tree(child, indent + "  ")
    # print("[INFO] Prefix Tree Structure:")
    # print_tree(prefix_manager.tree.root)