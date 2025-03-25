
from typing import List, Optional, Dict
import pdb
import torch
import graphviz
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache

class TrieNode:
    """A node in the prefix-sharing trie that stores KV cache tensors."""
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.kv_cache: DynamicCache = None  # Stores KV cache if computed
        self.is_end_of_query: bool = False

class PrefixSharingTrie:
    """Trie structure to manage shared prefixes and their KV caches."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens: List[int], model: GPT2LMHeadModel):
        """Insert a tokenized query into the trie and compute only necessary KV cache."""
        node = self.root
        cached_tokens = []  # Tokens that already have KV cache
        computed_tokens = []  # Tokens that need to be computed

        # Step 1: Traverse the trie to find the longest cached prefix
        prefix_ending_node: TrieNode = None
        for token in tokens:
            # children = list(node.children.keys())
            if token in node.children and node.children[token].kv_cache is not None:
                node = node.children[token]
                cached_tokens.append(token)
                # cached = True
            else:
                if prefix_ending_node is None and node.kv_cache is not None:
                    prefix_ending_node = node
                # First time seeing this token -> Needs computation
                if token not in node.children:
                    node.children[token] = TrieNode()
                computed_tokens.append(token)  
                node = node.children[token]
                # cached = False
            # print(f"\ttoken {token}, node children {children}, cached {cached}")

        # If everything is already cached, no need to compute anything
        if not computed_tokens:
            return

        # Step 2: Compute KV cache only for new tokens
        real_cached_prefix_len = 0 if prefix_ending_node is None else prefix_ending_node.kv_cache.key_cache[0].shape[2]
        prefix_len = len(cached_tokens)
        assert prefix_len == real_cached_prefix_len, "KV cache shape mismatch!"
        # if prefix_len != real_cached_prefix_len:
        #     pdb.set_trace()
        # new_input_ids = torch.tensor([computed_tokens]).cuda()
        new_input_ids = torch.tensor([tokens]).cuda()
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


    def find_longest_prefix(self, tokens):
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

    def generate_with_prefix_sharing(self, model: GPT2LMHeadModel, input_tokens: List[int], use_prefix: bool = True, max_new_tokens: Optional[int] = None):
        """Generate text using the longest shared prefix KV cache."""
        if use_prefix:
            # Tokenize input
            prefix_node, prefix_tokens, remaining_tokens = self.find_longest_prefix(input_tokens)
            # print(f"  ** Found prefix {prefix_tokens} (children {list(prefix_node.children.keys())}), remaining tokens {remaining_tokens}.  **\n")
            input_ids = torch.tensor([input_tokens]).cuda()

            # Use stored KV cache if available
            if prefix_node.kv_cache:
                # input_ids = torch.tensor([remaining_tokens]).cuda()
                # input_text = tokenizer.decode(input_ids[0])
                past_key_values = prefix_node.kv_cache
                # Prefix is the input minus the remaining tokens
                # print(f"  ** Input IDs {input_ids.shape} ({input_text}), prefix key {past_key_values.key_cache[0].shape}, prefix value {past_key_values.value_cache[0].shape}.  **\n")
                # with torch.no_grad():
                #     outputs = self.model(input_ids=input_ids, past_key_values=prefix_node.kv_cache, use_cache=True)
            else:
                # Compute from scratch and store KV cache
                # input_ids = torch.tensor([input_tokens]).cuda()
                # input_text = tokenizer.decode(input_ids[0])
                past_key_values = None
                # print(f"  **  No KV cache found. Input text ({input_text}). **\n")
                # with torch.no_grad():
                #     outputs = self.model(input_ids=input_ids, use_cache=True)
        else:
            input_ids = torch.tensor([input_tokens]).cuda()
            past_key_values = None

        outputs = model.generate(
            inputs=input_ids,
            # attention_mask=torch.ones_like(input_ids, device=input_ids.device),
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
        return outputs
    


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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_prefix", action="store_true", help="Use prefix sharing")
    args = parser.parse_args()
    # Example Usage
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # model_name = "facebook/opt-125m"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    prefix_trie = PrefixSharingTrie()

    # Insert some queries
    if args.use_prefix:
        queries = [
            "What is AI?", 
            "What is AI used for?", 
            "What is AI's impact on society?",
        ]
        for q in queries:
            print(f"Inserting query: {q}")
            tokenized = tokenizer.encode(q)  # list of token IDs
            prefix_trie.insert(tokenized, model)  

    # Example usage
    if args.use_prefix:
        # Visualize the trie structure
        visualizer = TrieVisualizer(prefix_trie, tokenizer)
        os.makedirs("figures", exist_ok=True)
        visualizer.save("figures/trie_graph", file_format="png")  # Saves as trie_graph.png

    # Process new query using prefix sharing
    text = "What is AI used in medicine?"
    print(f"Processing query: {text}")
    input_tokens = tokenizer.encode(text)
    context_length = len(input_tokens)

    # GPU Memory profiling
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = prefix_trie.generate_with_prefix_sharing(model, input_tokens, args.use_prefix, max_new_tokens=50)
    end_event.record()
    torch.cuda.synchronize()  # Ensure all CUDA kernels finish
    latency = start_event.elapsed_time(end_event)  # Time in milliseconds
    memory_used = torch.cuda.max_memory_allocated(model.device) / (1024**2)  # MB
    print(f"Latency: {latency:.2f} ms, Memory used: {memory_used:.2f} MB")
    # print(output.logits.shape)  # (1, seq_len, vocab_size)
    # print(tokenizer.decode(output.logits[0].argmax(-1).tolist()))
    batch_outputs = tokenizer.batch_decode(output.sequences[:, context_length:], skip_special_tokens=True)
    print(f"Generated text: {batch_outputs[0]}")

