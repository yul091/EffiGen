import math
import warnings
from typing import Optional, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaAttention, _prepare_4d_causal_attention_mask, apply_rotary_pos_emb, repeat_kv, logger,
)


class LlamaAttentionSparse(LlamaAttention):

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # We enable memory coalescing by predefining 
        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        active_heads: Optional[torch.Tensor] = None,  # New argument
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """head_mask (`torch.Tensor`, optional): binary mask of shape `(num_heads,)` indicating active heads."""
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # If active_heads is empty, return early
        if active_heads is not None and active_heads.numel() == 0:
            return hidden_states, None, past_key_value

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply `head_mask` to select active heads
        num_heads = self.num_heads
        if active_heads is not None:
            query_states = query_states[:, active_heads]
            key_states = key_states[:, active_heads]
            value_states = value_states[:, active_heads]
            num_heads = active_heads.size(0)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # Repair the attn_output for non-active heads (zero filling)
        if active_heads is not None:
            full_attn_output = torch.zeros(bsz, q_len, self.num_heads, self.head_dim, device=attn_output.device, dtype=attn_output.dtype)
            full_attn_output[:, :, active_heads, :] = attn_output
            attn_output = full_attn_output

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


if __name__ == "__main__":
    import time
    from transformers import AutoTokenizer, AutoConfig
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the actual model name
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    config = AutoConfig.from_pretrained(model_name, token=access_token)

    sample_texts = [
        "What is the capital of France?",
        "Write me a poem on working from home please.",
        "I just got a new car and I'm loving it!",
    ]
    embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    attention_layer = LlamaAttention(config, layer_idx=0)
    sparse_attention_layer = LlamaAttentionSparse(config, layer_idx=0)
    
    inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_position_embeddings)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, seq_length = input_ids.shape[:2]
    inputs_embeds = embedding_layer(input_ids)
    print(f"Input: {inputs_embeds.shape}")
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length=0
    )

    # Test the standard attention layer
    num_exp = 50
    st = time.time()
    for _ in range(num_exp):
        with torch.no_grad():
            output, attn_weights, _ = attention_layer(inputs_embeds, attention_mask=attention_mask)
    print(f"Time (standard attention): {time.time() - st:.4f} -> Output: {output.shape}")
    
    # Test the sparse attention layer
    st = time.time()
    active_heads = torch.randint(0, 2, (config.num_attention_heads,)).nonzero().squeeze()
    print(f"Sparsity: {active_heads.numel()} / {config.num_attention_heads}")
    for _ in range(num_exp):
        with torch.no_grad():
            output, attn_weights, _ = sparse_attention_layer(
                inputs_embeds, attention_mask=attention_mask, active_heads=active_heads,
            )
    print(f"Time (sparse attention): {time.time() - st:.4f} -> Output: {output.shape}")