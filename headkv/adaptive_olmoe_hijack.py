
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.utils import logging, is_flash_attn_2_available
from transformers.activations import ACT2FN
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeMLP,
    OlmoeSparseMoeBlock,
    OlmoeDecoderLayer,
    repeat_kv,
    apply_rotary_pos_emb,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
from line_profiler import profile

logger = logging.get_logger(__name__)



def norm_olmoe_attention_forward(
    self: OlmoeAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_norm(self.q_proj(hidden_states))
    key_states = self.k_norm(self.k_proj(hidden_states))
    value_states = self.v_proj(hidden_states)

    if self.config.clip_qkv is not None:
        query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cache_position is None or cache_position[0] == 0:
        phase = "prefilling"
    else:
        phase = "decoding"

    return attn_output, attn_weights, past_key_value, phase



def norm_olmoe_flash_attn2_forward(
    self: OlmoeAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_norm(self.q_proj(hidden_states))
    key_states = self.k_norm(self.k_proj(hidden_states))
    value_states = self.v_proj(hidden_states)
    if self.config.clip_qkv is not None:
        query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (OlmoeRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cache_position is None or cache_position[0] == 0:
        phase = "prefilling"
    else:
        phase = "decoding"

    return attn_output, attn_weights, past_key_value, phase



def norm_olmoe_mlp_forward(
    self: OlmoeMLP, 
    x: torch.Tensor, 
    neuron_mask: Optional[torch.BoolTensor] = None,
):
    if neuron_mask is not None:
        x_masked = x[:, neuron_mask]  # Shape: (X * B, H')

        # Gate projection
        gate_proj = F.linear(x_masked, self.gate_proj.weight[:, neuron_mask])  # Shape: (X * B, intermediate_size)
        up_proj = F.linear(x_masked, self.up_proj.weight[:, neuron_mask])      # Shape: (X * B, intermediate_size)

        # Down projection
        down_proj = F.linear(self.act_fn(gate_proj) * up_proj, self.down_proj.weight)  # Shape: (X * B, H)
        
    else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    return down_proj

# class OlmoeMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]

#         # Generate a random mask across the neurons (hidden size)
#         self.sparsity = 0.5
#         # Ensure 50% of the neurons are kept, by ranking and selecting the top 50% of the neurons
#         self.neuron_mask = torch.randperm(self.hidden_size) < int(self.sparsity * self.hidden_size)
#         # self.create_indexed_weights(self.neuron_mask)
#         h_prime = self.neuron_mask.sum().item()  # The reduced input dimension H'
#         dtype = self.gate_proj.weight.dtype
#         self.gate_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype)
#         self.up_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype)
#         self.down_proj_masked = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype)
#         with torch.no_grad():
#             # Assign the masked weights
#             self.gate_proj_masked.weight.copy_(self.gate_proj.weight[:, self.neuron_mask])
#             self.up_proj_masked.weight.copy_(self.up_proj.weight[:, self.neuron_mask])
#             self.down_proj_masked.weight.copy_(self.down_proj.weight)

#     def forward(self, x, neuron_mask: Optional[torch.BoolTensor] = None,):
#         if neuron_mask is not None:
#             x_masked = x[:, neuron_mask]  # Shape: (X * B, H')

#             # # Gate projection
#             # gate_proj = F.linear(x_masked, self.gate_proj.weight[:, neuron_mask])  # Shape: (X * B, intermediate_size)
#             # up_proj = F.linear(x_masked, self.up_proj.weight[:, neuron_mask])      # Shape: (X * B, intermediate_size)

#             # # Down projection
#             # down_proj = F.linear(self.act_fn(gate_proj) * up_proj, self.down_proj.weight)  # Shape: (X * B, H)
#             down_proj = self.down_proj_masked(self.act_fn(self.gate_proj_masked(x_masked)) * self.up_proj_masked(x_masked))

#         else:
#             down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
#         return down_proj

# def create_indexed_weights(
#     self: OlmoeMLP, 
#     neuron_mask: torch.BoolTensor,
# ):
#     """
#     Create indexed weights based on neuron_mask and move them to GPU.
#     This avoids repeated indexing overhead during inference.
#     """
#     h_prime = neuron_mask.sum().item()  # The reduced input dimension H'
#     self.original_device = self.gate_proj.weight.device
#     dtype = self.gate_proj.weight.dtype

#     # Create new Linear layers with modified input size
#     self.gate_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype).to(self.original_device)
#     self.up_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype).to(self.original_device)
#     self.down_proj_masked = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype).to(self.original_device)

#     with torch.no_grad():
#         # Assign the masked weights
#         self.gate_proj_masked.weight.copy_(self.gate_proj.weight[:, neuron_mask])
#         self.up_proj_masked.weight.copy_(self.up_proj.weight[:, neuron_mask])
#         self.down_proj_masked.weight.copy_(self.down_proj.weight)

    # # Offload the original weights to CPU
    # self.gate_proj.to("cpu")
    # self.up_proj.to("cpu")
    # self.down_proj.to("cpu")


# def update_indexed_weights(
#     self: OlmoeMLP, 
#     neuron_mask: torch.BoolTensor,
# ):
#     """
#     Create indexed weights based on neuron_mask and move them to GPU.
#     This avoids repeated indexing overhead during inference.
#     """
#     with torch.no_grad():
#         # Assign the masked weights
#         self.gate_proj_masked.weight.copy_(self.gate_proj.weight[:, neuron_mask])
#         self.up_proj_masked.weight.copy_(self.up_proj.weight[:, neuron_mask])
#         self.down_proj_masked.weight.copy_(self.down_proj.weight)

    # # Offload the original weights to CPU
    # self.gate_proj.to("cpu")
    # self.up_proj.to("cpu")
    # self.down_proj.to("cpu")



def norm_olmoe_sparse_block_forward(
    self: OlmoeSparseMoeBlock,
    hidden_states: torch.Tensor,
    neuron_mask: Optional[torch.BoolTensor] = None,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be selected
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)  # Shape: (top_x, hidden_dim)
        # print(f"current_state: {current_state.shape}")
        current_hidden_states = expert_layer(current_state, neuron_mask) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits



@profile
def norm_olmoe_decoder_layer_indexing_forward(
    self: OlmoeDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value, phase = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if phase == "prefilling":
        hidden_states, router_logits = self.mlp(hidden_states, )
        mlp_norm = torch.norm(hidden_states, p=2, dim=1).mean(dim=0)  # L2 norm -> Shape: (hidden_size)
        sparsity = 0.8
        self.neuron_mask = mlp_norm > mlp_norm.topk(int(sparsity * mlp_norm.shape[0]), largest=False).values[-1]
        # print(f"sparsity: {1 - self.neuron_mask.sum().item() / self.neuron_mask.numel()}")
        # for expert in self.mlp.experts:
        #     if not hasattr(expert, 'gate_proj_masked'):
        #         create_indexed_weights(expert, self.neuron_mask)

        # if self.self_attn.layer_idx == 0:
            # print(f"[prefilling] last decode step: {self.decode_step if hasattr(self, 'decode_step') else None}")
            # self.decode_step = 1

    else:  # decoding, we need to use the neuron mask which is obtained during prefilling
        # TO-DO: this part is really time-consuming  (64%)
        hidden_states, router_logits = self.mlp(hidden_states, self.neuron_mask)  
        # if self.self_attn.layer_idx == 0:
        #     self.decode_step += 1
        
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs