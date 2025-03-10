
import warnings
from typing import List, Optional, Tuple, Union, Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import (
    ACT2FN,
    MixtralConfig,
    MixtralBLockSparseTop2MLP,
    MixtralSparseMoeBlock,
    MixtralDecoderLayer,
)


def sparsity_init(self: MixtralBLockSparseTop2MLP, config: MixtralConfig):
    # Originally, super().__init__() is used to call the __init__() of the parent class, which is nn.Module, 
    # we manually call the parent class __init__() method
    nn.Module.__init__(self)

    self.ffn_dim = config.intermediate_size
    self.hidden_dim = config.hidden_size

    self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
    self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
    self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    self.act_fn = ACT2FN[config.hidden_act]
    self.sparsity = config.mlp_sparsity
    self.neuron_mask = None



def mixtral_mlp_masked_forward(
    self: MixtralBLockSparseTop2MLP, 
    hidden_states: torch.Tensor,
    phase: str = "prefilling",
):
    if phase == "prefilling":
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        # Compute norm of the hidden states
        mlp_norm = torch.norm(hidden_states, p=2, dim=0).detach()  # L2 norm -> Shape: (H)
        # Compute the sparsity mask
        self.neuron_mask = mlp_norm > mlp_norm.topk(int(self.sparsity * mlp_norm.shape[0]), largest=False).values[-1]
    elif phase == "decoding":
        masked_hidden_states = hidden_states[:, self.neuron_mask].contiguous()  # shape: (X * B, H')
        w1_project = F.linear(masked_hidden_states, self.w1.weight[:, self.neuron_mask].contiguous())  # shape: (X * B, ffn_dim)
        w3_project = F.linear(masked_hidden_states, self.w3.weight[:, self.neuron_mask].contiguous())  # shape: (X * B, ffn_dim)
        current_hidden_states = self.w2(self.act_fn(w1_project) * w3_project)  # shape: (X * B, H)
    else:
        raise ValueError(f"Unknown phase: {phase}!")
        
    return current_hidden_states
        


def mixtral_sparse_block_forward(
    self: MixtralSparseMoeBlock,
    hidden_states: torch.Tensor,
    phase: str = "prefilling",
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state, phase=phase) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits



def mixtral_decoder_layer_nomlp_forward(
    self: MixtralDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def mixtral_decoder_layer_masked_forward(
    self: MixtralDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    phase = self.self_attn.generation_phase
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, router_logits = self.block_sparse_moe(hidden_states, phase=phase)

    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs