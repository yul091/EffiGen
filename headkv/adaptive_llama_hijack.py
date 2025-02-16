import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
    is_flash_attn_2_available,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
)

from headkv.snapkv_utils import (
    init_headkv,
    init_reason_snapkv,  
    init_reason_normkv,
    DynamicCacheSplitHeadFlatten,
)
from line_profiler import profile


logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

def adaptive_LlamaModel_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    is_prefill = past_key_values is None
    if is_prefill:
        import time
        torch.cuda.synchronize()
        t = time.time()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            # NOTE: adakv
            # past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    if is_prefill:
        torch.cuda.synchronize()
        t = time.time() - t
        self.prefill_time = t

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def adaptive_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: adakv
    init_headkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if key_states.shape[-2] == kv_seq_len: # Prefilling: [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

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
            # in fp32. (LlamaRMSNorm handles it correctly)

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

            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        else:  # Decoding
            self.kv_seq_len += q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1,1,self.head_dim)
            value_states = value_states.view(-1,1,self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True).reshape(
                bsz, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def reason_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: reasonkv
    init_reason_snapkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

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
            # in fp32. (LlamaRMSNorm handles it correctly)

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

            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        else:
            self.kv_seq_len += q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1,1,self.head_dim)
            value_states = value_states.view(-1,1,self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True).reshape(
                bsz, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def norm_llama_flash_attn2_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # NOTE: reasonkv
    # init_reason_snapkv(self)
    init_reason_normkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    status = "prefilling"
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster: Pre-filling
            self.kv_seq_len = kv_seq_len

            # # ORIGINALLY, UPDATE KV CACHE IS HERE BEFORE THE FLASH ATTENTION
            # key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
            # past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

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
            # in fp32. (LlamaRMSNorm handles it correctly)

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

            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )  # shape (bsz, q_len, num_heads, head_dim)

            # Calculate output norm and reallocate capacity
            attention_norm = torch.norm(attn_output, p=2, dim=-1)  # shape: (bsz, q_len, num_heads)
            self.kv_cluster.head_capacity[self.layer_idx] = self.kv_cluster.reallocate_capacity(attention_norm.mean(dim=1))

            # ALTERNATIVELY, IF WE FIRST COMPUTE OUTPUT NORM (AFTER ATTENTION) AND THEN UPDATE KV CACHE
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states.transpose(1, 2), query_states.transpose(1, 2), value_states.transpose(1, 2),
            )
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            status = "prefilling"

        else:  # Decoding
            self.kv_seq_len += q_len

            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # NOTE: update meta data
            self.kv_cluster.klen_sum += self.num_heads
            self.kv_cluster.max_seqlen_k += 1
            self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
            self.kv_cluster.head_lens += 1

            query_states = query_states.view(-1, 1, self.head_dim)
            key_states = key_states.view(-1,1,self.head_dim)
            value_states = value_states.view(-1,1,self.head_dim)

            cu_seqlens_q = self.kv_cluster.cu_qlen
            cu_seqlens_k = self.kv_cluster.cu_klen
            max_seqlen_q = 1
            max_seqlen_k = self.kv_cluster.max_seqlen_k

            attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q,
                                                 cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True).reshape(
                bsz, self.num_heads, q_len, self.head_dim)
            attn_output = attn_output.transpose(0, 1).reshape(bsz, q_len, self.hidden_size)
            status = "decoding"

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, status


# def create_indexed_weights(
#     self: LlamaMLP, 
#     neuron_mask: torch.BoolTensor,
# ):
#     """
#     Create indexed weights based on neuron_mask and move them to GPU.
#     This avoids repeated indexing overhead during inference.
#     """
#     self.original_device = self.gate_proj.weight.device
#     dtype = self.gate_proj.weight.dtype
#     h_prime = neuron_mask.sum().item()  # The reduced input dimension H'

#     # Create new Linear layers with modified input size
#     self.gate_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype).to(self.original_device)
#     self.up_proj_masked = nn.Linear(h_prime, self.intermediate_size, bias=False, dtype=dtype).to(self.original_device)
#     self.down_proj_masked = nn.Linear(self.intermediate_size, h_prime, bias=False, dtype=dtype).to(self.original_device)

#     with torch.no_grad():
#         # Assign the masked weights
#         self.gate_proj_masked.weight.copy_(self.gate_proj.weight[:, neuron_mask])
#         self.up_proj_masked.weight.copy_(self.up_proj.weight[:, neuron_mask])
#         self.down_proj_masked.weight.copy_(self.down_proj.weight[neuron_mask, :])

    # # Offload the original weights to CPU
    # self.gate_proj.to("cpu")
    # self.up_proj.to("cpu")
    # self.down_proj.to("cpu")


# def reload_wieghts(
#     self: LlamaMLP,
#     device: Optional[torch.device] = None,
# ):
#     """
#     Reload the original weights to GPU. Assume they are current on GPU.
#     """
#     device = device if device is not None else self.original_device
#     self.gate_proj.to(device)
#     self.up_proj.to(device)
#     self.down_proj.to(device)


def norm_llama_mlp_forward(
    self: LlamaMLP, 
    x: torch.Tensor, 
    neuron_mask: Optional[torch.BoolTensor] = None,
):
    if neuron_mask is not None:
        # Gate projection
        masked_x = x[:, :, neuron_mask]  # Shape: (B, T, H')
        gate_proj = F.linear(masked_x, self.gate_proj.weight[:, neuron_mask])  # Shape: (B, T, intermediate_size)
        up_proj = F.linear(masked_x, self.up_proj.weight[:, neuron_mask])      # Shape: (B, T, intermediate_size)

        # Down projection
        down_proj = F.linear(self.act_fn(gate_proj) * up_proj, self.down_proj.weight[neuron_mask, :])  # Shape: (B, T, H')

    else:
        # Directly call the original forward function
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


def norm_llama_decoder_layer_nomlp_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
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
    hidden_states, self_attn_weights, present_key_value, status = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    # hidden_states = self.mlp(hidden_states)
        
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def norm_llama_decoder_layer_indexing_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
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
    hidden_states, self_attn_weights, present_key_value, status = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    if status == "prefilling":
        hidden_states = self.mlp(hidden_states)
        # Compute neuron mask
        mlp_norm = torch.norm(hidden_states, p=2, dim=1).mean(dim=0)  # L2 norm -> Shape: (hidden_size)
        sparsity = 0.8  # keep (1 - sparsity) of the neurons
        self.neuron_mask = mlp_norm > mlp_norm.topk(int(sparsity * mlp_norm.shape[0]), largest=False).values[-1]
        # if self.self_attn.layer_idx == 0:
        #     # print(f"sparsity: {1 - self.neuron_mask.sum().item() / self.neuron_mask.numel()}")
        #     # print(f"last decode step: {self.decode_step if hasattr(self, 'decode_step') else None}")
        #     self.decode_step = 1
    else:  # decoding, we need to use the neuron mask which is obtained during prefilling
        hidden_states[:, :, self.neuron_mask] = self.mlp(hidden_states, self.neuron_mask)
        # if self.self_attn.layer_idx == 0:
        #     self.decode_step += 1
        
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            # max_cache_length = None
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs




if __name__ == "__main__":
    # Test latency of llama mlp with neuron mask
    import time
    import transformers 
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    transformers.models.llama.modeling_llama.LlamaMLP.forward = norm_llama_mlp_forward
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda:2"
    mlp = LlamaMLP(config).to(device)
    mlp.eval()

    # Test dummy input
    input_texts = [
        "Hello, my dog is cute. But let's generate some very long context, so that we can test the speed of the model. This is a very long sentence. I am not sure how long it is. But it is long enough to test the speed of the model.",
        "Passage 1:\nWaldrada of Lotharingia\nWaldrada was the mistress, and later the wife, of Lothair II of Lotharingia.\n\nBiography\nWaldrada's family origin is uncertain. The prolific 19th-century French writer Baron Ernouf suggested that Waldrada was of noble Gallo-Roman descent, sister of Thietgaud, the bishop of Trier, and niece of Gunther, archbishop of Cologne. However, these suggestions are not supported by any evidence, and more recent studies have instead suggested she was of relatively undistinguished social origins, though still from an aristocratic milieu.\nThe Vita Sancti Deicoli states that Waldrada was related to Eberhard II, Count of Nordgau (included Strasbourg) and the family of Etichonids, though this is a late 10th-century source and so may not be entirely reliable on this question.In 855 the Carolingian king Lothar II married Teutberga, a Carolingian aristocrat and the daughter of Bosonid Boso the Elder. The marriage was arranged by Lothar's father Lothar I for political reasons. It is very probable that Waldrada was already Lothar II's mistress at this time.Teutberga was allegedly not capable of bearing children and Lothar's reign was chiefly occupied by his efforts to obtain an annulment of their marriage, and his relations with his uncles Charles the Bald and Louis the German were influenced by his desire to obtain their support for this endeavour. Lothair, whose desire for annulment was arguably prompted by his affection for Waldrada, put away Teutberga. However, Hucbert took up arms on his sister's behalf, and after she had submitted successfully to the ordeal of water, Lothair was compelled to restore her in 858. Still pursuing his purpose, he won the support of his brother, Emperor Louis II, by a cession of lands and obtained the consent of the local clergy to the annulment and to his marriage with Waldrada, which took place in 862. However, Pope Nicholas I was suspicious of this and sent legates to investigate at the Council of Metz in 863. The Council found in favour of Lothair's divorce, which led to rumours that the papal legates may have bribed and thus meant that Nicholas order Lothair to take Teutberga back or face excommunication. \nWith the support of Charles the Bald and Louis the German, Teutberga appealed the annulment to Pope Nicholas. Nicholas refused to recognize the annulment and excommunicated Waldrada in 866, forcing Lothair to abandon Waldrada in favour of Teutberga. Lothair accepted this begrudgingly for a time, but shortly afterward at the end of 867 Pope Nicholas I died. Thus, Lothair began to seek the permission of the newly appointed Pope Adrian II to again put Teutberga aside and marry Waldrada, riding to Rome to speak with him on the matter in 869. However, on his way home, Lothair died.\n\nChildren\nWaldrada and Lothair II had some sons and probably three daughters, all of whom were declared illegitimate:\n\nHugh (c. 855–895), Duke of Alsace (867–885)\nGisela (c. 865–908), who in 883 married Godfrey, the Viking leader ruling in Frisia, who was murdered in 885\nBertha (c. 863–925), who married Theobald of Arles (c. 854–895), count of Arles, nephew of Teutberga. They had two sons, Hugh of Italy and Boso of Tuscany. After Theobald's death, between 895 and 898 she married Adalbert II of Tuscany (c. 875–915) They had at least three children: Guy, who succeeded his father as count and duke of Lucca and margrave of Tuscany, Lambert succeeded his brother in 929, but lost the titles in 931 to his half-brother Boso of Tuscany, and Ermengard.\nErmengarde (d. 90?)\nOdo (d. c.879)\nPassage 2:\nFrancis I Rákóczi\nFrancis I Rákóczi (February 24, 1645, Gyulafehérvár, Transylvania – July 8, 1676, Zboró, Royal Hungary) was a Hungarian aristocrat, elected prince of Transylvania and father of Hungarian national hero Francis Rákóczi II.Francis Rákóczi was the son of George Rákóczi II, prince of Transylvania, and Sophia Báthory. He was elected prince by the Transylvanian Diet in 1652, during his father's life. However, because of the disastrous Polish campaign of 1657 and its consequences, the Ottoman Empire removed his father from the throne in 1660, and prohibited any Rákóczi to ascend the Transylvanian throne. This left Francis unable to come by his father's legacy; he therefore withdrew to his estates in Royal Hungary.\nNotably, the Rákóczi family was Calvinist, and they were staunch supporters of the Reformed Church in Hungary. However, Francis' mother, Sophia Báthory, had converted to Calvinism merely for the sake of her marriage. After her husband's death, she returned to Catholicism and supported the Counter Reformation. Francis Rákóczi also became a Catholic, thus acquiring favour with the Catholic Habsburg Court. His mother converted him to Catholicism. He was made a count in 1664.\nIn 1666 Francis married Jelena Zrinska (Hungarian: Zrínyi Ilona), a Croatian countess, and joined the Wesselényi conspiracy (Zrinski-Frankopan conspiracy in Croatia), one leader of which was Jelena's father, Petar Zrinski (Hungarian: Zrínyi Péter). Francis soon became the leader of the conspiracy, and, as a culmination of their anti-Habsburg stratagems, started an armed uprising of nobles in Upper Hungary, while the other conspirators were supposed to start the fight in Croatia. Due to poor organization and discord between the conspirators, however, the Austrian authorities were well informed; they quickly suppressed the Croatian branch of the revolt.\nWhen Rákóczi learned that Petar Zrinski had been captured by the Austrians, he laid down his arms and applied for mercy. All other leaders of the conspiracy were executed for high treason; Rákóczi, due to his mother's intervention, and for a ransom of 300,000 forints and several castles, was pardoned. \n\nIssue\nFrancis I had three  children:\n\nGyörgy (1667)\nJulianna Borbála (1672–1717), married Count Ferdinand Gobert von Aspremont-Lynden (1643-1708)\nFrancis Rákóczi II (1676–1735)Francis II was born only three months before his father's death. He led a rebellion against Austrian rule (Rákóczi's War of Independence) and died in exile.\nPassage 3:\nMary Fiennes (lady-in-waiting)\nMary Fiennes (1495–1531) was an English courtier. She was the wife of Henry Norris. Norris was executed for treason as one of the alleged lovers of her cousin, Anne Boleyn, the second wife of King Henry VIII of England. Mary lived for six years at the French court as a Maid of Honour to queens consort Mary Tudor, wife of Louis XII; and Claude of France, wife of Francis I.\n\nFamily and early years\nMary was born at Herstmonceux Castle in Sussex in 1495, the only daughter of Thomas Fiennes, 8th Baron Dacre and Anne Bourchier. By both her father and mother she was descended from Edward III. She had two younger brothers, Sir Thomas and John. Her mother was an elder half-sister of Elizabeth Howard and Lord Edmund Howard, making queen consorts Anne Boleyn and Catherine Howard a cousin of Mary. Her paternal grandmother, Alice FitzHugh, was sister to Elizabeth FitzHugh, grandmother of Catherine Parr, making her cousin to yet another queen consort of Henry VIII.\nIn 1514, Mary was appointed a Maid of Honour to Princess Mary Tudor and accompanied her to France when the latter married King Louis XII of France; afterwards she served in the capacity to Queen Mary's successor, Queen Claude, consort of the new king Francis I of France. Among her fellow Maids of Honour were her cousins, Mary (a mistress of Henry VIII) and Anne Boleyn.\n\nMarriage and issue\nIn 1520 upon her return to England, she married the courtier, Henry Norreys (1491 – 17 May 1536) of Yattendon in Berkshire, whom she had met that same year at the Field of the Cloth of Gold in France.\nNorris served King Henry VIII of England as a Gentleman of the Bedchamber, and was held in high favour by the King. He was later appointed Groom of the Stool and continued to enjoy the King's favour. According to biographer Eric Ives, Norris was \"perhaps the nearest thing Henry had to a friend.\" Norris had control of King Henry's Privy chamber.\nHenry and Mary had three children:\nEdward Norris (died 1529)\nHenry Norris, 1st Baron Norreys (c. 1525 – 1601), married Margaret Williams of Rycote, by whom he had issue.\nMary Norris, married firstly Sir George Carew, and secondly Sir Arthur Champernowne, by whom she had issue.\n\nDeath\nMary died in 1531, a year after her mother. Five years later her husband was attainted and executed for treason as one of the five alleged lovers of her cousin Queen Anne Boleyn, who herself was beheaded at the Tower of London on 19 May 1536.\nTheir four orphaned children were raised by Norris's brother Sir John Norris.\n\nAncestry\nPassage 4:\nAgatha (wife of Samuel of Bulgaria)\nAgatha (Bulgarian: Агата, Greek: Άγάθη; fl. late 10th century) was the wife of Emperor Samuel of Bulgaria.\n\nBiography\nAccording to a later addition to the history of the late-11th-century Byzantine historian John Skylitzes, Agatha was a captive from Larissa, and the daughter of the magnate of Dyrrhachium, John Chryselios. Skylitzes explicitly refers to her as the mother of Samuel's heir Gavril Radomir, which means that she was probably Samuel's wife. On the other hand, Skylitzes later mentions that Gavril Radomir himself also took a beautiful captive, named Irene, from Larissa as his wife. According to the editors of the Prosopographie der mittelbyzantinischen Zeit, this may have been a source of confusion for a later copyist, and Agatha's real origin was not Larissa, but Dyrrhachium. According to the same work, it is likely that she had died by ca. 998, when her father surrendered Dyrrhachium to the Byzantine emperor Basil II.Only two of Samuel's and Agatha's children are definitely known by name: Gavril Radomir and Miroslava. Two further, unnamed, daughters are mentioned in 1018, while Samuel is also recorded as having had a bastard son.Agatha is one of the central characters in Dimitar Talev's novel Samuil.\nPassage 5:\nEmpress Shōken\nEmpress Dowager Shōken (昭憲皇太后, Shōken-kōtaigō, 9 May 1849 – 9 April 1914), born Masako Ichijō (一条勝子, Ichijō Masako), was the wife of Emperor Meiji of Japan. She is also known under the technically incorrect name Empress Shōken (昭憲皇后, Shōken-kōgō). She was one of the founders of the Japanese Red Cross Society, whose charity work was known throughout the First Sino-Japanese War.\n\nEarly life\nLady Masako Ichijō was born on 9 May 1849, in Heian-kyō, Japan. She was the third daughter of Tadayoshi Ichijō, former Minister of the Left and head of the Fujiwara clan's Ichijō branch. Her adoptive mother was one of Prince Fushimi Kuniie's daughters, but her biological mother was Tamiko Niihata, the daughter of a doctor from the Ichijō family. Unusually for the time, she had been vaccinated against smallpox. As a child, Masako was somewhat of a prodigy: she was able to read poetry from the Kokin Wakashū by the age of 4 and had composed some waka verses of her own by the age of 5. By age seven, she was able to read some texts in classical Chinese with some assistance and was studying Japanese calligraphy. By the age of 12, she had studied the koto and was fond of Noh drama. She excelled in the studies of finances, ikebana and Japanese tea ceremony.The major obstacle to Lady Masako's eligibility to become empress consort was the fact that she was 3 years older than Emperor Meiji, but this issue was resolved by changing her official birth date from 1849 to 1850. They became engaged on 2 September 1867, when she adopted the given name Haruko (美子), which was intended to reflect her \nserene beauty and diminutive size.\nThe Tokugawa Bakufu promised 15,000 ryō in gold for the wedding and assigned her an annual income of 500 koku, but as the Meiji Restoration occurred before the wedding could be completed, the promised amounts were never delivered. The wedding was delayed partly due to periods of mourning for Emperor Kōmei,  for her brother Saneyoshi, and the political disturbances around Kyoto between 1867 and 1868.\n\nEmpress of Japan\nLady Haruko and Emperor Meiji's wedding was finally officially celebrated on 11 January 1869. She was the first imperial consort to receive the title of both nyōgō and of kōgō (literally, the emperor's wife, translated as \"empress consort\"), in several hundred years. However, it soon became clear that she was unable to bear children. Emperor Meiji already had 12 children by 5 concubines, though: as custom in Japanese monarchy, Empress Haruko adopted Yoshihito, her husband's eldest son by Lady Yanagihara Naruko, who became Crown Prince. On 8 November 1869, the Imperial House departed from Kyoto for the new capital of Tokyo. In a break from tradition, Emperor Meiji insisted that the Empress and the senior ladies-in-waiting should attend the educational lectures given to the Emperor on a regular basis about national conditions and developments in foreign nations.\n\nInfluence\nOn 30 July 1886, Empress Haruko attended the Peeresses School's graduation ceremony in Western clothing. On 10 August, the imperial couple received foreign guests in Western clothing for the first time when hosting a Western Music concert.From this point onward, the Empress' entourage wore only Western-style clothes in public, to the point that in January 1887 \nEmpress Haruko issued a memorandum on the subject: traditional Japanese dress was not only unsuited to modern life, but Western-style dress was closer than the kimono to clothes worn by Japanese women in ancient times.In the diplomatic field, Empress Haruko hosted the wife of former US President Ulysses S. Grant during his visit to Japan. She was also present for her husband's meetings with Hawaiian King Kalākaua in 1881. Later that same year, she helped host the visit of the sons of future British King Edward VII: Princes Albert Victor and George (future George V), who presented her with a pair of pet wallabies from Australia.On  26 November 1886, Empress Haruko accompanied her husband to Yokosuka, Kanagawa to observe the new Imperial Japanese Navy cruisers Naniwa and Takachiho firing torpedoes and performing other maneuvers. From 1887, the Empress was often at the Emperor's side in official visits to army maneuvers. When Emperor Meiji fell ill in 1888, Empress Haruko took his place in welcoming envoys from Siam, launching warships and visiting Tokyo Imperial University. In 1889, Empress Haruko accompanied Emperor Meiji on his official visit to Nagoya and Kyoto. While he continued on to visit naval bases at Kure and Sasebo, she went to Nara to worship at the principal Shinto shrines.Known throughout her tenure for her support of charity work and women's education during the First Sino-Japanese War (1894–95), Empress Haruko worked for the establishment of the Japanese Red Cross Society. She participated in the organization's administration, especially in their peacetime activities in which she created a money fund for the International Red Cross. Renamed \"The Empress Shōken Fund\", it is presently used for international welfare activities. After Emperor Meiji moved his military headquarters from Tokyo to Hiroshima to be closer to the lines of communications with his troops, Empress Haruko joined her husband in March 1895. While in Hiroshima, she insisted on visiting hospitals full of wounded soldiers every other day of her stay.\n\nDeath\nAfter Emperor Meiji's death in 1912, Empress Haruko was granted the title Empress Dowager (皇太后, Kōtaigō) by her adoptive son, Emperor Taishō. She died in 1914 at the Imperial Villa in Numazu, Shizuoka and was buried in the East Mound of the Fushimi Momoyama Ryo in Fushimi, Kyoto, next to her husband. Her soul was enshrined in Meiji Shrine in Tokyo. On 9 May 1914, she received the posthumous name Shōken Kōtaigō (昭憲皇太后). Her railway-carriage can be seen today in the Meiji Mura Museum, in Inuyama, Aichi prefecture.\n\nHonours\nNational\nGrand Cordon of the Order of the Precious Crown, 1 November 1888\n\nForeign\nShe received the following orders and decorations:\n Russian Empire: Grand Cross of the Order of St. Catherine, 13 December 1887\n Spain: Dame of the Order of Queen Maria Luisa, 29 November 1889\n Siam: Dame of the Order of the Royal House of Chakri, 12 October 1899\n German Empire: Dame of the Order of Louise, 1st Class, 19 May 1903\n Kingdom of Bavaria: Dame of Honour of the Order of Theresa, 29 February 1904\n Korean Empire: Grand Cordon of the Order of the Auspicious Phoenix, 27 July 1908\n\nAncestry\nSee also\nEmpress of Japan\nŌmiya Palace\n\nNotes\nPassage 6:\nEunoë (wife of Bogudes)\nEunoë Maura was the wife of Bogudes, King of Western Mauretania. Her name has also been spelled Euries or Euryes or Eunoa.\n\nBiography\nEarly life\nEunoë Maura was thought to be descended from Berbers, but her name is Greek so it appears she might have been from there or had Greek ancestry. She was likely of very high status, as she is mentioned by historian Suetonius in the same context as Cleopatra.\n\nMarriage\nAt an unspecified early date in her marriage to her husband Bogud he mounted an expedition along the Atlantic coast, seemingly venturing into the tropics. When he returned he presented his wife Eunoë with gigantic reeds and asparagus he had found on the journey.She is believed to have been a mistress of Julius Caesar. She may have replaced Cleopatra in Caesar's affections, when he arrived in North Africa prior to the Battle of Thapsus on 6 April 46 BC, the two were among several queens courted by Caesar. It is also possible that they first met in Spain if she accompanied her husband there on a campaign. Only a brief romance for the Roman, both Eunoe and Bogudes profited through gifts bestowed on them by Caesar. Caesar departed from Africa in June 46 BC, five and a half months after he landed.\n\nCultural depictions\nEunoë and Caesar's affair is greatly exaggerated and expanded on in the Medieval French prose work Faits des Romains. Jeanette Beer in her book A Medieval Caesar states that the Roman general is \"transformed into Caesar, the medieval chevalier\" in the text, and that the author is more interested in Caesar's sexual dominance over the queen than the political dominance he held over her husband Bogud. The text describes her; \"Eunoe was the most beautiful woman in four kingdoms — nevertheless, she was Moorish\", which Beer further analysed as being indicative of the fact that it was unimaginable to audiences of the time to believe that a lover of Caesar could be ugly, but that Moors still represented everything that was ugly to them.Eunoë has also been depicted in several novels about Caesar, as well as serialized stories in The Cornhill Magazine. In such fiction her character often serves as a foil for the relationship between Caesar and another woman, mostly Cleopatra, such as in The Memoirs of Cleopatra, The Bloodied Toga and When We Were Gods. In Song of the Nile she also plays a posthumous role as a person of interest for Cleopatra's daughter Selene II who became queen of Mauritania after her.Eunoe has also been depicted in a numismatic drawing by Italian artist and polymath Jacopo Strada, who lived in the 16th century.",
        "AL PACINO is a great actor. He has been in many great movies. I like his acting. He is very talented. I wish I could meet him in person. That would be a dream come true. One amazing thing about him is that he is very versatile. He can play any role. He is a true legend. I highly recommend one of his movies called 'The Godfather', where he acts as Michael Corleone who is a mafia boss in New York. The most iconic scene is when he says 'I'm gonna make him an offer he can't refuse'.",
    ]
    torch.random.manual_seed(0)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=False)
    embedding = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id).to(device)
    hidden_states = embedding(inputs["input_ids"].to(device))
    neuron_mask = torch.rand(config.hidden_size, device=device) > 0.5
    print(f"sparsity: {1 - neuron_mask.sum().item() / config.hidden_size}")
    print(f"input hidden {hidden_states.shape}")
    steps = 500
    testID = 1
    
    # Test latency of llama mlp without neuron mask
    # # torch.cuda.synchronize()
    if testID == 0:
        # reload_wieghts(mlp, device)
        t = time.perf_counter()
        # y = mlp(hidden_states)
        # Let's run 100 times to get a more accurate measurement
        for _ in range(steps):
            y = mlp(hidden_states)
        # torch.cuda.synchronize()
        print("Time: ", time.perf_counter() - t)

    else:
        # torch.cuda.synchronize()
        # masked_hidden_states = hidden_states[:, :, neuron_mask]
        # create_indexed_weights(mlp, neuron_mask)
        t = time.perf_counter()
        # y = mlp(masked_hidden_states, neuron_mask)
        # Let's run 100 times to get a more accurate measurement
        for _ in range(steps):
            y = mlp(hidden_states, neuron_mask)
        # torch.cuda.synchronize()
        print("Time (neuron_mask): ", time.perf_counter() - t)
