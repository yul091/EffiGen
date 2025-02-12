import sys 
sys.dont_write_bytecode = True
from importlib.metadata import version
import warnings
import transformers
from headkv.fixed_mistral_hijack import pyramidkv_mistral_flash_attn2_forward, fixed_mistral_flash_attn2_forward, fixed_MistralModel_forward
from headkv.fixed_mistral_hijack import prepare_inputs_for_generation_mistral as fixed_prepare_inputs_for_generation_mistral
from headkv.adaptive_mistral_hijack import reason_mistral_flash_attn2_forward, adaptive_mistral_flash_attn2_forward, adaptive_MistralModel_forward, norm_mistral_flash_attn2_forward, norm_mistral_decoder_layer_indexing_forward, norm_mistral_decoder_layer_nomlp_forward
from headkv.adaptive_mistral_hijack import prepare_inputs_for_generation_mistral as ada_prepare_inputs_for_generation_mistral
from headkv.adaptive_mixtral_hijack import reason_mixtral_flash_attn2_forward, adaptive_mixtral_flash_attn2_forward, adaptive_MixtralModel_forward, norm_mixtral_flash_attn2_forward, norm_mixtral_mlp_forward, norm_mixtral_sparse_block_forward, norm_mixtral_decoder_layer_indexing_forward, norm_mixtral_decoder_layer_nomlp_forward
from headkv.adaptive_mixtral_hijack import prepare_inputs_for_generation_mixtral as ada_prepare_inputs_for_generation_mixtral

from headkv.fixed_llama_hijack import pyramidkv_llama_flash_attn2_forward, fixed_llama_flash_attn2_forward, fixed_LlamaModel_forward
from headkv.fixed_llama_hijack import prepare_inputs_for_generation_llama as fixed_prepare_inputs_for_generation_llama
from headkv.adaptive_llama_hijack import reason_llama_flash_attn2_forward, adaptive_llama_flash_attn2_forward,adaptive_LlamaModel_forward, norm_llama_flash_attn2_forward, norm_llama_mlp_forward, norm_llama_decoder_layer_nomlp_forward, norm_llama_decoder_layer_indexing_forward
from headkv.adaptive_llama_hijack import prepare_inputs_for_generation_llama as ada_prepare_inputs_for_generation_llama
from headkv.fixed_mixtral_hijack import pyramidkv_mixtral_flash_attn2_forward, fixed_mixtral_flash_attn2_forward, fixed_MixtralModel_forward
from headkv.fixed_mixtral_hijack import prepare_inputs_for_generation_mixtral as fixed_prepare_inputs_for_generation_mixtral
# from headkv.adaptive_olmoe_hijack import norm_olmoe_mlp_forward, norm_olmoe_sparse_block_forward, norm_olmoe_decoder_layer_indexing_forward, norm_olmoe_attention_forward, norm_olmoe_flash_attn2_forward


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.37']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def replace_mistral_fixed():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = fixed_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = fixed_MistralModel_forward

def replace_mistral_adaptive():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward

def replace_llama_fixed():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = fixed_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = fixed_LlamaModel_forward

def replace_llama_adaptive():
    check_version()
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward

def replace_mixtral_fixed():
    check_version()
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mixtral
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = fixed_mixtral_flash_attn2_forward
    transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = fixed_MixtralModel_forward

def replace_mixtral_adaptive():
    check_version()
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = adaptive_mixtral_flash_attn2_forward
    transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = adaptive_MixtralModel_forward


# def replace_olmoe(method):
#     if method == 'NormKV_indexmlp':
#         transformers.models.olmoe.modeling_olmoe.OlmoeMLP.forward = norm_olmoe_mlp_forward
#         transformers.models.olmoe.modeling_olmoe.OlmoeAttention.forward = norm_olmoe_attention_forward
#         transformers.models.olmoe.modeling_olmoe.OlmoeFlashAttention2.forward = norm_olmoe_flash_attn2_forward
#         transformers.models.olmoe.modeling_olmoe.OlmoeSparseMoeBlock.forward = norm_olmoe_sparse_block_forward
#         transformers.models.olmoe.modeling_olmoe.OlmoeDecoderLayer.forward = norm_olmoe_decoder_layer_indexing_forward



def replace_mistral(method):

    if method == "AdativeKV":
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward
    elif method == "ReasonKV":
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = reason_mistral_flash_attn2_forward
    elif method == 'SnapKV':
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = fixed_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = fixed_mistral_flash_attn2_forward
    elif method == 'PyramidKV':
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = fixed_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = pyramidkv_mistral_flash_attn2_forward
    elif method == 'NormKV':
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = norm_mistral_flash_attn2_forward
    elif method == 'NormKV_nomlp':
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = norm_mistral_flash_attn2_forward
        transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = norm_mistral_decoder_layer_forward


def replace_llama(method):
    check_version()

    if method == "AdativeKV":    
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = adaptive_llama_flash_attn2_forward
    elif method == "ReasonKV":
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = reason_llama_flash_attn2_forward
    elif method == 'SnapKV':
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = fixed_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = fixed_llama_flash_attn2_forward
    elif method == 'PyramidKV':
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = fixed_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = pyramidkv_llama_flash_attn2_forward
    elif method == 'NormKV':
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = norm_llama_flash_attn2_forward
        transformers.models.llama.modeling_llama.LlamaMLP.forward = norm_llama_mlp_forward  # MLP sparsity
    elif method == 'NormKV_nomlp':
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = norm_llama_flash_attn2_forward
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = norm_llama_decoder_layer_nomlp_forward
    elif method == 'NormKV_indexmlp':
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = norm_llama_flash_attn2_forward
        transformers.models.llama.modeling_llama.LlamaMLP.forward = norm_llama_mlp_forward  # MLP sparsity
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = norm_llama_decoder_layer_indexing_forward


def replace_mixtral(method):
    check_version()

    if method == "AdativeKV":
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = adaptive_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = adaptive_mixtral_flash_attn2_forward
    elif method == "ReasonKV":
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = adaptive_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = reason_mixtral_flash_attn2_forward
    elif method == 'SnapKV':
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = fixed_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = fixed_mixtral_flash_attn2_forward
    elif method == 'PyramidKV':
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = fixed_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = pyramidkv_mixtral_flash_attn2_forward
    elif method == 'NormKV':
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = adaptive_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = norm_mixtral_flash_attn2_forward
    elif method == 'NormKV_indexmlp':
        transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mixtral
        transformers.models.mixtral.modeling_mixtral.MixtralModel.forward = adaptive_MixtralModel_forward
        transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = norm_mixtral_flash_attn2_forward
        transformers.models.mixtral.modeling_mixtral.MixtralBLockSparseTop2MLP.forward = norm_mixtral_mlp_forward  # MLP sparsity
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward = norm_mixtral_sparse_block_forward  # Sparse block
        transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer.forward = norm_mixtral_decoder_layer_indexing_forward