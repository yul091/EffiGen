import os
import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import *
from peft import LoraConfig, get_peft_model

import datasets
from datasets import Dataset
from tqdm.auto import tqdm
from typing import List, Optional, Union, Dict

import time
import json
import numpy as np
# from pathlib import Path
from reason_needle.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from reason_needle.metrics import compare_answers, TASK_LABELS

from datasets import load_from_disk, load_dataset
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



def cut_context(tokens: List[str], length: str, tokenizer=AutoTokenizer) -> str:
    return tokenizer.convert_tokens_to_string(tokens[:length])



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

# class CompactJSONEncoder(json.JSONEncoder):
#     def iterencode(self, obj, _one_shot=False):
#         if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
#             yield '[' + ', '.join(json.dumps(item, separators=(',', ':')) for item in obj) + ']'
#         else:
#             yield from super().iterencode(obj, _one_shot)

class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        if isinstance(obj, dict):
            formatted_items = []
            for key, value in obj.items():
                if key == "generation_profile" and isinstance(value, list):
                    # Format the list with each dict on a new line but compact within
                    formatted_items.append(f'"{key}": [\n        ' + ",\n        ".join(json.dumps(item, separators=(",", ":")) for item in value) + "\n    ]")
                else:
                    formatted_items.append(f'"{key}": {json.dumps(value, indent=4)}')
            return "{\n    " + ",\n    ".join(formatted_items) + "\n}"
        return super().encode(obj)


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. `The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None
    profile_res: Optional[List[Dict[str, Union[int, float]]]] = None


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        is_compileable = is_compileable and not self.generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    profile_res = []
    is_prefill = True
    while self._has_unfinished_sequences(
        this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
    ):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})


        # GPU Memory profiling
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        profile = ()
        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
            profile = {
                "operation": "prefill",
                "cur_length": cur_len,
            }
            # profile += ("prefill", cur_len)
        else:
            outputs = model_forward(**model_inputs, return_dict=True)
            profile = {
                "operation": "decode",
                "cur_length": cur_len,
            }
            # profile += ("decode", cur_len)

        # Profiling Results
        end_event.record()
        torch.cuda.synchronize()  # Ensure all CUDA kernels finish
        latency = start_event.elapsed_time(end_event)  # Time in milliseconds
        memory_used = torch.cuda.max_memory_allocated(input_ids.device) / (1024**2)  # MB
        profile["latency"] = latency
        profile["memory_used"] = memory_used
        # profile += (latency, memory_used)
        profile_res.append(profile)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].clone().float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                profile_res=profile_res,
            )
    else:
        return input_ids


@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())
    self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    device = inputs_tensor.device
    self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    # decoder-only models must use left-padding for batched generation.
    if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    # 4. Define other model kwargs
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        generation_config.use_cache = True

    if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )
    elif kwargs_has_attention_mask:
        # TODO (joao): generalize this check with other types of inputs
        if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if generation_config.token_healing:
        input_ids = self.heal_tokens(input_ids, tokenizer)

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = self._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    # dynamically overrides this value as it can need more than the last token logits
    if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
        model_kwargs["logits_to_keep"] = 1

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. Prepare the cache.
    # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    # - different models have a different cache name expected by the model (default = "past_key_values")
    # - `max_length`, prepared above, is used to determine the maximum cache length
    max_cache_length = generation_config.max_length - 1
    if (
        inputs_tensor.shape[1] != input_ids_length
        and model_input_name == "inputs_embeds"
        and not self.config.is_encoder_decoder
    ):
        max_cache_length += inputs_tensor.shape[1]
    self._prepare_cache_for_generation(
        generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    )

    # 8. determine generation mode
    # generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 9. prepare logits processors and stopping criteria
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        device=inputs_tensor.device,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    )

    # Set model_kwargs `use_cache` so we can use it later in forward runs
    model_kwargs["use_cache"] = generation_config.use_cache

    # 10. go into different generation modes
    # 11. expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )

    # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
    result = _sample(
        self,
        input_ids,
        logits_processor=prepared_logits_processor,
        stopping_criteria=prepared_stopping_criteria,
        generation_config=generation_config,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )


    # Convert to legacy cache format if requested
    if (
        generation_config.return_legacy_cache is True
        and not is_torchdynamo_compiling()
        and hasattr(result, "past_key_values")
        and getattr(result.past_key_values, "to_legacy_cache") is not None
    ):
        result.past_key_values = result.past_key_values.to_legacy_cache()
    return result


    


def main(args):
    model_path = args.model_path.lower()
    lengths = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096,]
    batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
    # if args.experiment == "fine-tune":
    #     batch_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]
    # else:
    if args.experiment != "fine-tune":
        # batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 32]
        if args.experiment == "prefill":
            output_max_len = 1
        else:
            lengths = [16,]  # for decode, we only need to test one length
            output_max_len = 4096
        
    # output_max_len = args.length
    model_name = model_path.split("/")[-1]
    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}_{args.experiment}.json"), "w")
    # memory_fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", args.dataset, f"{args.method}_memory.json"), "w")

    data_dir = 'reason_needle/babilong-100examples/64k/qa1/'
    file = os.path.join(data_dir, 'data-00000-of-00001.arrow')
    dataset = load_dataset('arrow', data_files=file, split='train')
    all_prompt = ''
    for i in range(4):
        all_prompt += dataset[i]['input'] + '\n'
    tokens = tokenizer.tokenize(all_prompt)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    pbar = tqdm(total=len(lengths))
    for index, length in enumerate(lengths):
        context = cut_context(tokens, length=length, tokenizer=tokenizer)
        bbar = tqdm(total=len(batch_sizes))
        for batch_size in batch_sizes:
            inputs = tokenizer(
                [context] * batch_size, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            # print(f"Context: {context}")
            # inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True).to("cuda")
            context_length = inputs.input_ids.shape[-1]
            max_memory_allocated_after_input_to_cuda = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
            args.result_data[f"{context_length}:after_input_to_cuda"] = f"{max_memory_allocated_after_input_to_cuda} GB"

            example = {}
            example["prompt"] = context
            example["prompt_length"] = length
            example["batch_size"] = batch_size

            if args.experiment in ["prefill", "decode"]:
                model.eval() 
                start_time = time.perf_counter()
                try:
                    output = model.generate(
                        **inputs,
                        output_attentions = args.output_attentions,
                        max_new_tokens=output_max_len,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0,
                        min_length=context_length+output_max_len,
                        eos_token_id=[tokenizer.eos_token_id],
                        return_dict_in_generate=True,
                    )
                
                    end_time = time.perf_counter()

                    max_memory_allocated_after_generate = get_max_memory_allocated(all_devices) / (1024 * 1024 * 1024)
                    args.result_data[f"{context_length}:after_generate"] = f"{max_memory_allocated_after_generate} GB"
                    args.result_data[f'{context_length}:time'] = end_time - start_time

                    # batch_outputs =tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)
                    batch_outputs = tokenizer.batch_decode(output.sequences[:, context_length:], skip_special_tokens=True)
                    torch.cuda.empty_cache()
                
                    example["pred"] = batch_outputs[0]
                    example["pred_length"] = len(tokenizer.encode(example["pred"]))
                    example["latency"] = end_time - start_time
                    example["memory"] = max_memory_allocated_after_generate - max_memory_allocated_after_input_to_cuda
                    example["output_max_len"] = output_max_len
                    example["generation_profile"] = output.profile_res
                except Exception as e:
                    print(e)
                    example["pred"] = "OOM"
                    example["pred_length"] = "OOM"
                    example["latency"] = "OOM"
                    example["memory"] = "OOM"
                    example["output_max_len"] = output_max_len
                    example["generation_profile"] = []
            
            elif args.experiment == "fine-tune":
                example["generation_profile"] = []
                model.train()
                inputs["labels"] = inputs.input_ids

                # GPU Memory profiling
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                try:
                    loss = model(**inputs).loss
                    example["loss"] = loss.item()
                    loss.backward()
                    # Profiling Results
                    end_event.record()
                    torch.cuda.synchronize()  # Ensure all CUDA kernels finish
                    latency = start_event.elapsed_time(end_event)  # Time in milliseconds
                    memory_used = torch.cuda.max_memory_allocated(inputs.input_ids.device) / (1024**2)  # MB

                    example["generation_profile"].append({
                        "operation": "fine-tune",
                        "cur_length": length,
                        "latency": latency,
                        "memory_used": memory_used,
                    })

                except Exception as e:
                    print(e)
                    example["generation_profile"].append({
                        "operation": "fine-tune",
                        "cur_length": length,
                        "latency": "OOM",
                        "memory_used": "OOM",
                    })


            # Dump with indent for better readability
            fout.write(json.dumps(example, indent=4, cls=CustomJSONEncoder) + "\n")
            bbar.update(1)
        # bbar.close()
        pbar.update(1)

    # memory_fout.write(json.dumps(args.result_data) + "\n")
    pbar.close()
 

if __name__ == "__main__":
    import transformers
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
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument("--experiment", type=str, default="decode", choices=["prefill", "decode", "fine-tune"])

    parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1, help="")
    parser.add_argument("--steps", type=int, default=-1, help="maximum number of examples to evaluate per task.")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    # Patch generation method
    transformers.generation.utils.GenerationMixin.generate = generate


    if args.device is not None:
       os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
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
        from headkv.monkeypatch import (
            replace_llama, 
            replace_mistral, 
            replace_mixtral,
            # replace_olmoe,
        ) 
        replace_llama(args.method)
        replace_mistral(args.method)
        replace_mixtral(args.method)
        # replace_olmoe(args.method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    
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
    # Patch generation method
    transformers.generation.utils.GenerationMixin.generate = generate
    
    save_dir = args.save_dir

    max_capacity_prompts = args.max_capacity_prompts
    
    for idx, dataset in enumerate(datasets):
        
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        print(f'base capacity: {args.max_capacity_prompts}\thead_choice:{args.head_choice}\tbeta:{args.beta}\ttemp:{args.temp}\talpha:{args.alpha}')

        args.dataset = dataset
        
        # args.data_file = f"/mnt/users/t-yufu/HeadAllocation_share_2/data/LongBench/{args.dataset}.jsonl"
        name = datasets2name[args.dataset]
        main(args)










