from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset
from evaluate import logging
import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def compute_perplexity(data, tokenizer, model, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None):
    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    # Tokenize the dataset
    encodings = tokenizer(
        data,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}



if __name__ == "__main__":
    # Load LLaMA model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # "facebook/blenderbot-3B", "gpt2", "microsoft/DialoGPT-large", "meta-llama/Llama-2-7b-chat-hf"
    access_token = "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"
    batch_size = 5
    num_samples = 100
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=access_token)
    if 'gpt' in model_name or 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        if 'llama' in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).to("cuda")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=access_token).to("cuda")
    model.eval()  # Move model to GPU for faster evaluation
    max_length = model.config.max_position_embeddings

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Pick a subset and filter those 'text' strip are empty
    dataset = dataset.select(range(num_samples)).filter(lambda x: x["text"].strip())
    predictions = dataset["text"]

    results = compute_perplexity(predictions, tokenizer, model, batch_size=batch_size, device='cuda', max_length=max_length)
    print("perplexities: ", [round(x, 2) for x in results["perplexities"]])
    print("mean_perplexity: ", round(results["mean_perplexity"], 2)) 
    # 244.06 (GPT-2), 26434433.53 (dialogpt-large), 676.65 (blenderbot-3B), 145.81 (Llama-2-7b-chat-hf)


    # # Evaluate perplexity
    # import evaluate
    # perplexity = evaluate.load("perplexity", module_type="measurement")
    # results = perplexity.compute(model_id=model_name, data=predictions, device='cuda', batch_size=batch_size, max_length=max_length)
    # print("perplexities: ", [round(x, 2) for x in results["perplexities"]])
    # print("mean_perplexity: ", round(results["mean_perplexity"], 2))


