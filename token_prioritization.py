# data-driven token prioritization pipeline
import math
import torch
from datasets import Dataset, load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.generation.logits_process import LogitsProcessorList
from collections import Counter
from typing import Union, List, Dict, Optional, Callable

# Phase 1: Token scoring based on dataset
def compute_token_scores(
        dataset: Union[Dataset, List[str]], 
        tokenizer: GPT2Tokenizer,
        model: GPT2LMHeadModel, 
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        prompt_fn: Optional[Callable[[Dict[str, str]], str]] = None,
        size: Optional[int] = None,
    ):
    start_counts = Counter()
    end_counts = Counter()
    total_tokens = Counter()
    entropy_scores = Counter()
    count = 0
    for response in dataset:
        # print(f'brefore prompt template, input: {response}')
        if prompt_fn is not None:
            response = prompt_fn(response)
        # print(f'after prompt template, input: {response}')
        tokens = tokenizer.encode(response)  # List of token IDs
        total_tokens.update(tokens)
        if len(tokens) > 0:
            start_counts[tokens[0]] += 1
            end_counts[tokens[-1]] += 1

        # Calculate entropy for each token in the response
        input_ids = torch.tensor(tokens[:-1]).unsqueeze(0)  # Input without last token
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]  # Predict next token
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            for idx, token in enumerate(tokens[:-1]):  # Skip last token as no next prediction
                token_prob = probs[token].item()
                entropy = -math.log2(token_prob) if token_prob > 0 else 0
                entropy_scores[token] += entropy

        count += 1
        if size is not None and count >= size:
            break
    
    # Normalize entropy scores
    for token in entropy_scores:
        entropy_scores[token] /= total_tokens[token]

    # Compute final scores: S = alpha * P_start + beta * P_end + gamma * H
    total_responses = len(dataset)
    scores = {}
    for token, count in total_tokens.items():
        start_prob = start_counts[token] / total_responses
        end_prob = end_counts[token] / total_responses
        avg_entropy = entropy_scores[token] if token in entropy_scores else 0
        scores[token] = alpha * start_prob + beta * end_prob + gamma * avg_entropy  # Weighted combination 
    return scores


def create_prompt(inputs: Dict[str, str]):
    context = inputs['context']
    question = inputs['instruction']
    prompt = f"{context} \n### Question: {question} \n### Response: "
    return prompt


def preprocess_and_tokenize(examples):
    # Generate prompts
    prompts = [
        create_prompt({'context': context, 'instruction': instruction}) 
        for context, instruction in zip(examples['context'], examples['instruction'])
    ]
    
    # Tokenize the prompts
    tokenized_inputs = tokenizer(
        prompts, 
        padding=False, 
        truncation=True,
    )
    
    # Tokenize the responses
    labels = tokenizer(
        examples['response'], 
        padding=False, 
        truncation=True,
    )
    
    # Align labels and return
    tokenized_inputs['labels'] = labels['input_ids']
    return tokenized_inputs


class DataDrivenTokenLogitsProcessor:
    def __init__(self, token_scores, boost_factor=5.0):
        self.token_scores = token_scores
        self.boost_factor = boost_factor

    def __call__(self, input_ids, logits):
        for token, score in self.token_scores.items():
            logits[:, token] += self.boost_factor * score
        return logits


# Example dataset (responses)
# dataset = [
#     "Yes, I can help you.",
#     "No, I am not sure.",
#     "Sure, let me check that for you.",
#     "I don't know, sorry.",
#     "Yes.",
# ]
if __name__ == "__main__":
    ds = load_dataset("databricks/databricks-dolly-15k")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("Start computing token scores for the dataset ...")
    token_scores = compute_token_scores(dataset, tokenizer, model)
    # Sort tokens by importance
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
    print("Top critical tokens (top-5):", [tokenizer.decode([t]) for t, score in sorted_tokens[:5]])

    # Phase 2: Integrate token scores into decoding
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    # Wrap the custom processor in a LogitsProcessorList
    logits_processor = LogitsProcessorList([DataDrivenTokenLogitsProcessor(token_scores)])

    # Input for generation
    input_text = "Can you help me?"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate with token prioritization
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        logits_processor=logits_processor,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,  # Set the pad token to eos if missing
    )

    print("Generated response:", tokenizer.decode(outputs[0], skip_special_tokens=True))
