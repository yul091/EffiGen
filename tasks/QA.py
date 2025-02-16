
import os
import tqdm
import evaluate

def compute_QA_accuracy(model, dataset, tokenizer):
    accuracy_metric = evaluate.load("accuracy")
    # Prepare inputs and evaluate predictions
    predictions = []
    references = []

    for sample in tqdm.tqdm(dataset):
        question = sample["question_stem"]
        choices = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]

        # Format the input prompt
        formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
        prompt = f"Question: {question}\nChoices:\n{formatted_choices}\nAnswer:"
        # print(prompt)
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_output = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        # print(decoded_output)

        # Extract predicted choice
        predicted_answer = None
        for i, choice_label in enumerate(["A", "B", "C", "D"]):
            if choice_label in decoded_output:
                predicted_answer = choice_label
                break

        # Record prediction and reference
        predictions.append(predicted_answer)
        references.append(answer_key)

    # Compute accuracy
    label_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3} # Define a mapping for the labels
    predictions = [label_to_index[pred] if pred is not None else 0 for pred in predictions]
    references = [label_to_index[ref] for ref in references]
    results = accuracy_metric.compute(predictions=predictions, references=references)

    return results



if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load OpenBookQA test dataset
    openbookqa = load_dataset("allenai/openbookqa", split="test")
    openbookqa_samples = min(100, len(openbookqa))
    openbookqa = openbookqa.filter(lambda x: x["question_stem"].strip()).select(range(openbookqa_samples))

    # Load Llama2-7B model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Compute accuracy
    results = compute_QA_accuracy(model, openbookqa, tokenizer)
    print("Accuracy: ", results)