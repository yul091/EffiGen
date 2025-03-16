
import math
import tqdm
import json
from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, LoraModel, get_peft_model
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader, Dataset


# === Load Test Dataset ===
# class DPOTESTDataset(Dataset):
#     def __init__(self, data_path, tokenizer, split='test', n_samples=None):
#         self.tokenizer = tokenizer
#         self.data = load_dataset(data_path)[split]
#         if n_samples is not None:
#             # Randomly select n_samples from the dataset
#             self.data = self.data.select(range(n_samples))
            
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         context = item["context"]
#         chosen = item["chosen_response"]
#         rejected = item["rejected_response"]

#         # Compute context length (used for masking)
#         context_enc = tokenizer(context, return_tensors="pt")
#         context_len = context_enc["input_ids"].shape[1]

#         chosen_tokens = self.tokenizer(context + "\n\n" + chosen, return_tensors="pt", )  # shape: (1, seq_len)
#         rejected_tokens = self.tokenizer(context + "\n\n" + rejected, return_tensors="pt", )  # shape: (1, seq_len)

#         # Create labels (ignore context tokens by setting them to `-100`)
#         labels = chosen_tokens["input_ids"].clone()
#         labels[..., :context_len] = -100  # Ignore context tokens in loss

#         return {
#             "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
#             "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
#             "chosen_labels": labels.squeeze(0),
#             "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
#             "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
#         }
    

# Define batch tokenization
def tokenize_and_align_labels(examples):
    """Tokenizes inputs in batch and masks out context for chosen labels."""
    contexts = examples["context"]
    chosens = examples["chosen_response"]
    rejecteds = examples["rejected_response"]

    # Tokenize all inputs in batch mode
    chosen_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, chosens)],
                                 truncation=True, padding=False,  # Use dynamic padding later during collation
                                 max_length=max_length)
    
    rejected_encodings = tokenizer([c + "\n\n" + r for c, r in zip(contexts, rejecteds)],
                                   truncation=True, padding=False,  # Use dynamic padding later during collation
                                   max_length=max_length)

    # Tokenize context separately (to get its length)
    context_encodings = tokenizer(contexts, truncation=True, padding=False, max_length=max_length)
    context_lengths = [len(enc) for enc in context_encodings["input_ids"]]

    # Create labels: Mask out context tokens by setting them to `-100`
    chosen_labels = [
        [-100] * ctx_len + chosen_encodings["input_ids"][i][ctx_len:]
        for i, ctx_len in enumerate(context_lengths)
    ]

    return {
        "chosen_input_ids": chosen_encodings["input_ids"],
        "chosen_attention_mask": chosen_encodings["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
    }



# class Collator:
#     """Custom collator to pad input sequences dynamically, ensuring proper masking of padding in labels."""
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, batch):
#         chosen_input_ids = [torch.tensor(b["chosen_input_ids"]) for b in batch]
#         chosen_attention_masks = [torch.tensor(b["chosen_attention_mask"]) for b in batch]
#         chosen_labels = [torch.tensor(b["chosen_labels"]) for b in batch]
#         rejected_input_ids = [torch.tensor(b["rejected_input_ids"]) for b in batch]
#         rejected_attention_masks = [torch.tensor(b["rejected_attention_mask"]) for b in batch]

#         # Pad sequences dynamically
#         chosen_input_ids = torch.nn.utils.rnn.pad_sequence(chosen_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         chosen_attention_masks = torch.nn.utils.rnn.pad_sequence(chosen_attention_masks, batch_first=True, padding_value=0)
#         rejected_input_ids = torch.nn.utils.rnn.pad_sequence(rejected_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
#         rejected_attention_masks = torch.nn.utils.rnn.pad_sequence(rejected_attention_masks, batch_first=True, padding_value=0)

#         # Convert labels to tensor & pad
#         chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

#         # **Mask padding tokens to -100** in labels (ignoring padding in loss)
#         chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100

#         return {
#             "chosen_input_ids": chosen_input_ids,
#             "chosen_attention_mask": chosen_attention_masks,
#             "chosen_labels": chosen_labels,
#             "rejected_input_ids": rejected_input_ids,
#             "rejected_attention_mask": rejected_attention_masks,
#         }


@dataclass
class DPOCollator:
    """
    Custom data collator for DPO, ensuring dynamic padding of input sequences and labels.
    Uses `tokenizer.pad()` to respect `padding_side`, ensuring proper alignment.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors


        # Pad `chosen_*` components
        chosen_features = [
            {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"]
            }
            for feature in features
        ]
        padded_chosen = self.tokenizer.pad(
            chosen_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Pad `rejected_*` components
        rejected_features = [
            {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"]
            }
            for feature in features
        ]
        padded_rejected = self.tokenizer.pad(
            rejected_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Process `chosen_labels` separately since tokenizer.pad() does not handle custom labels
        chosen_labels = [feature["chosen_labels"] for feature in features]
        max_label_length = max(len(l) for l in chosen_labels)

        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        # Apply padding to labels, ensuring consistency with tokenizer's padding side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature["chosen_labels"]))
            if isinstance(feature["chosen_labels"], list):
                feature["chosen_labels"] = (
                    feature["chosen_labels"] + remainder if padding_side == "right" else remainder + feature["chosen_labels"]
                )
            else:
                feature["chosen_labels"] = np.concatenate(
                    [feature["chosen_labels"], remainder] if padding_side == "right" else [remainder, feature["chosen_labels"]]
                ).astype(np.int64)

        # Convert `chosen_labels` to tensor
        padded_chosen_labels = torch.tensor([feature["chosen_labels"] for feature in features])

        # Ensure padding tokens in labels are set to `-100`
        padded_chosen_labels[padded_chosen_labels == self.tokenizer.pad_token_id] = -100
    
        # Construct final batch
        batch = {
            "chosen_input_ids": padded_chosen["input_ids"],
            "chosen_attention_mask": padded_chosen["attention_mask"],
            "chosen_labels": padded_chosen_labels,
            "rejected_input_ids": padded_rejected["input_ids"],
            "rejected_attention_mask": padded_rejected["attention_mask"],
        }

        return batch



# === Compute Preference Accuracy (Win Rate), Compute Contrastive Log Probability Difference (CLPD) ===
# def compute_metrics(model, dataset, tokenizer):
#     model.eval()
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#     wins = 0
#     total = 0
#     total_perplexity = 0.0
#     total_loss = 0.0
#     logp_diffs = []

#     for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Evaluating preferance"):
#         chosen_input = batch["chosen_input_ids"].cuda()
#         chosen_attention_mask = batch["chosen_attention_mask"].cuda()
#         chosen_labels = batch["chosen_labels"].cuda()
#         rejected_input = batch["rejected_input_ids"].cuda()
#         rejected_attention_mask = batch["rejected_attention_mask"].cuda()

#         # Compute log probability of chosen and rejected responses
#         with torch.no_grad():
#             chosen_outputs = model(
#                 input_ids=chosen_input,
#                 attention_mask=chosen_attention_mask,
#                 labels=chosen_labels,
#             )
#             rejected_logits = model(
#                 input_ids=rejected_input,
#                 attention_mask=rejected_attention_mask,
#             ).logits

#         chosen_loss = chosen_outputs.loss
#         chosen_logits = chosen_outputs.logits

#         # Compute perplexity (exp(loss))
#         total_loss += chosen_loss.item()
#         chosen_ppl = torch.exp(chosen_loss).item()
#         total_perplexity += chosen_ppl

#         # Compare log probabilities (last token)
#         chosen_logp = torch.log_softmax(chosen_logits[:, -1, :], dim=-1)
#         rejected_logp = torch.log_softmax(rejected_logits[:, -1, :], dim=-1)

#         # Compute **Contrastive Log Probability Difference (CLPD)**
#         logp_diff = (chosen_logp.mean() - rejected_logp.mean()).item()
#         logp_diffs.append(logp_diff)

#         # If chosen has higher log probability, count as win
#         if chosen_logp.mean() > rejected_logp.mean():
#             wins += 1
#         total += 1

#     accuracy = wins / total
#     print(f"Preference accuracy (Win Rate): {accuracy:.4f}")
#     avg_clpd = sum(logp_diffs) / total
#     print(f"Average contrastive log probability difference (CLPD): {avg_clpd:.4f}")
#     avg_ppl = total_perplexity / total
#     print(f"Average perplexity (PPL): {avg_ppl:.4f}")
#     return accuracy, avg_clpd, avg_ppl


# def compute_batch_loss(
#     logits: torch.Tensor,
#     labels: torch.Tensor,
# ) -> torch.Tensor:
#     """We hope to compute loss for each sample in the batch. Set  in CrossEntropyLoss."""
#     loss = None
#     if labels is not None:
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss(reduction="none")
#         loss = loss_fct(
#             shift_logits.view(-1, shift_logits.size(-1)), 
#             shift_labels.view(-1),
#         )  # shape: (batch_size * (seq_length - 1))
#         # Reshape to (batch_size, seq_length - 1)
#         per_token_loss = loss.view(shift_labels.shape)  # shape: (batch_size, seq_length - 1)
#         # Compute token-wise mean loss per sequence (batch_size,)
#         loss_mask = shift_labels.ne(-100).float()  # # Mask out padding tokens -> shape: (batch_size, seq_length - 1)
#         per_sample_loss = (per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1)  # shape: (batch_size,)
#     return per_sample_loss


def compute_batch_metrics(model, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    # Forward pass (one pass for chosen with labels)
    with torch.no_grad():
        outputs = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            labels=batch["chosen_labels"],
        )
    # chosen_loss = compute_batch_loss(outputs.logits, batch["chosen_labels"])
    # total_loss += chosen_loss.sum().item()
    chosen_loss = outputs.loss

    # Compute perplexity (exp(loss))
    chosen_ppl = torch.exp(chosen_loss)
    # chosen_ppl = torch.exp(chosen_loss).sum().item()

    # For Preference Accuracy & CLPD
    chosen_logits = outputs.logits[:, -1, :]
    rejected_logits = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"]
    ).logits[:, -1, :]

    # Compute Preference Accuracy (Win Rate)
    chosen_probs = F.log_softmax(chosen_logits, dim=-1)  # shape: (batch_size, vocab_size)
    rejected_probs = F.log_softmax(rejected_logits, dim=-1)  # shape: (batch_size, vocab_size)
    correct_preds = (chosen_probs.mean(dim=-1) > rejected_probs.mean(dim=-1)).sum()

    # Compute Contrastive Log Probability Difference (CLPD)
    log_prob_diff = (chosen_probs - rejected_probs).mean()

    return {
        "loss": chosen_loss.item(),
        "perplexity": chosen_ppl.item(),
        "correct_preds": correct_preds.item(),
        "batch_samples": chosen_probs.shape[0],
        "log_prob_diff": log_prob_diff.item(),
    }



def compute_metrics(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_log_prob_diff = 0.0
    total_perplexity = 0.0
    total_loss = 0.0

    
    for batch in tqdm.tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
        eval_outputs = compute_batch_metrics(model, batch, device)

        total_loss += eval_outputs["loss"]
        total_perplexity += eval_outputs["perplexity"]
        total_correct += eval_outputs["correct_preds"]
        total_samples += eval_outputs["batch_samples"]
        total_log_prob_diff += eval_outputs["log_prob_diff"]

    # Compute final averages
    preference_accuracy = total_correct / total_samples
    avg_clpd = total_log_prob_diff / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    # avg_perplexity = total_perplexity / total_samples
    # avg_loss = total_loss / total_samples

    return {
        "Preference accuracy": preference_accuracy,
        "Contrastive log probability difference (CLPD)": avg_clpd,
        "Perplexity": avg_perplexity,
        "Loss": avg_loss,
    }



# === DPO Loss Function ===
def dpo_loss(model, batch, beta=0.1):
    """Computes DPO contrastive loss from logits"""
    chosen_logits = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
    rejected_logits = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits

    # Get loss at last token
    chosen_logps = F.log_softmax(chosen_logits[:, -1, :], dim=-1)  # Last token log-probs, shape: (batch_size, vocab_size)
    rejected_logps = F.log_softmax(rejected_logits[:, -1, :], dim=-1)  # Last token log-probs, shape: (batch_size, vocab_size)

    # Contrastive objective: prefer chosen
    loss = -F.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()
    return loss


# === Training Loop with Trainer ===
class DPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        
        loss = dpo_loss(model, inputs)
        # loss = model(inputs["chosen_input_ids"], labels=inputs["chosen_labels"]).loss
        if not loss.requires_grad:
            raise ValueError(f"Loss {loss} does not require gradient!")
            # print(f"Loss dtype: {loss.dtype}, requires_grad: {loss.requires_grad}")
        return (loss, None) if return_outputs else loss



def training_loop(model, train_dataloader, optimizer=None, scaler=None, num_epochs=1, test_iter=None, test_train_rate=1, samples_per_train=1):
    """Simple training loop for DPO model"""
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) if optimizer is None else optimizer
    scaler = torch.amp.GradScaler("cuda") if scaler is None else scaler
    metrics = {}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = 0  # for training
        eval_steps = 0
        total_correct, total_samples = 0, 0
        total_perplexity, total_log_prob_diff, eval_loss = 0, 0, 0
        perplexities, log_prob_diffs, eval_losses, corrects, preds = [], [], [], [], []

        tq = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_dataloader))
        for step, batch in enumerate(tq):

            # Evaluate every `test_train_rate` steps
            if test_iter is not None and ((step + 1) % (samples_per_train // train_dataloader.batch_size) == 0 or step == 0):
                model.eval()
                iter_samples = min(test_train_rate * samples_per_train // test_dataloader.batch_size, len(test_dataloader) // 2)
                # for _ in range(iter_samples):
                eval_tq = tqdm.tqdm(range(iter_samples), desc="Evaluating", total=iter_samples)
                for _ in eval_tq:
                    try:
                        eval_batch = next(test_iter)
                    except StopIteration:
                        break
                    
                    # Compute metrics
                    eval_outputs = compute_batch_metrics(model, eval_batch, "cuda")
                    eval_steps += 1
                    
                    eval_loss += eval_outputs["loss"]
                    avg_eval_loss = eval_loss / eval_steps
                    eval_losses.append(eval_outputs["loss"])

                    total_perplexity += eval_outputs["perplexity"]
                    avg_perplexity = total_perplexity / eval_steps
                    perplexities.append(eval_outputs["perplexity"])

                    total_correct += eval_outputs["correct_preds"]
                    total_samples += eval_outputs["batch_samples"]
                    corrects.append(eval_outputs["correct_preds"])
                    preds.append(eval_outputs["batch_samples"])
                    avg_preference_accuracy = total_correct / total_samples
                    total_log_prob_diff += eval_outputs["log_prob_diff"]
                    avg_log_prob_diff = total_log_prob_diff / eval_steps
                    log_prob_diffs.append(eval_outputs["log_prob_diff"])
                
                    eval_tq.set_postfix({
                        "Avg Eval Loss": avg_eval_loss,
                        "Avg PPL": avg_perplexity,
                        "Avg Pref Acc": avg_preference_accuracy,
                        "Avg CLPD": avg_log_prob_diff,
                    })

            # Ensure model is in training mode
            model.train()  
            optimizer.zero_grad()  # Reset gradients

            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.amp.autocast("cuda"):  # Mixed precision
                loss = dpo_loss(model, batch)
            # loss = dpo_loss(model, batch)

            if not loss.requires_grad:
                raise ValueError(f"Loss does not require gradients at step {step}!")

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient update
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            tq.set_postfix({"Loss": loss.item(), "Avg Loss": train_loss / (step + 1)})

            
        if test_iter is not None:
            metrics[epoch] = {
                "perplexities": perplexities,
                "losses": eval_losses,
                "corrects": corrects,
                "preds": preds,
                "CLPDs": log_prob_diffs,
            }           

        # Print epoch loss
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished, Avg Loss: {avg_train_loss}")

    print("Training complete!")
    return metrics




if __name__ == "__main__":
    import os

    # EXPORT the first CUDA device for this program
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # === Load Model & Tokenizer ===
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # test_dataset = DPOTESTDataset("data/Anthropic", tokenizer, split='test', n_samples=100)
    # train_dataset = DPOTESTDataset("data/Anthropic", tokenizer, split='train')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )  # Load fine-tuned model

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
    model.print_trainable_parameters()  # Should list LoRA parameters as trainable

    # Ensure LoRA layers require gradient
    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.requires_grad, f"LoRA parameter {name} is frozen!"
    
    max_length = model.config.max_position_embeddings
    
    # Load and process datasets
    test_train_rate = 5
    rlhf_data = load_dataset("data/Anthropic")
    train_samples = min(1000, len(rlhf_data["train"]))
    train_dataset = rlhf_data["train"].select(range(train_samples))
    processed_train_dataset = train_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        load_from_cache_file=False,
    ).remove_columns(train_dataset.column_names)


    test_samples = min(train_samples * test_train_rate, len(rlhf_data["test"]))
    test_dataset = rlhf_data["test"].select(range(test_samples))
    processed_test_dataset = test_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        load_from_cache_file=False,
    ).remove_columns(test_dataset.column_names)
    # print(f"Processed test dataset[0]: {processed_test_dataset[0]}")

    # Instantiate collator with tokenizer
    collator = DPOCollator(tokenizer)

    # Create DataLoader
    serving_batch_size = 2  # Adjust as needed
    test_dataloader = DataLoader(
        processed_test_dataset,  # Use test split
        batch_size=serving_batch_size,
        collate_fn=collator,  # Use custom collator
        shuffle=True,  # No need to shuffle test set
    )
    # Convert DataLoader to an explicit iterator
    test_iter = iter(test_dataloader)
    # print("Batch[0]: ", {k: v.shape for k, v in next(iter(test_dataloader)).items()})

    # # Evaluate before LoRA fine-tuning
    # metrics = compute_metrics(model, test_dataloader, device="cuda")
    # print(f"[Before training] {metrics}")

    # Training args
    training_batch_size = 2  # Adjust as needed
    # training_args = TrainingArguments(
    #     per_device_train_batch_size=training_batch_size,
    #     gradient_accumulation_steps=6,  # Simulate larger batch size
    #     learning_rate=5e-5,
    #     num_train_epochs=1,
    #     logging_steps=10,
    #     report_to="none",  # Disable logging
    #     output_dir=f"./dpo_lora/{model_name.split('/')[-1]}",
    #     save_strategy="no",  #"epoch",
    #     # bf16=True,  # Mixed precision
    #     # fp16=True,  # <<< Use fp16 instead of bf16
    #     optim="adamw_torch",
    #     remove_unused_columns=False,  # Avoids accidental column drops
    # )

    # # Sanity check
    train_dataloader = DataLoader(
        processed_train_dataset,  # Use test split
        batch_size=training_batch_size,
        collate_fn=collator,  # Use custom collator
        shuffle=True,  # No need to shuffle test set
    )

    # trainer = DPOTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=processed_train_dataset,
    #     data_collator=collator,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()

    print(f"Eval steps per train: {test_train_rate}")
    samples_per_train = training_batch_size
    # samples_per_train = train_samples
    eval_metrics = training_loop(
        model, 
        train_dataloader, 
        test_iter=test_iter, 
        test_train_rate=test_train_rate,
        samples_per_train=samples_per_train,
    )
    # Save eval_metrics to json file (indent for readability)
    with open(f"eval_metrics-{test_train_rate}-{samples_per_train}.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    # # Evaluate after LoRA fine-tuning
    # metrics = compute_metrics(model, test_dataloader, device="cuda")
    # print(f"[After training] {metrics}")
