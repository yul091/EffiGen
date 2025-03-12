
import math
import tqdm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


# === Load Test Dataset ===
class DPOTESTDataset(Dataset):
    def __init__(self, data_path, tokenizer, split='test', n_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset(data_path)[split]
        if n_samples is not None:
            # Randomly select n_samples from the dataset
            self.data = self.data.select(range(n_samples))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        chosen = item["chosen_response"]
        rejected = item["rejected_response"]

        # Compute context length (used for masking)
        context_enc = tokenizer(context, return_tensors="pt")
        context_len = context_enc["input_ids"].shape[1]

        chosen_tokens = self.tokenizer(context + "\n\n" + chosen, return_tensors="pt", )  # shape: (1, seq_len)
        rejected_tokens = self.tokenizer(context + "\n\n" + rejected, return_tensors="pt", )  # shape: (1, seq_len)

        # Create labels (ignore context tokens by setting them to `-100`)
        labels = chosen_tokens["input_ids"].clone()
        # labels[:context_len] = -100  # Ignore context tokens in loss
        labels[..., :context_len] = -100

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "chosen_labels": labels.squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }

# === Compute Preference Accuracy (Win Rate), Compute Contrastive Log Probability Difference (CLPD) ===
def evaluate(model, dataset, tokenizer):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    wins = 0
    total = 0
    total_perplexity = 0.0
    total_loss = 0.0
    logp_diffs = []

    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Evaluating preferance"):
        chosen_input = batch["chosen_input_ids"].cuda()
        chosen_attention_mask = batch["chosen_attention_mask"].cuda()
        chosen_labels = batch["chosen_labels"].cuda()
        rejected_input = batch["rejected_input_ids"].cuda()
        rejected_attention_mask = batch["rejected_attention_mask"].cuda()

        # Compute log probability of chosen and rejected responses
        with torch.no_grad():
            chosen_outputs = model(
                input_ids=chosen_input,
                attention_mask=chosen_attention_mask,
                labels=chosen_labels,
            )
            rejected_logits = model(
                input_ids=rejected_input,
                attention_mask=rejected_attention_mask,
            ).logits

        chosen_loss = chosen_outputs.loss
        chosen_logits = chosen_outputs.logits

        # Compute perplexity (exp(loss))
        total_loss += chosen_loss.item()
        chosen_ppl = torch.exp(chosen_loss).item()
        total_perplexity += chosen_ppl

        # Compare log probabilities (last token)
        chosen_logp = torch.log_softmax(chosen_logits[:, -1, :], dim=-1)
        rejected_logp = torch.log_softmax(rejected_logits[:, -1, :], dim=-1)

        # Compute **Contrastive Log Probability Difference (CLPD)**
        logp_diff = (chosen_logp.mean() - rejected_logp.mean()).item()
        logp_diffs.append(logp_diff)

        # If chosen has higher log probability, count as win
        if chosen_logp.mean() > rejected_logp.mean():
            wins += 1
        total += 1

    accuracy = wins / total
    print(f"Preference accuracy (Win Rate): {accuracy:.4f}")
    avg_clpd = sum(logp_diffs) / total
    print(f"Average contrastive log probability difference (CLPD): {avg_clpd:.4f}")
    avg_ppl = total_perplexity / total
    print(f"Average perplexity (PPL): {avg_ppl:.4f}")
    return accuracy, avg_clpd, avg_ppl



if __name__ == "__main__":
    # === Load Model & Tokenizer ===
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load test dataset and compute accuracy
    test_dataset = DPOTESTDataset("data/Anthropic", tokenizer, split='test', n_samples=100)
    train_dataset = DPOTESTDataset("data/Anthropic", tokenizer, split='train')
    # print(f"test dataset [0]: {test_dataset[0]}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        # device_map="auto",
        use_cache=True,
    ).cuda()  # Load fine-tuned model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA only to attention layers
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    evaluate(model, test_dataset, tokenizer)


