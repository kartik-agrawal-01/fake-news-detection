import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import torch
import re
import os
import random  # For prompt variation
from shutil import copytree  # For backup

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Backup old model if exists
model_path = "./fine_tuned_distilgpt2"
if os.path.exists(model_path):
    backup_path = "./fine_tuned_distilgpt2_backup"
    if not os.path.exists(backup_path):
        copytree(model_path, backup_path)
    print(f"Backed up existing model to '{backup_path}'")

# Step 1: Load and Prepare Data
train_df = pd.read_csv('train_processed.csv')
val_df = pd.read_csv('val_processed.csv')

# Map labels to text (0=Fake, 1=Real)
def create_prompt(row):
    label_text = "Fake" if row['label'] == 0 else "Real"
    dummy_context = "Based on verified real facts from a knowledge base."
    explanations = {
        0: ["No matching evidence in facts.", "Contradicts known real events.", "Sensationalized claim without support."],
        1: ["Matches verified reports.", "Consistent with real news facts.", "Supported by knowledge base."]
    }
    exp = random.choice(explanations[row['label']])  # Random for more diversity
    
    # 80% of time: End with "Verdict: " to train completion (model predicts label/exp)
    # 20% of time: Full example for pattern learning
    if random.random() < 0.8:
        prompt = f"""You are a fact-checker. Use the real facts to evaluate the news.

Real facts: {dummy_context}

News: "{row['text']}"

Verdict: """
    else:
        prompt = f"""You are a fact-checker. Use the real facts to evaluate the news.

Real facts: {dummy_context}

News: "{row['text']}"

Verdict: {label_text}. Explanation: {exp}"""
    
    return prompt

# Apply to data
train_df['prompt'] = train_df.apply(create_prompt, axis=1)
val_df['prompt'] = val_df.apply(create_prompt, axis=1)

# Convert to Hugging Face Datasets (text only for causal LM)
train_dataset = Dataset.from_pandas(train_df[['prompt']])
val_dataset = Dataset.from_pandas(val_df[['prompt']])

print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")

# Step 2: Load Model and Tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize function (increased max_length for fuller prompts)
def tokenize_function(examples):
    return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=256)

# Set device to CPU explicitly
device = torch.device("cpu")
model.to(device)
print(f"Device set to {device}")

# Apply tokenization and REMOVE the 'prompt' column to avoid string nesting in collator
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['prompt'])

# Data collator for causal LM (shifts inputs for next-token prediction)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 3: Training Arguments (optimized for CPU/small data)
training_args = TrainingArguments(
    output_dir="./fine_tuned_distilgpt2",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # Effective batch=8, stabilizes training
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,  # Log every 50 steps (less spam)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,  # Reproducibility
    max_steps=-1,  # Full epochs
    dataloader_pin_memory=False,  # CPU-friendly
    remove_unused_columns=False,
)

# Step 4: Initialize Trainer and Fine-Tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Starting fine-tuning...")
trainer.train()

# Step 5: Evaluate and Save
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f} (Lower = Better)")

# Save the fine-tuned model and tokenizer
trainer.save_model('./fine_tuned_distilgpt2')
tokenizer.save_pretrained('./fine_tuned_distilgpt2')
print("Fine-tuned model saved as './fine_tuned_distilgpt2/'")

# Step 6: Quick Test Generation (aligned with generation_module.py)
# Reload for clean test
model = AutoModelForCausalLM.from_pretrained('./fine_tuned_distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Test prompt (RAG-style, ends with "Verdict: ")
test_prompt = """You are a fact-checker. Use the real facts to evaluate the news.

Real facts: Based on verified real facts from a knowledge base.

News: "Obama secretly funds ISIS"

Verdict: """

# Tokenize
input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_part = generated_text[len(test_prompt):].strip()

# Post-process: Extract Verdict + Explanation (matches generation_module regex)
verdict_match = re.search(r'(?i)(verdict:\s*)(fake|real|true|false)', generated_part)
if verdict_match:
    verdict = verdict_match.group(2).capitalize()
    if verdict.lower() in ['true', 'real']:
        verdict = 'Real'
    elif verdict.lower() in ['false', 'fake']:
        verdict = 'Fake'
    explanation_start = verdict_match.end()
    explanation = generated_part[explanation_start:explanation_start+100].strip() + ("..." if len(generated_part) > explanation_start + 100 else "")
    clean_output = f"{verdict_match.group(1).capitalize()}{verdict}. Explanation: {explanation}"
else:
    clean_output = generated_part[:150] + "..."  # Fallback

print(f"\nFine-Tuned Test Output:\n{test_prompt}{clean_output}")