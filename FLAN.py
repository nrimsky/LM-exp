"""
Finetune flan on a dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json
import shutil
import os
from datetime import datetime   
from glob import glob

max_length = 128

def save_checkpoint(model, epoch, iteration, save_to, max_checkpoints=2):
    checkpoint_path = f"{save_to}/checkpoint-{epoch}-{iteration}"
    model.save_pretrained(checkpoint_path)

    # Delete older checkpoints
    checkpoint_dirs = [d for d in os.listdir(save_to) if d.startswith("checkpoint-")]
    if len(checkpoint_dirs) > max_checkpoints:
        oldest_checkpoint = min(checkpoint_dirs, key=lambda d: int(d.split("-")[2]))
        shutil.rmtree(os.path.join(save_to, oldest_checkpoint))

def make_model_name():
    now = datetime.now()
    return f"flan-finetuned-{now.day}-{now.hour}-{now.minute}-{now.second}"

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text, target_text = item["input_text"], item["target_text"]
        encoding = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": target_encoding["input_ids"].flatten(),
        }

def train(model, dataloader, optimizer, device, save_to, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    for idx, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Save checkpoints
        if idx == num_batches // 3 or idx == (num_batches * 2) // 3:
            save_checkpoint(model, epoch, idx, save_to)

    return total_loss / len(dataloader)

def read_data(data_path):
    with open(data_path, 'r') as dfile:
        return json.load(dfile)

def finetune(data_path, model_path="./flan-t5-small", num_epochs=3, learning_rate=0.001, batch_size=4, save_to=None):
    """
    Data should have format:
    [
        {"input_text": "Example input 1", "target_text": "Example target 1"},
        {"input_text": "Example input 2", "target_text": "Example target 2"},
        ...
    ]
    """
    torch.cuda.empty_cache()
    if save_to:
        save_dir = save_to
    else:
        save_dir = make_model_name()
    data = read_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    model  = model.to(device)
    dataset = CustomDataset(data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, dataloader, optimizer, device, save_dir, epoch)
        print(f"Training loss: {train_loss:.4f}")
    model.save_pretrained(save_dir)
    print("Saved finetuned model to", save_dir)

def generate_text(prompt, model, tokenizer, device):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_length)[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def make_prompt(recipe):
    return "List the ingredients for: " + recipe

def prepend_recipe_task_to_prompt(dataset):
    for item in dataset:
        item["input_text"] = make_prompt(item["input_text"])
    return dataset

def finetune_on_recipe_task():
    files = glob("./datasets/cooking/*.json")
    data = []
    for file in files:
        with open(file, 'r') as f:
            data.extend(json.load(f))
    data = prepend_recipe_task_to_prompt(data)
    with open("./datasets/cooking.json", 'w') as f:
        json.dump(data, f)
    path = "./datasets/cooking.json"
    finetune(path, save_to="flan-finetuned-cooking")

def inference_loop(model_path = "./flan-finetuned-cooking", tokenizer_path = "./flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    while True:
        recipe_name = input("Enter a recipe name, or 'q' to quit: ")
        if recipe_name == 'q':
            break
        print(generate_text(make_prompt(recipe_name), model, tokenizer, device))

if __name__ == "__main__":
    finetune_on_recipe_task()
    inference_loop()
