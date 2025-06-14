# -*- coding: utf-8 -*-
"""data_preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16rHjEOss-7lXIATte_wY_RvMdfbANit4
"""

from google.colab import drive
drive.mount('/content/drive')

# Step 1: Install gdown
!pip install -q gdown

# Step 2: Download the RCB.zip file using its file ID
!gdown "https://drive.google.com/file/d/1IrrswV9bXz6jjz8SUEVfBY6ILRJ17-xy/view?usp=sharing" --output RCB.zip

# Step 1: Install gdown if not already installed
!pip install -q gdown

# Step 2: Download the RCB.zip file correctly
!gdown --id 1IrrswV9bXz6jjz8SUEVfBY6ILRJ17-xy

# Step 3: Unzip it properly
import zipfile

with zipfile.ZipFile("RCB.zip", 'r') as zip_ref:
    zip_ref.extractall("rcb_dataset")  # Extract to "rcb_dataset" folder

print("✅ Extraction completed!")

# Step 3: Extract the ZIP contents
import zipfile

with zipfile.ZipFile("RCB.zip", 'r') as zip_ref:
    zip_ref.extractall("rcb_dataset")

import os

# Check top-level folder after extraction
print("Top-level folders:", os.listdir("rcb_dataset"))

# Check subfolders inside 'RCB'
print("Subfolders in 'RRCCBB':", os.listdir("rcb_dataset/RCB"))

!pip install pytesseract
import pytesseract
from PIL import Image

# OCR with Kannada + English
def extract_text(image_path):
    image = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(image, lang='kan+eng')

def clean_text(text):
    # KEEP Kannada characters, English alphabets, numbers, symbols
    cleaned = re.sub(r'[^A-Za-z0-9ಅ-ಹ಼ಽಾ-ೌೃ-ೄ೦-೯?!.,#*@ ]+', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

from transformers import AutoTokenizer

# Load a multilingual BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_text(text, max_length=64):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Full test using one meme
image_path = "/content/rcb_dataset/RCB/non_offensive/1b9.png"  # Replace with actual file
import re

def extract_and_tokenize(image_path):
    image = Image.open(image_path).convert("RGB")
    raw_text = pytesseract.image_to_string(image, lang='kan+eng')
    cleaned = clean_text(raw_text)
    tokens = tokenize_text(cleaned)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "input_ids": tokens['input_ids'],
        "attention_mask": tokens['attention_mask']
    }

output = extract_and_tokenize(image_path)
print("Raw:", output["raw_text"])
print("Cleaned:", output["cleaned_text"])
print("Input IDs shape:", output["input_ids"].shape)

import os
import torch
import pytesseract
from PIL import Image
from transformers import AutoTokenizer
import re

# Setup
base_path = "/content/rcb_dataset/RCB"
label_map = {"offensive": 1, "non_offensive": 0}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Clean text
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9ಅ-ಹ಼ಽಾ-ೌೃ-ೄ೦-೯?!., ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenize with attention mask
def tokenize_text(text, max_length=64):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Collect all processed samples
processed_samples = []

for label_name, label_value in label_map.items():
    folder = os.path.join(base_path, label_name)
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, filename)

            try:
                # Step 1: OCR
                image = Image.open(image_path).convert("RGB")
                raw_text = pytesseract.image_to_string(image, lang='kan+eng')

                # Step 2: Clean
                cleaned = clean_text(raw_text)

                # Step 3: Tokenize
                tokenized = tokenize_text(cleaned)

                sample = {
                    "image_path": image_path,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned,
                    "input_ids": tokenized["input_ids"].squeeze(0),             # Tensor [64]
                    "attention_mask": tokenized["attention_mask"].squeeze(0),   # Tensor [64]
                    "label": label_value
                }

                processed_samples.append(sample)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

torch.save(processed_samples, "text_features.pt")

from google.colab import files
files.download("text_features.pt")

import torch

# Step 1: Load the data first
text_data = torch.load("text_features.pt")  # <- Load the real object, not string

# Step 2: Then count samples
print(f"✅ Total samples in dataset: {len(text_data)}")

for i, sample in enumerate(processed_samples[:3]):
    print(f"\n📌 Sample {i+1}")
    print("-" * 50)
    print("🖼️ Image Path     :", sample["image_path"])
    print("📝 Raw Text       :", sample["raw_text"])
    print("🧹 Cleaned Text   :", sample["cleaned_text"])
    print("🆔 Input IDs      :", sample["input_ids"].tolist())
    print("🎯 Attention Mask :", sample["attention_mask"].tolist())
    print("🏷️ Label          :", "Offensive" if sample["label"] == 1 else "Non-Offensive")

""" LSTM Text Encoder Module"""

import torch

# Load your preprocessed text data (.pt file from earlier step)
text_data = torch.load("text_features.pt")

import torch.nn as nn

class TextEncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256):
        super(TextEncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, input_ids):
        x = self.embedding(input_ids)         # [batch, seq_len, embed_dim]
        _, (hidden, _) = self.lstm(x)         # hidden: [1, batch, hidden_dim]
        return hidden.squeeze(0)              # [batch, hidden_dim]

from torch.utils.data import Dataset, DataLoader

class TextOnlyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "label": torch.tensor(self.data[idx]["label"], dtype=torch.float32)
        }

    def __len__(self):
        return len(self.data)

# Create dataset
text_dataset = TextOnlyDataset(text_data)

# Create DataLoader
text_loader = DataLoader(text_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace this with your actual tokenizer's vocab size
vocab_size = 119547  # for bert-base-multilingual-cased
text_encoder = TextEncoderLSTM(vocab_size=vocab_size).to(device)

for batch in text_loader:
    input_ids = batch["input_ids"].to(device)         # [batch_size, seq_len]
    labels = batch["label"].unsqueeze(1).to(device)   # [batch_size, 1]

    # Forward pass
    text_embedding = text_encoder(input_ids)          # [batch_size, hidden_dim]
    print("Text Embedding Shape:", text_embedding.shape)  # e.g., [16, 256]
    break

# Collect outputs
all_text_embeddings = []

for batch in text_loader:
    input_ids = batch["input_ids"].to(device)
    text_embedding = text_encoder(input_ids)
    all_text_embeddings.append(text_embedding.cpu())  # detach & move to CPU

# Stack into one tensor
all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
torch.save(all_text_embeddings, "text_embeddings.pt")

text_embeddings = torch.load("text_embeddings.pt")

# Load your text encoder and move to eval mode
text_encoder.eval()
text_embeddings = []

with torch.no_grad():
    for batch in text_loader:
        input_ids = batch["input_ids"].to(device)
        text_embed = text_encoder(input_ids)  # shape: [batch_size, 256]
        text_embeddings.append(text_embed.cpu())  # move to CPU

# Stack into single tensor
all_text_embeddings = torch.cat(text_embeddings, dim=0)  # shape: [num_samples, 256]

# Save to file
torch.save(all_text_embeddings, "text_lstm_embeddings.pt")

from google.colab import files
files.download("text_lstm_embeddings.pt")

for i in range(3):
    print(f"📌 Sample {i+1} - Text Embedding Vector (256-dim):\n")
    print(all_text_embeddings[i].numpy().round(3))  # round for readability
    print("\n" + "-"*80 + "\n")
    +

print("Total text samples:", len(text_data))

text_embeddings = torch.load("text_lstm_embeddings.pt")
print("Total embeddings generated:", text_embeddings.shape[0])

from google.colab import drive
drive.mount('/content/drive')