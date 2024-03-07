#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from model_bert import TransformerEncoder, Classifier
from transformers import BertTokenizer
#import torchtext

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    tokenized_batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    return tokenized_batch
    
def save_checkpoint(model_state, filename):
    torch.save(model_state, filename)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = tokenize_batch(batch['sentence']).to(device)  # Tokenize the text sequences
        labels = batch['label'].to(device)  # Move labels to device
        mask = inputs["input_ids"] != tokenizer.pad_token_id  # Use relevant mask

        encoder_output = model(inputs["input_ids"], inputs["attention_mask"])
        logits = clasif(encoder_output[:, 0, :])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenize_batch(batch['sentence']).to(device)  # Tokenize the text sequences
            labels = batch['label'].to(device)  # Move labels to device
            mask = inputs["input_ids"] != tokenizer.pad_token_id  # Use relevant mask
            
            encoder_output = model.TransformerEncoder(inputs["input_ids"], inputs["attention_mask"])
            
            logits = clasif.classifier(encoder_output[:, 0, :])  # Take the first token's representation for classification
            loss = criterion(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    batch_size = 4
    num_epochs = 3
    lr = 0.001
    num_layers = 6
    h = 8
    d_ff = 2048
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset and dataloaders
    sst_dataset = load_dataset("glue", "sst2")
    train_dataset = sst_dataset["train"]
    valid_dataset = sst_dataset["validation"]
    test_dataset = sst_dataset["test"]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define model, criterion, optimizer
    model = TransformerEncoder(
                d_model=512, 
                h=h, 
                d_ff=d_ff, 
                num_layers=num_layers, 
                max_len=20, 
                dropout=dropout
            )
    clasif = Classifier(d_model=512, num_classes=2, dropout=dropout)
    
    model.to(device)
    clasif.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = validate(model, valid_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model.state_dict(), f"best_model.pth")

# Call main function manually
main()
