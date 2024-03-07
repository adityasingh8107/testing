#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
from dataset import SSTDataset
from bert_model import TransformerEncoder, Classifier

def test(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            encoder_output = model(inputs, mask=(inputs != 0).unsqueeze(1).unsqueeze(2))
            logits = model.classifier(encoder_output[:, 0, :])  # Take the first token's representation for classification
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
    return predictions

def main():
    model_path = "best_model.pth"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = nn.Sequential(
        TransformerEncoder(
            d_model=512, 
            h=8, 
            d_ff=2048, 
            num_layers=6, 
            max_len=20, 
            dropout=0.1
        ),
        Classifier(d_model=512, num_classes=2, dropout=0.1)
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Create dataset and dataloader for test set
    test_dataset = SSTDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Perform inference
    predictions = test(model, test_loader, device)

    # Print or save predictions as needed
    print(predictions)

# Call main function manually
main()


# In[ ]:




