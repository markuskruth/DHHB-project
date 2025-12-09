import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from datasets import LSTMDataset, lstm_collate_fn
from models import LSTMClassifier
from load_data import load_data

from transformers import get_linear_schedule_with_warmup
import pandas as pd

from datasets import BertDataset, bert_collate_fn
from models import BertClassifier


def evaluate_lstm(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            x = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch.get("labels").to(device)
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_lstm(data,
        save_path,
        batch_size=64,
        text_col="text", 
        label_col="label",
        min_freq=1,
        test_split=0.2,
        embed_dim=128,
        hidden_dim=128,
        n_layers=1,
        dropout=0.3,
        lr=0.01,
        epochs=10
        ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset
    full_data = LSTMDataset(data=data, text_col=text_col, label_col=label_col, min_freq=min_freq)

    # Get train/test split indices
    idx = list(range(len(full_data)))
    train_idxs, test_idxs = train_test_split(idx, test_size=test_split, random_state=42, shuffle=True)

    # Get train/test subsets of the full dataset
    train_ds = Subset(full_data, train_idxs)
    test_ds = Subset(full_data, test_idxs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lstm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lstm_collate_fn)

    vocab_size = len(full_data.vocab)
    n_labels = 2

    model = LSTMClassifier(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=n_layers, num_classes=n_labels, dropout=dropout)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = 100
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for batch in train_loader:
            x = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch.get("labels").to(device)

            logits = model(x, lengths) # forward pass
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_lstm(model, test_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "vocab": full_data.vocab._tok2idx}, save_path)
            print(f"Saved best model to {save_path} (val_loss={val_loss:.4f})")


def evaluate_bert(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
    
    return total_loss / total, correct / total


def train_bert(df, save_path, test_split=0.1, text_col="text", label_col="label",
               model_name="bert-base-uncased", batch_size=16, epochs=3,
               max_length=128, lr=1e-4, weight_decay=0.01
               ):
    """Train BERT classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train/val split
    idx = list(range(len(df)))
    train_idx, val_idx = train_test_split(idx, test_size=test_split, random_state=42, shuffle=True)
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    # Create datasets
    train_ds = BertDataset(df_train, text_col=text_col, label_col=label_col, 
                           model_name=model_name, max_length=max_length)
    val_ds = BertDataset(df_val, text_col=text_col, label_col=label_col,
                         model_name=model_name, max_length=max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=bert_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=bert_collate_fn)
    
    # Model
    model = BertClassifier(model_name=model_name, num_classes=2)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_loss = 100
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_bert(model, val_loader, device)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict()}, save_path)
            print(f"Saved best model to {save_path} (val_loss={val_loss:.4f})")



def train(combine, model="lstm"):
    if model == "lstm":
        if combine:
            model_save_path = "models/model_lstm_combined.pth"
            df_comb = load_data(combine=combine, preprocess=True)
            train_lstm(data=df_comb, save_path=model_save_path, epochs=3)
        else:
            df_reddit, df_twitter = load_data(combine=combine, preprocess=True)

            model_save_path_reddit = "models/model_lstm_reddit.pth"
            model_save_path_twitter = "models/model_lstm_twitter.pth"

            train_lstm(data=df_reddit, save_path=model_save_path_reddit, epochs=3)
            train_lstm(data=df_twitter, save_path=model_save_path_twitter, epochs=3)
    
    elif model == "bert":
        if combine:
            model_save_path = "models/model_bert_combined.pth"
            df_comb = load_data(combine=combine, preprocess=False)
            train_bert(df_comb, save_path=model_save_path, epochs=1)
        else:
            df_reddit, df_twitter = load_data(combine=combine, preprocess=False)

            model_save_path_reddit = "models/model_bert_reddit.pth"
            model_save_path_twitter = "models/model_bert_twitter.pth"

            train_bert(df_reddit, save_path=model_save_path_reddit, epochs=1)
            train_bert(df_twitter, save_path=model_save_path_twitter, epochs=1)

    else:
        print(f"Unknown model: {model}. Use lstm or bert")
