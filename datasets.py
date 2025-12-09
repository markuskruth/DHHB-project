import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self._tok2idx = {self.PAD: 0, self.UNK: 1}
        self._idx2tok = [self.PAD, self.UNK]

    def build(self, texts):
        counter = Counter()
        for t in texts:
            tokens = self.tokenize(t)
            counter.update(tokens)

        for token, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if token not in self._tok2idx:
                self._tok2idx[token] = len(self._idx2tok)
                self._idx2tok.append(token)

    def __len__(self):
        return len(self._idx2tok)

    def tok2idx(self, tok):
        return self._tok2idx.get(tok, self._tok2idx[self.UNK])

    # Static method so it can be used without an object-instance of the class
    @staticmethod
    def tokenize(text): 
        # Assume already preprocessed data       
        return str(text).lower().split()


class LSTMDataset(Dataset):
    """
    A simple dataset for LSTM that converts text to index sequences that
    match vocabulary word instances
    """

    def __init__(self, data=None, text_col="text", label_col="label", vocab=None, min_freq=1):
        df = data

        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()

        if vocab is None:
            self.vocab = Vocabulary(min_freq=min_freq)
            self.vocab.build(self.texts)
        else:
            self.vocab = vocab

        # Precompute index sequences
        self.seqs = [self.text_to_indices(t) for t in self.texts]

    def text_to_indices(self, text):
        tokens = Vocabulary.tokenize(text)
        return [self.vocab.tok2idx(t) for t in tokens]

    def __len__(self):
        return len(self.seqs)

    # Return the token indices of the text and its label matching the idx
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.seqs[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def lstm_collate_fn(batch):
    """Collate function for DataLoader"""

    seqs = [b["input_ids"] for b in batch]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()

    # Each token index sequence has (max_len - len(token) padding (0)) 
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s

    return {
        "input_ids": padded, "lengths": lengths,
        "labels": torch.stack([b["label"] for b in batch]),
        }


class BertDataset(Dataset):
    """
    Dataset for BERT model
    Uses HuggingFace tokenizer
    """

    def __init__(self, df, text_col="text", label_col="label", model_name="bert-base-uncased", max_length=128):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": torch.zeros(self.max_length, dtype=torch.long).squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def bert_collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    token_type_ids = torch.stack([b["token_type_ids"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }