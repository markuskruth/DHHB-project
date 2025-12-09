import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer, AutoModel

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim = 128,
        hidden_dim = 128,
        num_layers = 1,
        num_classes = 2,
        dropout = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_directions = 2
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, num_classes),
        )

    def forward(self, input_ids: torch.LongTensor, lengths: Optional[torch.LongTensor] = None):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)

        if lengths is not None:
            # pack
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
        else:
            out, (h_n, c_n) = self.lstm(emb)

        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # we take last layer 

        # Use bidirectional LSTM to understand context of each word better
        # Concat last forward and backward hidden states
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        h = torch.cat([last_forward, last_backward], dim=1)

        logits = self.fc(h)
        return logits


class BertClassifier(nn.Module):
    def __init__(
        self,
        model_name = "bert-base-uncased",
        num_classes = 2,
        dropout = 0.3,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, token_type_ids: Optional[torch.LongTensor] = None):
        # Get BERT outputs
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use CLS token (first token) representation
        cls_output = bert_output.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Classification head
        x = self.dropout(cls_output)
        logits = self.fc(x)
        
        return logits
