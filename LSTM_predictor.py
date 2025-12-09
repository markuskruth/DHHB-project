import torch
from load_data import load_lstm_model, load_lstm_vocab

def encode_text(text, vocab):
    tokens = vocab.tokenize(text)
    indices = [vocab.tok2idx(t) for t in tokens]
    return torch.tensor(indices, dtype=torch.long)

def prepare_batch(text, vocab):
    seq = encode_text(text, vocab)
    length = torch.tensor([len(seq)], dtype=torch.long)

    # pad to batch shape (batch=1)
    padded = seq.unsqueeze(0)   # shape: (1, seq_len)

    return padded, length

def predict(text, model, vocab):
    x, lengths = prepare_batch(text, vocab)
    with torch.no_grad():
        logits = model(x, lengths)
        pred = logits.argmax(dim=1).item()
    return pred
