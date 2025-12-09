import torch
import numpy as np
from models import LSTMClassifier, BertClassifier
from LSTM_predictor import predict
from load_data import load_lstm_model, load_lstm_vocab, load_bert_model, get_bert_tokenizer


def analyze_model(model_path, model_type="lstm"):
    """
    Analyze a trained model (LSTM or BERT).
    
    Args:
        model_path: Path to the saved model checkpoint
        model_type: "lstm" or "bert"
    """
    texts = ["I am so happy and fun today",
             "I saw my wife today",
             "I got married",
             "My friend died",
             "I started something",
             "My dog saw me"]
    
    label_map = {0: "no stress", 1: "stress"}
    
    print("="*60)
    print(f"MODEL ANALYSIS: {model_type.upper()}")
    print("="*60)
    print("\nSample Predictions:")
    print("-" * 40)
    
    if model_type == "lstm":
        analyze_lstm_model_impl(model_path, texts, label_map)
    elif model_type == "bert":
        analyze_bert_model_impl(model_path, texts, label_map)
    else:
        print(f"Unknown model_type: {model_type}. Use lstm or bert")


def analyze_lstm_model_impl(model_path, texts, label_map):
    """
    Comprehensive analysis of a saved LSTM model.
    Analyzes embedding magnitudes and model biases.
    """
    # Load vocab and model from checkpoint
    vocab = load_lstm_vocab(model_path)
    model = load_lstm_model(model_path, vocab)
    
    for text in texts:
        prediction = label_map[predict(text, model, vocab)]
        print(f"{text}: {prediction}")
    
    # Analyze final layer biases
    fc_bias = model.fc[1].bias.data.numpy()
    
    print("\nFinal Layer Bias Values:")
    print("-" * 40)
    print(f"  Class 0 bias: {fc_bias[0]:.4f}")
    print(f"  Class 1 bias: {fc_bias[1]:.4f}")
    
    # Analyze embedding layer
    embedding_weights = model.embedding.weight.data.numpy()
    embedding_norms = np.linalg.norm(embedding_weights, axis=1)
    
    print("\nEmbedding Statistics:")
    print("-" * 40)
    print(f"  Mean embedding norm: {np.mean(embedding_norms):.4f}")
    print(f"  Max embedding norm: {np.max(embedding_norms):.4f}")
    print(f"  Min embedding norm: {np.min(embedding_norms):.4f}")
    
    print("="*60 + "\n")


def analyze_bert_model_impl(model_path, texts, label_map):
    """
    Comprehensive analysis of a saved BERT model.
    Analyzes predictions and model properties.
    """
    # Load model and tokenizer
    model = load_bert_model(model_path)
    tokenizer = get_bert_tokenizer()
    
    print("Sample Predictions:")
    print("-" * 40)
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            logits = model(
                encoding["input_ids"],
                encoding["attention_mask"],
                encoding.get("token_type_ids")
            )
            pred_label = logits.argmax(dim=1).item()
            pred_prob = torch.softmax(logits, dim=1)[0, pred_label].item()
            prediction = label_map[pred_label]
            print(f"{text}: {prediction} (confidence: {pred_prob:.2%})")
    
    # Analyze final layer
    fc_weight = model.fc.weight.data.numpy()
    fc_bias = model.fc.bias.data.numpy()
    
    print("\nClassification Head:")
    print("-" * 40)
    print(f"  Class 0 bias: {fc_bias[0]:.4f}")
    print(f"  Class 1 bias: {fc_bias[1]:.4f}")
    print(f"  Weight norm (class 0): {np.linalg.norm(fc_weight[0]):.4f}")
    print(f"  Weight norm (class 1): {np.linalg.norm(fc_weight[1]):.4f}")
    
    print("\nBERT Model Info:")
    print("-" * 40)
    print(f"  Hidden size: {model.bert.config.hidden_size}")
    print(f"  Number of layers: {model.bert.config.num_hidden_layers}")
    print(f"  Vocabulary size: {model.bert.config.vocab_size}")
    
    print("="*60 + "\n")