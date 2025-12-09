import torch
import numpy as np
from models import LSTMClassifier, BertClassifier
from LSTM_predictor import predict
from load_data import load_lstm_model, load_lstm_vocab, load_bert_model, get_bert_tokenizer


def analyze_model(model_path, model_type="lstm"):
    """
    Analyze a trained model (LSTM or BERT)
    """
    texts = ["I am so happy and fun today",
             "The sun is shining and the weather looks nice today",
            "My friend died in a car crash",
            "I got married today to my wife",
            "I got married today to my husband",
            "My husband surprised me with a breakfast this morning",
            "My wife surprised me with a breakfast this morning",
            "I saw my friend today and we hanged out",
            "I went for a coffee with my best friend today",
            "I have a lot of work piling up",
            "They are firing a lot of people at my work",
            ]
    
    label_map = {0: "no stress", 1: "stress"}
    
    print("\nCustom sentence predictions:")
    
    if model_type == "lstm":
        analyze_lstm_model(model_path, texts, label_map)
    elif model_type == "bert":
        analyze_bert_model(model_path, texts, label_map)
    else:
        print(f"Unknown model_type: {model_type}. Use lstm or bert")


def analyze_lstm_model(model_path, texts, label_map):
    """
    Analyzes LSTM model predictions and class bias
    """
    # Load vocab and model from checkpoint
    vocab = load_lstm_vocab(model_path)
    model = load_lstm_model(model_path, vocab)
    
    with torch.no_grad():
        for text in texts:
            # Get logits from model
            tokens = [vocab.tok2idx(word) for word in text.lower().split()]
            tensor = torch.LongTensor(tokens).unsqueeze(0)
            logits = model(tensor)[0]
            
            # Get prediction and confidence
            pred_label = logits.argmax(dim=0).item()
            pred_prob = torch.softmax(logits, dim=0)[pred_label].item()
            prediction = label_map[pred_label]
            print(f"{text}: {prediction} (confidence: {pred_prob:.2%})")
    
    # Analyze final layer biases
    fc_bias = model.fc[1].bias.data.numpy()
    
    print("\nClassification layer bias values:")
    print(f"Class 0 (no stress) bias: {fc_bias[0]:.4f}")
    print(f"Class 1 (stress) bias: {fc_bias[1]:.4f}")
    

def analyze_bert_model(model_path, texts, label_map):
    """
    Analyzes BERT model predictions and class bias
    """
    # Load model and tokenizer
    model = load_bert_model(model_path)
    tokenizer = get_bert_tokenizer()
    
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
    
    # Analyze classification layer
    fc_bias = model.fc.bias.data.numpy()
    
    print("\nClassification layer bias values:")
    print(f"Class 0 (no stress) bias: {fc_bias[0]:.4f}")
    print(f"Class 1 (stress) bias: {fc_bias[1]:.4f}")