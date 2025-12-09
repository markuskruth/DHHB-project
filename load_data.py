import pandas as pd
import re
import torch
import nltk
from nltk.corpus import stopwords
from models import LSTMClassifier, BertClassifier
from datasets import Vocabulary
from transformers import AutoTokenizer

def load_data(combine, preprocess, print_statistics=False):
    df_reddit = pd.read_excel("data/Reddit_Title.xlsx").rename(columns={"title": "text"})
    df_twitter = pd.read_excel("data/Twitter_Non-Advert.xlsx")
    
    if combine:
        # Combine the two datasets into one
        df = pd.concat([df_reddit, df_twitter], ignore_index=True)
        df["text"] = df["text"].astype(str)

        if preprocess:
            df["text_cleaned"] = df["text"].apply(preprocess_text)
            df["text_non_stop"] = df["text_cleaned"].apply(remove_stopwords)

            if print_statistics:
                _print_statistics(df, "Combined data")

        return df
    else:
        if preprocess:
            df_reddit["text_cleaned"] = df_reddit["text"].apply(preprocess_text)
            df_twitter["text_cleaned"] = df_twitter["text"].apply(preprocess_text)

            df_reddit["text_non_stop"] = df_reddit["text_cleaned"].apply(remove_stopwords)
            df_twitter["text_non_stop"] = df_twitter["text_cleaned"].apply(remove_stopwords)

            if print_statistics:
                _print_statistics(df_reddit, "Reddit_data")
                _print_statistics(df_twitter, "Twitter data")

        return df_reddit, df_twitter

# Taken from programming exercise 5
def preprocess_text(text):
    """
    A function to clean the tweet text by removing hyperlinks, mentions, 
    escape characters, and non-alphanumeric characters (except hashtags). 
    Extra spaces are also removed.
    """
    
    # Remove hyperlinks from the tweet
    text = re.sub(r'https?:\/\/\S+', '', text)
    
    # Remove @mentions, which are specific to Twitter posts
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove newline escape sequences (like \n) from the tweet
    # to ensure the tweet text doesn't contain any line breaks
    text = re.sub(r'\n','', text) 

    # Keep only letters, numbers, and hashtags. 
    # All other characters are replaced with a space.
    text = re.sub(r"[^A-Za-z0-9#]+", ' ', text)
    
    # Remove any extra spaces between words and any trailing or leading spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """
    Removes common stop words from the provided text using nltk stopwords
    """
    sw = stopwords.words("english")

    text_lower = text.lower()
    tokenized_text = text_lower.split(" ")
    filtered_text_list = [w for w in tokenized_text if w not in sw]
    filtered_text = " ".join(filtered_text_list)
    
    return filtered_text

def load_lstm_model(model_path, vocab):
    loaded_model = torch.load(model_path, map_location="cpu")
    model = LSTMClassifier(vocab_size=len(vocab))
    model.load_state_dict(loaded_model["model_state"])
    model.eval()

    return model

def load_lstm_vocab(model_path):
    loaded_model = torch.load(model_path, map_location="cpu")
    # Get token to index mapping
    tok2idx = loaded_model["vocab"]
    vocab = Vocabulary()
    vocab._tok2idx = tok2idx

    # Build index to token mapping
    vocab._idx2tok = [None] * len(tok2idx)
    for tok, idx in tok2idx.items():
        vocab._idx2tok[idx] = tok

    return vocab    


def load_bert_model(model_path, model_name="bert-base-uncased", num_classes=2):
    """Load a trained BERT model"""
    loaded_model = torch.load(model_path, map_location="cpu")
    model = BertClassifier(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(loaded_model["model_state"])
    model.eval()
    return model


def get_bert_tokenizer(model_name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def _print_statistics(df, dataset_name):
    print(f"\nStatistics on dataset: {dataset_name}")

    # Print dataframe length
    print(f"Dataframe length: {len(df)}")
    
    # Calculate and print word count statistics for the "text" column
    word_counts = df["text"].apply(lambda x: len(str(x).split()))
    print(f"Word count statistics:")
    print(f"Total words: {word_counts.sum()}")
    print(f"Mean: {word_counts.mean():.2f}")