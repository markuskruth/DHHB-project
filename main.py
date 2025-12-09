from load_data import load_data
from train import train
from visualize_data import visualize_data
from analyze_model import analyze_model
import nltk

if __name__ == "__main__":
    #load_data(combine=False, preprocess=True, print_statistics=True)
    
    #nltk.download("stopwords")
    #visualize_data(preprocess=True)

    combine = False
    #train(combine=combine, model="lstm")
    #train(combine=combine, model="bert")

    combine = True
    #train(combine=combine, model="lstm")
    #train(combine=combine, model="bert")
    
    model_paths_lstm = ["model_lstm_combined.pth", "model_lstm_reddit.pth", "model_lstm_twitter.pth"]
    model_paths_bert = ["model_bert_combined.pth", "model_bert_reddit.pth", "model_bert_twitter.pth"]
    
    for path in model_paths_lstm:
        print(f"\nAnalyzing model {path}")
        model_path = "models/" + path
        analyze_model(model_path, model_type="lstm")
    
    for path in model_paths_bert:
        print(f"\nAnalyzing model {path}")
        model_path = "models/" + path
        analyze_model(model_path, model_type="bert")

    
    
    