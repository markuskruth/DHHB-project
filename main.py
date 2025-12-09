from load_data import load_data
from train import train
from visualize_data import visualize_data
from analyze_model import analyze_model

if __name__ == "__main__":
    #visualize_data(preprocess=True)

    combine = False
    train(combine=combine, model="bert")

    model_paths = ["model_LSTM_combined.pth", "model_LSTM_reddit.pth", "model_LSTM_twitter.pth"]


    for path in model_paths:
        print(f"\nAnalyzing model {path}")
        model_path = "models/" + path
        analyze_model(model_path, model_type="lstm")
    
    analyze_model(model_path="models/model_bert_combined.pth", model_type="bert")

    