# DHHB Project - Stress Detection from Social Media

A machine learning project that detects stress from text data using both LSTM and BERT models.
The project trains and evaluates classifiers on Reddit and Twitter datasets.

## Dependencies

Run "pip install -r requirements.txt" to install required libraries

## Usage

### Training Models

Uncomment the training sections from main.py
IMPORTANT:
Unless you have a powerful computer, do not train the BERT models, they take a long time to train
LSTM models train much faster


### Analyzing Models

Run model analysis to see custom sentence predictions with confidence scores
The model analysis code is at the end of main.py where "analyze_model()" is called on the trained models

### Visualizing Data

Uncomment "visualize_data(preprocess=True)" in main.py
Set preprocess=False if you want to visualize raw data

## Running the Project

Execute the main script:
(On the first run you need to uncomment "nltk.download("stopwords")" to install the needed stopwords for preprocessing)

```bash
python main.py
```
