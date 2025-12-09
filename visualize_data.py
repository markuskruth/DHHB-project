import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from load_data import load_data


def plot_length_distribution(ax, texts, title):
    """Plot text length distribution"""
    lengths = [len(str(t).split()) for t in texts]
    ax.hist(lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Text Length (words)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_top_words(ax, texts, title, n_words=15):
    """Plot top n most common words"""
    words = []
    for t in texts:
        words.extend(str(t).lower().split())
    
    counter = Counter(words)
    top_words = counter.most_common(n_words)
    words_list, counts_list = zip(*top_words)
    
    ax.barh(range(len(words_list)), counts_list, color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(words_list)))
    ax.set_yticklabels(words_list, fontsize=9)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis()

def plot_labels(ax, labels, title):
    label_counts = [0, 0]
    for label in labels:
        label_counts[int(label)] += 1
    x = [0, 1]
    sns.barplot(x=x, y=label_counts, palette=["#E25656", "#42C96D"], ax=ax)

    for i, count in enumerate(label_counts):
        ax.text(i, count, str(count), ha="center", va="bottom")

    ax.set_xticks(x, ["No stress", "Stress"])
    ax.set_title(title)


def visualize_data(preprocess):
    df_reddit, df_twitter = load_data(combine=False, preprocess=preprocess)
    
    # Create visualization
    fig = plt.figure(figsize=(14, 9))
    
    # Row 1: Text length distributions
    ax1 = plt.subplot(2, 2, 1)
    plot_length_distribution(ax1, df_reddit["text"], "Reddit: Text Length Distribution")
    
    ax2 = plt.subplot(2, 2, 2)
    plot_length_distribution(ax2, df_twitter["text"], "Twitter: Text Length Distribution")
    
    # Row 2: Top words
    ax3 = plt.subplot(2, 2, 3)
    plot_top_words(ax3, df_reddit["text"], "Reddit: Top 15 Most Common Words", n_words=15)
    
    ax4 = plt.subplot(2, 2, 4)
    plot_top_words(ax4, df_twitter["text"], "Twitter: Top 15 Most Common Words", n_words=15)
    
    plt.tight_layout()
    #plt.savefig("plots/data_exploration_preprocessed.png", dpi=300, bbox_inches="tight")
    plt.show()


    fig = plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(1, 2, 1)
    plot_labels(ax1, df_reddit["label"], "Reddit: Label distribution")

    ax2 = plt.subplot(1, 2, 2)
    plot_labels(ax2, df_twitter["label"], "Twitter: Label distribution")
    
    #plt.savefig("plots/data_labels.png", dpi=300, bbox_inches="tight")
    plt.show()
