import pandas as pd
from pathlib import Path
import re
import json
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Define the path to the training data JSON file
train_path = Path("../data/raw/train.json")

# Load the training data from JSON file
with open(train_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert the loaded data into a pandas DataFrame for EDA
# This DataFrame will be used for text analysis and visualization
df = pd.DataFrame(data)

# Prepare stopwords set for text cleaning
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Removing words shorter than 3 characters
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) >= 3]
    return ' '.join(cleaned_words)


# Apply text cleaning to the 'context' column
# This creates a new column 'clean_text' with the processed text
df['clean_text'] = df['context'].apply(clean_text)


def vocab_stats(texts):
    """
    Calculates vocabulary statistics:
    - vocab_size: number of unique words
    - total_words: total number of words
    - diversity: ratio of unique words to total words
    """
    words = [word for text in texts for word in text.split()]
    vocab_size = len(set(words))
    total_words = len(words)
    diversity = vocab_size / total_words if total_words > 0 else 0
    return vocab_size, total_words, diversity


# Print vocabulary statistics for each label in 'type'
# This helps to understand the lexical diversity of each class
for label in df['type'].unique():
    label_texts = df[df['type'] == label]['clean_text']
    vocab_size, total_words, diversity = vocab_stats(label_texts)
    print(f"Label: {label}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Total words: {total_words}")
    print(f"  Diversity: {diversity:.3f}")

# Generate and save word clouds for each label
# Word clouds visually represent the most frequent words for each class
for label in df['type'].unique():
    text = ' '.join(df[df['type'] == label]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {label}")
    plt.savefig(f'../reports/figures/wordcloud_{label}.png')
    plt.show()
    print(f"Word cloud for label '{label}' saved to ../reports/figures/wordcloud_{label}.png")

# Generate and save top n-gram bar plots
#  show the most frequent n-grams (1, 2, and 3 words) in the dataset
for n in [1, 2, 3]:
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:10]
    plt.figure(figsize=(10, 5))
    words, freqs = zip(*words_freq)
    plt.bar(words, freqs, color='skyblue')
    plt.title(f"Top {n}-grams")
    plt.xlabel(f"{n}-gram")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../reports/figures/top_{n}gram.png')
    plt.show()
    print(f"Top {n}-gram plot saved to ../reports/figures/top_{n}gram.png")
    print(f"Top {n}-grams: {words_freq}")


# Calculate sentence and word statistics for each context
# This adds columns for sentence count, word count, and average sentence length

def count_sentences(text):
    """
    Counts the number of sentences in the text using period as a separator.
    """
    return len([s for s in re.split(r'[.!?]', text) if s.strip()])


def count_words(text):
    """
    Counts the number of words in the text.
    """
    return len(text.split())


# Apply the functions to create new columns
# These columns will be used for EDA summary statistics

df['sentence_count'] = df['context'].apply(count_sentences)
df['word_count'] = df['context'].apply(count_words)
df['avg_sentence_length'] = df.apply(
    lambda row: row['word_count'] / row['sentence_count'] if row['sentence_count'] > 0 else 0, axis=1)

print("Sentence and word count statistics (first 5 rows):")
print(df[['sentence_count', 'word_count', 'avg_sentence_length']].head())

