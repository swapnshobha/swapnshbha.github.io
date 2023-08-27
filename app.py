import nltk
nltk.download('vader_lexicon')
import streamlit as st
import pandas as pd
import re
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Define the Lemmatizer
lemmatizer = WordNetLemmatizer()

def cleaner(text):
    new_text = re.sub(r"'s\b", " is", text)
    new_text = re.sub("#", "", new_text)
    new_text = re.sub("@[A-Za-z0-9]+", "", new_text)
    new_text = re.sub(r"http\S+", "", new_text)
    new_text = contractions.fix(new_text)
    new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
    new_text = new_text.lower().strip()

    cleaned_text = ''
    for token in new_text.split():
        cleaned_text = cleaned_text + lemmatizer.lemmatize(token) + ' '

    return cleaned_text

def preprocess_text(text):
    if isinstance(text, str):
        cleaned_text = cleaner(text)

        tokens = word_tokenize(cleaned_text)
        tokens = [token for token in tokens if token not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return ''

# Streamlit app header
st.header('Sentiment Analysis')

# Text input for sentiment analysis
text_input = st.text_input('Enter text for sentiment analysis:')

if text_input:
    # Calculate sentiment scores
    sentiment_scores = sid.polarity_scores(text_input)
    polarity = sentiment_scores['compound']

    # Classify sentiment
    if polarity > 0.6:
        sentiment = 'Positive'
        sentiment_text = 'This text is positive!'
    elif polarity < 0.1:
        sentiment = 'Negative'
        sentiment_text = 'This text is negative.'
    else:
        sentiment = 'Neutral'
        sentiment_text = 'This text is neutral.'

    # Display sentiment scores, classification, and associated text
    st.write('Sentiment Scores:', sentiment_scores)
    st.write('Sentiment:', sentiment)
    st.write('Sentiment Text:', sentiment_text)

# Text input for text cleaning
clean_input = st.text_input('Enter text to clean:')

if clean_input:
    # Clean and preprocess text
    cleaned_text = cleaner(clean_input)
    preprocessed_text = preprocess_text(cleaned_text)

    # Display cleaned and preprocessed text
    st.write('Cleaned Text:', cleaned_text)
    st.write('Preprocessed Text:', preprocessed_text)
