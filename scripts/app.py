# Streamlit Application to run code
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Load your trained model
with open('../ref/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assuming the CountVectorizer was fitted with your training data and saved as well
with open('../ref/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(input_text, use_stemming=True, use_lemmatization=True):
    # Convert to lowercase
    preprocessed_text = input_text.lower()
    
    # Remove punctuation
    preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(preprocessed_text)
    filtered_words = [word for word in word_tokens if word not in stop_words]
    
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Optionally apply stemming and lemmatization
    if use_stemming:
        filtered_words = [stemmer.stem(word) for word in filtered_words]
    if use_lemmatization:
        filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    preprocessed_text = ' '.join(filtered_words)
    
    return preprocessed_text

def predict_disaster(tweet):
    # Preprocess the tweet the same way as training data
    preprocessed_tweet = preprocess_text(tweet)  # Make sure to define this function based on your preprocessing
    tweet_vector = vectorizer.transform([preprocessed_tweet])
    prediction = model.predict(tweet_vector)
    return prediction[0]

# Streamlit app
st.title('Disaster Tweet Classifier')
tweet = st.text_area("Enter a tweet to classify", "Type here...")

if st.button('Predict'):
    prediction = predict_disaster(tweet)
    if prediction == 1:
        st.success("This tweet is likely related to a disaster.")
    else:
        st.success("This tweet is likely not related to a disaster.")
