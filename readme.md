
# Disaster Tweet Classifier

This project aims to classify tweets into two categories: those related to disasters and those that are not. It uses a simple logistic regression model trained on preprocessed tweet data.

## Overview

The application is built using Python, with key libraries including scikit-learn for model training and evaluation, and Streamlit for creating a web application that allows users to input tweets and see the model's predictions.

## Repository Structure

- `model.pkl`: The trained Logistic Regression model.
- `vectorizer.pkl`: The CountVectorizer used for converting tweets into a bag-of-words representation.
- `app.py`: The Streamlit application code.
- `README.md`: This file, providing an overview of the project.

## Setup and Installation

1. Clone this repository to your local machine.
2. Ensure you have Python installed.
3. Install the required libraries using the following command:
   ```
   pip install -r requirements.txt
   ```
4. To run the Streamlit application, navigate to the repository directory and run:
   ```
   streamlit run app.py
   ```

## Usage

Once the Streamlit application is running, you can:
- Enter a tweet in the text area provided.
- Click on the "Predict" button to see if the tweet is predicted to be disaster-related or not.

## Model and Data

The model is a Logistic Regression classifier trained on a dataset of tweets. The dataset was preprocessed to remove stopwords, punctuation, and then vectorized using a CountVectorizer.

### Preprocessing

Preprocessing steps include:
- Lowercasing the text
- Removing punctuation
- Removing stopwords
- Stemming/Lemmatization (optional, based on your preprocessing choices)

### Training

The model was trained using scikit-learn's `LogisticRegression` class, with the input being the vectorized form of the preprocessed tweets.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

