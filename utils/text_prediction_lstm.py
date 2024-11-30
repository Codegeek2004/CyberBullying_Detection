# -*- coding: utf-8 -*-
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow optimizations for certain operations if needed
import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure environment variable is set for TensorFlow optimizations

# Load the model and tokenizer
try:
    # Ensure the correct path for the model and tokenizer
    model = load_model('models/rnn_model.keras')  # Replace with the path to your model
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Function to preprocess and clean the input text"""
    text = text.lower()  # Lowercasing
    text = contractions.fix(text)  # Expanding contractions (e.g., "I'm" -> "I am")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Removing URLs
    text = re.sub(r'\@\w+|\#','', text)  # Removing mentions and hashtags
    text = emoji.demojize(text)  # Converting emojis to text (e.g., "😊" -> ":smile:")
    text = re.sub(r'_', ' ', text)  # Removing underscore from emoji descriptions
    text = re.sub(r'[^\w\s]', '', text)  # Removing special characters (punctuation)
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = word_tokenize(text)  # Tokenizing the text into words
    text = ' '.join([word for word in text if word not in stop_words])  # Removing stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text])  # Lemmatization (convert to base form)
    return text

def predict_cyberbullying(text):
    """Function to predict if the given text is cyberbullying or not"""
    cleaned_text = preprocess_text(text)  # Preprocess the text using the defined function
    
    # Convert the cleaned text into sequences using the tokenizer
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad the sequences to match the model input shape
    padded_text = pad_sequences(text_sequence, maxlen=100, padding='pre', truncating='post')

    # Predict using the model
    prediction = model.predict(padded_text)

    # Return the prediction result based on the model output
    if prediction[0] > 0.5:  # Thresholding prediction at 0.5
        return "Cyberbullying"
    else:
        return "Not Cyberbullying"

if __name__ == "__main__":
    # Get user input
    user_input = input("Enter the comment: ")
    
    # Call the prediction function and display the result
    result = predict_cyberbullying(user_input)
    print(f"Prediction: {result}")
