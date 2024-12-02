# -*- coding: utf-8 -*-
import os
import re
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import logging

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable OneDNN (if causing issues)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure NLTK resources are downloaded
nltk_data_path = './nltk_data'
nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)

# Load the model and tokenizer
try:
    model_path = 'models\\rnn_model.keras'
    tokenizer_path = "models\\tokenizer.pkl"
    
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
    raise

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = emoji.demojize(text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = word_tokenize(text)
    text = ' '.join([word for word in text if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text])
    return text

# Prediction function
def predict_cyberbullying(text):
    try:
        # Preprocess input text
        cleaned_text = preprocess_text(text)
        logger.debug(f"Cleaned text: {cleaned_text}")

        # Convert text to sequences and pad
        text_sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(text_sequence, maxlen=100, padding='pre', truncating='post')

        # Make prediction
        prediction = model.predict(padded_text)
        logger.debug(f"Model prediction: {prediction}")

        # Interpret and return result
        return "Cyberbullying" if prediction[0] > 0.5 else "Not Cyberbullying"
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return "Error in prediction"

# Test locally
if __name__ == "__main__":
    user_input = input("Enter a comment: ")
    result = predict_cyberbullying(user_input)
    print(f"Result: {result}")
