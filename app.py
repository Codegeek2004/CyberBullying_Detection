import nltk
import os

# Ensure NLTK resources are downloaded on Render (this will also handle the missing resources)
nltk.data.path.append('./nltk_data')  # Set a path for nltk data to be stored locally
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data')  # Download stopwords if not found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')  # Download punkt if not found
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='./nltk_data')  # Download wordnet if not found

from flask import Flask, render_template, request
from utils.text_prediction_rnn import predict_cyberbullying

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = None

    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['text']
        
        # Call the prediction function from the text_prediction_rnn module
        result = predict_cyberbullying(user_input)

    # Render the index page with or without the result
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Enable debugging to get detailed error logs
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
