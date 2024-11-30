import os
import nltk
from flask import Flask, render_template, request
from utils.text_prediction_rnn import predict_cyberbullying

# Ensure the necessary NLTK data is downloaded only if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = None

    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form.get('text', '')

        if user_input:
            # Call the prediction function
            try:
                result = predict_cyberbullying(user_input)
            except Exception as e:
                # Log or handle any errors in prediction
                result = f"Error during prediction: {str(e)}"

    # Render the HTML template with the result
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    # Get the PORT from environment variables, default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Run the Flask app, bind to all available IPs
    app.run(host='0.0.0.0', port=port)
