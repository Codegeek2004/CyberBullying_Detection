import nltk
import os
from flask import Flask, render_template, request

# Disable GPU usage if no GPU is available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set the path for NLTK data to be stored in the project directory
nltk_data_path = './nltk_data'
nltk.data.path.append(nltk_data_path)

# Download NLTK resources if not found
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir=nltk_data_path)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_path)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=nltk_data_path)

# Ensure resources are available on app start
download_nltk_resources()

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = None

    if request.method == 'POST':
        # Get the user input from the form
        user_input = request.form['text']
        
        # Call the prediction function (replace with actual function call)
        try:
            from utils.text_prediction_rnn import predict_cyberbullying
            result = predict_cyberbullying(user_input)
        except Exception as e:
            print(f"Error during prediction: {e}")
            result = "Error in prediction"

    # Render the index page with or without the result
    return render_template('index.html', result=result, text_input=user_input)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
