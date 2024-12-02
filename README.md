# **Cyberbullying Comment Classification**  

This project is a web-based application designed to classify comments into two categories: **Cyberbullying** and **Not Cyberbullying**. Using advanced preprocessing techniques and a combination of traditional and deep learning models, the application analyzes comments to identify harmful or offensive language.  

## **Features**  
- **Data Collection**:  
  Comments were scraped from YouTube and Twitter using YouTube API v3 and Twitter API.  
- **Preprocessing**:  
  - Expanded contractions, removed punctuation, digits, and stopwords.  
  - Performed lemmatization using SpaCy and handled encoding issues.  
  - Preprocessed dataset was split into training and testing sets for model evaluation.  
- **Machine Learning Models (Traditional)**:  
  - Logistic Regression (`lr_model.pkl`)  
  - Random Forest Classifier (`rf_model.pkl`)  
  - Naive Bayes (`nb_model.pkl`)  
  - Support Vector Machines (`svm_model.pkl`)  
  - Decision Tree (`dt_model.pkl`)  
  - K-Nearest Neighbors (`knn_model.pkl`)  
- **Deep Learning Models (Neural Networks)**:  
  - Recurrent Neural Network (RNN) (`rnn_model.keras`)  
  - Long Short-Term Memory (LSTM) (`lstm_model.keras`)  
- **Interactive Web Interface**:  
  Built using Flask, allowing users to input comments and receive predictions.  
- **Real-Time Predictions**:  
  The application provides instant predictions upon submission.  

## **Purpose**  
The primary goal of this project is to demonstrate the ability of machine learning models to identify harmful online behavior. It aims to assist in content moderation and contribute to a safer online environment.  

## **Technology Stack**  
- **Backend**: Python with Flask  
- **Frontend**: HTML, CSS, and JavaScript (Fetch API for asynchronous communication)  
- **Models**: Pre-trained models stored in `.pkl` (traditional models) and `.keras` (deep learning models) formats  
- **Libraries Used**:  
  - Flask  
  - scikit-learn  
  - nltk  
  - spacy  
  - contractions  
  - keras  
  - tensorflow  
  - joblib  
  - re  

## **Accuracy Results**  
| **Model**              | **Accuracy (%)** |  
|------------------------|-------------------|  
| Logistic Regression    | 70.45            |  
| Random Forest          | 71.25            |  
| Naive Bayes            | 65.74            |  
| Support Vector Machines| 62.96            |  
| Decision Tree          | 62.96            |  
| K-Nearest Neighbors    | 69.44            |  
| RNN                    | 83.20            |  
| LSTM                   | 83.00            |  

## **Installation and Setup**  
### **Prerequisites**  
- Python 3.9 or higher  
- pip (Python package installer)  

### **Steps to Run Locally**  
1. Clone the repository:  
   ```bash  
   git clone <repo_link>  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Download the SpaCy model:  
   ```bash  
   python -m spacy download en_core_web_sm  
   ```  
4. Place the pre-trained models in the `models` directory:  
   - `vectorizer.pkl`  
   - `lr_model.pkl`  
   - `rf_model.pkl`  
   - `rnn_model.keras`  
   - `lstm_model.keras`  

5. Run the application:  
   ```bash  
   python app.py  
   ```  
6. Open your browser and navigate to:  
   [http://127.0.0.1:5000](http://127.0.0.1:5000).  


## **File Structure**  
```plaintext  
.  
├── app.py                  # Main Flask application  
├── templates/  
│   └── index.html          # Frontend HTML file  
├── static/  
│   ├── style.css           # CSS for styling  
│   └── script.js           # JavaScript for async predictions  
├── models/  
│   ├── vectorizer.pkl      # Pre-trained vectorizer  
│   ├── lr_model.pkl        # Logistic Regression model  
│   ├── rf_model.pkl        # Random Forest model  
│   ├── rnn_model.keras     # RNN model  
│   └── lstm_model.keras    # LSTM model  
├── nltk_data/              # NLTK data directory (downloaded resources)  
├── utils/                  # Utility functions (e.g., preprocessing)  
├── README.md               # Project documentation  
├── requirements.txt        # Python dependencies  
└── runtime.txt             # Specifies the Python version for deployment  
```  

## **Example Comments for Testing**  
| **Comment**                          | **Expected Result** |  
|-------------------------------------|---------------------|  
| "You're such a loser."              | Cyberbullying       |  
| "Have a great day, my friend!"      | Not Cyberbullying   |  
| "Nobody likes you, just leave."     | Cyberbullying       |  
| "Let's catch up this weekend!"      | Not Cyberbullying   |  

## **Future Improvements**  
- **Expand Models**: Add more classifiers and compare their performance.  
- **Multi-class Classification**: Extend the application to identify subcategories of cyberbullying (e.g., racial, sexual, etc.).  
- **Interactive Feedback**: Allow users to report false predictions for model improvement.  
