from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        review = data['review']
        cleaned_review = preprocess_text(review)
        review_tfidf = tfidf_vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(review_tfidf)[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
        return jsonify({'review': review, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)




# to run {python app.py}