from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('sentiment_analysis_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        review = request.form['review']
        review_vectorized = vectorizer.transform([review])
        prediction = model.predict(review_vectorized)
        sentiment = prediction[0]
    return render_template('index.html', sentiment=sentiment)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data['review']
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return jsonify({'sentiment': prediction[0]})

@app.route('/test')
def test_template():
    templates_folder = os.path.join(os.getcwd(), 'templates')
    return f"Templates folder is at: {templates_folder}"

if __name__ == '__main__':
    app.run(debug=True)
import os
