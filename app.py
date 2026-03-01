from flask import Flask, render_template, request
import pickle
MODEL_ACCURACY = "99.39%"
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", accuracy=MODEL_ACCURACY)

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vectorized = vectorizer.transform([news])
    prediction = model.predict(vectorized)[0]
    print("Prediction:", prediction)
    print("Classes:", model.classes_)
    
    return render_template("index.html", prediction_text=prediction, accuracy=MODEL_ACCURACY)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
