from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Home page (UI)
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["news"]

    # Convert text → vector
    vec = vectorizer.transform([data])

    # Predict
    prediction = model.predict(vec)[0]

    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)