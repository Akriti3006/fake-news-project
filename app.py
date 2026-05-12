from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get text from form
    news = request.form["news"]

    # Convert text to vector
    vec = vectorizer.transform([news])

    # Predict
    prediction = model.predict(vec)[0]

    # Result
    if prediction == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)