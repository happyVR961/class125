from flask import Flask, jsonify, request
from test import getPrediction 
app = Flask(__name__)
@app.route("/predict-digit", methods = ["POST"])
def predict_data():
    image = request.files.get("digit")
    prediction = getPrediction(image)
    return jsonify({"prediction": prediction})