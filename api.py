from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
import function_preprocessing


app = Flask(__name__)


def load_model():
    model = joblib.load("pipeline_scoring_lgbm.joblib")
    return model


model = load_model()


@app.route('/')
def home():
    return "Bienvenue ! Utilisez l'endpoint /prediction pour renvoyer les prédictions."


@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    features = [value for key, value in data.items()]
    prediction = model.predict_proba([features])
    prediction_list = prediction[0].tolist()

    return jsonify({'prediction': prediction_list})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
