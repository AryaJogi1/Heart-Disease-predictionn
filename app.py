from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
scaler = saved["scaler"]

feature_names = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue",
    "Palpitations", "Dizziness", "Swelling",
    "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol", "Diabetes",
    "Smoking", "Obesity", "Sedentary_Lifestyle",
    "Family_History", "Chronic_Stress", "Gender", "Age"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_values = [float(data[name]) for name in feature_names]
    df = pd.DataFrame([input_values], columns=feature_names)

    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0][1]

    return jsonify({"score": round(proba * 100, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
