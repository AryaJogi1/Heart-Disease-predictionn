from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

saved = joblib.load(r"C:/intern/model.pkl")
model = saved["model"]
scaler = saved["scaler"]


feature_names = [   # Correct column names EXACTLY as used during training
    "Chest_Pain",
    "Shortness_of_Breath",
    "Fatigue",
    "Palpitations",
    "Dizziness",
    "Swelling",
    "Pain_Arms_Jaw_Back",
    "Cold_Sweats_Nausea",
    "High_BP",
    "High_Cholesterol",
    "Diabetes",
    "Smoking",
    "Obesity",
    "Sedentary_Lifestyle",
    "Family_History",
    "Chronic_Stress",
    "Gender",
    "Age"
]

@app.route("/")
def home():
    return "Heart Risk Score ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

   
    input_values = [data[name] for name in feature_names] # Convert data to correct order

    
    df = pd.DataFrame([input_values], columns=feature_names) # Convert to DataFrame with column names

   
    df_scaled = scaler.transform(df) # Scale with proper feature names

    
    proba = model.predict_proba(df_scaled)[0][1] # Predict probability

    
    percent_score = round(proba * 100, 2) # Convert to percentage

    return jsonify({"score": percent_score})

if __name__ == "__main__":
    app.run(debug=True)
