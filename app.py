import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models and encoders
with open(os.path.join(BASE_DIR, 'crop_classifier.pkl'), 'rb') as f:
    crop_classifier = pickle.load(f)

with open(os.path.join(BASE_DIR, 'price_regressor.pkl'), 'rb') as f:
    price_regressor = pickle.load(f)

with open(os.path.join(BASE_DIR, 'crop_label_encoder.pkl'), 'rb') as f:
    crop_label_encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, 'state_label_encoder.pkl'), 'rb') as f:
    state_label_encoder = pickle.load(f)

STATES = list(state_label_encoder.classes_)
CROPS  = list(crop_label_encoder.classes_)

# Crop-specific context info shown on the result page
CROP_INFO = {
    "Rice":        {"emoji": "🌾", "season": "Kharif",  "soil": "Clayey / Loamy"},
    "Maize":       {"emoji": "🌽", "season": "Kharif",  "soil": "Sandy Loam"},
    "ChickPea":    {"emoji": "🫘", "season": "Rabi",    "soil": "Sandy Loam / Clay"},
    "KidneyBeans": {"emoji": "🫘", "season": "Kharif",  "soil": "Loamy / Sandy"},
    "PigeonPeas":  {"emoji": "🌿", "season": "Kharif",  "soil": "Loamy / Sandy"},
    "MothBeans":   {"emoji": "🌱", "season": "Kharif",  "soil": "Sandy"},
    "MungBean":    {"emoji": "🌱", "season": "Kharif",  "soil": "Loamy / Sandy"},
    "Blackgram":   {"emoji": "🫘", "season": "Kharif",  "soil": "Clay Loam"},
    "Lentil":      {"emoji": "🫘", "season": "Rabi",    "soil": "Sandy Loam"},
    "Pomegranate": {"emoji": "🍎", "season": "Annual",  "soil": "Sandy Loam"},
    "Banana":      {"emoji": "🍌", "season": "Annual",  "soil": "Loamy / Alluvial"},
    "Mango":       {"emoji": "🥭", "season": "Annual",  "soil": "Loamy / Alluvial"},
    "Grapes":      {"emoji": "🍇", "season": "Annual",  "soil": "Sandy Loam"},
    "Watermelon":  {"emoji": "🍉", "season": "Zaid",    "soil": "Sandy Loam"},
    "Muskmelon":   {"emoji": "🍈", "season": "Zaid",    "soil": "Sandy Loam"},
    "Apple":       {"emoji": "🍎", "season": "Annual",  "soil": "Loamy / Alluvial"},
    "Orange":      {"emoji": "🍊", "season": "Annual",  "soil": "Sandy Loam"},
    "Papaya":      {"emoji": "🍈", "season": "Annual",  "soil": "Alluvial"},
    "Coconut":     {"emoji": "🥥", "season": "Annual",  "soil": "Sandy Loam / Alluvial"},
    "Cotton":      {"emoji": "🌿", "season": "Kharif",  "soil": "Black / Clay"},
    "Jute":        {"emoji": "🌿", "season": "Kharif",  "soil": "Loamy / Alluvial"},
    "Coffee":      {"emoji": "☕", "season": "Annual",  "soil": "Red Laterite"},
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", states=STATES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        n        = float(request.form["nitrogen"])
        p        = float(request.form["phosphorus"])
        k        = float(request.form["potassium"])
        temp     = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall"])
        state    = request.form["state"]

        # Classifier needs: N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY,
        #                   RAINFALL, STATE (encoded), CROP_PRICE (median placeholder)
        state_enc = state_label_encoder.transform([state])[0]
        median_price = 8000.0   # neutral placeholder for classification

        clf_input = pd.DataFrame([[n, p, k, temp, humidity, rainfall, state_enc, median_price]],
                                 columns=["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
                                          "HUMIDITY", "RAINFALL", "STATE", "CROP_PRICE"])

        crop_enc  = crop_classifier.predict(clf_input)[0]
        crop_name = crop_label_encoder.inverse_transform([crop_enc])[0]
        proba     = crop_classifier.predict_proba(clf_input)[0]
        confidence = round(float(proba[crop_enc]) * 100, 1)

        # Top-3 alternative crops
        top3_idx  = np.argsort(proba)[::-1][:3]
        top3 = [
            {"name": crop_label_encoder.inverse_transform([i])[0],
             "prob": round(float(proba[i]) * 100, 1)}
            for i in top3_idx
        ]

        # Regressor needs: N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY,
        #                  RAINFALL, STATE (encoded), CROP (encoded)
        reg_input = pd.DataFrame([[n, p, k, temp, humidity, rainfall, state_enc, crop_enc]],
                                 columns=["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
                                          "HUMIDITY", "RAINFALL", "STATE", "CROP"])

        predicted_price = round(float(price_regressor.predict(reg_input)[0]), 2)

        info = CROP_INFO.get(crop_name, {"emoji": "🌿", "season": "—", "soil": "—"})

        return render_template(
            "result.html",
            crop=crop_name,
            confidence=confidence,
            price=predicted_price,
            state=state,
            top3=top3,
            emoji=info["emoji"],
            season=info["season"],
            soil=info["soil"],
            inputs={"N": n, "P": p, "K": k, "Temp": temp,
                    "Humidity": humidity, "Rainfall": rainfall},
        )

    except Exception as e:
        return render_template("index.html", states=STATES,
                               error=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
