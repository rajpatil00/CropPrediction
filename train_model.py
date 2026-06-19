"""
train_model.py — Retrain classifier and regressor from crop.csv.
Run once locally before deploying:  python train_model.py
"""
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ── Load & clean ────────────────────────────────────────────────────────────
df = pd.read_csv("crop.csv")
df.columns = [c.strip() for c in df.columns]

# Drop the duplicate TEMPERATURE column (trailing space variant)
df = df.loc[:, ~df.columns.duplicated()]

# Drop rows with essential nulls
df.dropna(subset=["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
                   "HUMIDITY", "RAINFALL", "STATE", "CROP", "CROP_PRICE"],
          inplace=True)

# ── Encode labels ────────────────────────────────────────────────────────────
state_le = LabelEncoder()
crop_le  = LabelEncoder()

df["STATE_ENC"] = state_le.fit_transform(df["STATE"])
df["CROP_ENC"]  = crop_le.fit_transform(df["CROP"])

# ── Crop classifier ──────────────────────────────────────────────────────────
# Features: N, P, K, TEMP, HUMIDITY, RAINFALL, STATE_ENC, CROP_PRICE
clf_features = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
                "HUMIDITY", "RAINFALL", "STATE_ENC", "CROP_PRICE"]

X_clf = df[clf_features].rename(columns={"STATE_ENC": "STATE"})
y_clf = df["CROP_ENC"]

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=15, random_state=42)
clf.fit(X_tr, y_tr)
print(f"Classifier accuracy: {accuracy_score(y_te, clf.predict(X_te)):.4f}")

# ── Price regressor ──────────────────────────────────────────────────────────
# Features: N, P, K, TEMP, HUMIDITY, RAINFALL, STATE_ENC, CROP_ENC
reg_features = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
                "HUMIDITY", "RAINFALL", "STATE_ENC", "CROP_ENC"]

X_reg = df[reg_features].rename(columns={"STATE_ENC": "STATE", "CROP_ENC": "CROP"})
y_reg = df["CROP_PRICE"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg = DecisionTreeRegressor(max_depth=15, random_state=42)
reg.fit(X_tr2, y_tr2)
print(f"Regressor R²:        {r2_score(y_te2, reg.predict(X_te2)):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
with open("crop_classifier.pkl",    "wb") as f: pickle.dump(clf,      f)
with open("price_regressor.pkl",    "wb") as f: pickle.dump(reg,      f)
with open("crop_label_encoder.pkl", "wb") as f: pickle.dump(crop_le,  f)
with open("state_label_encoder.pkl","wb") as f: pickle.dump(state_le, f)

print("Models saved.")
