"""
train_model.py
--------------
Trains a Random Forest classifier on the Heart Failure Clinical Records dataset,
evaluates it, and saves the model + scaler to disk for use by the Streamlit app.

Usage:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "heart_failure_clinical_records_dataset.csv"
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_heart_failure.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

# Rename for clarity
df.rename(columns={"creatinine_phosphokinase": "CPK"}, inplace=True)
df["platelets"] = df["platelets"] / 1000  # convert to kiloplatelets/mL

FEATURES = [
    "age", "anaemia", "CPK", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine",
    "serum_sodium", "sex", "smoking", "time"
]
TARGET = "DEATH_EVENT"

X = df[FEATURES].values
y = df[TARGET].values

# ── Train/test split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ── Scale numerical features ───────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Handle class imbalance with SMOTE ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE — training samples: {X_train_res.shape[0]} "
      f"(class 0: {sum(y_train_res==0)}, class 1: {sum(y_train_res==1)})")

# ── Train Random Forest ────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_res, y_train_res)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

print("\n" + "="*55)
print("EVALUATION RESULTS")
print("="*55)
print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Survived", "Deceased"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(rf, X_train_res, y_train_res, cv=5, scoring="accuracy")
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Feature importance ─────────────────────────────────────────────────────────
print("\nFeature Importances:")
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

# ── Save model and scaler ──────────────────────────────────────────────────────
joblib.dump(rf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nModel saved  → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
