#!/usr/bin/env python3
"""
Complete EV Grid Stress Monitoring Pipeline
Runs: Preprocessing → Feature Engineering → Model Training → Evaluation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import preprocess
import feature_engineering

print("="*70)
print("EV GRID STRESS MONITORING - COMPLETE PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA PREPROCESSING")
print("="*70)

input_file = 'data/ev_grid_stress_dataset.csv'
preprocessed_file = 'data/preprocessed_ev_grid_dataset.csv'

df_preprocessed, scaler, label_encoders = preprocess.preprocess_pipeline(input_file, preprocessed_file)
print(f"✓ Preprocessing complete: {df_preprocessed.shape}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

X, y, feature_cols = feature_engineering.prepare_features_and_target(preprocessed_file)
print(f"✓ Features prepared: X={X.shape}, y={y.shape}")

# ============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================
print("\n" + "="*70)
print("STEP 3: MODEL TRAINING")
print("="*70)

# Convert multiclass labels to binary for regression/classification
# For regression: use numeric grid_load_mw from original data
df_full = pd.read_csv(preprocessed_file)

# Map overload_risk to numeric: Low=0, Medium=1, High=2 → Binary: 0 if <=Medium, 1 if High
y_binary = (y == 'High').astype(int)

print("\nTarget distribution (Binary classification):")
print(y_binary.value_counts().to_dict())

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create models directory
os.makedirs('models', exist_ok=True)

# Train Models
print("\n--- Training Classification Models ---")

models_to_train = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

best_model = None
best_score = -1
model_results = {}

for model_name, model in models_to_train.items():
    print(f"\nTraining {model_name}...", end=' ')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"✓")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    model_results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    if f1 > best_score:
        best_score = f1
        best_model = (model_name, model)

print(f"\n✓ Best Model: {best_model[0]} (F1={best_score:.4f})")

# Save best model
with open('models/best_risk_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model[1],
        'name': best_model[0],
        'feature_cols': feature_cols,
        'scaler': scaler,
        'label_encoders': label_encoders
    }, f)
print("✓ Saved: models/best_risk_model.pkl")

# ============================================================================
# STEP 4: EVALUATION
# ============================================================================
print("\n" + "="*70)
print("STEP 4: MODEL EVALUATION")
print("="*70)

best_model_name, best_clf = best_model
y_pred = model_results[best_model_name]['y_pred']
y_pred_proba = model_results[best_model_name]['y_pred_proba']

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print(f"\n{best_model_name} - Classification Report:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

if y_pred_proba is not None:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  ROC-AUC:   {roc_auc:.4f}")

print(f"\nConfusion Matrix:")
print(cm)

# Save evaluation report
report_text = f"""
EV GRID STRESS - MODEL EVALUATION REPORT
{'='*60}

Best Model: {best_model_name}

CLASSIFICATION METRICS:
  Accuracy:  {accuracy:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1 Score:  {f1:.4f}

CONFUSION MATRIX:
{cm}

FEATURE IMPORTANCE (Top 10):
"""

if hasattr(best_clf, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_clf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in importances.iterrows():
        report_text += f"\n  {row['feature']}: {row['importance']:.4f}"

with open('models/evaluation_report.txt', 'w') as f:
    f.write(report_text)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✓ PIPELINE COMPLETE!")
print("="*70)
print("\nOutput Files:")
print("  ✓ data/preprocessed_ev_grid_dataset.csv")
print("  ✓ models/best_risk_model.pkl")
print("  ✓ models/evaluation_report.txt")
print("\nNext Steps:")
print("  1. Run Jupyter notebook for visualizations")
print("  2. Start Flask dashboard: python dashboard/app.py")
print("  3. Open browser: http://localhost:5000")
print("="*70)
