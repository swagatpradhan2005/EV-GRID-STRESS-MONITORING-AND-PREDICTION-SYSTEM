"""
Model Evaluation Module for EV Grid Stress System
Evaluates model performance with detailed metrics
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def evaluate_regression_model(model, X_test, y_test):
    """Evaluate regression model"""
    print("\nREGRESSION MODEL EVALUATION")
    print("-" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }


def evaluate_classification_model(model, X_test, y_test):
    """Evaluate classification model"""
    print("\nCLASSIFICATION MODEL EVALUATION")
    print("-" * 50)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ROC-AUC if binary classification
    roc_auc = None
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved confusion matrix plot: {save_path}")
    
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    if y_pred_proba is None:
        print("ROC curve requires probability predictions")
        return
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Overload Risk Prediction')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved ROC curve plot: {save_path}")
    
    plt.close()


def create_evaluation_report(X_test, y_test_regression, y_test_classification):
    """Create comprehensive evaluation report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*60)
    
    # Load models
    regression_model_data = load_model('models/best_load_model.pkl')
    classification_model_data = load_model('models/best_risk_model.pkl')
    
    regression_model = regression_model_data['model']
    classification_model = classification_model_data['model']
    
    # Evaluate regression model
    print(f"\n{'='*60}")
    print(f"MODEL: {regression_model_data['name']}")
    print(f"TASK: Grid Load Prediction")
    print(f"{'='*60}")
    
    reg_metrics = evaluate_regression_model(regression_model, X_test, y_test_regression)
    
    # Evaluate classification model
    print(f"\n{'='*60}")
    print(f"MODEL: {classification_model_data['name']}")
    print(f"TASK: Overload Risk Prediction")
    print(f"{'='*60}")
    
    clf_metrics = evaluate_classification_model(classification_model, X_test, y_test_classification)
    
    # Create visualizations
    try:
        plot_confusion_matrix(clf_metrics['confusion_matrix'], 
                            'models/confusion_matrix.png')
        plot_roc_curve(y_test_classification, clf_metrics['probabilities'],
                      'models/roc_curve.png')
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Create summary report
    report = f"""
EV GRID STRESS MONITORING SYSTEM - EVALUATION REPORT
{'='*60}

REGRESSION MODEL (Grid Load Prediction)
Model: {regression_model_data['name']}
RMSE: {reg_metrics['rmse']:.4f}
MAE: {reg_metrics['mae']:.4f}
R² Score: {reg_metrics['r2']:.4f}

CLASSIFICATION MODEL (Overload Risk Prediction)
Model: {classification_model_data['name']}
Accuracy: {clf_metrics['accuracy']:.4f}
Precision: {clf_metrics['precision']:.4f}
Recall: {clf_metrics['recall']:.4f}
F1 Score: {clf_metrics['f1']:.4f}
ROC-AUC: {clf_metrics['roc_auc']:.4f}

{'='*60}
"""
    
    print(report)
    
    # Save report
    with open('models/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Saved evaluation report: models/evaluation_report.txt")
    
    return {
        'regression': reg_metrics,
        'classification': clf_metrics,
        'models': {
            'regression': regression_model,
            'classification': classification_model
        }
    }


if __name__ == "__main__":
    # This will be called from the training pipeline
    print("Evaluation module loaded successfully")
