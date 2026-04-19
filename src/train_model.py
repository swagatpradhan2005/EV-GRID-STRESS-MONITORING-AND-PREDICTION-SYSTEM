"""
Model Training Module for EV Grid Stress System
Trains regression and classification models
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def load_engineered_data(filepath):
    """Load engineered features dataset"""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def prepare_data_for_training(df, target_col='overload_risk'):
    """Prepare data for training/testing"""
    print(f"\nPreparing data with target: {target_col}...")
    
    # Select features (exclude identifiers and targets)
    exclude_cols = ['overload_risk', 'grid_load_mw']
    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
    y = df[target_col].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Handle any remaining NaN values
    X = X.fillna(X.mean())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, X.columns


def train_regression_models(X_train, X_test, y_train, y_test):
    """Train regression models for grid load prediction"""
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS")
    print("="*60)
    
    models_config = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [15, 20]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.1, 0.05]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, verbosity=0),
            'params': {
                'n_estimators': [100],
                'max_depth': [5, 7]
            }
        }
    }
    
    trained_models = {}
    
    for model_name, config in models_config.items():
        print(f"\n{model_name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,
            n_jobs=-1,
            scoring='r2'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        trained_models[model_name] = {
            'model': best_model,
            'train_score': train_score,
            'test_score': test_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Train R² Score: {train_score:.4f}")
        print(f"  Test R² Score: {test_score:.4f}")
    
    return trained_models


def train_classification_models(X_train, X_test, y_train, y_test):
    """Train classification models for overload risk prediction"""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [10, 15]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.1, 0.05]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, verbosity=0),
            'params': {
                'n_estimators': [100],
                'max_depth': [5, 7]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.1, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'kernel': ['rbf'],
                'C': [0.1, 1.0]
            }
        }
    }
    
    trained_models = {}
    
    for model_name, config in models_config.items():
        print(f"\n{model_name}...")
        
        # Grid search (simplified for SVM due to computation)
        cv_folds = 2 if model_name == 'SVM' else 3
        
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=cv_folds,
            n_jobs=-1,
            scoring='f1'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        
        trained_models[model_name] = {
            'model': best_model,
            'train_score': train_score,
            'test_score': test_score,
            'best_params': grid_search.best_params_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Test Accuracy: {test_score:.4f}")
    
    return trained_models


def select_best_model(trained_models, task_type='classification'):
    """Select best model based on test score"""
    print(f"\n{'='*60}")
    print(f"SELECTING BEST {task_type.upper()} MODEL")
    print('='*60)
    
    best_model_name = None
    best_score = -1
    best_model_obj = None
    
    for model_name, model_info in trained_models.items():
        test_score = model_info['test_score']
        print(f"{model_name}: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model_name = model_name
            best_model_obj = model_info['model']
    
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"✓ Test Score: {best_score:.4f}")
    
    return best_model_name, best_model_obj


def train_all_models(input_file, output_dir='models'):
    """Complete model training pipeline"""
    print("="*60)
    print("EV GRID STRESS - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_engineered_data(input_file)
    
    # Train regression models (for grid load prediction)
    print("\n" + "-"*60)
    print("REGRESSION MODELS (Grid Load Prediction)")
    print("-"*60)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_cols = (
        prepare_data_for_training(df, target_col='grid_load_mw')
    )
    
    regression_models = train_regression_models(
        X_train_reg, X_test_reg, y_train_reg, y_test_reg
    )
    
    best_reg_name, best_reg_model = select_best_model(
        regression_models, task_type='regression'
    )
    
    # Train classification models (for overload risk prediction)
    print("\n" + "-"*60)
    print("CLASSIFICATION MODELS (Overload Risk Prediction)")
    print("-"*60)
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, _ = (
        prepare_data_for_training(df, target_col='overload_risk')
    )
    
    classification_models = train_classification_models(
        X_train_clf, X_test_clf, y_train_clf, y_test_clf
    )
    
    best_clf_name, best_clf_model = select_best_model(
        classification_models, task_type='classification'
    )
    
    # Save best models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print('='*60)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save regression model
    reg_model_path = os.path.join(output_dir, 'best_load_model.pkl')
    with open(reg_model_path, 'wb') as f:
        pickle.dump({
            'model': best_reg_model,
            'name': best_reg_name,
            'feature_cols': feature_cols.tolist()
        }, f)
    print(f"✓ Saved regression model: {reg_model_path}")
    
    # Save classification model
    clf_model_path = os.path.join(output_dir, 'best_risk_model.pkl')
    with open(clf_model_path, 'wb') as f:
        pickle.dump({
            'model': best_clf_model,
            'name': best_clf_name,
            'feature_cols': feature_cols.tolist()
        }, f)
    print(f"✓ Saved classification model: {clf_model_path}")
    
    print("\n" + "="*60)
    print("Model Training Complete!")
    print("="*60)
    
    return {
        'regression': {
            'name': best_reg_name,
            'model': best_reg_model,
            'all_models': regression_models
        },
        'classification': {
            'name': best_clf_name,
            'model': best_clf_model,
            'all_models': classification_models
        }
    }


if __name__ == "__main__":
    input_file = 'data/engineered_features.csv'
    train_all_models(input_file)
