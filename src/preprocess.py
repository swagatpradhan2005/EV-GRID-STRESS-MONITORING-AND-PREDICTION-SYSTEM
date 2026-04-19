"""
Data Preprocessing Module for EV Grid Stress System
Handles cleaning, normalization, and encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def handle_missing_values(df):
    """Handle missing values"""
    print("\nHandling missing values...")
    
    initial_missing = df.isnull().sum().sum()
    
    if initial_missing > 0:
        print(f"Found {initial_missing} missing values")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  - Filled {col} with median")
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  - Filled {col} with mode")
    
    print(f"✓ Missing values: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """Remove duplicate rows"""
    print("\nRemoving duplicates...")
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    
    if removed > 0:
        print(f"✓ Removed {removed} duplicate rows")
    else:
        print("✓ No duplicates found")
    
    return df


def remove_outliers(df, columns=None, threshold=3):
    """Remove outliers using z-score method"""
    print("\nRemoving outliers...")
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    initial_rows = len(df)
    
    # Calculate z-scores
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    
    # Remove rows with z-score > threshold
    df = df[(z_scores < threshold).all(axis=1)]
    
    removed = initial_rows - len(df)
    print(f"✓ Removed {removed} outlier rows")
    
    return df


def encode_categorical(df, categorical_cols=None):
    """Encode categorical variables"""
    print("\nEncoding categorical variables...")
    
    if categorical_cols is None:
        categorical_cols = ['state']
    
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"✓ Encoded {col}: {len(le.classes_)} unique values")
            
            # Drop original categorical column
            df = df.drop(col, axis=1)
    
    return df, label_encoders


def normalize_features(df, feature_columns=None):
    """Normalize numerical features using StandardScaler"""
    print("\nNormalizing numerical features...")
    
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't scale encoded categorical or target variables
    cols_to_exclude = ['overload_risk', 'state_encoded', 'hour', 'day_of_week', 'peak_hour_flag']
    feature_columns = [col for col in feature_columns if col not in cols_to_exclude]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    print(f"✓ Normalized {len(feature_columns)} features")
    
    return df_scaled, scaler, feature_columns


def extract_timestamp_features(df):
    """Extract time-based features from timestamp"""
    print("\nExtracting timestamp features...")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        print("✓ Extracted: month, quarter, is_weekend")
        
        # Drop original timestamp
        df = df.drop('timestamp', axis=1)
    
    return df


def preprocess_pipeline(input_file, output_file):
    """Complete preprocessing pipeline"""
    print("="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data(input_file)
    print(f"Initial shape: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Remove outliers
    outlier_cols = ['grid_load_mw', 'energy_consumed_kwh', 'transformer_load_percent']
    existing_outlier_cols = [col for col in outlier_cols if col in df.columns]
    if existing_outlier_cols:
        df = remove_outliers(df, columns=existing_outlier_cols)
    
    # Extract timestamp features
    df = extract_timestamp_features(df)
    
    # Encode categorical variables
    df, label_encoders = encode_categorical(df, categorical_cols=['state'])
    
    # Normalize numerical features
    df, scaler, feature_columns = normalize_features(df)
    
    print(f"\nFinal shape: {df.shape}")
    
    # Save preprocessed data
    df.to_csv(output_file, index=False)
    print(f"✓ Saved preprocessed data to {output_file}")
    
    return df, scaler, label_encoders


def preprocess_for_prediction(data_dict, scaler, label_encoders):
    """Preprocess data for prediction"""
    df = pd.DataFrame([data_dict])
    
    # Encode state if present
    if 'state' in df.columns and 'state' in label_encoders:
        df['state_encoded'] = label_encoders['state'].transform(df['state'])
        df = df.drop('state', axis=1)
    
    # Normalize features
    feature_columns = ['total_charging_stations', 'fast_chargers', 'slow_chargers',
                      'charging_sessions', 'energy_consumed_kwh', 'grid_load_mw',
                      'voltage_v', 'frequency_hz', 'transformer_load_percent',
                      'renewable_share_percent']
    existing_features = [col for col in feature_columns if col in df.columns]
    
    if existing_features:
        df[existing_features] = scaler.transform(df[existing_features])
    
    return df


if __name__ == "__main__":
    input_file = '../data/ev_grid_stress_dataset.csv'
    output_file = '../data/preprocessed_ev_grid_dataset.csv'
    
    df, scaler, label_encoders = preprocess_pipeline(input_file, output_file)
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
