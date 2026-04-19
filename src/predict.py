"""
Prediction Module for EV Grid Stress System
Makes predictions on new data using trained models
"""

import pandas as pd
import numpy as np
import pickle
import os


class EVGridPredictor:
    """Wrapper class for making predictions"""
    
    def __init__(self, load_model_path='models/best_load_model.pkl',
                 risk_model_path='models/best_risk_model.pkl'):
        """Initialize predictor with trained models"""
        self.load_model_path = load_model_path
        self.risk_model_path = risk_model_path
        self.load_model_data = None
        self.risk_model_data = None
        self.feature_cols = None
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(self.load_model_path):
                with open(self.load_model_path, 'rb') as f:
                    self.load_model_data = pickle.load(f)
                print("✓ Loaded grid load prediction model")
            
            if os.path.exists(self.risk_model_path):
                with open(self.risk_model_path, 'rb') as f:
                    self.risk_model_data = pickle.load(f)
                print("✓ Loaded overload risk prediction model")
            
            # Store feature columns
            if self.load_model_data:
                self.feature_cols = self.load_model_data.get('feature_cols')
        
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def prepare_features(self, data_dict):
        """Prepare features for prediction"""
        # Create DataFrame from input
        df = pd.DataFrame([data_dict])
        
        # Fill missing columns with default values
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select only required features in correct order
        X = df[self.feature_cols]
        
        return X
    
    def predict_grid_load(self, data_dict):
        """Predict grid load in MW"""
        if self.load_model_data is None:
            return None
        
        X = self.prepare_features(data_dict)
        model = self.load_model_data['model']
        
        prediction = model.predict(X)[0]
        return max(0, prediction)  # Ensure non-negative
    
    def predict_overload_risk(self, data_dict):
        """Predict overload risk (0 or 1)"""
        if self.risk_model_data is None:
            return None
        
        X = self.prepare_features(data_dict)
        model = self.risk_model_data['model']
        
        prediction = model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]
        
        return int(prediction), probability
    
    def get_risk_level(self, risk_prediction, risk_probability):
        """Convert risk prediction to risk level"""
        if risk_prediction == 0:
            return "LOW"
        elif risk_probability and risk_probability > 0.7:
            return "HIGH"
        else:
            return "MEDIUM"
    
    def predict(self, data_dict):
        """Make complete prediction"""
        grid_load = self.predict_grid_load(data_dict)
        risk_pred, risk_prob = self.predict_overload_risk(data_dict)
        risk_level = self.get_risk_level(risk_pred, risk_prob)
        
        return {
            'predicted_grid_load_mw': grid_load,
            'overload_risk': risk_pred,
            'risk_probability': risk_prob,
            'risk_level': risk_level
        }


def get_risk_mitigation_steps(risk_level):
    """Get mitigation steps based on risk level"""
    steps = {
        'LOW': [
            "✓ Grid is stable",
            "Continue monitoring grid parameters",
            "Maintain current charging distribution"
        ],
        'MEDIUM': [
            "1. Schedule fast charging during off-peak hours",
            "2. Enable smart charging load distribution",
            "3. Monitor transformer temperatures",
            "4. Increase renewable energy supply gradually"
        ],
        'HIGH': [
            "1. URGENT: Reduce fast charging during peak hours",
            "2. Enable smart charging scheduling immediately",
            "3. Increase transformer capacity or add new transformers",
            "4. Distribute charging load across multiple stations",
            "5. Maximize renewable energy generation",
            "6. Implement demand response programs",
            "7. Encourage off-peak charging incentives"
        ]
    }
    
    return steps.get(risk_level, [])


def format_prediction_result(prediction_result):
    """Format prediction result for display"""
    result_text = f"""
    ╔════════════════════════════════════════════╗
    ║    EV GRID STRESS PREDICTION RESULT        ║
    ╚════════════════════════════════════════════╝
    
    Predicted Grid Load: {prediction_result['predicted_grid_load_mw']:.2f} MW
    
    Overload Risk Status: {prediction_result['risk_level']}
    Risk Probability: {prediction_result['risk_probability']*100:.2f}% if prediction_result['risk_probability'] else 'N/A'
    
    ════════════════════════════════════════════
    """
    
    return result_text


def example_prediction():
    """Example prediction"""
    print("="*60)
    print("EV GRID STRESS - PREDICTION EXAMPLE")
    print("="*60)
    
    # Initialize predictor
    predictor = EVGridPredictor()
    
    # Example input data
    example_data = {
        'state_encoded': 5,
        'total_charging_stations': 300,
        'fast_chargers': 150,
        'slow_chargers': 150,
        'charging_sessions': 800,
        'energy_consumed_kwh': 25000,
        'voltage_v': 230,
        'frequency_hz': 50.0,
        'transformer_load_percent': 75,
        'renewable_share_percent': 35,
        'hour': 14,
        'day_of_week': 2,
        'ev_population': 150000,
        'charger_ev_ratio': 0.002,
        'peak_hour_flag': 1,
        'public_charging_stations': 300,
        'charging_growth_rate': 0.20,
        'average_station_capacity_kw': 75,
        'peak_demand_mw': 5000,
        'average_daily_load_mw': 3000,
        'total_energy_consumption_gwh': 50000,
        'solar_capacity_mw': 1000,
        'wind_capacity_mw': 500,
        'hydro_capacity_mw': 1500,
        'ev_registered': 150000,
        'ev_growth_rate': 0.25,
        'month': 7,
        'quarter': 3,
        'is_weekend': 0
    }
    
    print("\nInput Parameters:")
    for key, value in example_data.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    result = predictor.predict(example_data)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Predicted Grid Load: {result['predicted_grid_load_mw']:.2f} MW")
    print(f"Overload Risk: {result['risk_level']}")
    if result['risk_probability']:
        print(f"Risk Probability: {result['risk_probability']*100:.2f}%")
    
    # Get mitigation steps
    mitigation_steps = get_risk_mitigation_steps(result['risk_level'])
    print("\nRecommended Actions:")
    for step in mitigation_steps:
        print(f"  {step}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    example_prediction()
