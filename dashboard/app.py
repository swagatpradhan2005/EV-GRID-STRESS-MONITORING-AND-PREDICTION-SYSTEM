"""
Flask Dashboard for EV Grid Stress Monitoring and Prediction System
Provides interactive interface for predictions and grid analysis
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import EVGridPredictor, get_risk_mitigation_steps

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize predictor
try:
    predictor = EVGridPredictor()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    predictor = None

# Load state data
try:
    df_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_grid_stress_dataset.csv'))
    print(f"✓ Dataset loaded: {len(df_data)} rows")
except Exception as e:
    print(f"Warning: Could not load dataset: {e}")
    df_data = None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/parameters', methods=['POST'])
def parameters():
    """Parameters selection page"""
    state = request.form.get('state', '')
    
    if not state or df_data is None:
        return render_template('index.html', error="Please select a state")
    
    # Get state data
    state_data = df_data[df_data['state'] == state]
    
    if state_data.empty:
        return render_template('index.html', error=f"No data found for {state}")
    
    # Calculate average parameters
    params = {
        'state': state,
        'charging_sessions': int(state_data['charging_sessions'].mean()),
        'total_charging_stations': int(state_data['total_charging_stations'].mean()),
        'fast_chargers': int(state_data['fast_chargers'].mean()),
        'slow_chargers': int(state_data['slow_chargers'].mean()),
        'transformer_load_percent': round(state_data['transformer_load_percent'].mean(), 2),
        'voltage_v': round(state_data['voltage_v'].mean(), 2),
        'frequency_hz': round(state_data['frequency_hz'].mean(), 2),
        'renewable_share_percent': round(state_data['renewable_share_percent'].mean(), 2),
    }
    
    # Get state statistics
    # Convert risk strings to numeric for calculation
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 1}
    risk_numeric = state_data['overload_risk'].map(risk_mapping).fillna(0)
    
    stats = {
        'avg_grid_load': round(state_data['grid_load_mw'].mean(), 2),
        'max_grid_load': round(state_data['grid_load_mw'].max(), 2),
        'min_grid_load': round(state_data['grid_load_mw'].min(), 2),
        'risk_percentage': round((risk_numeric.sum() / len(state_data)) * 100, 2),
    }
    
    return render_template('parameters.html', params=params, stats=stats, states=sorted(df_data['state'].unique()))


@app.route('/api/get-states')
def get_states():
    """API endpoint to get list of states"""
    if df_data is None:
        return jsonify({'states': []})
    
    states = sorted(df_data['state'].unique().tolist())
    return jsonify({'states': states})


@app.route('/api/get-state-data')
def get_state_data():
    """API endpoint to get state parameters"""
    state = request.args.get('state', '')
    
    if not state or df_data is None:
        return jsonify({'error': 'Invalid state'}), 400
    
    state_data = df_data[df_data['state'] == state]
    
    if state_data.empty:
        return jsonify({'error': 'No data found'}), 404
    
    # Calculate hourly pattern
    hourly_pattern = state_data.groupby('hour').agg({
        'charging_sessions': 'mean',
        'grid_load_mw': 'mean',
        'transformer_load_percent': 'mean'
    }).round(2).to_dict(orient='index')
    
    # Fast vs slow chargers
    charger_dist = {
        'fast': int(state_data['fast_chargers'].mean()),
        'slow': int(state_data['slow_chargers'].mean())
    }
    
    return jsonify({
        'hourly_pattern': hourly_pattern,
        'charger_distribution': charger_dist,
        'avg_transformer_load': round(state_data['transformer_load_percent'].mean(), 2)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on parameters"""
    try:
        # Get form data
        state = request.form.get('state', '')
        charging_sessions = float(request.form.get('charging_sessions', 0))
        total_stations = float(request.form.get('total_charging_stations', 0))
        fast_chargers = float(request.form.get('fast_chargers', 0))
        slow_chargers = float(request.form.get('slow_chargers', 0))
        transformer_load = float(request.form.get('transformer_load_percent', 0))
        voltage = float(request.form.get('voltage_v', 230))
        frequency = float(request.form.get('frequency_hz', 50))
        renewable_share = float(request.form.get('renewable_share_percent', 30))
        
        if not predictor:
            return render_template('result.html', 
                                 error="Prediction model not loaded. Please train models first.",
                                 state=state)
        
        # Prepare prediction data
        prediction_data = {
            'state_encoded': 5,  # Placeholder
            'total_charging_stations': total_stations,
            'fast_chargers': fast_chargers,
            'slow_chargers': slow_chargers,
            'charging_sessions': charging_sessions,
            'energy_consumed_kwh': charging_sessions * 25,  # Estimate
            'voltage_v': voltage,
            'frequency_hz': frequency,
            'transformer_load_percent': transformer_load,
            'renewable_share_percent': renewable_share,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'ev_population': 150000,
            'charger_ev_ratio': 0.002,
            'peak_hour_flag': 1 if 6 <= datetime.now().hour <= 22 else 0,
            'public_charging_stations': total_stations,
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
            'month': datetime.now().month,
            'quarter': (datetime.now().month - 1) // 3 + 1,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0
        }
        
        # Make prediction
        result = predictor.predict(prediction_data)
        
        # Get mitigation steps
        mitigation_steps = get_risk_mitigation_steps(result['risk_level'])
        
        return render_template('result.html',
                             state=state,
                             predicted_grid_load=result['predicted_grid_load_mw'],
                             risk_level=result['risk_level'],
                             risk_probability=result['risk_probability'],
                             charging_sessions=charging_sessions,
                             total_stations=total_stations,
                             transformer_load=transformer_load,
                             mitigation_steps=mitigation_steps)
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"Prediction error: {str(e)}",
                             state=state)


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


if __name__ == '__main__':
    print("="*60)
    print("EV GRID STRESS MONITORING - FLASK DASHBOARD")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='localhost', port=5000)
