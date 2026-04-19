"""
Dataset Builder for EV Grid Stress Monitoring System
Generates synthetic datasets for training and analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Random seed for reproducibility
np.random.seed(42)

# Indian states
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

def create_ev_grid_stress_dataset(num_rows=25000):
    """Create main EV Grid Stress dataset"""
    print("Creating EV Grid Stress Dataset...")
    
    # Generate timestamps over 1 year
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i * 6) for i in range(num_rows)]
    
    data = {
        'timestamp': timestamps,
        'state': np.random.choice(INDIAN_STATES, num_rows),
        'total_charging_stations': np.random.randint(50, 800, num_rows),
        'fast_chargers': np.random.randint(10, 400, num_rows),
        'slow_chargers': np.random.randint(10, 500, num_rows),
        'charging_sessions': np.random.randint(50, 2000, num_rows),
        'energy_consumed_kwh': np.random.uniform(5000, 50000, num_rows),
        'grid_load_mw': np.random.uniform(500, 5000, num_rows),
        'voltage_v': np.random.uniform(220, 240, num_rows),
        'frequency_hz': np.random.uniform(49.5, 50.5, num_rows),
        'transformer_load_percent': np.random.uniform(20, 120, num_rows),
        'renewable_share_percent': np.random.uniform(10, 60, num_rows),
        'hour': [t.hour for t in timestamps],
        'day_of_week': [t.weekday() for t in timestamps],
        'ev_population': np.random.randint(10000, 500000, num_rows),
        'charger_ev_ratio': np.random.uniform(0.001, 0.02, num_rows),
    }
    
    df = pd.DataFrame(data)
    
    # Create peak_hour_flag (6-22 hours are peak hours)
    df['peak_hour_flag'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 22 else 0)
    
    # Create overload_risk based on transformer load
    df['overload_risk'] = (
        (df['transformer_load_percent'] > 90).astype(int)
    )
    
    # Add some correlation
    for i in range(num_rows):
        if df.loc[i, 'peak_hour_flag'] == 1:
            df.loc[i, 'charging_sessions'] *= 1.3
            df.loc[i, 'grid_load_mw'] *= 1.2
            df.loc[i, 'transformer_load_percent'] *= 1.15
        
        if df.loc[i, 'overload_risk'] == 1:
            df.loc[i, 'energy_consumed_kwh'] *= 1.2
    
    # Add some missing values (realistic)
    missing_indices = np.random.choice(num_rows, int(num_rows * 0.02), replace=False)
    df.loc[missing_indices, 'voltage_v'] = np.nan
    
    missing_indices = np.random.choice(num_rows, int(num_rows * 0.01), replace=False)
    df.loc[missing_indices, 'frequency_hz'] = np.nan
    
    output_path = 'data/ev_grid_stress_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path} ({len(df)} rows)")
    
    return df


def create_ev_charging_infrastructure():
    """Create EV Charging Infrastructure dataset"""
    print("Creating EV Charging Infrastructure Dataset...")
    
    data = {
        'state': INDIAN_STATES,
        'public_charging_stations': np.random.randint(50, 1000, len(INDIAN_STATES)),
        'fast_chargers': np.random.randint(20, 600, len(INDIAN_STATES)),
        'slow_chargers': np.random.randint(30, 800, len(INDIAN_STATES)),
        'charging_growth_rate': np.random.uniform(0.05, 0.35, len(INDIAN_STATES)),
        'average_station_capacity_kw': np.random.uniform(30, 150, len(INDIAN_STATES)),
    }
    
    df = pd.DataFrame(data)
    output_path = 'data/ev_charging_infrastructure.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path} ({len(df)} rows)")
    
    return df


def create_state_electricity_demand():
    """Create State Electricity Demand dataset"""
    print("Creating State Electricity Demand Dataset...")
    
    data = {
        'state': INDIAN_STATES,
        'year': 2023,
        'peak_demand_mw': np.random.uniform(1000, 20000, len(INDIAN_STATES)),
        'average_daily_load_mw': np.random.uniform(500, 12000, len(INDIAN_STATES)),
        'total_energy_consumption_gwh': np.random.uniform(10000, 150000, len(INDIAN_STATES)),
    }
    
    df = pd.DataFrame(data)
    output_path = 'data/state_electricity_demand.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path} ({len(df)} rows)")
    
    return df


def create_renewable_energy():
    """Create Renewable Energy by State dataset"""
    print("Creating Renewable Energy Dataset...")
    
    data = {
        'state': INDIAN_STATES,
        'solar_capacity_mw': np.random.uniform(100, 5000, len(INDIAN_STATES)),
        'wind_capacity_mw': np.random.uniform(50, 3000, len(INDIAN_STATES)),
        'hydro_capacity_mw': np.random.uniform(50, 4000, len(INDIAN_STATES)),
        'renewable_share_percent': np.random.uniform(10, 70, len(INDIAN_STATES)),
    }
    
    df = pd.DataFrame(data)
    output_path = 'data/renewable_energy_by_state.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path} ({len(df)} rows)")
    
    return df


def create_ev_adoption():
    """Create EV Adoption by State dataset"""
    print("Creating EV Adoption Dataset...")
    
    data = {
        'state': INDIAN_STATES,
        'year': 2023,
        'ev_registered': np.random.randint(10000, 500000, len(INDIAN_STATES)),
        'ev_growth_rate': np.random.uniform(0.15, 0.50, len(INDIAN_STATES)),
    }
    
    df = pd.DataFrame(data)
    output_path = 'data/ev_adoption_by_state.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Created: {output_path} ({len(df)} rows)")
    
    return df


def create_india_geojson():
    """Create India GeoJSON file"""
    print("Creating India Map GeoJSON...")
    
    # Simplified India states GeoJSON (representative centroids)
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    state_coords = {
        'Andhra Pradesh': [79.7400, 15.9129],
        'Arunachal Pradesh': [93.6100, 28.2180],
        'Assam': [91.7898, 26.2006],
        'Bihar': [85.3131, 25.0961],
        'Chhattisgarh': [81.6296, 21.2787],
        'Goa': [73.8278, 15.2993],
        'Gujarat': [72.6369, 22.2587],
        'Haryana': [77.0373, 29.0588],
        'Himachal Pradesh': [77.1734, 31.7894],
        'Jharkhand': [85.2799, 23.6102],
        'Karnataka': [75.7139, 15.3173],
        'Kerala': [76.2711, 10.8505],
        'Madhya Pradesh': [78.6569, 22.9375],
        'Maharashtra': [75.7139, 19.7515],
        'Manipur': [94.7868, 24.6637],
        'Meghalaya': [91.8960, 25.4670],
        'Mizoram': [93.9063, 23.1645],
        'Nagaland': [94.5614, 26.1584],
        'Odisha': [84.1239, 20.9517],
        'Punjab': [75.5941, 31.1471],
        'Rajasthan': [75.7139, 27.0238],
        'Sikkim': [88.5122, 27.5330],
        'Tamil Nadu': [78.6569, 11.1271],
        'Telangana': [78.4711, 18.1124],
        'Tripura': [91.9882, 23.9408],
        'Uttar Pradesh': [79.0193, 26.8467],
        'Uttarakhand': [79.0193, 30.0668],
        'West Bengal': [88.3639, 24.5155],
    }
    
    for state in INDIAN_STATES:
        if state in state_coords:
            coords = state_coords[state]
            feature = {
                "type": "Feature",
                "properties": {
                    "name": state,
                    "state": state
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": coords
                }
            }
            geojson_data["features"].append(feature)
    
    output_path = 'data/india_states.geojson'
    with open(output_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    
    print(f"✓ Created: {output_path}")
    
    return geojson_data


def merge_datasets():
    """Merge all datasets for training"""
    print("\nMerging datasets...")
    
    # Read main dataset
    main_df = pd.read_csv('data/ev_grid_stress_dataset.csv')
    infra_df = pd.read_csv('data/ev_charging_infrastructure.csv')
    demand_df = pd.read_csv('data/state_electricity_demand.csv')
    renewable_df = pd.read_csv('data/renewable_energy_by_state.csv')
    adoption_df = pd.read_csv('data/ev_adoption_by_state.csv')
    
    # Merge datasets on state
    merged_df = main_df.copy()
    
    # Merge infrastructure data
    merged_df = merged_df.merge(infra_df, on='state', how='left')
    
    # Merge electricity demand
    merged_df = merged_df.merge(demand_df, on='state', how='left')
    
    # Merge renewable energy
    merged_df = merged_df.merge(renewable_df, on='state', how='left')
    
    # Merge EV adoption
    merged_df = merged_df.merge(adoption_df, on='state', how='left')
    
    # Save merged dataset
    output_path = 'data/final_ev_grid_dataset.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Created merged dataset: {output_path} ({len(merged_df)} rows, {len(merged_df.columns)} columns)")
    
    return merged_df


def build_all_datasets():
    """Build all datasets"""
    print("="*60)
    print("EV GRID STRESS MONITORING SYSTEM - DATASET BUILDER")
    print("="*60)
    print()
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create all datasets
    create_ev_grid_stress_dataset(num_rows=25000)
    create_ev_charging_infrastructure()
    create_state_electricity_demand()
    create_renewable_energy()
    create_ev_adoption()
    create_india_geojson()
    
    # Merge datasets
    merge_datasets()
    
    print()
    print("="*60)
    print("✓ All datasets created successfully!")
    print("="*60)


if __name__ == "__main__":
    build_all_datasets()
