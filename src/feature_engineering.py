"""
Feature Engineering Module for EV Grid Stress System
Creates derivative features for improved model performance
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """Create engineered features"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    df = df.copy()
    
    # 1. Fast Charger Ratio
    if 'fast_chargers' in df.columns and 'total_charging_stations' in df.columns:
        df['fast_charger_ratio'] = (
            df['fast_chargers'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: fast_charger_ratio")
    
    # 2. Charging Intensity
    if 'charging_sessions' in df.columns and 'total_charging_stations' in df.columns:
        df['charging_intensity'] = (
            df['charging_sessions'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: charging_intensity")
    
    # 3. Grid Stress Index
    if 'transformer_load_percent' in df.columns and 'charging_sessions' in df.columns:
        df['grid_stress_index'] = (
            df['transformer_load_percent'] * df['charging_sessions'] / 100
        )
        print("✓ Created: grid_stress_index")
    
    # 4. Slow Charger Ratio
    if 'slow_chargers' in df.columns and 'total_charging_stations' in df.columns:
        df['slow_charger_ratio'] = (
            df['slow_chargers'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: slow_charger_ratio")
    
    # 5. Energy Per Session
    if 'energy_consumed_kwh' in df.columns and 'charging_sessions' in df.columns:
        df['energy_per_session'] = (
            df['energy_consumed_kwh'] / (df['charging_sessions'] + 1)
        )
        print("✓ Created: energy_per_session")
    
    # 6. Load Per Station
    if 'grid_load_mw' in df.columns and 'total_charging_stations' in df.columns:
        df['load_per_station'] = (
            df['grid_load_mw'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: load_per_station")
    
    # 7. Charger Efficiency (Energy consumed / Charger count)
    if 'energy_consumed_kwh' in df.columns and 'total_charging_stations' in df.columns:
        df['charger_efficiency'] = (
            df['energy_consumed_kwh'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: charger_efficiency")
    
    # 8. Peak Hour Impact
    if 'peak_hour_flag' in df.columns and 'grid_load_mw' in df.columns:
        df['peak_load_impact'] = (
            df['peak_hour_flag'] * df['grid_load_mw']
        )
        print("✓ Created: peak_load_impact")
    
    # 9. Renewable Integration Ratio
    if 'renewable_share_percent' in df.columns and 'grid_load_mw' in df.columns:
        df['renewable_load_ratio'] = (
            df['renewable_share_percent'] * df['grid_load_mw'] / 100
        )
        print("✓ Created: renewable_load_ratio")
    
    # 10. Voltage Stability Index
    if 'voltage_v' in df.columns and 'frequency_hz' in df.columns:
        # Normal voltage is 230V, normal frequency is 50Hz
        df['voltage_stability'] = (
            np.abs(df['voltage_v'] - 230) + np.abs(df['frequency_hz'] - 50)
        )
        print("✓ Created: voltage_stability")
    
    # 11. Grid Demand Ratio
    if 'ev_population' in df.columns and 'total_charging_stations' in df.columns:
        df['ev_demand_ratio'] = (
            df['ev_population'] / (df['total_charging_stations'] + 1)
        )
        print("✓ Created: ev_demand_ratio")
    
    # 12. Fast Charger Stress
    if 'fast_chargers' in df.columns and 'transformer_load_percent' in df.columns:
        df['fast_charger_stress'] = (
            df['fast_chargers'] * df['transformer_load_percent'] / 100
        )
        print("✓ Created: fast_charger_stress")
    
    print(f"\n✓ Total features created: 12")
    print(f"✓ Final dataset shape: {df.shape}")
    
    return df


def add_interaction_features(df):
    """Create interaction features for better model performance"""
    print("\nAdding interaction features...")
    
    df = df.copy()
    
    # Interaction: Peak hour x transformer load
    if 'peak_hour_flag' in df.columns and 'transformer_load_percent' in df.columns:
        df['peak_transformer_interaction'] = (
            df['peak_hour_flag'] * df['transformer_load_percent']
        )
        print("✓ Created: peak_transformer_interaction")
    
    # Interaction: Charging sessions x renewable share
    if 'charging_sessions' in df.columns and 'renewable_share_percent' in df.columns:
        df['charging_renewable_interaction'] = (
            df['charging_sessions'] * df['renewable_share_percent'] / 100
        )
        print("✓ Created: charging_renewable_interaction")
    
    # Interaction: EV population x charger ratio
    if 'ev_population' in df.columns and 'charger_ev_ratio' in df.columns:
        df['ev_charger_demand'] = (
            df['ev_population'] * df['charger_ev_ratio']
        )
        print("✓ Created: ev_charger_demand")
    
    return df


def select_features(df, target_col=None):
    """Select relevant features for modeling"""
    print("\nSelecting features for modeling...")
    
    # Define feature groups
    base_features = [
        'total_charging_stations', 'fast_chargers', 'slow_chargers',
        'charging_sessions', 'energy_consumed_kwh', 'grid_load_mw',
        'voltage_v', 'frequency_hz', 'transformer_load_percent',
        'renewable_share_percent', 'hour', 'day_of_week', 'ev_population',
        'charger_ev_ratio', 'peak_hour_flag'
    ]
    
    engineered_features = [
        'fast_charger_ratio', 'charging_intensity', 'grid_stress_index',
        'slow_charger_ratio', 'energy_per_session', 'load_per_station',
        'charger_efficiency', 'peak_load_impact', 'renewable_load_ratio',
        'voltage_stability', 'ev_demand_ratio', 'fast_charger_stress'
    ]
    
    interaction_features = [
        'peak_transformer_interaction', 'charging_renewable_interaction',
        'ev_charger_demand'
    ]
    
    # Combine all available features
    all_features = base_features + engineered_features + interaction_features
    available_features = [f for f in all_features if f in df.columns]
    
    # Add encoded state if present
    if 'state_encoded' in df.columns:
        available_features.insert(0, 'state_encoded')
    
    # Add time features if present
    time_features = ['month', 'quarter', 'is_weekend']
    for tf in time_features:
        if tf in df.columns and tf not in available_features:
            available_features.append(tf)
    
    print(f"✓ Selected {len(available_features)} features")
    
    return available_features


def prepare_features_and_target(df, target_col='overload_risk'):
    """Prepare features and target for modeling"""
    print("\n" + "="*60)
    print("PREPARING FEATURES AND TARGET")
    print("="*60)
    
    # Load from CSV if string path is provided
    if isinstance(df, str):
        df = pd.read_csv(df)
    
    # Select features
    feature_cols = select_features(df, target_col)
    
    # Ensure target exists
    if target_col not in df.columns:
        print(f"⚠ Target column '{target_col}' not found!")
        return None, None
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols


if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_ev_grid_dataset.csv')
    
    # Engineer features
    df = engineer_features(df)
    
    # Add interaction features
    df = add_interaction_features(df)
    
    # Prepare for modeling
    X, y, feature_cols = prepare_features_and_target(df)
    
    # Save engineered features
    df.to_csv('data/engineered_features.csv', index=False)
    print("\n✓ Saved engineered features to data/engineered_features.csv")
    
    print("\n" + "="*60)
    print("Feature Engineering Complete!")
    print("="*60)
