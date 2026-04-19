# QUICK START GUIDE - EV GRID STRESS MONITORING SYSTEM

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Generate Datasets
```
cd src
python dataset_builder.py
```

### 3. Preprocess & Engineer Features
```
python preprocess.py
python feature_engineering.py
```

### 4. Train Models
```
python train_model.py
```

### 5. Start Dashboard
```
cd ../dashboard
python app.py
```

### 6. Open Browser
Go to: **http://localhost:5000**

---

## 📊 Project Files Created

### Source Code
- `src/dataset_builder.py` - Creates 25,000 rows of synthetic EV grid data
- `src/preprocess.py` - Data cleaning, normalization, and encoding
- `src/feature_engineering.py` - Creates 12+ engineered features
- `src/train_model.py` - Trains 5 regression and 5 classification models
- `src/evaluate_model.py` - Evaluates model performance
- `src/predict.py` - Makes predictions on new data

### Data Files
- `data/ev_grid_stress_dataset.csv` - Main dataset (25,000 rows)
- `data/ev_charging_infrastructure.csv` - Infrastructure data
- `data/state_electricity_demand.csv` - Baseline electricity demand
- `data/renewable_energy_by_state.csv` - Renewable capacity
- `data/ev_adoption_by_state.csv` - EV adoption trends
- `data/final_ev_grid_dataset.csv` - Merged dataset
- `data/india_states.geojson` - Geographic coordinates

### Models
- `models/best_load_model.pkl` - Trained regression model for grid load
- `models/best_risk_model.pkl` - Trained classification model for risk

### Dashboard
- `dashboard/app.py` - Flask application
- `dashboard/templates/index.html` - Home page
- `dashboard/templates/parameters.html` - Parameter adjustment
- `dashboard/templates/result.html` - Results display
- `dashboard/templates/about.html` - Documentation
- `dashboard/static/style.css` - Styling

### Documentation
- `notebooks/ev_grid_analysis.ipynb` - Analysis and visualizations
- `README.md` - Comprehensive documentation
- `requirements.txt` - Python dependencies

---

## 🎯 System Features

### Prediction Engine
- **Grid Load Prediction**: Forecasts MW of electricity demand
- **Risk Assessment**: Determines overload risk (Low/Medium/High)
- **Recommendations**: Provides mitigation strategies

### Interactive Dashboard
- State selection from all 28 Indian states
- Parameter adjustment interface
- Real-time prediction results
- Visual risk indicators with color coding
- Hourly and daily pattern analysis

### Machine Learning Models
- **Regression**: Random Forest, XGBoost, Gradient Boosting
- **Classification**: 5 models with automatic selection
- **Evaluation**: RMSE, MAE, R², Accuracy, F1-Score, ROC-AUC

---

## 🔄 Data Flow

```
Raw Data → Preprocessing → Feature Engineering → Model Training
    ↓           ↓              ↓                      ↓
25,000+     Handle Missing   Create 12+         Trained Models
Records     Values, Outliers Engineered         Selected
            Remove Dups      Features           Automatically
            Normalize        Interactions
            Encode           Added
```

---

## 📈 Features Created

**Base Features** (15):
- Charging stations, sessions, energy consumption
- Grid metrics (load, voltage, frequency)
- Transformer load, renewable share
- EV population, charger ratios
- Time-based features

**Engineered Features** (12):
- fast_charger_ratio
- charging_intensity
- grid_stress_index
- load_per_station
- energy_per_session
- renewable_load_ratio
- voltage_stability
- And 5 more...

**Interaction Features** (3):
- peak_transformer_interaction
- charging_renewable_interaction
- ev_charger_demand

---

## 📊 Dataset Structure

### Main Dataset (25,000 rows)
```
timestamp | state | total_charging_stations | fast_chargers | ...
2023-01-01 | Gujarat | 250 | 125 | ...
2023-01-02 | Gujarat | 250 | 125 | ...
...
```

### Features (40+ total)
- 3 time-based features
- 15 base input features
- 12 engineered features
- 3 interaction features
- 2 target variables

---

## 🎨 Dashboard Workflow

1. **Home**: Select state
   ↓
2. **Parameters**: Adjust inputs (charging_sessions, load, etc.)
   ↓
3. **Predict**: Click PREDICT button
   ↓
4. **Results**: See predicted grid load and risk level
   ↓
5. **Recommendations**: Follow mitigation strategies

---

## 📋 Risk Levels

### 🟢 LOW RISK
- Status: Safe
- Action: Continue monitoring

### 🟡 MEDIUM RISK
- Status: Approaching limits
- Action: Enable smart charging, load shifting

### 🔴 HIGH RISK
- Status: Critical
- Action: Immediate intervention required

---

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| Models not found | Run `python src/train_model.py` |
| Data not found | Run `python src/dataset_builder.py` |
| Port in use | Change port in `app.py` |
| Missing modules | Run `pip install -r requirements.txt` |

---

## 📚 Learn More

- **Full Documentation**: See README.md
- **Data Analysis**: Open notebooks/ev_grid_analysis.ipynb
- **Source Code**: Review src/ directory
- **API Docs**: Check dashboard/app.py

---

## ✨ Key Highlights

✓ 25,000+ synthetic data points across all 28 Indian states
✓ 5 machine learning algorithms trained and compared
✓ Automatic best model selection
✓ Interactive Flask-based dashboard
✓ Comprehensive jupyter notebook with 7+ visualizations
✓ Complete preprocessing and feature engineering pipeline
✓ Production-ready prediction module
✓ Detailed documentation and examples

---

**Status**: ✅ COMPLETE AND READY TO USE

**Next Step**: Run `cd src && python dataset_builder.py`
