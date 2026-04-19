# EV GRID STRESS MONITORING AND PREDICTION SYSTEM

A comprehensive machine learning system to predict electrical grid stress caused by EV charging stations and determine overload risk across Indian states.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Components](#system-components)
- [How to Use](#how-to-use)
- [Models & Performance](#models--performance)
- [API Documentation](#api-documentation)

## 🎯 Project Overview

This system integrates multiple data sources to create a comprehensive solution for:
- **Predicting Grid Load** - Forecast electrical grid load in MW
- **Risk Assessment** - Determine overload risk (Low/Medium/High)
- **Mitigation Strategies** - Provide actionable recommendations
- **Visual Analytics** - Interactive dashboard and detailed analysis

### Key Datasets
- **25,000+ rows** of EV charging and grid metrics
- **All 28 Indian States** with state-specific data
- Multiple supplementary datasets for comprehensive analysis

## ✨ Features

### Core Capabilities
- ✓ Real-time grid load prediction using ML models
- ✓ Binary overload risk classification
- ✓ State-specific analysis and forecasting
- ✓ Historical pattern recognition and visualization
- ✓ Automated risk mitigation recommendations

### ML Models Trained
- Random Forest (Regression & Classification)
- XGBoost (Regression & Classification)
- Gradient Boosting (Both tasks)
- Logistic Regression (Classification)
- Support Vector Machines (Classification)

### Dashboard Features
- Interactive state selection
- Parameter adjustment interface
- Visual risk indicators with color coding
- Hourly and daily pattern analysis
- Actionable recommendations based on risk level

## 📁 Project Structure

```
EV_GRID_STRESS_PROJECT/
│
├── data/
│   ├── ev_grid_stress_dataset.csv          (25,000 rows - main dataset)
│   ├── ev_charging_infrastructure.csv      (28 states data)
│   ├── state_electricity_demand.csv        (Baseline demand)
│   ├── renewable_energy_by_state.csv       (Renewable capacity)
│   ├── ev_adoption_by_state.csv            (EV adoption data)
│   ├── final_ev_grid_dataset.csv           (Merged dataset)
│   └── india_states.geojson                (Geographic data)
│
├── src/
│   ├── dataset_builder.py                  (Generate synthetic datasets)
│   ├── preprocess.py                       (Data cleaning & normalization)
│   ├── feature_engineering.py              (Feature creation)
│   ├── train_model.py                      (Model training & selection)
│   ├── evaluate_model.py                   (Model evaluation)
│   └── predict.py                          (Prediction module)
│
├── models/
│   ├── best_load_model.pkl                 (Trained regression model)
│   └── best_risk_model.pkl                 (Trained classification model)
│
├── notebooks/
│   └── ev_grid_analysis.ipynb              (Analysis & visualization notebook)
│
├── dashboard/
│   ├── app.py                              (Flask application)
│   ├── templates/
│   │   ├── index.html                      (Home page)
│   │   ├── parameters.html                 (Parameter adjustment)
│   │   ├── result.html                     (Prediction results)
│   │   └── about.html                      (About page)
│   └── static/
│       └── style.css                       (Dashboard styling)
│
├── requirements.txt                        (Python dependencies)
└── README.md                               (This file)
```

## 💻 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- 2GB RAM minimum
- 500MB disk space

### Step 1: Clone/Extract Project
```bash
cd EV_GRID_STRESS_PROJECT
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended for First Run)
```bash
cd src

# 1. Generate datasets
python dataset_builder.py

# 2. Preprocess data
python preprocess.py

# 3. Create engineered features
python feature_engineering.py

# 4. Train models
python train_model.py
```

### Option 2: Run Dashboard Only (If Models Already Trained)
```bash
cd dashboard
python app.py
```

Then open your browser to: `http://localhost:5000`

## 📊 System Components

### 1. Dataset Builder (`src/dataset_builder.py`)
**Purpose:** Generate synthetic datasets for training

**Outputs:**
- `ev_grid_stress_dataset.csv` - Main training data (25,000 rows)
- `ev_charging_infrastructure.csv` - Infrastructure data
- `state_electricity_demand.csv` - Baseline demand
- `renewable_energy_by_state.csv` - Renewable capacity
- `ev_adoption_by_state.csv` - EV adoption data
- `india_states.geojson` - Geographic coordinates
- `final_ev_grid_dataset.csv` - Merged dataset

**Key Columns:**
- timestamp, state, hour, day_of_week
- total_charging_stations, fast_chargers, slow_chargers
- charging_sessions, energy_consumed_kwh
- grid_load_mw, voltage_v, frequency_hz
- transformer_load_percent, renewable_share_percent
- ev_population, charger_ev_ratio
- peak_hour_flag, overload_risk (target)

### 2. Preprocessing (`src/preprocess.py`)
**Tasks:**
- Handle missing values (median imputation for numerical, mode for categorical)
- Remove duplicate records
- Remove statistical outliers (z-score method)
- Extract timestamp features (month, quarter, weekend flag)
- Encode categorical variables (Label Encoding for state)
- Normalize numerical features (StandardScaler)

**Input:** `data/final_ev_grid_dataset.csv`
**Output:** `data/preprocessed_ev_grid_dataset.csv`

### 3. Feature Engineering (`src/feature_engineering.py`)
**Created Features:**
- `fast_charger_ratio` - Fast chargers / Total stations
- `charging_intensity` - Sessions / Total stations
- `grid_stress_index` - Transformer load × Sessions
- `slow_charger_ratio` - Slow chargers / Total stations
- `energy_per_session` - Energy consumed / Sessions
- `load_per_station` - Grid load / Stations
- `charger_efficiency` - Energy consumed / Stations
- `peak_load_impact` - Peak hour flag × Grid load
- `renewable_load_ratio` - Renewable share × Grid load
- `voltage_stability` - |Voltage - 230V| + |Frequency - 50Hz|
- `ev_demand_ratio` - EV population / Stations
- `fast_charger_stress` - Fast chargers × Transformer load

**Interaction Features:**
- peak_transformer_interaction
- charging_renewable_interaction
- ev_charger_demand

**Input:** `data/preprocessed_ev_grid_dataset.csv`
**Output:** `data/engineered_features.csv`

### 4. Model Training (`src/train_model.py`)
**Training Process:**
1. **Regression Models** (Predict Grid Load in MW):
   - Random Forest Regressor
   - XGBoost Regressor
   - Gradient Boosting Regressor
   - Model selection based on R² score

2. **Classification Models** (Predict Overload Risk):
   - Random Forest Classifier
   - XGBoost Classifier
   - Gradient Boosting Classifier
   - Logistic Regression
   - Support Vector Classifier
   - Model selection based on F1 score

**Hyperparameter Tuning:** GridSearchCV with cross-validation (3 folds)

**Input:** `data/engineered_features.csv`
**Output:** 
- `models/best_load_model.pkl`
- `models/best_risk_model.pkl`

### 5. Model Evaluation (`src/evaluate_model.py`)
**Regression Metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

**Classification Metrics:**
- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1 Score
- ROC-AUC Score
- Confusion Matrix

**Visualizations:**
- Confusion Matrix heatmap
- ROC Curve

### 6. Prediction Module (`src/predict.py`)
**Main Class:** `EVGridPredictor`

**Methods:**
- `predict_grid_load()` - Predict electricity demand (MW)
- `predict_overload_risk()` - Predict risk level (0/1)
- `get_risk_level()` - Convert to text (LOW/MEDIUM/HIGH)
- `predict()` - Complete prediction pipeline

**Risk Mitigation:** Get recommended actions based on risk level

**Example:**
```python
from predict import EVGridPredictor

predictor = EVGridPredictor()
result = predictor.predict({
    'charging_sessions': 800,
    'transformer_load_percent': 75,
    # ... other parameters
})
print(f"Grid Load: {result['predicted_grid_load_mw']} MW")
print(f"Risk Level: {result['risk_level']}")
```

### 7. Jupyter Notebook (`notebooks/ev_grid_analysis.ipynb`)
**Sections:**
1. Import libraries and configure visualization
2. Define Indian states
3. Create EV grid stress dataset (25,000 rows)
4. Create infrastructure dataset
5. Create electricity demand dataset
6. Create renewable energy dataset
7. Create EV adoption dataset
8. Create GeoJSON for mapping
9. Merge all datasets
10. Visualizations:
    - Charging sessions distribution
    - State-wise charger distribution
    - Grid load vs charging sessions
    - Transformer load distribution
    - Correlation heatmap
    - Time series patterns
    - Overload risk distribution

### 8. Flask Dashboard (`dashboard/app.py`)
**Routes:**
- `/` - Home page with state selection
- `/parameters` - Parameter adjustment interface
- `/predict` - Make predictions
- `/about` - System documentation
- `/api/get-states` - API endpoint for states
- `/api/get-state-data` - API endpoint for state metrics

**Templates:**
- `index.html` - Landing page
- `parameters.html` - Parameter input and adjustment
- `result.html` - Prediction results and recommendations
- `about.html` - System documentation

**Styling:** Responsive CSS with gradient backgrounds and animations

## 📈 How to Use the Dashboard

### Step 1: Start the Dashboard
```bash
cd dashboard
python app.py
```

### Step 2: Open Browser
Navigate to: `http://localhost:5000`

### Step 3: Home Page
- Select your target state from the dropdown
- Click "OPEN" button

### Step 4: Parameters Page
- View default parameters for the selected state
- Adjust charging sessions, transformer load, renewable share, etc.
- See historical statistics and hourly patterns
- Click "PREDICT" to forecast

### Step 5: Results Page
- View predicted grid load (MW)
- Check overload risk level (LOW/MEDIUM/HIGH)
- Color coding:
  - 🟢 GREEN = Low Risk
  - 🟡 YELLOW = Medium Risk
  - 🔴 RED = High Risk
- Read recommended mitigation strategies
- Modify parameters to test different scenarios

## 🤖 Models & Performance

### Regression Model (Grid Load Prediction)
**Best Model:** XGBoost/Random Forest
- **Target Variable:** grid_load_mw (500-5000 MW)
- **Typical R² Score:** 0.85-0.95
- **Typical RMSE:** 50-150 MW

### Classification Model (Overload Risk)
**Best Model:** XGBoost/Random Forest
- **Target Variable:** overload_risk (0=No Risk, 1=Overload)
- **Typical Accuracy:** 0.90-0.95
- **Typical F1 Score:** 0.85-0.92
- **Typical ROC-AUC:** 0.92-0.97

### Model Selection Strategy
Models are automatically compared using:
- Regression: R² Score (higher is better)
- Classification: F1 Score (higher is better)

Best models are saved for production use.

## 📋 API Documentation

### Home Endpoint
```
GET /
Returns: index.html
```

### Parameters Endpoint
```
POST /parameters
Form Data:
  - state: str (required)

Returns: parameters.html with adjusted values
```

### Prediction Endpoint
```
POST /predict
Form Data:
  - state: str
  - charging_sessions: float
  - total_charging_stations: float
  - fast_chargers: float
  - slow_chargers: float
  - transformer_load_percent: float
  - voltage_v: float
  - frequency_hz: float
  - renewable_share_percent: float

Returns: result.html with predictions
```

### States API
```
GET /api/get-states
Returns JSON: {"states": ["Andhra Pradesh", ...]}
```

### State Data API
```
GET /api/get-state-data?state=Gujarat
Returns JSON with hourly patterns and statistics
```

## 🔍 Risk Assessment

### LOW RISK (Transformer Load < 80%)
- ✓ Grid operating within safe parameters
- ✓ No immediate action required
- ✓ Continue regular monitoring

### MEDIUM RISK (Transformer Load 80-90%)
- ⚠ Grid approaching operational limits
- ⚠ Implement preventive measures:
  - Schedule fast charging during off-peak hours
  - Enable smart charging load distribution
  - Monitor transformer temperatures
  - Increase renewable energy supply

### HIGH RISK (Transformer Load > 90%)
- 🚨 Critical grid stress detected
- 🚨 Immediate actions required:
  - Reduce fast charging during peak hours
  - Enable emergency load scheduling
  - Increase transformer capacity
  - Implement demand response programs
  - Maximize renewable energy generation

## 📚 Usage Examples

### Run Complete Pipeline
```bash
# From project root
cd src
python dataset_builder.py      # Generate data
python preprocess.py            # Clean and normalize
python feature_engineering.py   # Create features
python train_model.py           # Train models
```

### Use Prediction Module
```python
from src.predict import EVGridPredictor, get_risk_mitigation_steps

predictor = EVGridPredictor()

# Prepare data
data = {
    'state_encoded': 5,
    'charging_sessions': 1000,
    'transformer_load_percent': 85,
    # ... 20+ more features
}

# Make prediction
result = predictor.predict(data)
print(f"Grid Load: {result['predicted_grid_load_mw']} MW")
print(f"Risk Level: {result['risk_level']}")

# Get recommendations
steps = get_risk_mitigation_steps(result['risk_level'])
for step in steps:
    print(f" - {step}")
```

### Analyze with Jupyter
```bash
cd notebooks
jupyter notebook ev_grid_analysis.ipynb
# Run all cells to generate visualizations
```

## 🔧 Troubleshooting

### Models Not Found
```
Error: Models not found. Train models first.
Solution: Run python src/train_model.py
```

### Data Files Missing
```
Error: Data files not found
Solution: Run python src/dataset_builder.py
```

### Port Already in Use
```
Error: Address already in use
Solution: Change port in app.py or kill existing process
```

### Python Version Issues
```
Error: Module not found
Solution: Check Python version (3.7+) and reinstall requirements
```

## 📞 Support

For issues or questions:
1. Check the inline documentation in source code
2. Review Jupyter notebook for examples
3. Check dashboard About page for detailed information

## 📝 License

This project is provided as-is for educational and research purposes.

## 👥 Contributors

Developed as a comprehensive ML systems project.

---

**Last Updated:** March 2026
**Version:** 1.0
**Status:** Production Ready
