#  EV Grid Stress & Risk Analysis

A machine learning pipeline to predict and visualize electrical grid stress caused by Electric Vehicle (EV) charging loads — helping utility providers proactively manage grid stability.

---

##  Problem Statement

As EV adoption accelerates, uncoordinated charging creates dangerous stress peaks on power grids. This project builds a predictive model to identify **high-risk grid zones** before failures occur, using historical load data and engineered features.

---

##  Screenshots

### Landing Page — EV Grid Stress Assessment
<img width="1842" height="825" alt="Screenshot 2026-04-19 222004" src="https://github.com/user-attachments/assets/e8bd3596-4b57-4fce-beba-2c8113d39859" />

> Select your Indian state and monitoring window (6h / 12h / 24h) to predict transformer overload risk across India's EV charging infrastructure — powered by Random Forest ML.

---

### Adjust Parameters Panel
<img width="948" height="851" alt="Screenshot 2026-04-19 222051" src="https://github.com/user-attachments/assets/941bdc9f-8b19-4541-ba25-658f8071f9c2" />

> Slide each parameter to reflect current grid conditions — Grid Load, Transformer Load, Charging Sessions, EV Population, Renewable Share, Charging Stations, and Fast Chargers — then run the assessment.

---

### Assessment Result
<img width="857" height="291" alt="Screenshot 2026-04-19 222104" src="https://github.com/user-attachments/assets/31758d81-4f51-4868-a9ca-5c1de7db2517" />

> The model returns a risk score and actionable analysis factors. In this example, Odisha scores 2/7 (Moderate Risk) due to grid load above safe threshold (7,221 MW > 7,200 MW) and insufficient charging infrastructure (375 stations < 500).

---

##  Project Structure

```
ev-grid-analysis/
│
src
|  ├── dataset_builder.py               # Generates synthetic/real EV grid dataset
|  ├── preprocess.py                    # Data cleaning and normalization
|  ├── feature_engineering.py           # Feature creation (load variance, peak hours, etc.)
|  ├── train_model.py                   # Model training (Random Forest / XGBoost)
|  ├── evaluate_model.py                # Evaluation metrics (accuracy, F1, confusion matrix)
|  ├── predict.py                       # Run predictions on new data
|  ├── run_pipeline.py                  # End-to-end pipeline runner
|
data
|  ├── ev_grid_stress_dataset.csv       # Raw dataset
|  ├── preprocessed_ev_grid_dataset.csv # Cleaned dataset
|  ├── engineered_features.csv          # Final feature set used for training
│  ├── ev_grid_risk_map.html            # Interactive risk visualization map
|
notebook
|  ├── ev_grid_analysis.ipynb           # Exploratory Data Analysis notebook
|  
│
└── README.md
```

---

##  Pipeline Overview

```
Raw Data → Preprocess → Feature Engineering → Train Model → Evaluate → Predict → Risk Map
```

| Step | Script | Description |
|------|--------|-------------|
| 1 | `dataset_builder.py` | Build/load EV grid dataset |
| 2 | `preprocess.py` | Handle nulls, normalize, encode |
| 3 | `feature_engineering.py` | Create load variance, peak hour flags |
| 4 | `train_model.py` | Train classifier on stress levels |
| 5 | `evaluate_model.py` | Print accuracy, F1, classification report |
| 6 | `predict.py` | Predict stress for new grid readings |
| 7 | `ev_grid_risk_map.html` | Visual risk map output |

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SidharthSatapathy04/ev-grid-analysis.git
cd ev-grid-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python run_pipeline.py
```

### 4. Or Run Step by Step
```bash
python preprocess.py
python feature_engineering.py
python train_model.py
python evaluate_model.py
python predict.py
```

---

##  Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
folium
xgboost
jupyter
```

> Install via: `pip install -r requirements.txt`

---

##  Key Features Engineered

- **Peak Hour Flag** — Whether the reading occurred during high-demand hours
- **Load Variance** — Rolling variance of grid load over time windows
- **EV Density Score** — Estimated EV concentration per grid zone
- **Stress Label** — Target variable: `Low / Medium / High` grid stress

---

##  Risk Map

Open `ev_grid_risk_map.html` in any browser to explore an interactive map showing:
- 🟢 Low stress zones
- 🟡 Medium stress zones
- 🔴 High stress / at-risk zones

---

##  Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 1.00 |
| F1 Score (Macro) | 1.00 |
| Precision | 1.00 |
| Recall | 1.00 |



##  Future Improvements

- [ ] Integrate real-time grid sensor data via API
- [ ] Add time-series forecasting (LSTM/Prophet)
- [ ] Expand to multi-city grid datasets

---

## 👤 Authors

**Sidharth Satapathy**
- GitHub: [@SidharthSatapathy04](https://github.com/SidharthSatapathy04)

**Biswaranjan Panda**
- GitHub: [@biswa2006](https://github.com/biswa2006)
  
**Tumulu Mihika**
- GitHub: [@TumuluMihika](https://github.com/TumuluMihika)

**Swagat Pradhan**
- GitHub: [@swagatpradhan2005](https://github.com/swagatpradhan2005)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
