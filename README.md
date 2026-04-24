# Bangladesh Electricity Demand Forecasting
## IITG.AI Recruitment Task

## Objective --

The goal of this project is to predict **The next hour demand i.e. demand_mw** using historical consumption data, weather and economic indicators.

## Dataset Description --

The project uses three datasets that were provided:

* **Electricity Demand Data:** Hourly records of power demand and generation
* **Weather Data:** Temperature, humidity, and precipitation (hourly)
* **Economic Data:** Annual macroeconomic indicators

These datasets were aligned temporally to create a unified feature set for modeling.


## Key Challenges

* Representing time for non-sequential ML models
* Preventing data leakage in time-based features
* Handling extreme spikes in demand data
* Integrating multi-frequency data (hourly + yearly)

---

## Approach

### Data Preparation

* Converted timestamps to a consistent datetime format and fixed the duplicate and half hourly time stamps.
* Removed anomalies using IQR-based outlier detection. On top of that, unrealistically low demands that could not be caught by IQR were also removed. 

### Feature Engineering

* **Time Features:** Hour, day of week, month, weekend indicator
* **Cyclical Encoding:** Captured periodic patterns using sine/cosine transformations
* **Lag Features:** Previous demand values to model temporal dependency
* **Rolling Features:** Moving averages and variability to capture short-term trends
* **External Features:** Integrated weather and economic indicators

### Data Integrity

* Strict chronological train-test split
* All features constructed using past data only (no leakage)

## Modeling

The following models were trained and compared:

* LightGBM
* Random Forest
* XGBoost

LightGBM was selected as the final model due to its superior performance and efficiency.

---

## 📈 Results

* **MAPE:** ~2.46%
* **R² Score:** ~0.97


## Key Insights

* Electricity demand shows strong dependence on recent values (lag features dominate)
* Clear daily and seasonal consumption patterns exist
* Weather conditions significantly influence demand levels


## Limitations

* Extreme demand spikes remain difficult to predict


## Project Structure

```
electricity-demand-forecasting/
│
├── electricity_demand_forecasting.ipynb
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

1. Open the notebook
2. Run all cells sequentially
3. View predictions and evaluation metrics

---

Built as part of the *Predictive Paradox* recruitment task.
