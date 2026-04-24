# Predictive Paradox — Power Demand Forecasting

**IITG.ai Recruitment Task**

A machine learning pipeline to forecast the next hour's electricity demand on Bangladesh's national grid — built using only classical ML, no deep learning allowed.

---

## My Results

| Metric | Score |
|--------|-------|
| **MAPE** | 2.46% |
| **R² Score** | 0.97 |
| **MAE** | 273.9 MW |
| **Best Model** | LightGBM |

---

## What's the problem?

Getting electricity demand forecasting wrong is costly in both directions — overestimate and you waste generation capacity, underestimate and the grid becomes unstable. The goal here is to predict `demand_mw` (next hour's grid demand) using historical demand, weather, and macroeconomic data.

The catch: no LSTMs, no Transformers, no ARIMA, no Prophet. Classical ML only. That means I had to manually engineer everything that a sequential model would learn on its own — lags, rolling averages, cyclic time encodings — and bake the concept of "time" directly into the feature set.

---

## Dataset

Three files were provided:

| File | What it contains |
|------|-----------------|
| `PGCB_date_power_demand.xlsx` | Hourly demand & generation data — this is the target |
| `weather_data.xlsx` | Hourly temperature, humidity, precipitation, cloud cover etc. |
| `economic_full_1.csv` | Annual World Bank macroeconomic indicators for Bangladesh |

---

## How I approached it

### 1. Cleaning the data

The raw PGCB dataset was messy — 432 duplicate timestamps from overlapping hourly and half-hourly readings. Instead of just dropping them, I did a weighted merge giving 2/3 weight to the on-the-hour reading and 1/3 to the half-hour reading. Missing timestamps after merging were filled with time-based interpolation.

For outliers, the raw demand ranged from 6 MW to 117,000 MW — both completely unrealistic for Bangladesh's grid. I used IQR-based detection with a factor of 3 (wider than the usual 1.5, because the valid range itself is wide) plus a hard floor of 3,000 MW. Crucially, I replaced outliers with a 5-hour backward rolling mean rather than dropping them — dropping rows would create holes that break the lag features downstream.

### 2. Feature Engineering

This was the most important step. Since tree models see every row independently, I had to encode time explicitly:

**Cyclic calendar features** — raw hour integers make hour 0 and hour 23 look far apart when they're actually adjacent. Sine/cosine encoding fixes that.

**Bangladesh-specific flags** — Friday and Saturday are the weekend here, not Saturday/Sunday. Peak demand hours are 17:00–23:00 (confirmed by heatmap analysis). Monsoon months (June–October) consistently show higher demand.

**Lag features** — the model's "memory":
- `lag_1h`, `lag_2h`, `lag_3h` — recent history
- `lag_24h` — same hour yesterday, captures daily seasonality

**Rolling features** — `rolling_2h` and `rolling_6h` backward means to capture short-term trends.

**Weather** — merged hourly on datetime. Temperature and humidity are big drivers of AC load.

**Economic indicator** — `electric_power_consumption_per_capita` from World Bank, merged by year to capture long-term structural growth in demand.

### 3. Train / Test Split

| Split | Period | Rows |
|-------|--------|------|
| Train | 2015–2023 | 76,296 |
| Test  | 2024 | 8,784 |

Strictly chronological. The model never sees 2024 during training. Outlier bounds and weather interpolation were computed separately on train and test to prevent any cross-boundary leakage.

### 4. Models

Trained and compared three models:

| Model | MAPE | R² |
|-------|------|----|
| **LightGBM** | **2.46%** | **0.97** |
| XGBoost | 2.47% | ~0.97 |
| Random Forest | higher | lower |

LightGBM won by a thin margin. The near-identical scores between LightGBM and XGBoost suggest the feature set is doing most of the heavy lifting.

---

## What drives demand the most?

Top features by importance:

| Feature | Importance | Why it makes sense |
|---------|------------|-------------------|
| `lag_1h` | ~9% | Demand doesn't jump suddenly — the last hour is the best single predictor |
| `temperature` | ~7.5% | AC load is a massive driver in Bangladesh summers |
| `hour_sin` | ~7.5% | Time of day shapes everything |
| `coal` | ~7% | Acts as a proxy for industrial/base load |
| `lag_24h` | ~6.7% | Same-hour-yesterday patterns are strong |

Importance is well spread across all 23 features — no single variable is doing all the work, which is a good sign.

---

## Constraints followed

- Classical ML only (LightGBM, XGBoost, Random Forest)
- No deep learning (LSTMs, Transformers)
- No autoregressive packages (ARIMA, Prophet)
- Strictly chronological train/test split
-  Zero data leakage all features computed from past data only

---

Submitted by - Shivanshi

R. No - 250106066

BSBE'29
