# Report — Predictive Paradox
## Methodology, Decisions & Findings

---

## 1. Handling Missing Data & Outliers

### The duplicate timestamp problem

When I first loaded the PGCB dataset, I found 432 duplicate timestamps. After digging in, I realised these weren't errors in the traditional sense — they were overlapping hourly (`:00`) and half-hourly (`:30`) readings that had been recorded together. Simply dropping duplicates would throw away real measurements.

My fix: a weighted merge. For each duplicate pair, I computed:

```
merged_value = (2 × hourly_reading + 1 × half_hourly_reading) / 3
```

This gives more trust to the on-the-hour reading (which is what we care about) while still incorporating the half-hour information. After merging, any gaps in the hourly timeline were filled using time-based linear interpolation.

### Outlier detection

The raw `demand_mw` column had values as extreme as 117,000 MW (maximum) and 6 MW (minimum). For context, Bangladesh's grid realistically operates between roughly 3,000 and 17,000 MW — so these values are clearly erroneous.

I went with IQR-based detection, but used a factor of 3 instead of the standard 1.5. The reason: the legitimate demand range is already wide (~14,000 MW spread), and a factor of 1.5 would clip values that are actually real. I also added a hard floor at 3,000 MW based on domain knowledge.

**Why replace instead of drop?**
Dropping outlier rows creates holes in the time series. Those holes would then make `lag_1h`, `lag_2h` etc. reference the wrong timestamps for every row that follows. Instead, I replaced flagged values with a 5-hour backward rolling mean — this fills the gap smoothly while only using past information.

One important detail: the IQR bounds were computed **only from the training set** and then applied to both train and test. This way, cleaning thresholds aren't contaminated by future data.

---

## 2. Temporal Feature Engineering

Since tree-based models treat every row as completely independent, the model has no idea that row 1000 comes one hour after row 999. I had to encode the concept of "time" manually, as explicit columns.

### Cyclic calendar encoding

The naive approach — using raw integers for hour (0–23) or month (1–12) — creates a false discontinuity. Hour 23 and hour 0 look as far apart as hour 0 and hour 12, when in reality they are adjacent. Sine/cosine projection onto a unit circle fixes this:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Same treatment for month and day of week. This lets the model understand cyclical proximity correctly.

### Bangladesh-specific flags

A few things I learned from EDA that justified custom binary features:

- **`is_fri_sat`**: Bangladesh's weekend is Friday–Saturday, not Saturday–Sunday. The heatmap of demand by day and hour shows a clear and consistent dip on these two days — factories, offices, schools are closed.
- **`peak_hours`**: The 17:00–23:00 window is visibly the highest-demand period every day. People return home, lights go on, ACs kick in.
- **`peak_month`**: June through October (monsoon and pre-monsoon) show consistently higher demand — heat and humidity drive AC usage.

### Lag features

Lags are how a tree model "looks back" at history without being a sequential model. Each lag creates a new column where the value at row `t` is the demand from `t-n` hours ago:

| Feature | Offset | What it captures |
|---------|--------|-----------------|
| `lag_1h` | t−1 | Immediate past — strongest predictor (~0.95 correlation with target) |
| `lag_2h` | t−2 | Short-term momentum |
| `lag_3h` | t−3 | Short-term trend |
| `lag_24h` | t−24 | Same hour yesterday — daily seasonality |

All lags use `.shift(n)` with n ≥ 1, so no row ever sees its own current value or anything from the future.

### Rolling statistics

`rolling_2h` and `rolling_6h` are backward-looking rolling means that smooth out noise and capture the average level of demand over recent windows. Both use `.shift(1).rolling(n).mean()` — the shift ensures the window starts at t−1, not t.

### Weather features

Merged on `datetime` from `weather_data.xlsx`. Temperature, humidity, precipitation and cloud cover all influence demand through heating/cooling loads. Interpolation was done separately on train and test — no cross-boundary leakage.

### Economic feature

`electric_power_consumption_per_capita` from the World Bank CSV was merged by calendar year. The idea: this indicator captures Bangladesh's long-term structural growth in electricity demand. It stays constant within each year (as expected for an annual figure) and shifts gradually across years.

---

## 3. Feature Importance & Key Insights

After training, LightGBM's feature importances (normalised to 100%) revealed the following:

```
lag_1h                    ~9.0%   ← strongest single predictor
temperature               ~7.5%
hour_sin                  ~7.5%
coal                      ~7.0%   ← industrial/base load proxy
lag_24h                   ~6.7%
hour_cos                  ~6.5%
lag_2h                    ~5.8%
rolling_6h                ~5.5%
humidity                  ~5.2%
... rest spread across remaining features
```

**What this tells us:**

`lag_1h` leading the list makes complete sense — electricity demand evolves smoothly, and the last hour is the strongest signal of what the next hour will look like. A MAPE of 2.46% suggests the model is capturing this well.

Temperature ranking second alongside hour_sin confirms that weather and time-of-day are nearly co-equal drivers of demand in Bangladesh. The summer AC load is real and large.

`coal` appearing as a top feature is interesting. It's a generation-side variable, but it correlates strongly with industrial demand patterns — it's essentially acting as a proxy for how much heavy industry is running at any given hour.

`lag_24h` at ~6.7% validates the decision to include a daily seasonality lag. The grid repeats its daily cycle reliably, and the model is using it.

The binary flags (`is_fri_sat`, `peak_hours`, `peak_month`) rank lower — not because they're unimportant, but because the cyclic encodings and lag features already carry much of that information. The model learns to trust the richer continuous features more.

Overall the importance is well-distributed across all 23 features. No single feature is doing all the work, which suggests the pipeline is capturing demand from multiple angles without redundancy.

---

## 4. Validation & Final Score

| | Detail |
|--|--------|
| Train period | 2015–2023 (76,296 rows) |
| Test period | 2024 (8,784 rows) — never seen during training |
| Primary metric | MAPE |
| **Final test MAPE** | **2.46%** |
| R² | 0.97 |
| MAE | 273.9 MW |

A MAPE of 2.46% means the model's predictions are within ~2.5% of actual demand on average across the full year 2024. For reference, sub-5% MAPE is considered operationally useful for grid management; sub-3% is strong performance for classical ML without any sequential architecture.

The residual distribution is approximately normal and centered near zero — no systematic bias. The scatter plot of residuals vs. predicted values shows uniform spread across the demand range, meaning the model performs consistently at both low and high demand levels.

---

*Submitted for IITG.ai Club Recruitment — Predictive Paradox*
