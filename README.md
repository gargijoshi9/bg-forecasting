# Blood Glucose, Hormone, & Stress Dynamics (bg-forecasting)

This repository contains continuous glucose monitoring (CGM) data analysis and forecasting models, with specific focus on diurnal dynamics, menstrual cycle / hormonal phase associations, and acute psychosocial/academic stress events.

## Repository Structure

```text
bg-forecasting/
├── data/
│   ├── raw/                  # Apple Health export XML files (export.xml, export_cda.xml)
│   └── processed/            # Modular processed datasets:
│       ├── glucose_5min.csv      # Primary 5-minute CGM time series
│       ├── cycle_events.csv      # Period start dates and cycle boundaries
│       ├── stress_events.csv     # Manually annotated stress windows
│       ├── wearable_metrics.csv  # Apple Watch HR and HRV measurements
│       └── glucose_features.csv  # Final aligned modeling dataset
├── src/
│   └── build_datasets.py     # Deterministic pipeline script to generate all datasets
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis, Hormone & Stress Mapping
│   ├── 02_baseline_lstm.ipynb# Baseline LSTM forecasting model
│   └── 03_tft_model.ipynb    # Temporal Fusion Transformer (TFT) model
├── models/                   # Serialized model checkpoints & metadata:
│   ├── scaler.pkl                # Fitted StandardScaler on training partition
│   ├── ridge_h30m.pkl            # Trained Ridge model (30m horizon)
│   ├── ridge_h60m.pkl            # Trained Ridge model (60m horizon)
│   ├── ridge_h120m.pkl           # Trained Ridge model (120m horizon)
│   ├── xgboost_h30m.json         # Trained XGBoost model (30m horizon)
│   ├── xgboost_h60m.json         # Trained XGBoost model (60m horizon)
│   ├── xgboost_h120m.json        # Trained XGBoost model (120m horizon)
│   └── metadata.json             # Model training metadata & validation metrics
├── results/                  # Generated figures, plots, and metric reports
├── DATA_DICTIONARY.md        # Comprehensive schemas, audit checklist & reconciliation
└── README.md                 # Project documentation
```

## Dataset & Mapped Context

- **Source:** Apple Health CGM export (`LinX` sensor)
- **Timeframe:** June 8, 2026 – October 27, 2026 (~4.5 months)
- **Menstrual Cycles Mapped:**
  - May 27 (26d), June 22 (25d), July 17 (33d)
  - September: **Skipped (73-day cycle)**
  - October 31 (Menses resumed)
- **Key Stress Events Mapped:**
  - **Aug 26–29:** Hackathon, conflict/interpersonal fights, 1 AM Aug 29 big fight, physical exertion (spiked to 327.2 mg/dL).
  - **Aug 30 – Sep 4:** Post-hackathon anxiety.
  - **Sep 13–17:** Severe exam failure panic. Coincided with period due date $\rightarrow$ triggered functional amenorrhea (skipped menses) & acute hyperglycemia (Mean 194.1 mg/dL, Median 205.0 mg/dL, 64.3% TAR).
  - **Sep 18 – Oct 5:** Chronic anticipatory anxiety awaiting results.
  - **Oct 6+:** Recovery & stress resolution.

## Visualizations in `results/`

- `agp_diurnal_profile.png` — Ambulatory Glucose Profile across 24 hours.
- `hormone_cycle_glucose_pattern.png` — Glucose across cycle days (Ovulation peak & Pre-menses drop).
- `skipped_cycle_september_impact.png` — Trajectory of the skipped September cycle.
- `stress_timeline_glucose_mapping.png` — Full multi-month timeline overlaid with stress event bands.
- `aug29_hackathon_fight_micro_trace.png` — High-resolution trace of the 1 AM fight and 327 mg/dL morning surge.
- `stress_vs_glycemic_metrics.png` — Bar chart comparison of Mean, Median, and TAR% across stress events.
