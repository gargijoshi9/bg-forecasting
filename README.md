# Blood Glucose, Hormone, & Stress Dynamics (bg-forecasting)

This repository contains continuous glucose monitoring (CGM) data analysis and forecasting models, with specific focus on diurnal dynamics, menstrual cycle / hormonal phase associations, and acute psychosocial/academic stress events.

## Repository Structure

```text
bg-forecasting/
├── data/
│   ├── raw/                  # Apple Health export XML files (export.xml, export_cda.xml)
│   └── processed/            # Cleaned CGM datasets (glucose_with_cycle.csv)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis, Hormone & Stress Mapping
│   ├── 02_baseline_lstm.ipynb# Baseline LSTM forecasting model
│   └── 03_tft_model.ipynb    # Temporal Fusion Transformer (TFT) model
├── models/                   # Saved model checkpoints
├── results/                  # Generated figures, plots, and metric reports
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
