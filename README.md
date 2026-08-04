# Blood Glucose Forecasting (bg-forecasting)

This repository contains code and notebooks for forecasting blood glucose levels.

## Repository Structure

```text
bg-forecasting/
├── data/
│   ├── raw/          # OhioT1DM XML files go here
│   └── processed/    # cleaned CSVs
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_baseline_lstm.ipynb  # Baseline LSTM model
│   └── 03_tft_model.ipynb      # Temporal Fusion Transformer (TFT) model
├── models/                     # Saved model artifacts
├── results/                    # Outputs, plots, evaluation results
└── README.md                   # Project documentation (this file)
```

## Getting Started

1. Place your raw OhioT1DM XML dataset files in the `data/raw/` directory.
2. Run through the notebooks in numerical order to perform exploratory analysis, train the baseline LSTM model, and develop the TFT model.
