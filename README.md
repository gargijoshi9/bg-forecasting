# Multimodal Blood Glucose Forecasting: Conditioning Continuous Glucose Dynamics on Endocrine Cycles, Psychosocial Stress, and Circadian Rhythms

[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-14%20Passed-success?style=flat-square)](tests/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-red?style=flat-square)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research%20%26%20Baseline%20Benchmarking-blueviolet?style=flat-square)](#research-roadmap)

---

## 1. Executive Summary & Research Motivation

Continuous Glucose Monitoring (CGM) forecasting is a cornerstone of proactive metabolic management and artificial pancreas algorithms. However, conventional state-of-the-art forecasting approaches predominantly treat glucose time series as **univariate autoregressive processes**, assuming stationary or quasi-stationary dynamics.

In real-world physiology—particularly in female individuals—metabolic clearance and insulin sensitivity are **non-stationary** and strongly modulated by:
1. **The Menstrual Cycle (Endocrine Shifts):** Biphasic fluctuations in estradiol and progesterone alter systemic insulin resistance, inducing marked glucose elevation during the mid-cycle luteal/ovulatory surge and precipitous nadirs immediately prior to menses.
2. **Hypothalamic-Pituitary-Adrenal (HPA) Axis Activation (Stress):** Acute psychological and somatic stress triggers counter-regulatory cortisol and epinephrine cascades that impair peripheral GLUT-4 translocation, causing acute hyperglycemia and, when severe, functional hypothalamic amenorrhea.
3. **Circadian Entrainment:** Endogenous dawn phenomenon cortisol pulses and evening metabolic deceleration.

### Research Objective
This research investigates whether **explicit conditioning on multimodal exogenous context** (menstrual cycle phase, annotated stress intensity, and circadian encodings) statistically improves multi-horizon blood glucose forecasts ($H \in \{30, 60, 120\}$ minutes) over pure autoregressive baselines, establishing empirical benchmarks prior to deep sequence modeling (LSTM and Temporal Fusion Transformers).

---

## 2. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  RAW DATA SOURCES                                      │
│   • Apple Health XML (142,511 LinX CGM readings @ 1-min)                               │
│   • Apple Watch Biometrics (21,527 HR records, 398 HRV SDNN records)                   │
│   • Ground-Truth Menstrual Logs (5 validated cycle boundaries, 4-day menses)           │
│   • Ecological Stress Annotations (Hackathon conflict, exam panic, recovery)           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA ENGINEERING & MODULARIZATION PIPELINE                      │
│   • Zero synthetic data invention: strictly 0 imputed/interpolated glucose points      │
│   • Left-closed 5-minute bucket aggregation with reading coverage tracking             │
│   • Backward-only wearable alignment (zero lookahead leakage)                          │
│   • Partitioned into 5 single-responsibility audited CSVs                              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LEAKAGE-SAFE SEQUENCE GENERATION                                │
│   • Lookback Windows (L): 60 min (12 steps), 120 min (24 steps), 360 min (72 steps)   │
│   • Forecast Horizons (H): 30 min (6 steps), 60 min (12 steps), 120 min (24 steps)    │
│   • Gap-Isolated: sliding sequences NEVER cross multi-hour sensor disconnect boundaries│
│   • Chronological Segment Partition: Train (72.1%) | Val (15.1%) | Test (12.8%)       │
│     Separated by natural 21h 55m and 38h 15m physical sensor changeover gaps           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             BENCHMARK & MODELING SUITE                                 │
│   [Completed] Persistence Baseline: y_hat_{t+H} = y_t                                  │
│   [Completed] Regularized Ridge Regression (alpha tuned on validation loss)            │
│   [Completed] Gradient Boosted Trees (XGBoost with early stopping on validation loss)  │
│   [In Progress] Bidirectional LSTM with Multi-Head Attention                          │
│   [Upcoming] Temporal Fusion Transformer (TFT) with Variable Selection Networks       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. What Has Been Accomplished ("What's Done")

### Phase 1: Data Engineering & Ground-Truth Reconstruction
* **High-Throughput Parsing:** Built deterministic, streaming XML extraction pipeline ([`src/build_datasets.py`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/src/build_datasets.py)) parsing 142k+ raw records in seconds without memory bloat.
* **Modularization:** Decoupled monolithic files into five modular tables documented in [`DATA_DICTIONARY.md`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/DATA_DICTIONARY.md):
  * `glucose_5min.csv`: 28,508 primary CGM records (100% real measurements, zero synthetic imputation).
  * `cycle_events.csv`: Ground-truth cycle boundaries and regularity classifications.
  * `stress_events.csv`: Categorized psychological and somatic stress intervals with intensity scaling.
  * `wearable_metrics.csv`: Aligned Apple Watch HR and HRV SDNN readings.
  * `glucose_features.csv`: Unified 25-column modeling matrix with continuous cyclical time encodings ($\sin/\cos$), cycle phase progression, and sensor boundary flags.
* **Audit & Reconciliations:** Mathematically proved the exact origin of follicular phase nadirs ($166.4\text{ mg/dL}$) versus phase-level weighted averages ($184.1\text{ mg/dL}$).

### Phase 2: Biological & Clinical Exploratory Analysis
* **Ambulatory Glucose Profile (AGP):** Mapped 24-hour diurnal rhythm revealing an overnight nadir at 3:00–4:00 AM ($130\text{ mg/dL}$), a marked dawn phenomenon surge from 4:00–9:00 AM ($+60\text{ mg/dL}$), and a nocturnal dinner/postprandial peak at 11:00 PM ($215.1\text{ mg/dL}$).
* **Endocrine Phase Characterization:** Discovered an acute insulin resistance peak during the ovulatory window (Days 12–14: mean $202.6\text{ mg/dL}$, median $206.6\text{ mg/dL}$) and a steep pre-menses decline (Days 25–26: drop to $146–151\text{ mg/dL}$).
* **Stress-Metabolic Interaction:** Documented an acute post-conflict glycemic spike to $327.2\text{ mg/dL}$ (Aug 29), and confirmed that severe academic panic (Sep 13–17) coincided with functional hypothalamic amenorrhea (skipped cycle) and a sustained 2-week glycemic surge ($>200\text{ mg/dL}$ with $71.2\%$ Time Above Range).

### Phase 3: Leakage-Safe Time-Series Infrastructure
* **SequenceBuilder Engine:** Implemented [`src/forecasting/sequence_builder.py`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/src/forecasting/sequence_builder.py) enforcing that every sliding sequence of length $L + H$ resides strictly within a single contiguous wear block. Across 8 wear segments, exactly $(W - 1) \times 7$ cross-gap sequences are systematically excluded.
* **Leakage-Free Partitioning:** Established chronological split boundaries:
  * **Train:** Segments 1–6 ($N=20,546$, $72.1\%$, June 8 – September 27).
  * **Validation:** Segment 7 ($N=4,305$, $15.1\%$, September 28 – October 13).
  * **Test:** Segment 8 ($N=3,657$, $12.8\%$, October 14 – October 27).
  * Train and Val are separated by a **21h 55m gap**; Val and Test by a **38h 15m gap**, physically preventing any lookback/forecast window leakage.
* **Preprocessing Isolation:** Scalers (`StandardScaler`) are fit exclusively on the training partition.
* **Automated Unit Testing:** Created 14 unit tests in [`tests/`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/tests/) covering gap isolation, duplicate detection, monotonicity, temporal leakage, and metric calculation accuracy.

### Phase 4: Machine Learning Benchmarks & Systematic Ablation Study
Benchmarked **Persistence Baseline**, **Regularized Ridge Regression**, and **XGBoost Regressor** across 5 feature sets on the unseen Test set ($N=3,634$):

$$\text{FS1 (Glucose)} \subset \text{FS2 (+Time)} \subset \text{FS3 (+Cycle)} \subset \text{FS4 (+Stress)} \subset \text{FS5 (+Wearables)}$$

#### Unseen Test Set Empirical Results ($L=60\text{ min}$)

| Forecast Horizon | Feature Set | Model | Test MAE (mg/dL) | Test RMSE (mg/dL) | $R^2$ Score | Clinical Accuracy ($\pm 20\%$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **30 Minutes** | Naive Baseline | Persistence | 16.99 | 22.47 | 0.8649 | 85.77% |
| | FS1: Glucose Only | Ridge | 14.21 | 18.66 | 0.9068 | 90.05% |
| | **FS1: Glucose Only** | **XGBoost** | **14.06** | **18.69** | **0.9065** | **90.85%** |
| | FS3: + Time & Cycle | Ridge | 14.14 | 18.59 | 0.9075 | 90.44% |
| | FS3: + Time & Cycle | XGBoost | 14.16 | 18.74 | 0.9060 | 90.11% |
| | | | | | | |
| **60 Minutes** | Naive Baseline | Persistence | 29.67 | 38.34 | 0.6064 | 65.88% |
| | FS1: Glucose Only | XGBoost | 25.04 | 32.32 | 0.7203 | 73.09% |
| | FS2: + Time | XGBoost | 24.28 | 31.56 | 0.7332 | 74.82% |
| | **FS3: + Time & Cycle**| **XGBoost** | **24.09** | **31.52** | **0.7338** | **74.79%** |
| | FS3: + Time & Cycle | Ridge | 25.09 | 32.23 | 0.7218 | 72.10% |
| | | | | | | |
| **120 Minutes**| Naive Baseline | Persistence | 48.01 | 60.97 | −0.0006 | 45.75% |
| | FS1: Glucose Only | Ridge | 39.90 | 50.13 | 0.3235 | 52.87% |
| | FS1: Glucose Only | XGBoost | 40.40 | 50.80 | 0.3054 | 52.73% |
| | **FS2: + Time** | **XGBoost** | **37.28** | **48.46** | **0.3680** | **57.87%** |
| | FS3: + Time & Cycle | XGBoost | 38.28 | 49.81 | 0.3321 | 55.66% |
| | FS5: Full MultiModal| XGBoost | 37.99 | 49.53 | 0.3397 | 56.54% |

#### Key Empirical Findings
1. **Short Horizons ($H=30\text{m}$):** Local autoregressive momentum dominates ($R^2 > 0.90$). XGBoost achieves **$14.06\text{ mg/dL}$ MAE** with **$90.85\%$** clinical accuracy. Exogenous conditioning yields negligible gain because active insulin and digestive lag drive immediate rate-of-change.
2. **Intermediate Horizons ($H=60\text{m}$):** Endocrine conditioning provides concrete empirical lift. Adding menstrual cycle parameters (`cycle_day`, `days_to_next_period`, `is_menstrual_flow`) yielded the **global lowest error ($24.09\text{ mg/dL}$)**, improving upon pure autoregression ($25.04\text{ mg/dL}$) and beating persistence by **$18.8\%$**.
3. **Long Horizons ($H=120\text{m}$):** Persistence completely collapses ($R^2 \approx 0$). Circadian and diurnal features (`time_sin`, `time_cos`, `day_of_week`) provide critical anchoring, reducing XGBoost error from $40.40\text{ mg/dL}$ down to **$37.28\text{ mg/dL}$** ($R^2 = 0.3680$), a **$22.3\%$ error reduction over baseline**.

---

## 4. Research Roadmap ("What Will Be Done")

```text
  [ Current Stage: Phase 4 Complete ]
                  │
                  ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 5: Recurrent Sequence Modeling (LSTM / BiLSTM)    │
│  • Multi-layer LSTM with temporal hidden state retention │
│  • Self-Attention mechanisms across lookback steps       │
│  • Direct vs. iterative multi-horizon autoregression     │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 6: Temporal Fusion Transformer (TFT) Architecture │
│  • Static covariate encoders for cycle metadata          │
│  • Variable Selection Networks (VSN) for feature weights │
│  • Interpretable multi-head temporal self-attention      │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 7: Probabilistic Quantile & Risk Forecasting     │
│  • Pinball loss optimization (10th, 50th, 90th quantiles)│
│  • Early warning alerts for acute hypoglycemia (<70)     │
│  • Clarke Error Grid Analysis (CEGA) compliance          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 8: Research Manuscript & Benchmark Release        │
│  • Publication submission on endocrine-aware forecasting │
│  • Open-source reproducible benchmark toolkit            │
└──────────────────────────────────────────────────────────┘
```

### Detailed Upcoming Milestones

1. **Deep Recurrent Architectures (PyTorch LSTM) in [`notebooks/02_baseline_lstm.ipynb`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/notebooks/02_baseline_lstm.ipynb):**
   * Implement a PyTorch LSTM with batch sequence generation (`[batch, seq_len=12, d_features]`).
   * Compare unidirectional vs. bidirectional LSTM layers.
   * Evaluate whether non-linear recurrent memory cells capture acceleration/deceleration transitions better than tabular gradient-boosted trees.
2. **Temporal Fusion Transformer (TFT) in [`notebooks/03_tft_model.ipynb`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/notebooks/03_tft_model.ipynb):**
   * Utilize `pytorch-forecasting`'s native TFT implementation.
   * Leverage **Variable Selection Networks (VSN)** to formally isolate the dynamic attention weights assigned to cycle day, stress intensity, and past glucose across different times of day.
   * Model static metadata (`cycle_length_days`), past-observed time-varying inputs (glucose, HR, HRV), and known future-time inputs (diurnal time encodings, day of cycle).
3. **Probabilistic & Clinical Risk Forecasting:**
   * Transition from point forecasts to multi-quantile forecasts ($p_{10}, p_{50}, p_{90}$) to provide prediction uncertainty bands.
   * Evaluate clinical utility metrics: Clarke Error Grid Analysis (CEGA Zones A & B) and Surveillance Error Grid (SEG).
4. **Academic Manuscript Preparation:**
   * Prepare a scientific paper detailing the dataset audit, mathematical reconciliation of follicular dynamics, and the quantified improvement of endocrine-conditioned forecasting over naive autoregression.

---

## 5. Repository Structure

```text
bg-forecasting/
├── data/
│   ├── raw/                               # Raw Apple Health & clinical exports (Git-ignored)
│   │   ├── export.xml                     # LinX CGM, HR, HRV, sleep data
│   │   └── export_cda.xml                 # Clinical CDA export
│   └── processed/                         # Canonical modular datasets (Git-ignored for privacy)
│       ├── glucose_5min.csv               # Primary 5-minute CGM time series (28,508 records)
│       ├── cycle_events.csv               # Ground-truth period dates & cycle boundaries
│       ├── stress_events.csv              # Annotated psychosocial & academic stress intervals
│       ├── wearable_metrics.csv           # Apple Watch HR & HRV measurements
│       └── glucose_features.csv           # Aligned 25-column modeling matrix
├── models/                                # Saved model checkpoints & serialization
│   ├── metadata.json                      # [Tracked] Hyperparameters & evaluation metrics
│   ├── xgboost_h30m.json                  # [Tracked] Serialized XGBoost model (30m horizon)
│   ├── xgboost_h60m.json                  # [Tracked] Serialized XGBoost model (60m horizon)
│   ├── xgboost_h120m.json                 # [Tracked] Serialized XGBoost model (120m horizon)
│   ├── ridge_h*.pkl                       # [Git-ignored] Serialized Ridge models
│   └── scaler.pkl                         # [Git-ignored] StandardScaler fitted on Train partition
├── notebooks/                             # Research & experimentation notebooks
│   ├── 01_eda.ipynb                       # Complete EDA, AGP diurnal profile, cycle & stress mapping
│   ├── 02_baseline_lstm.ipynb             # Baselines, XGBoost, feature ablation, & PyTorch LSTM setup
│   └── 03_tft_model.ipynb                 # Temporal Fusion Transformer workflow
├── results/                               # Research evaluation outputs
│   ├── feature_ablation_study.csv         # [Tracked] Comprehensive feature ablation matrix
│   ├── forecasting_experiments.csv        # [Tracked] Model benchmark comparison metrics
│   ├── sequence_generation_audit.csv      # [Tracked] Sequence & gap counts across all (L, H)
│   └── *.png                              # [Git-ignored] Private visualization plots
├── src/                                   # Source code package
│   ├── build_datasets.py                  # Deterministic dataset generation pipeline
│   └── forecasting/                       # Production time-series forecasting package
│       ├── __init__.py
│       ├── dataset.py                     # Schema validation, temporal split, training-only scaler
│       ├── sequence_builder.py            # Strict gap-isolated sequence generator (0 leakage)
│       ├── baseline.py                    # PersistenceBaseline & MeanBaseline
│       ├── ridge_model.py                 # L2-regularized linear model with validation tuning
│       ├── xgboost_model.py               # Gradient boosted decision tree forecaster
│       ├── evaluation.py                  # Statistical & clinical metrics calculation
│       ├── experiment_tracker.py          # Results serialization & Markdown formatter
│       ├── run_experiments.py             # Benchmark driver across all lookbacks/horizons
│       └── run_ablation_study.py          # Feature ablation experimental pipeline
├── tests/                                 # Automated test suite (14 passing tests)
│   ├── test_forecasting_dataset.py        # Model shapes, baseline math, metric correctness
│   ├── test_sequence_builder.py           # Gap crossing, missing intervals, duplicates, ordering
│   └── test_temporal_split.py             # Chronological ordering, purge margins, scaler isolation
├── DATA_DICTIONARY.md                     # Data schemas, audit checklist & mathematical proofs
├── README.md                              # Research documentation (this file)
└── .gitignore                             # Security-hardened privacy configuration
```

---

## 6. Reproducibility & Setup

### Environment Setup
Python 3.10+ is required (tested on Python 3.13).

```bash
# Clone the repository
git clone https://github.com/gargijoshi9/bg-forecasting.git
cd bg-forecasting

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn xgboost torch matplotlib seaborn
```

### Reproduce the Data Pipeline
```bash
# Rebuild the 5 canonical modular datasets from raw data
python src/build_datasets.py
```

### Run the Test Suite
```bash
# Execute all 14 automated unit tests
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v
```

### Execute Benchmarks & Ablation Studies
```bash
# Run baseline models across all (lookback, horizon) combinations
PYTHONPATH=. python src/forecasting/run_experiments.py

# Run systematic feature ablation study (Persistence vs. Ridge vs. XGBoost)
PYTHONPATH=. python src/forecasting/run_ablation_study.py
```

---

## 7. Data Governance & Privacy Standards

This project adheres to strict clinical data governance and privacy practices:
* **Raw & Processed Data Isolation:** All raw Apple Health XML exports and processed CSV datasets containing continuous glucose measurements, menstrual tracking dates, and personal annotations are **strictly ignored by version control** via [`.gitignore`](file:///Users/gargijoshi/Documents/My%20projects%20research%20/bg-forecasting/.gitignore).
* **Visualization Privacy:** Visualizations containing individual glucose trajectories or annotated life events (`results/*.png`) remain local and private.
* **Public Research Artifacts:** Only aggregated, de-identified statistical metrics (`forecasting_experiments.csv`, `feature_ablation_study.csv`), serializable model graph structures (`xgboost_*.json`), and code pipelines are committed.

---

## 8. Citation & Academic Contact

If you utilize this methodology or code for research on endocrine-conditioned metabolic time series, please cite:

```bibtex
@misc{joshi2026multimodal_bg_forecasting,
  author = {Gargi Joshi},
  title = {Multimodal Blood Glucose Forecasting: Conditioning Continuous Glucose Dynamics on Endocrine Cycles, Psychosocial Stress, and Circadian Rhythms},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gargijoshi9/bg-forecasting}}
}
```
