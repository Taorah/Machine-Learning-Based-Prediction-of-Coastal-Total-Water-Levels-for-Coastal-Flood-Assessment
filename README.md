# Machine Learning-Based Prediction of Coastal Total Water Levels Using BiLSTM Networks

This repository contains MATLAB code for predicting **coastal Total Water Levels (TWL)** using Bidirectional Long Short-Term Memory (BiLSTM) neural networks. The workflow combines observed tidal data and non-tidal residuals to model nonlinear tide–surge interactions and improve total water level prediction in hydraulically constrained estuaries for coastal application. Panama City, Florida, is used as the demonstration site.

This script also includes baseline models (Linear Superposition and Bagged Tree Ensembles) for comparison.

---

## 📘 Features of This Repository

- Loads and preprocesses NOAA and ADCIRC water-level datasets  
- Computes nonlinear TWL residuals  
- Builds supervised learning inputs using lag features  
- Constructs sequence-to-one BiLSTM inputs using sliding windows  
- Trains a BiLSTM network **in parallel (6 cores)**  
- Produces TWL predictions from Linear, Bagged Tree, and BiLSTM models  
- Computes RMSE and correlation for each model  
- Generates time-series and scatter comparison plots  

---

## 📁 Repository Contents

```
├── Improved_LSTM.m        # Main script containing the full workflow
├── README.md              # Documentation
├── noaa_*.csv             # NOAA water-level dataset (uploaded by user)
└── Panama_City_*.csv      # ADCIRC NTR dataset
```

---

## 🧠 How the Model Works

### 1. Linear Prediction
TWL_linear = tide + ntr

### 2. Bagged Regression Tree Ensemble
Learns nonlinear corrections to the linear model.

### 3. BiLSTM Prediction (Main Model)
- Sequence length = 72 hours  
- Inputs per timestep = tide + NTR  
- Output = nonlinear residual at time t + 72  

The BiLSTM learns:
- tidal cycles and lags  
- storm-driven memory effects  
- nonlinear tide–surge interactions  

---

## 🔧 Requirements

### Software
- MATLAB R2021b or newer  
- Deep Learning Toolbox  
- Statistics and Machine Learning Toolbox  
- Parallel Computing Toolbox (optional but recommended)

### Hardware
- CPU with at least 4–6 cores for parallel training  
- Minimum 8 GB RAM (16 GB recommended)

---

## 📥 Input Data Required

Place your input CSV files in any directory and update the folder path inside the script:

- noaa_*.csv → NOAA tide + TWL dataset  
- Panama_City_*.csv → ADCIRC non-tidal residual dataset  

Expected columns:

NOAA:
time, TWL, tide

ADCIRC:
Datetime, NTR

---

## 🚀 How to Run

1. Open MATLAB  
2. Load the script:
```
open Improved_LSTM.m
```

3. Update dataset folder path:
```
folder = "D:\...\Processed";
```

4. Run the script:
```
run('Improved_LSTM.m')
```

---

## 📊 Outputs

### Console Metrics
Example:
```
Linear RMSE: 0.195 m
Bagged Tree RMSE: 0.139 m
BiLSTM RMSE: 0.130 m
BiLSTM Correlation: 0.895
```

### Plots
- Time-series comparisons  
- Scatter plots with 1:1 line  
- Model performance evaluation  

---

## 🛠 Troubleshooting

- **File not found** → check folder path  
- **GPU issues** → set ExecutionEnvironment to "cpu"  
- **Out-of-memory** → reduce MiniBatchSize  
- **Parallel issues** → restart pool using:
```
delete(gcp('nocreate'))
parpool(4)
```

---

## 📚 Citation
Taorah (2025).  
*Machine Learning-Based Prediction of Coastal Total Water Levels Using BiLSTM Networks.*  
GitHub Repository.

---

## 🙌 Contact
GitHub: https://github.com/Taorah  
Email: yusuftaorah@gmail.com
