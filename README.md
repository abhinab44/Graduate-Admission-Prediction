# Graduate Admission Prediction — Comparative Regression Study

> Predict a student's probability of graduate admission using GRE, TOEFL, CGPA, and more — benchmarking Linear Regression, Ridge, SVR, and Random Forest with leak-free preprocessing and cross-validation.

---

## GitHub Repository Description

> Comparative regression study on the Graduate Admissions dataset — EDA, MinMaxScaler with no data leakage, cross-validated Linear Regression, Ridge, SVR & Random Forest, MAE/RMSE/R² evaluation, residual analysis, feature importance, and a full model comparison summary.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Models Compared](#models-compared)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset Setup](#dataset-setup)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Overview

This project builds and benchmarks a complete regression pipeline to predict the **Chance of Admission** (0–1) for graduate school applicants. It emphasizes correct ML methodology — particularly leak-free scaling (fit only on training data) and honest cross-validated model comparison — to produce reliable, reproducible results.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) |
| **Records** | 400 student profiles |
| **Features** | 7 input features + 1 target |
| **Target** | `Chance of Admit` — continuous value between 0 and 1 |

### Features

| Feature | Description |
|---|---|
| `GRE Score` | Graduate Record Examination score (290–340) |
| `TOEFL Score` | Test of English as a Foreign Language score (92–120) |
| `University Rating` | Undergraduate university rating (1–5) |
| `SOP` | Strength of Statement of Purpose (1–5) |
| `LOR` | Strength of Letter of Recommendation (1–5) |
| `CGPA` | Undergraduate CGPA (out of 10) |
| `Research` | Research experience (0 = No, 1 = Yes) |

### Summary Statistics

| Statistic | GRE Score | TOEFL Score | Univ_Rating | SOP | LOR | CGPA | Research | Chance_of_Admit |
|---|---|---|---|---|---|---|---|---|
| count | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| mean | 316.81 | 107.41 | 3.09 | 3.40 | 3.45 | 8.60 | 0.55 | 0.724 |
| std | 11.47 | 6.07 | 1.14 | 1.01 | 0.90 | 0.60 | 0.50 | 0.143 |
| min | 290 | 92 | 1 | 1.0 | 1.0 | 6.80 | 0 | 0.34 |
| 25% | 308 | 103 | 2 | 2.5 | 3.0 | 8.17 | 0 | 0.64 |
| 50% | 317 | 107 | 3 | 3.5 | 3.5 | 8.61 | 1 | 0.73 |
| 75% | 325 | 112 | 4 | 4.0 | 4.0 | 9.06 | 1 | 0.83 |
| max | 340 | 120 | 5 | 5.0 | 5.0 | 9.92 | 1 | 0.97 |

---

## Pipeline Steps

```
1. Data Loading & EDA
        ↓
2. Preprocessing        (strip/rename columns, drop Serial No., check nulls)
        ↓
3. Train/Test Split      (80/20, random_state=42)  →  Train: 320 | Test: 80
        ↓
4. Feature Scaling       (MinMaxScaler — fit on train only, transform both)
        ↓
5. Model Training & Cross-Validation   (5-fold CV, 4 models)
        ↓
6. Test Set Evaluation   (MAE, RMSE, R²)
        ↓
7. Actual vs Predicted + Residual Analysis
        ↓
8. Feature Importance    (Linear coefficients + Random Forest importances)
        ↓
9. Results Summary
```

---

## Models Compared

| Model | Key Characteristic |
|---|---|
| **Linear Regression** | Baseline — assumes full linearity |
| **Ridge Regression** (α=1.0) | L2 regularization — handles multicollinearity |
| **SVR (RBF kernel)** | Captures non-linear patterns; sensitive to scaling |
| **Random Forest** | Ensemble tree method; captures feature interactions |

All models are evaluated with **5-fold cross-validation (CV MAE)** and independently tested on the held-out test set using **MAE, RMSE, and R²**.

---

## Results

| Model | CV MAE (mean) | CV MAE (±std) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| **Linear Regression** | **0.0451** | ±0.0049 | **0.0480** | **0.0679** | **0.8212** |
| Ridge (α=1.0) | 0.0455 | ±0.0047 | 0.0488 | 0.0695 | 0.8129 |
| Random Forest | 0.0507 | ±0.0049 | 0.0497 | 0.0704 | 0.8081 |
| SVR (RBF) | 0.0648 | ±0.0033 | 0.0665 | 0.0804 | 0.7496 |

- **Best model by R²:** Linear Regression (R² = 0.8212)
- **Most important feature:** `CGPA` — dominant across all models
- Ridge closely matches Linear Regression, confirming the near-linear nature of the problem
- Mean residual: **−0.0123** (near zero — no systematic bias)
- Std residual: **0.0668**

---

## Project Structure

```
graduate-admission-prediction/
│
├── data/
│   └── Admission_Predict.csv          # Dataset (download from Kaggle)
├── graduate_admission_prediction.ipynb  # Main notebook (full pipeline)
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies
```

---

## Dataset Setup

The dataset is **not included** in this repository due to Kaggle's terms of use. Follow these steps to set it up:

### Option A — Manual Download (Recommended for local use)

1. Go to [Kaggle — Graduate Admissions](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
2. Download `Admission_Predict.csv`
3. Place it inside a `data/` folder in the project root:

```
graduate-admission-prediction/
└── data/
    └── Admission_Predict.csv
```

The notebook reads it with:

```python
df = pd.read_csv('data/Admission_Predict.csv')
```

### Option B — Kaggle API (Automated)

```bash
pip install kaggle
kaggle datasets download -d mohansacharya/graduate-admissions
unzip graduate-admissions.zip -d data/
```

### Option C — Google Colab

Upload the CSV file directly in Colab, then update the path:

```python
from google.colab import files
uploaded = files.upload()                    # Upload Admission_Predict.csv
df = pd.read_csv('Admission_Predict.csv')    # No subfolder needed in Colab
```

Or mount Google Drive:

```python
from google.drive import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/data/Admission_Predict.csv')
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## How to Run

**Option 1 — Jupyter Notebook (local):**

```bash
git clone https://github.com/abhinab44/graduate-admission-prediction.git
cd graduate-admission-prediction

# Place Admission_Predict.csv inside the data/ folder
mkdir data
mv /path/to/Admission_Predict.csv data/

jupyter notebook graduate_admission_prediction.ipynb
```

**Option 2 — Google Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> Remember to upload the CSV or mount Drive, and update the `pd.read_csv` path accordingly (see [Dataset Setup](#dataset-setup)).

---

## Key Findings

- **CGPA** is the single strongest predictor of admission probability (correlation = 0.87) — agreed upon by both Linear Regression coefficients and Random Forest feature importances (Gini ≈ 0.72).
- **GRE Score** and **TOEFL Score** are the second and third most correlated features (0.80 and 0.79 respectively), but contribute far less than CGPA in tree-based models.
- **Linear Regression and Ridge** perform nearly identically (R² of 0.8212 vs 0.8129), confirming the relationship between features and admission chance is largely linear.
- **Random Forest** captures non-linear interactions (R² = 0.8081) but does not dramatically outperform linear models — suggesting diminishing returns from complexity on this dataset.
- **SVR (RBF)** performs worst in this configuration (R² = 0.7496), likely due to suboptimal default hyperparameters (no grid search applied).
- **Residuals are centered around zero** (mean = −0.0123) with no systematic bias — the model does not consistently over- or under-predict.
- **Correct scaling protocol** (MinMaxScaler fit only on training data, then applied to both splits) prevents data leakage — a common mistake in many public implementations of this dataset.
- High inter-feature correlations exist (e.g., GRE ↔ TOEFL = 0.84, GRE ↔ CGPA = 0.83), indicating multicollinearity — which Ridge regression is specifically designed to handle, though it confers minimal benefit here.

---

## Concepts Demonstrated

- Proper train/test splitting and **leak-free preprocessing**
- `MinMaxScaler` applied fit-on-train-only methodology
- **5-fold cross-validation** for honest model comparison
- Multi-metric evaluation: MAE, RMSE, and R²
- Residual analysis for model diagnostics
- Feature importance from both linear coefficients and tree-based ensembles
- Side-by-side model benchmarking with visual comparison

---

## License

This project is open-source under the [MIT License](LICENSE).

---

*Built with Python 3.10 · scikit-learn · pandas · NumPy · Matplotlib · Seaborn*
