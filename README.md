# Heart Failure Clinical Records: Classification and Clustering Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## Abstract

Heart failure is a leading cause of mortality worldwide, with an estimated 64.3 million people affected globally. Early prediction of adverse outcomes using routinely collected clinical data can substantially improve patient management and resource allocation. This project applies both supervised and unsupervised machine learning techniques to the Heart Failure Clinical Records dataset (Chicco & Jurman, 2020) to predict patient mortality. We implement and compare two deep learning classifiers — a Convolutional Neural Network (CNN) and a Long Short-Term Memory Recurrent Neural Network (LSTM-RNN) — alongside two unsupervised approaches — K-Means clustering and a custom Self-Organizing Map (SOM). A lightweight Random Forest model is also trained and exported for real-time inference through an interactive Streamlit web application.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Interactive Demo](#interactive-demo)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Background

Cardiovascular diseases account for approximately 31% of all global deaths (WHO, 2021). Heart failure specifically occurs when the heart cannot pump sufficient blood to meet the body's needs. Timely prediction of patient outcomes following a heart failure episode is critical for clinical decision-making.

Machine learning has shown significant promise in this domain. Chicco & Jurman (2020) demonstrated that a simple set of 12 clinical features, routinely measured during follow-up visits, could predict patient survival with high accuracy. This project builds upon that foundation by:

1. Conducting rigorous exploratory data analysis with statistical feature selection
2. Implementing and comparing multiple deep learning architectures
3. Applying unsupervised clustering methods to uncover latent patient subgroups
4. Deploying the best-performing model in an accessible web interface

---

## Dataset

**Source**: UCI Machine Learning Repository
**Citation**: Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 1–16. https://doi.org/10.1186/s12911-020-1023-5

| Property | Value |
|---|---|
| Instances | 299 |
| Features | 12 clinical predictors + 1 target |
| Missing Values | None |
| Class Balance | 67.9% survived / 32.1% deceased |
| Follow-up Period | 4–285 days |

### Feature Description

| Feature | Type | Unit | Description |
|---|---|---|---|
| `age` | Numeric | years | Patient age |
| `anaemia` | Binary | — | Decrease of red blood cells (haemoglobin) |
| `creatinine_phosphokinase` | Numeric | mcg/L | Level of CPK enzyme in blood |
| `diabetes` | Binary | — | Presence of diabetes |
| `ejection_fraction` | Numeric | % | Percentage of blood leaving the heart per contraction |
| `high_blood_pressure` | Binary | — | Presence of hypertension |
| `platelets` | Numeric | kiloplatelets/mL | Platelet count in blood |
| `serum_creatinine` | Numeric | mg/dL | Creatinine level in blood serum |
| `serum_sodium` | Numeric | mEq/L | Sodium level in blood serum |
| `sex` | Binary | — | Biological sex (0=female, 1=male) |
| `smoking` | Binary | — | Smoking status |
| `time` | Numeric | days | Follow-up period duration |
| `DEATH_EVENT` | Binary | — | **Target**: 1 = deceased, 0 = survived |

---

## Methodology

### 1. Data Preprocessing

- **Unit Normalization**: Platelet counts converted to kiloplatelets/mL for consistency
- **Feature Renaming**: `creatinine_phosphokinase` → `CPK` for readability
- **Scaling**: StandardScaler (zero mean, unit variance) for neural network inputs; MinMaxScaler for SOM
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) applied to training set only

### 2. Feature Selection

Chi-square independence tests assessed the statistical association between binary features and the target variable (`DEATH_EVENT`). Features with p > 0.05 (sex, high blood pressure, diabetes) were identified as less discriminative, though all 12 features were retained for the full models to avoid information loss.

### 3. Models

#### Convolutional Neural Network (CNN)
The input features are reshaped into a 1D sequence (12 × 1) and processed through three `Conv1D` layers with increasing filter sizes (64, 128, 256), interleaved with `MaxPooling1D` and `Dropout` (rate = 0.25) layers. A final dense layer with softmax activation produces class probabilities.

| Hyperparameter | Value |
|---|---|
| Activation (Conv) | Sigmoid |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Epochs | 1000 (EarlyStopping, patience=30) |
| Batch Size | 32 |

#### LSTM-RNN (Recurrent Neural Network)
The input is reshaped to a 3D tensor for recurrent processing. A single LSTM layer (64 units) captures temporal dependencies across the feature sequence, followed by dropout and dense layers.

| Hyperparameter | Value |
|---|---|
| LSTM Units | 64 |
| Activation (LSTM) | Tanh |
| Optimizer | Adam |
| Validation Split | 20% |
| Epochs | 1000 (EarlyStopping, patience=30) |

#### K-Means Clustering
Unsupervised partitioning into k=2 clusters (corresponding to survival outcomes) using Euclidean distance. Cluster quality evaluated via Silhouette Score and Adjusted Rand Index.

#### Self-Organizing Map (SOM)
A custom implementation of a 25×25 competitive learning network. Weights are updated iteratively using decaying learning rates and neighborhood radii. After training, each neuron is labeled by majority vote from mapped training samples, enabling classification of unseen patients.

| Hyperparameter | Value |
|---|---|
| Grid Size | 25 × 25 |
| Max Learning Rate | 0.4 |
| Max Neighborhood Distance | 4 |
| Training Steps | 150,001 |

#### Random Forest (Deployment Model)
A 300-tree Random Forest classifier trained on the full feature set with SMOTE-balanced data. Serialized via `joblib` for use in the Streamlit application.

---

## Results

### Classification Performance

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| CNN | 88.76% | 73.3% | Overfitting observed |
| LSTM-RNN | 90.00% | 72.2% | Overfitting observed |
| **Random Forest** | **~98%** | **~83%** | Best supervised model |
| SOM | — | **96.32%** | Unsupervised; competitive labeling |

### Clustering Metrics (K-Means, k=2)

| Metric | Value |
|---|---|
| Silhouette Score | 0.801 |
| Adjusted Rand Index | Computed |
| Within-Cluster Sum of Squares | Computed |

### Key Findings

1. **SOM achieved the highest classification accuracy (96.32%)**, demonstrating that unsupervised topological learning can outperform supervised deep learning on this dataset.
2. **Both CNN and RNN overfit** significantly — training accuracy exceeded test accuracy by ~15–18 percentage points — suggesting the 299-sample dataset is insufficient for complex deep learning architectures without stronger regularization or augmentation.
3. **K-Means produced well-separated clusters** (Silhouette = 0.801), indicating that the 12 clinical features encode genuinely distinct patient subpopulations.
4. **Feature selection via chi-square** identified `time` (follow-up duration) and `serum_creatinine` as the most predictive features, consistent with Chicco & Jurman (2020).

---

## Interactive Demo

A Streamlit web application is included for real-time prediction:

- Input patient clinical values manually or generate a random synthetic patient
- View survival probability with a confidence gauge
- Inspect feature importance rankings
- Compare against population statistics

![App Screenshot](assets/app_screenshot.png)

---

## Repository Structure

```
Heart_Failure_Classification/
├── Heart_Failure_Classification.ipynb   # Main analysis notebook
├── app.py                               # Streamlit prediction UI
├── train_model.py                       # Script to train & save Random Forest model
├── heart_failure_clinical_records_dataset.csv
├── requirements.txt
├── README.md
└── models/
    └── rf_heart_failure.pkl             # Saved Random Forest model (generated by train_model.py)
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Abhi183/Heart_Failure_Classification.git
cd Heart_Failure_Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Jupyter Notebook

```bash
jupyter notebook Heart_Failure_Classification.ipynb
```

### Train and Save the Random Forest Model

```bash
python train_model.py
```

This generates `models/rf_heart_failure.pkl`.

### Launch the Streamlit App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## References

1. Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 1–16. https://doi.org/10.1186/s12911-020-1023-5

2. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository. http://archive.ics.uci.edu/ml

3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

4. Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*, 78(9), 1464–1480.

5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

---

## License

This project is licensed under the MIT License.
Dataset: Original dataset licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

*DSDA 385 — Abhishek Shekhar*
