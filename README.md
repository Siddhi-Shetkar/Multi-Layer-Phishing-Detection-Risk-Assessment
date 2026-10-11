# Multi-Layer Phishing Detection & Risk Assessment

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)](https://pytest.org)
[![Accuracy](https://img.shields.io/badge/accuracy-96.95%25-success.svg)](results/reports/model_comparison.csv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

**Multi-Layer Phishing Detection & Risk Assessment** is a machine learning-driven cybersecurity platform engineered to identify malicious URLs and quantify their threat risk with transparent, explainable reasoning.

Traditional security filters frequently rely on static blocklists that lag behind zero-day phishing campaigns. This system introduces a defense-in-depth architecture: it extracts **28 lexical, structural, and domain-level characteristics** from raw URLs in real time without executing network requests, evaluates them against trained machine learning classifiers, and layers an empirical **heuristic risk assessment engine** to classify links into **LOW**, **MEDIUM**, or **HIGH** risk categories.

---

## 👥 Project Contributors

| Engineer | Focus Area | Scope of System Implementation | Status |
|---|---|---|---|
| **Siddhi Shetkar** | Data Engineering & Analysis | Dataset ingestion, data cleaning, 28-feature lexical extraction, exploratory data analysis (EDA), and leakage-free stratified dataset preparation. | **Completed & Verified** |
| **Sankeerthana Thoviti** | Machine Learning & Security | Machine learning model development (5 algorithms), evaluation & comparison, multi-layer risk assessment engine, end-to-end prediction CLI, web console, and unit test suite. | **Completed & Verified** |

---

## 🏗️ Multi-Layer System Architecture

```text
                                  User URL Input
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   Layer 1: Offline Lexical Extraction     │
                  │   - 28 structural, host & keyword features│
                  │   - 100% offline (zero network execution) │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    Layer 2: Machine Learning Inference    │
                  │   - HistGradientBoosting (Optimal Model)  │
                  │   - Predicts class (0=Legit, 1=Phish)     │
                  │   - Outputs calibrated probability P(phish)│
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   Layer 3: Heuristic Threat Assessment   │
                  │   - Configurable risk thresholds          │
                  │   - Escalation on critical red flags:     │
                  │     * Raw IP hosting                      │
                  │     * Authority '@' spoofing              │
                  │     * Shortened URL obfuscation           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          Comprehensive Diagnosis          │
                  │   - Classification: LEGITIMATE / PHISHING │
                  │   - Risk Level: LOW / MEDIUM / HIGH       │
                  │   - Risk Score: 0.0 - 100.0               │
                  │   - Observed Warnings & Safe Signals      │
                  │   - Security Advisory Disclaimer          │
                  └───────────────────────────────────────────┘
```

---

## 📊 Machine Learning Model Evaluation & Comparison

All models were trained on **160,000 URLs** and independently evaluated on a strictly held-out, stratified test set of **40,000 URLs** (20,000 legitimate and 20,000 phishing). All models utilize the identical 28 engineered feature columns.

| Algorithm | Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score | ROC-AUC | False Positives | False Negatives | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Histogram Gradient Boosting** *(Selected)* | **96.95%** | **99.00%** | **94.85%** | **96.88%** | **0.9876** | **192** (0.96%) | 1,030 | 8.85s |
| **Random Forest** (100 Trees) | 96.90% | 98.99% | 94.77% | 96.84% | 0.9876 | 193 (0.96%) | 1,046 | 2.25s |
| **Decision Tree** | 96.86% | 98.99% | 94.69% | 96.79% | 0.9868 | 193 (0.96%) | 1,063 | 0.42s |
| **Support Vector Machine** (Linear + Calibrated) | 96.50% | 98.48% | 94.45% | 96.42% | 0.9829 | 292 (1.46%) | 1,110 | 11.92s |
| **Logistic Regression** (StandardScaler) | 96.49% | 98.63% | 94.29% | 96.41% | 0.9828 | 262 (1.31%) | 1,142 | 1.35s |

### Model Selection Rationale
**Histogram-based Gradient Boosting** (`HistGradientBoostingClassifier`) was selected as the optimal production estimator based on the composite **Cybersecurity Score** (weighted F1-Score and Phishing Recall):
1. **Minimizes False Alarms:** Achieves **99.00% precision**, resulting in fewer than 1 false positive per 100 legitimate URLs (only 192 false positives across 20,000 legitimate test URLs).
2. **Superior Discrimination:** Highest ROC-AUC score (**0.9876**) and highest overall F1-score (**96.88%**).
3. **Calibrated Probabilities:** Native, well-calibrated posterior probabilities allow direct integration with threshold-based risk scoring.
4. **Fast Inference:** Microsecond inference speed per URL.

---

## 🛡️ Multi-Layer Risk Assessment Framework

The risk assessment system bridges machine learning statistical probability with cybersecurity threat context:

### Threshold Architecture
- **LOW Risk (`P < 0.35`):** Standard URL conforming to typical naming patterns without critical threat flags.
- **MEDIUM Risk (`0.35 <= P < 0.70`):** Borderline statistical uncertainty, or low probability links that trigger suspicious structural anomalies.
- **HIGH Risk (`P >= 0.70`):** Strong statistical likelihood of malicious intent, credential harvesting, or infrastructure spoofing.

### Heuristic Escalation Rules
To defend against adversarial evasion techniques where individual lexical features might barely fall under the detection boundary:
- **Raw IP Address Hosting:** Immediately escalates risk to at least **MEDIUM** or **HIGH**.
- **Authority Override Symbol (`@`):** Escalates risk due to high probability of browser domain spoofing.
- **URL Shortener Services (`bit.ly`, `tinyurl`):** Raises risk alert due to concealed final destination.

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/Multi-Layer-Phishing-Detection-Risk-Assessment.git
cd Multi-Layer-Phishing-Detection-Risk-Assessment
pip install -r requirements.txt
```

### 2. Check a Single URL
Analyze any URL instantly in the terminal:
```bash
python main.py --url "https://www.google.com"
```
Or test a suspicious link:
```bash
python main.py --url "http://192.168.1.100:8080/admin/login.php?user=admin@bank.com"
```

**Terminal Output Example:**
```text
==============================================================================
 MULTI-LAYER PHISHING DETECTION & RISK ASSESSMENT REPORT
==============================================================================
 Analyzed URL:     http://192.168.1.100:8080/admin/login.php?user=admin@bank.com
 Classification:   PHISHING (Confidence: 99.1%)
 Risk Level:       [HIGH RISK - MALICIOUS THREAT]
 Risk Score:       96.0 / 100.0
 Phishing Prob:    [########################] 99.12%
------------------------------------------------------------------------------
 Detection Reasoning:
   * Model estimated high phishing probability (99.1%), exceeding the high-risk threshold (70%).

 Critical Security Flags:
   ! Direct IP address hosting
   ! Authority '@' symbol spoofing

 Observed Suspicious Indicators (5):
   - Raw IP Address Hostname: URL hosts destination directly on an IP address.
   - Non-Standard Port: The URL explicitly specifies port 8080.
   - Sensitive Authentication Lure: Contains keywords ('login').
   - Authority Override Symbol ('@'): Presence of '@' can spoof true destination.
   - Unencrypted HTTP Protocol: URL transmits traffic unencrypted.

 Positive Baseline Indicators (0):
------------------------------------------------------------------------------
 Key Extracted Features:
   Length: 60 | Host Length: 13 | Dots: 4 | Subdomains: 0 | Digits: 14 | Special Chars: 11
------------------------------------------------------------------------------
 DISCLAIMER: This assessment is generated via statistical Machine Learning classification
 and lexical URL heuristics. It serves as an advisory decision-support signal and does not
 guarantee absolute safety or malice. Never enter credentials into unverified domains.
==============================================================================
```

### 3. Interactive Shell Mode
Analyze multiple links continuously in a REPL interface:
```bash
python main.py --interactive
```

### 4. Train & Evaluate Models from Scratch
To retrain all 5 models and re-generate metrics, confusion matrices, and ROC plots:
```bash
python main.py --train
```

### 5. Run Data Engineering Pipeline
To re-run data cleaning, 28-feature extraction, and dataset splitting:
```bash
python main.py --prepare-data
```

### 6. SENTINEL Web Dashboard & REST API
Start the integrated SENTINEL cybersecurity intelligence server:
```bash
python server.py --port 8000
# or alternatively
python main.py --serve
```
Then open your web browser at:
```text
http://127.0.0.1:8000/
```

#### Supported REST API Endpoints:
- **`GET /api/health`**: Returns server status, loaded model artifact, and the 28-feature schema.
- **`GET /api/model-metrics`**: Returns empirical evaluation metrics, classification reports, and algorithm comparison records for all 5 ML models.
- **`POST /api/predict`**: Accepts `{"url": "https://..."}` JSON payload. Extracts 28 lexical features, executes ML model inference, runs the heuristic risk assessment engine, and returns a structured threat verdict with composite risk score, warning flags, and reassuring signals.

---

## 🧪 Automated Test Suite

The project includes an automated test suite with **49 test cases** covering feature extraction, ML model candidate training, cybersecurity metrics, model serialization, risk thresholds, heuristic escalation, API server endpoints, and end-to-end inference:

```bash
pytest -v
```

```text
tests/test_api.py ...                                                    [  6%]
tests/test_feature_extraction.py ................                        [ 38%]
tests/test_model.py ..............                                       [ 67%]
tests/test_risk_assessment.py .....                                      [ 77%]
tests/test_prediction_system.py ...........                              [100%]

============================== 49 passed in 5.63s ==============================
```

---

## 📂 Repository Structure

```text
Multi-Layer-Phishing-Detection-Risk-Assessment/
├── dataset/
│   ├── raw/
│   │   └── phishing_url_dataset.csv            # 200,000 raw input records
│   └── processed/
│       ├── cleaned_dataset.csv                 # Cleaned dataset (200k rows)
│       ├── url_features.csv                    # Full 28-feature matrix
│       ├── feature_descriptions.md             # Complete data dictionary
│       └── splits/
│           ├── X_train.csv                     # 80% training features (160,000 x 28)
│           ├── y_train.csv                     # 80% training labels (160,000)
│           ├── X_test.csv                      # 20% testing features (40,000 x 28)
│           └── y_test.csv                      # 20% testing labels (40,000)
├── models/
│   ├── best_model.joblib                       # Serialized optimal estimator
│   ├── best_model_metadata.json                # Feature schema & audit metrics
│   └── phishing_model.joblib                   # Backup deployment artifact
├── notebooks/
│   ├── data_preprocessing.ipynb               # Data cleaning & deduplication
│   ├── feature_extraction.ipynb                # Feature engineering (28 indicators)
│   ├── exploratory_data_analysis.ipynb         # EDA distribution exploration
│   └── model_training.ipynb                    # Model training & comparative evaluation
├── results/
│   ├── reports/
│   │   ├── eda_report.md                       # Empirical EDA findings & statistical metrics
│   │   ├── model_comparison.csv                # Tabular performance metrics for all models
│   │   └── model_evaluation_report.json        # Complete audit evaluation report
│   └── visualizations/
│       ├── 01_class_counts.png                 # EDA visualization charts (1-11)
│       ├── ...
│       ├── 11_suspicious_patterns_by_class.png
│       ├── model_comparison.png                # Model comparison bar chart
│       ├── roc_curves.png                      # Overlaid ROC curves
│       ├── confusion_matrix_gradient_boosting.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_decision_tree.png
│       ├── confusion_matrix_support_vector_machine.png
│       └── confusion_matrix_logistic_regression.png
├── src/
│   ├── __init__.py                             # Core source package
│   ├── data_loader.py                          # Dataset ingestion & label mapping
│   ├── preprocessing.py                        # Cleaning & deduplication pipeline
│   ├── feature_extraction.py                   # 28-feature extraction logic
│   ├── data_preparation.py                     # Stratified train/test splitting
│   ├── eda.py                                  # EDA visualizer & reporter
│   ├── model.py                                # ML training, evaluation, comparison
│   └── risk_assessment.py                      # Multi-layer risk engine & scoring
├── tests/
│   ├── test_api.py                             # REST API server & endpoint tests
│   ├── test_feature_extraction.py              # Feature extraction unit tests
│   ├── test_model.py                           # Model training & serialization tests
│   ├── test_risk_assessment.py                 # Risk engine & threshold tests
│   └── test_prediction_system.py               # End-to-end integration tests
├── app.js                                      # SENTINEL client-side controller & local history
├── index.html                                  # SENTINEL glassmorphic security dashboard console
├── main.py                                     # Master CLI execution pipeline
├── requirements.txt                            # Python environment dependencies
├── server.py                                   # SENTINEL REST API backend & static server
├── styles.css                                  # Glassmorphic CSS design system
└── README.md                                   # Comprehensive project documentation
```

---

## 🔒 Security & Privacy Notice

- **Zero-Network Analysis:** The prediction and risk assessment pipeline functions entirely offline through lexical URL string parsing. At no point does the application make HTTP/HTTPS connections, send DNS queries, or download content from submitted URLs.
- **Safety Disclaimer:** Machine learning predictions represent statistical risk estimates based on lexical indicators. They serve as an advisory defense-in-depth indicator and should be combined with enterprise DNS filters and endpoint protection.

---

## 📄 License
This project is licensed under the MIT License.
