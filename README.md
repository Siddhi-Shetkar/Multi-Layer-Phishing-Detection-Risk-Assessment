# Multi-Layer Phishing Detection & Risk Assessment — Part 1

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](tests/)

An enterprise-grade, offline-first data engineering, URL feature extraction, and exploratory analysis pipeline for detecting phishing and malicious websites.

This repository implements **Part 1** of the two-part system:
- **Part 1 (Implemented)**: Dataset ingestion, quality audit, label normalization, lexical/structural feature extraction, exploratory data analysis, and preparation of leakage-free train/test splits.
- **Part 2 (Separate Phase)**: Multi-layer ML model training (tree ensembles, neural networks), model evaluation, risk assessment scoring, and final URL inference.

---

## Table of Contents

1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Dataset Overview & Verified Label Mapping](#2-dataset-overview--verified-label-mapping)
3. [Project Directory Structure](#3-project-directory-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [URL Feature Engineering](#6-url-feature-engineering)
7. [Exploratory Data Analysis (EDA)](#7-exploratory-data-analysis-eda)
8. [Dataset Outputs & Split Specifications](#8-dataset-outputs--split-specifications)
9. [How to Run the Pipeline](#9-how-to-run-the-pipeline)
10. [Running the Test Suite](#10-running-the-test-suite)
11. [Limitations of URL-Only Detection](#11-limitations-of-url-only-detection)
12. [Reusing Extracted Features in Part 2](#12-reusing-extracted-features-in-part-2)

---

## 1. Project Overview & Objectives

Phishing attacks remain the primary initial access vector for cyber attacks, credential harvesting, and financial fraud. While modern phishing attacks employ dynamic landing pages and cloaking, the Uniform Resource Locator (URL) itself contains critical lexical, structural, and topological signals.

**Key Objectives of Part 1:**
- **Zero-Network Static Extraction**: Extract all features purely via offline lexical parsing (`urllib.parse`, `ipaddress`, and Public Suffix List rules). No URLs are contacted or fetched over the network.
- **Data Quality & Label Integrity**: Rigorously verify ground truth labels, resolve conflicting or malformed URLs, and enforce standard binary encoding (`0` = Legitimate, `1` = Phishing).
- **Leakage Prevention**: Generate stratified, reproducible train/test splits without target leakage or duplicate URL overlap.
- **Modular Architecture**: Provide production-ready modules in `src/` that can be imported directly into notebooks or future ML pipelines.

---

## 2. Dataset Overview & Verified Label Mapping

The project utilizes the primary dataset `phishing_url_dataset.csv` located in the root workspace and mirrored in `dataset/raw/phishing_url_dataset.csv`.

- **Total Records**: 200,000 URLs
- **Total Columns**: 2 (`URL`, `Label`)
- **Missing Values**: 0 null records
- **Class Balance**: 100,000 samples per class (perfect 50/50 balance)

### Verified Label Encoding

In the raw CSV:
- **Raw `0`**: Contains malicious domains, typosquats, dynamic DNS, and credential phishing forms (e.g., `facilitaire-controle.tk`, `openseas.firebaseapp.com`, `paypal.aisgwb.org`). **Represents Phishing**.
- **Raw `1`**: Contains verified government, educational, and official corporate domains (e.g., `health.state.mn.us`, `mtt.gov.rs`, `snohd.org`). **Represents Legitimate**.

**Standardized Encoding Enforced Throughout the Project:**
- **`0`**: **Legitimate**
- **`1`**: **Phishing**

The pipeline applies a verified bijective mapping (`Raw 0 -> 1`, `Raw 1 -> 0`), preventing inverted classifier predictions.

---

## 3. Project Directory Structure

```text
Multi-Layer-Phishing-Detection/
├── dataset/
│   ├── raw/
│   │   └── phishing_url_dataset.csv          # Mirrored original dataset
│   └── processed/
│       ├── cleaned_dataset.csv               # Normalized, cleaned dataset
│       ├── url_features.csv                  # Full dataset with 28 features + label
│       ├── feature_descriptions.md           # Comprehensive feature data dictionary
│       └── splits/
│           ├── X_train.csv                   # Training feature matrix (80%)
│           ├── X_test.csv                    # Testing feature matrix (20%)
│           ├── y_train.csv                   # Training target labels
│           └── y_test.csv                    # Testing target labels
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Dataset loading, inspection, & label mapping
│   ├── preprocessing.py                      # Reusable cleaning and validation pipeline
│   ├── feature_extraction.py                 # Offline URL feature engineering (28 features)
│   ├── data_preparation.py                   # Split generation and feature validation
│   └── eda.py                                # Plot generation and EDA reporting
├── notebooks/
│   ├── data_preprocessing.ipynb              # Preprocessing interactive walkthrough
│   ├── feature_extraction.ipynb              # Feature extraction demonstration
│   └── exploratory_data_analysis.ipynb       # Interactive visual analysis
├── results/
│   ├── visualizations/                       # 11 publication-grade PNG charts
│   │   ├── 01_class_counts.png
│   │   ├── 02_class_distribution_percentage.png
│   │   ├── 03_missing_values.png
│   │   ├── 04_duplicate_analysis.png
│   │   ├── 05_url_length_distribution.png
│   │   ├── 06_hostname_length_distribution.png
│   │   ├── 07_url_length_by_class.png
│   │   ├── 08_structural_features_by_class.png
│   │   ├── 09_correlation_heatmap.png
│   │   ├── 10_extracted_features_distributions.png
│   │   └── 11_suspicious_patterns_by_class.png
│   └── reports/
│       └── eda_report.md                     # Data-backed EDA report
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py                   # Unit tests for loading and mapping
│   ├── test_preprocessing.py                 # Unit tests for cleaning pipeline
│   ├── test_feature_extraction.py            # Unit tests for feature extractors
│   └── test_data_preparation.py              # Unit tests for validation and splitting
├── main.py                                   # End-to-end pipeline runner
├── requirements.txt                          # Production dependencies
├── README.md                                 # Complete project documentation
└── .gitignore                                # Git ignore file
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.9+ (Windows, Linux, or macOS)
- pip

### Installation Steps

1. Clone or navigate to the workspace root:
   ```bash
   cd "d:/Siddhi/ML Project"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Data Preprocessing Pipeline

Implemented in [`src/preprocessing.py`](file:///d:/Siddhi/ML%20Project/src/preprocessing.py), the cleaning pipeline executes a sequential 10-step process:

1. **Missing Value Pruning**: Removes null, NaN, or non-string URLs.
2. **Whitespace Normalization**: Trims surrounding leading/trailing whitespace.
3. **Empty String Removal**: Drops records that become empty after trimming.
4. **Malformed URL Filtering**: Uses `urllib.parse.urlsplit` to filter unparseable strings, invalid control characters, or non-URI structures.
5. **Target Label Normalization**: Maps verified raw classes to standardized binary values (`0` = Legitimate, `1` = Phishing).
6. **Exact Duplicate Removal**: Drops redundant records sharing identical URLs and labels.
7. **Conflict Resolution**: Identifies URLs that appear with contradictory labels and purges them to protect training set integrity.
8. **Repeated URL Deduplication**: Ensures every URL in the processed dataset is strictly unique.
9. **Integrity Verification**: Asserts zero nulls, zero NaNs, and strictly binary labels `{0, 1}`.
10. **Storage**: Saves the cleaned data to `dataset/processed/cleaned_dataset.csv`.

---

## 6. URL Feature Engineering

Implemented in [`src/feature_extraction.py`](file:///d:/Siddhi/ML%20Project/src/feature_extraction.py), extracting **28 numeric features** per URL across 3 categories:

### A. Length and Structural Features (15 Features)
- `url_length`: Total character count of the URL.
- `hostname_length`: Character length of the parsed hostname / FQDN.
- `path_length`: Length of the path string.
- `query_length`: Length of query parameters following `?`.
- `num_dots`: Total count of `.` in the URL.
- `num_hyphens`: Total count of `-` in the URL (common in combosquatting).
- `num_underscores`: Total count of `_` characters.
- `num_slashes`: Total count of `/` characters (reflects server path nesting).
- `num_question_marks`: Count of `?` characters.
- `num_equal_signs`: Count of `=` characters (parameter density).
- `num_ampersands`: Count of `&` characters.
- `num_at_symbols`: Count of `@` characters (credential injection indicator).
- `num_percent_symbols`: Count of `%` characters (hex obfuscation).
- `num_digits`: Count of numeric digits `0`-`9`.
- `num_special_characters`: Count of all non-alphanumeric characters.

### B. Hostname and Domain Features (9 Features)
- `has_ip_address`: 1 if hostname is an IPv4 or IPv6 address, 0 otherwise.
- `hostname_has_digits`: 1 if hostname contains digits, 0 otherwise.
- `hostname_has_hyphen`: 1 if hostname contains a hyphen, 0 otherwise.
- `num_subdomains`: Estimated count of subdomains using Public Suffix List rules via `tldextract`.
- `has_multiple_subdomains`: 1 if `num_subdomains > 1`, 0 otherwise.
- `is_https`: 1 if scheme is `https`, 0 otherwise.
- `has_explicit_port`: 1 if an explicit port number is present, 0 otherwise.
- `has_unusual_port`: 1 if port is specified and outside standard HTTP (80) or HTTPS (443), 0 otherwise.
- `num_query_parameters`: Number of parsed key-value pairs in the query string.

### C. Suspicious-Pattern Features (4 Features)
- `has_url_shortener`: 1 if host matches known URL shorteners (`bit.ly`, `t.co`, `tinyurl.com`, etc.), 0 otherwise.
- `has_suspicious_keyword`: 1 if URL contains authentication lure terms (`login`, `verify`, `update`, `account`, `password`, `banking`, etc.), 0 otherwise.
- `has_encoded_characters`: 1 if percent-encoded characters are present, 0 otherwise.
- `has_at_symbol`: 1 if `@` symbol is present, 0 otherwise.
*(Note: `has_ip_address` is reused from Hostname features rather than creating a duplicate column).*

---

## 7. Exploratory Data Analysis (EDA)

The pipeline automatically generates **11 high-resolution visualization charts** in `results/visualizations/` and a comprehensive data-backed report in `results/reports/eda_report.md`:

| Visualization File | Description |
| :--- | :--- |
| `01_class_counts.png` | Bar plot of legitimate vs phishing sample counts. |
| `02_class_distribution_percentage.png` | Donut chart displaying class balance proportions. |
| `03_missing_values.png` | Quality audit of nulls, empty URLs, and malformed strings. |
| `04_duplicate_analysis.png` | Breakdown of duplicates, conflicts, and unique records (log scale). |
| `05_url_length_distribution.png` | Overall URL length histogram and KDE density curve. |
| `06_hostname_length_distribution.png` | Overall hostname length distribution. |
| `07_url_length_by_class.png` | Class-conditional density curves and boxplot comparisons. |
| `08_structural_features_by_class.png` | Comparative bar chart of dots, slashes, hyphens, and symbols. |
| `09_correlation_heatmap.png` | Triangular correlation matrix between extracted features and target label. |
| `10_extracted_features_distributions.png` | Subplot grid showing path length, query length, digit count, and host length. |
| `11_suspicious_patterns_by_class.png` | Subdomain counts and prevalence rates of suspicious indicators. |

---

## 8. Dataset Outputs & Split Specifications

The processed data is exported to `dataset/processed/`:

- **`url_features.csv`**: Contains all 200,000 URLs with the 28 engineered numeric features and the standardized target `label`.
- **`feature_descriptions.md`**: Full Markdown data dictionary detailing each feature's definition and security rationale.
- **`splits/X_train.csv` & `splits/y_train.csv`**: 80% stratified training partition (160,000 samples).
- **`splits/X_test.csv` & `splits/y_test.csv`**: 20% stratified testing partition (40,000 samples).

### Split Characteristics
- **Random Seed**: `42` (ensures exact reproducibility).
- **Stratification**: Enforces exactly 50% Legitimate and 50% Phishing in both train and test splits.
- **Leakage Prevention**: Feature matrices `X_train` and `X_test` contain strictly numeric features. Target label `label` and raw URL string are completely decoupled.

---

## 9. How to Run the Pipeline

Execute the full pipeline from the root directory:

```bash
python main.py
```

### CLI Arguments & Options

```bash
python main.py [OPTIONS]

Options:
  -d, --data-path PATH      Path to custom raw CSV dataset (default: auto-detects)
  -o, --output-dir PATH     Output directory for processed data (default: dataset/processed)
  -r, --results-dir PATH    Output directory for visualizations and reports (default: results)
  -s, --sample-size INT     Process a sample subset for fast dry-runs (e.g., -s 10000)
  -b, --batch-size INT      Batch size for feature extraction progress (default: 10000)
  --test-size FLOAT         Fraction reserved for test split (default: 0.2)
  --random-state INT        Random seed for splitting (default: 42)
  -h, --help                Show help message and exit
```

---

## 10. Running the Test Suite

The test suite contains 30 comprehensive unit and integration tests covering data loading, column validation, preprocessing, feature extraction on edge cases, and split reproducibility.

Execute pytest:
```bash
pytest -v
```

---

## 11. Limitations of URL-Only Detection

While static URL feature analysis is fast, non-intrusive, and safe:

1. **Cloud & CDN Hosting Abuse**: Attackers regularly host phishing forms on reputable platforms (`workers.dev`, `pages.dev`, `firebaseapp.com`, `github.io`). Their hostnames appear benign under standard domain reputation filters.
2. **Shortener Indirection**: Services like `bit.ly` obscure the true landing URL until an HTTP redirect is performed.
3. **Dynamic Evasion**: Cloaking techniques inspect user-agents, IP ranges, and referrers to serve benign content to security scanners and malicious content to human victims.
4. **Valid TLS Deployment**: Over 80% of modern phishing sites utilize free SSL certificates (Let's Encrypt / Cloudflare), meaning `is_https=1` does not indicate safety.

These limitations demonstrate why **Part 2** will integrate multi-layer ensemble models and risk scoring.

---

## 12. Reusing Extracted Features in Part 2

Part 2 can directly import the modular code or load the prepared CSV splits:

```python
import pandas as pd
from src.feature_extraction import extract_url_features

# 1. Load pre-split data ready for model training
X_train = pd.read_csv("dataset/processed/splits/X_train.csv")
y_train = pd.read_csv("dataset/processed/splits/y_train.csv")['label']

X_test = pd.read_csv("dataset/processed/splits/X_test.csv")
y_test = pd.read_csv("dataset/processed/splits/y_test.csv")['label']

# 2. Extract features from new live URLs in inference:
new_url = "http://secure-login.bank-update.com/account"
new_features_dict = extract_url_features(new_url)
X_inference = pd.DataFrame([new_features_dict])
```

---

*Part 1 completed and verified in workspace.*
