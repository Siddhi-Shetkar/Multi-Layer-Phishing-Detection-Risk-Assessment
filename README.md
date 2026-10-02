# Multi-Layer Phishing Detection & Risk Assessment

## 📌 Project Overview

**Multi-Layer Phishing Detection & Risk Assessment** is a Machine Learning-based cybersecurity project designed to identify potentially malicious or phishing URLs and assess their level of risk.

The system analyzes multiple characteristics of a URL instead of relying on a single indicator. It extracts relevant URL-based features, applies Machine Learning techniques, and classifies URLs as **legitimate or phishing** while providing a risk assessment.

The project aims to provide an additional layer of security by helping users identify suspicious links before accessing them.

---

## 🎯 Objectives

* Detect phishing and malicious URLs using Machine Learning.
* Extract meaningful features from URLs.
* Analyze multiple characteristics of a URL for better detection.
* Classify URLs as legitimate or phishing.
* Provide a risk assessment based on the detected characteristics.
* Evaluate the performance of the developed Machine Learning model.

---

## 🏗️ Project Workflow

```text
Input URL
    ↓
URL Preprocessing
    ↓
Feature Extraction
    ↓
Feature Analysis
    ↓
Machine Learning Model
    ↓
Phishing / Legitimate Classification
    ↓
Risk Assessment
    ↓
Final Result
```

---

# 👩‍💻 Team Contributions

## Part 1 — Siddhi Shetkar

### 1. Dataset Collection & Preprocessing

* Research and collect a suitable phishing URL dataset.
* Understand the dataset structure and available features.
* Handle missing or inconsistent data.
* Remove duplicate or irrelevant records.
* Perform data cleaning and preprocessing.
* Prepare the dataset for Machine Learning.

### 2. URL Feature Extraction

* Identify important URL-based characteristics.
* Extract features such as:

  * URL length
  * Number of dots
  * Number of special characters
  * Number of subdomains
  * Presence of IP address
  * Use of HTTPS
  * Number of digits
  * Suspicious symbols
  * Presence of URL shortening services
  * Other relevant URL characteristics

### 3. Exploratory Data Analysis

* Analyze the distribution of phishing and legitimate URLs.
* Study relationships between extracted features and URL classification.
* Generate relevant graphs and visualizations.
* Identify features that may contribute to phishing detection.

### 4. Data Preparation

* Encode categorical features where required.
* Select relevant features.
* Split the dataset into training and testing sets.
* Prepare the final dataset for model training.

---

# 👩‍💻 Part 2 — Sankeerthana Thoviti

### 1. Machine Learning Model Development

* Select suitable Machine Learning algorithms.
* Train the models using the processed dataset.
* Compare the performance of different models.
* Tune model parameters where necessary.

Possible models include:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine
* XGBoost / Gradient Boosting

### 2. Model Evaluation

Evaluate the trained models using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

Select the model based on its performance across relevant evaluation metrics.

### 3. Risk Assessment System

* Develop a risk assessment layer on top of the classification model.
* Assign a risk level based on the model's prediction and relevant URL characteristics.

Example:

```text
Low Risk       → Likely Legitimate
Medium Risk    → Suspicious URL
High Risk      → Likely Phishing
```

### 4. Final Prediction System

* Accept a URL as input from the user.
* Extract the required features automatically.
* Pass the features to the trained Machine Learning model.
* Display the predicted classification.
* Display the corresponding risk level.
* Present the result in a simple and understandable format.

---

## 🧠 Machine Learning Approach

The project follows a supervised Machine Learning approach.

### Input

A URL provided by the user.

### Feature Extraction

The URL is converted into numerical and categorical features based on its structural and security-related characteristics.

### Classification

The extracted features are passed to the trained Machine Learning model.

### Output

The system produces:

```text
URL Classification:
Phishing / Legitimate

Risk Level:
Low / Medium / High
```

---

## 📊 Evaluation Metrics

The Machine Learning models will be evaluated using:

| Metric           | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| Accuracy         | Measures overall correct predictions                            |
| Precision        | Measures how many predicted phishing URLs are actually phishing |
| Recall           | Measures how many actual phishing URLs are detected             |
| F1-Score         | Balances precision and recall                                   |
| Confusion Matrix | Shows correct and incorrect classifications                     |
| ROC-AUC          | Measures the model's ability to distinguish between classes     |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook / Google Colab**

---

## 📂 Project Structure

```text
Multi-Layer-Phishing-Detection/
│
├── dataset/
│   └── phishing_dataset.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── risk_assessment.py
│
├── models/
│   └── phishing_model.pkl
│
├── results/
│   └── evaluation_results.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Future Enhancements

* Integrate real-time URL checking.
* Add domain reputation analysis.
* Incorporate DNS and WHOIS-based features.
* Add webpage/content-based phishing detection.
* Use ensemble or deep learning models.
* Develop a browser extension for real-time detection.
* Provide a user-friendly web interface.
* Continuously update the model using newly detected phishing URLs.

The primary goal of this project is to develop a **multi-layer Machine Learning-based phishing detection system** that can analyze URL characteristics, identify potentially malicious links, and provide users with an understandable **risk assessment** before they interact with a suspicious URL.
