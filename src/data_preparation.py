"""
Module for feature dataset preparation, validation, and train/test splitting.

Part of Part 1: Multi-Layer Phishing Detection & Risk Assessment.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_extraction import FEATURE_COLUMNS, extract_features_from_dataset


FEATURE_DESCRIPTIONS_TEXT = """# URL Feature Descriptions

This document provides a comprehensive dictionary of all features extracted for the **Multi-Layer Phishing Detection & Risk Assessment** system (Part 1).

All features are strictly numeric (`int64` or `float64`) and derived without requiring internet connectivity or web page content execution.

---

## 1. Length & Structural Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `url_length` | Integer | Total length of the URL in characters. | Phishing URLs often conceal destinations via deep subdirectory chains, obfuscated parameters, or encoded payloads, resulting in elevated length. |
| `hostname_length` | Integer | Length of the parsed hostname / FQDN. | Attackers frequently embed brand names or high-entropy random character strings into subdomains and hosts. |
| `path_length` | Integer | Length of the URL path component. | Fraudulent login clones often reside deep in hijacked server directories (e.g., `/wp-content/plugins/.../login.php`). |
| `query_length` | Integer | Length of the query string following `?`. | Phishing campaigns frequently pass tracking tokens, victim emails, and redirect targets via query strings. |
| `num_dots` | Integer | Total count of dot (`.`) characters in the full URL. | Phishers often create multi-level subdomains or mimic domains using periods in paths/queries. |
| `num_hyphens` | Integer | Total count of hyphen (`-`) characters. | Brand spoofing frequently uses hyphens to simulate authentic domains (e.g., `paypal-security-update.com`). |
| `num_underscores` | Integer | Total count of underscore (`_`) characters. | Less common in genuine domain names, commonly seen in auto-generated phishing script paths. |
| `num_slashes` | Integer | Total count of forward slash (`/`) characters. | Indicates nesting depth of paths and directories on the hosting server. |
| `num_question_marks`| Integer | Total count of question mark (`?`) characters. | Demarcates query strings; multiple question marks often signal parameter obfuscation. |
| `num_equal_signs` | Integer | Total count of equal sign (`=`) characters. | Reflects the density of query parameter key-value pairs. |
| `num_ampersands` | Integer | Total count of ampersand (`&`) characters. | Delimits multiple parameters in complex phishing tracking schemes. |
| `num_at_symbols` | Integer | Total count of `@` symbols in the URL. | The `@` symbol causes browsers to ignore all preceding text in the authority component, treating the subsequent host as the actual target. |
| `num_percent_symbols` | Integer | Total count of percent (`%`) characters. | Percent signs indicate URL hexadecimal encoding, often used to bypass basic keyword filters. |
| `num_digits` | Integer | Total count of numeric digits (`0`-`9`) across the URL. | High digit density is characteristic of randomized hostnames, IP addresses, and tracking identifiers. |
| `num_special_characters`| Integer | Total count of non-alphanumeric characters in the URL. | Quantifies overall symbol complexity and character variance. |

---

## 2. Hostname & Domain Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `has_ip_address` | Binary (0/1) | Indicates whether the hostname is an IPv4 or IPv6 address. | Legitimate commercial services rely on registered domain names. IP address hostnames are a strong indicator of malicious infrastructure. |
| `hostname_has_digits` | Binary (0/1) | 1 if the hostname contains numeric digits, 0 otherwise. | Algorithmic domain generation (DGA) and cheap VPS hosting often introduce numbers into hostnames. |
| `hostname_has_hyphen` | Binary (0/1) | 1 if the hostname contains hyphens, 0 otherwise. | Hyphenated hostnames are a standard technique for domain squatting (typosquatting / combosquatting). |
| `num_subdomains` | Integer | Estimated count of subdomains using Public Suffix List rules. | Excessive subdomain layers are used to construct convincing lures on free hosting platforms (e.g., `login.secure.bank.pages.dev`). |
| `has_multiple_subdomains` | Binary (0/1) | 1 if `num_subdomains` > 1, 0 otherwise. | Flags nested subdomain hierarchies commonly exploited by adversaries. |
| `is_https` | Binary (0/1) | 1 if URL protocol is `https://`, 0 otherwise. | While modern phishing also adopts free SSL certs, protocol distribution remains informative when correlated with other structural signals. |
| `has_explicit_port` | Binary (0/1) | 1 if an explicit non-default port is designated in authority, 0 otherwise. | Standard web traffic uses default ports; explicit ports often indicate compromised rogue servers. |
| `has_unusual_port` | Binary (0/1) | 1 if port is specified and outside standard HTTP(80) / HTTPS(443), 0 otherwise. | Non-standard ports (e.g., `:8080`, `:8443`, `:8888`) are common on temporary attacker testing nodes. |
| `num_query_parameters`| Integer | Count of parsed key-value pairs in the query string. | Quantifies dynamic parameter density used in targeted credential harvesting. |

---

## 3. Suspicious-Pattern Features

| Feature Name | Type | Description | Security Rationale |
| :--- | :--- | :--- | :--- |
| `has_url_shortener` | Binary (0/1) | 1 if hostname matches known URL shorteners (bit.ly, t.co, etc.), 0 otherwise. | URL shorteners conceal the true destination and prevent immediate user inspection. |
| `has_suspicious_keyword` | Binary (0/1) | 1 if URL contains sensitive lure words (`login`, `verify`, `account`, `password`, etc.), 0 otherwise. | Social engineering lures concentrate heavily around authentication and account-action verbs. |
| `has_encoded_characters` | Binary (0/1) | 1 if URL contains percent-encoded hex sequences (`%`), 0 otherwise. | Hex-encoding is used to mask executable script names or deceptive host fragments. |
| `has_at_symbol` | Binary (0/1) | 1 if URL contains `@` symbol, 0 otherwise. | Flags credential injection or userinfo spoofing in URL authority. |

---

## Target Label Specification

- **`0`**: **Legitimate** (Verified authentic, benign web resources)
- **`1`**: **Phishing** (Deceptive, malicious, or credential-harvesting web resources)
"""


def generate_feature_descriptions(
    output_path: Union[str, Path] = "dataset/processed/feature_descriptions.md"
) -> Path:
    """
    Generate and save the comprehensive feature descriptions markdown document.

    Args:
        output_path: Path to save the markdown document.

    Returns:
        Path of the saved file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w', encoding='utf-8') as f:
        f.write(FEATURE_DESCRIPTIONS_TEXT)
    print(f"Feature descriptions saved to: '{out_p.resolve()}'")
    return out_p


def validate_features(df_features: pd.DataFrame) -> None:
    """
    Validate that feature DataFrame adheres to quality standards:
    - Exactly matches expected columns
    - All columns are numeric
    - No NaN, null, or infinite values exist

    Args:
        df_features: DataFrame containing features to validate.

    Raises:
        ValueError: If validation fails.
    """
    missing_cols = set(FEATURE_COLUMNS) - set(df_features.columns)
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    # Check dtypes
    non_numeric = []
    for col in FEATURE_COLUMNS:
        if not np.issubdtype(df_features[col].dtype, np.number):
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(f"Features must be numeric. Found non-numeric columns: {non_numeric}")

    # Check for NaN / Nulls
    null_counts = df_features[FEATURE_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Found null values in features: {bad_cols}")

    # Check for Infinite values
    inf_counts = np.isinf(df_features[FEATURE_COLUMNS]).sum()
    if inf_counts.sum() > 0:
        bad_cols = inf_counts[inf_counts > 0].to_dict()
        raise ValueError(f"Found infinite values in features: {bad_cols}")


def prepare_features_and_splits(
    cleaned_df: pd.DataFrame,
    output_dir: Union[str, Path] = "dataset/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 10000,
    precomputed_features: Optional[pd.DataFrame] = None,
    verbose: bool = True
) -> Dict[str, Union[int, str, Path]]:
    """
    Extract features, validate, assemble full dataset, and generate stratified train/test splits.

    Guarantees:
    - 80:20 stratified split
    - Fixed random state for strict reproducibility
    - Prevention of target and data leakage (disjoint URLs)
    - Consistent feature column order across train and test sets
    - Clean binary labels: 0=Legitimate, 1=Phishing

    Args:
        cleaned_df: Cleaned DataFrame with 'url' and 'label' columns.
        output_dir: Base directory for processed outputs.
        test_size: Fraction of data reserved for test set (default: 0.2).
        random_state: Seed for reproducibility.
        batch_size: Batch size for feature extraction logging.
        precomputed_features: Optional pre-extracted features DataFrame to avoid re-computation.
        verbose: Whether to log progress.

    Returns:
        Summary dictionary containing record counts and file paths.
    """
    out_dir = Path(output_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if 'url' not in cleaned_df.columns or 'label' not in cleaned_df.columns:
        raise ValueError("Input DataFrame must contain 'url' and 'label' columns.")

    # 1. Feature extraction
    if verbose:
        print("=" * 70)
        print("PREPARING FEATURE DATASET & TRAIN/TEST SPLITS")
        print("=" * 70)

    if precomputed_features is not None:
        if verbose:
            print("Reusing precomputed extracted features matrix...")
        X_features = precomputed_features.copy()
    else:
        X_features = extract_features_from_dataset(
            cleaned_df,
            url_col='url',
            batch_size=batch_size,
            verbose=verbose
        )

    # 2. Validation
    validate_features(X_features)

    # 3. Combine features with original URL and target label
    # Preserve original URL column for traceability alongside numeric features and label
    full_dataset = X_features.copy()
    full_dataset.insert(0, 'url', cleaned_df['url'].values)
    full_dataset['label'] = cleaned_df['label'].values.astype(int)

    # Save complete feature dataset
    full_csv_path = out_dir / "url_features.csv"
    full_dataset.to_csv(full_csv_path, index=False)
    if verbose:
        print(f"Full feature dataset saved to: '{full_csv_path.resolve()}' ({len(full_dataset):,} records)")

    # 4. Generate feature descriptions document
    desc_path = out_dir / "feature_descriptions.md"
    generate_feature_descriptions(desc_path)

    # 5. Train / Test Split
    # Features matrix X (only numeric feature columns in exact FEATURE_COLUMNS order)
    X = full_dataset[FEATURE_COLUMNS]
    y = full_dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Verify column orders match
    assert list(X_train.columns) == FEATURE_COLUMNS, "X_train columns do not match FEATURE_COLUMNS"
    assert list(X_test.columns) == FEATURE_COLUMNS, "X_test columns do not match FEATURE_COLUMNS"

    # Save split files
    x_train_path = splits_dir / "X_train.csv"
    x_test_path = splits_dir / "X_test.csv"
    y_train_path = splits_dir / "y_train.csv"
    y_test_path = splits_dir / "y_test.csv"

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False, header=['label'])
    y_test.to_csv(y_test_path, index=False, header=['label'])

    summary = {
        'total_records': len(full_dataset),
        'num_features': len(FEATURE_COLUMNS),
        'train_records': len(X_train),
        'test_records': len(X_test),
        'train_legit_count': int((y_train == 0).sum()),
        'train_phish_count': int((y_train == 1).sum()),
        'test_legit_count': int((y_test == 0).sum()),
        'test_phish_count': int((y_test == 1).sum()),
        'full_csv_path': str(full_csv_path),
        'x_train_path': str(x_train_path),
        'x_test_path': str(x_test_path),
        'y_train_path': str(y_train_path),
        'y_test_path': str(y_test_path),
        'feature_descriptions_path': str(desc_path),
    }

    if verbose:
        print("-" * 70)
        print("SPLIT SUMMARY:")
        print(f"  - Training Set: {len(X_train):,} samples (Legitimate: {summary['train_legit_count']:,}, Phishing: {summary['train_phish_count']:,})")
        print(f"  - Testing Set:  {len(X_test):,} samples (Legitimate: {summary['test_legit_count']:,}, Phishing: {summary['test_phish_count']:,})")
        print(f"  - Features:     {len(FEATURE_COLUMNS)} numeric features")
        print(f"  - Output Dir:   '{splits_dir.resolve()}'")
        print("=" * 70)

    return summary
