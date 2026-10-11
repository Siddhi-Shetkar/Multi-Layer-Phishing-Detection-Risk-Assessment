"""
Module for loading, validating, and inspecting URL phishing datasets.
Multi-Layer Phishing Detection & Risk Assessment.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd


# Verified mapping for the dataset:
# In the raw phishing_url_dataset.csv:
#   Raw label 0 represents Phishing URLs (e.g., credential harvesting pages, dynamic DNS, typosquats)
#   Raw label 1 represents Legitimate URLs (e.g., government, educational, official company domains)
# Target standardized specification:
#   0 = Legitimate
#   1 = Phishing
RAW_TO_STANDARDIZED_LABEL_MAP = {
    0: 1,  # Raw Phishing (0) -> Standardized Phishing (1)
    1: 0,  # Raw Legitimate (1) -> Standardized Legitimate (0)
}

STRING_LABEL_MAP = {
    'phishing': 1,
    'phish': 1,
    'bad': 1,
    'malicious': 1,
    'legitimate': 0,
    'legit': 0,
    'good': 0,
    'benign': 0,
}


def find_column_name(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """
    Find matching column name case-insensitively.

    Args:
        df: Input DataFrame.
        target_name: Target column name in lowercase (e.g., 'url', 'label').

    Returns:
        The actual column name in df, or None if not found.
    """
    for col in df.columns:
        if col.strip().lower() == target_name.lower():
            return col
    return None


def validate_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Validate that required URL and Label columns exist in the DataFrame.

    Args:
        df: Input DataFrame to check.

    Returns:
        Tuple of (actual_url_col, actual_label_col).

    Raises:
        ValueError: If either required column is missing.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input data must be a valid pandas DataFrame.")

    url_col = find_column_name(df, 'url')
    label_col = find_column_name(df, 'label')

    missing = []
    if url_col is None:
        missing.append("'url'")
    if label_col is None:
        missing.append("'label'")

    if missing:
        raise ValueError(
            f"Dataset validation failed: missing required column(s): {', '.join(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    return url_col, label_col


def inspect_dataset(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Union[int, float, dict]]:
    """
    Inspect the dataset dimensions, types, missing values, duplicates, and label distribution.

    Args:
        df: Input DataFrame to inspect.
        verbose: Whether to print inspection output to stdout.

    Returns:
        Dictionary containing inspection metrics and summary statistics.
    """
    url_col, label_col = validate_columns(df)

    num_rows, num_cols = df.shape
    missing_url = int(df[url_col].isnull().sum())
    missing_label = int(df[label_col].isnull().sum())
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_urls = int(df[url_col].duplicated().sum())
    unique_labels = df[label_col].value_counts(dropna=False).to_dict()

    # Conflicting URLs (same URL with different labels)
    url_label_groups = df.groupby(url_col)[label_col].nunique()
    conflicting_urls_count = int((url_label_groups > 1).sum())

    summary = {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'url_column': url_col,
        'label_column': label_col,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {
            url_col: missing_url,
            label_col: missing_label,
            'total': total_missing
        },
        'duplicate_rows': duplicate_rows,
        'duplicate_urls': duplicate_urls,
        'conflicting_urls': conflicting_urls_count,
        'label_distribution': unique_labels,
    }

    if verbose:
        print("=" * 60)
        print("DATASET INSPECTION SUMMARY")
        print("=" * 60)
        print(f"Dimensions: {num_rows:,} rows, {num_cols} columns")
        print(f"Identified URL column:   '{url_col}' (dtype: {df[url_col].dtype})")
        print(f"Identified Label column: '{label_col}' (dtype: {df[label_col].dtype})")
        print("-" * 60)
        print("Missing Values:")
        print(f"  - In '{url_col}': {missing_url:,}")
        print(f"  - In '{label_col}': {missing_label:,}")
        print(f"  - Total across dataframe: {total_missing:,}")
        print("-" * 60)
        print("Duplicates and Conflicts:")
        print(f"  - Exact duplicate rows: {duplicate_rows:,}")
        print(f"  - Duplicate URLs: {duplicate_urls:,}")
        print(f"  - Conflicting URLs (same URL, different label): {conflicting_urls_count:,}")
        print("-" * 60)
        print("Label Distribution (Raw):")
        for lbl, count in unique_labels.items():
            pct = (count / num_rows) * 100 if num_rows > 0 else 0
            print(f"  - Label {lbl!r}: {count:,} ({pct:.2f}%)")
        print("=" * 60)

    return summary


def get_verified_label_mapping(unique_labels: list) -> Dict[Union[int, str], int]:
    """
    Determine and return a verified mapping to standardize labels to {0: Legitimate, 1: Phishing}.

    Args:
        unique_labels: List of unique label values present in dataset.

    Returns:
        Mapping dictionary {raw_label: standardized_label}.

    Raises:
        ValueError: If labels cannot be safely resolved.
    """
    clean_labels = set(unique_labels)

    # Case 1: Raw dataset has {0, 1} where 0=Phishing and 1=Legitimate
    if clean_labels.issubset({0, 1}):
        return RAW_TO_STANDARDIZED_LABEL_MAP

    # Case 2: String labels
    mapping = {}
    for lbl in clean_labels:
        if isinstance(lbl, str):
            lbl_lower = lbl.strip().lower()
            if lbl_lower in STRING_LABEL_MAP:
                mapping[lbl] = STRING_LABEL_MAP[lbl_lower]
            else:
                raise ValueError(f"Unknown string label '{lbl}'. Cannot determine legitimate vs phishing.")
        else:
            raise ValueError(f"Unsupported label type or value '{lbl}'. Cannot determine legitimate vs phishing.")

    return mapping


def load_dataset(
    filepath: Union[str, Path],
    standardize_col_names: bool = True
) -> pd.DataFrame:
    """
    Load dataset from CSV with robust error handling and column validation.

    Args:
        filepath: Path to the CSV file.
        standardize_col_names: If True, rename URL and Label columns to lowercase 'url' and 'label'.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, corrupted, or missing required columns.
    """
    path_obj = Path(filepath)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: '{path_obj.resolve()}'. "
            "Please provide a valid path to an existing CSV file."
        )

    if not path_obj.is_file():
        raise ValueError(f"Specified path is not a file: '{path_obj.resolve()}'.")

    try:
        # Load CSV with pandas
        df = pd.read_csv(path_obj)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Dataset file is empty: '{path_obj.resolve()}'.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file '{path_obj.resolve()}': {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV from '{path_obj.resolve()}': {e}")

    if df.empty:
        raise ValueError(f"Dataset contains 0 rows: '{path_obj.resolve()}'.")

    url_col, label_col = validate_columns(df)

    if standardize_col_names:
        rename_map = {}
        if url_col != 'url':
            rename_map[url_col] = 'url'
        if label_col != 'label':
            rename_map[label_col] = 'label'
        if rename_map:
            df = df.rename(columns=rename_map)

    return df
