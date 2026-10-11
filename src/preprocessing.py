"""
Module for data cleaning, sanitization, and preprocessing.

Part of Part 1: Multi-Layer Phishing Detection & Risk Assessment.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import urllib.parse
import pandas as pd

from src.data_loader import get_verified_label_mapping, validate_columns


def is_malformed_url(url_val: Union[str, float, None]) -> bool:
    """
    Check if a URL value is malformed or unparseable.

    A URL is considered malformed if:
    - It is not a string
    - It is empty or only whitespace
    - It contains invalid control characters or cannot be parsed by urllib.parse.urlsplit
    - It has neither a scheme nor any valid hostname/domain structure

    Args:
        url_val: The URL candidate to check.

    Returns:
        True if the URL is malformed, False if parseable.
    """
    if url_val is None or not isinstance(url_val, str):
        return True

    stripped = url_val.strip()
    if not stripped:
        return True

    # Check for unprintable/control characters
    if any(ord(c) < 32 or ord(c) == 127 for c in stripped):
        return True

    # Try parsing
    try:
        # Prepend scheme if absent for parse check
        test_url = stripped if "://" in stripped else "http://" + stripped
        parsed = urllib.parse.urlsplit(test_url)
        if not parsed.netloc and not parsed.path:
            return True
    except Exception:
        return True

    return False


def clean_dataset(
    df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Union[int, dict]]]:
    """
    Execute complete end-to-end preprocessing and data cleaning pipeline.

    Steps performed:
    1. Validate input structure and copy data.
    2. Remove rows with missing or empty URLs.
    3. Normalize URL strings by trimming surrounding whitespace.
    4. Identify and report malformed URLs.
    5. Standardize target labels into verified binary format (0=Legitimate, 1=Phishing).
    6. Identify and drop exact duplicate records.
    7. Detect URLs with conflicting labels and resolve them explicitly.
    8. Deduplicate remaining repeated URLs.
    9. Verify zero remaining missing or invalid values.
    10. Report record counts before and after every cleaning stage.

    Args:
        df: Input raw or loaded DataFrame.
        verbose: Whether to log step-by-step progress to stdout.

    Returns:
        Tuple of (cleaned_df, cleaning_statistics_dict).
    """
    url_col, label_col = validate_columns(df)
    data = df.copy()

    initial_count = len(data)
    stats: Dict[str, Union[int, dict]] = {
        'initial_count': initial_count,
        'step_counts': {'0_initial': initial_count},
        'removed_counts': {},
    }

    if verbose:
        print("=" * 70)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"Step 0: Initial dataset record count: {initial_count:,}")

    # Step 1: Remove rows with null or non-string URLs/labels
    null_urls = data[url_col].isnull()
    null_labels = data[label_col].isnull()
    null_mask = null_urls | null_labels
    num_null = int(null_mask.sum())
    stats['removed_counts']['missing_values'] = num_null

    if num_null > 0:
        data = data[~null_mask].copy()
    stats['step_counts']['1_after_null_removal'] = len(data)
    if verbose:
        print(f"Step 1: Removed {num_null:,} records with missing URL or Label -> Remaining: {len(data):,}")

    # Step 2: Normalize URL strings by trimming surrounding whitespace
    data[url_col] = data[url_col].astype(str).str.strip()

    # Remove empty URLs after trimming
    empty_mask = data[url_col] == ""
    num_empty = int(empty_mask.sum())
    stats['removed_counts']['empty_urls'] = num_empty
    if num_empty > 0:
        data = data[~empty_mask].copy()
    stats['step_counts']['2_after_whitespace_and_empty_trimming'] = len(data)
    if verbose:
        print(f"Step 2: Trimmed whitespace and removed {num_empty:,} empty URLs -> Remaining: {len(data):,}")

    # Step 3: Identify malformed URLs
    malformed_mask = data[url_col].apply(is_malformed_url)
    num_malformed = int(malformed_mask.sum())
    stats['removed_counts']['malformed_urls'] = num_malformed
    if num_malformed > 0:
        if verbose:
            malformed_samples = data[malformed_mask][url_col].head(3).tolist()
            print(f"  [Info] Sample malformed URLs: {malformed_samples}")
        data = data[~malformed_mask].copy()
    stats['step_counts']['3_after_malformed_removal'] = len(data)
    if verbose:
        print(f"Step 3: Identified and removed {num_malformed:,} malformed URLs -> Remaining: {len(data):,}")

    # Step 4: Normalize target labels into binary format (0=Legitimate, 1=Phishing)
    unique_raw_labels = data[label_col].unique().tolist()
    label_mapping = get_verified_label_mapping(unique_raw_labels)
    data[label_col] = data[label_col].map(label_mapping)

    # Check for unmapped labels
    unmapped_mask = data[label_col].isnull()
    num_unmapped = int(unmapped_mask.sum())
    if num_unmapped > 0:
        data = data[~unmapped_mask].copy()
    data[label_col] = data[label_col].astype(int)
    stats['step_counts']['4_after_label_normalization'] = len(data)
    stats['label_mapping'] = {str(k): v for k, v in label_mapping.items()}
    if verbose:
        print(f"Step 4: Normalized labels using verified mapping {label_mapping} -> Remaining: {len(data):,}")

    # Step 5: Identify exact duplicate records (same URL and same Label)
    exact_duplicates = int(data.duplicated().sum())
    stats['removed_counts']['exact_duplicates'] = exact_duplicates
    if exact_duplicates > 0:
        data = data.drop_duplicates(keep='first').copy()
    stats['step_counts']['5_after_exact_duplicate_removal'] = len(data)
    if verbose:
        print(f"Step 5: Removed {exact_duplicates:,} exact duplicate records -> Remaining: {len(data):,}")

    # Step 6: Detect URLs appearing with conflicting labels
    url_label_counts = data.groupby(url_col)[label_col].nunique()
    conflicting_urls = url_label_counts[url_label_counts > 1].index
    num_conflicting = len(conflicting_urls)
    stats['removed_counts']['conflicting_urls'] = num_conflicting
    if num_conflicting > 0:
        # Explicit handling: Remove ambiguous URLs whose ground truth is contradictory
        data = data[~data[url_col].isin(conflicting_urls)].copy()
        if verbose:
            print(f"  [Warning] Found {num_conflicting:,} URLs with conflicting labels. Removed for safety.")
    stats['step_counts']['6_after_conflicts_removal'] = len(data)
    if verbose:
        print(f"Step 6: Handled conflicting labels (removed {num_conflicting:,} conflicting URLs) -> Remaining: {len(data):,}")

    # Step 7: Deduplicate remaining duplicate URLs (if any exist with identical labels)
    duplicate_urls_remaining = int(data[url_col].duplicated().sum())
    stats['removed_counts']['duplicate_urls'] = duplicate_urls_remaining
    if duplicate_urls_remaining > 0:
        data = data.drop_duplicates(subset=[url_col], keep='first').copy()
    stats['step_counts']['7_after_duplicate_url_deduplication'] = len(data)
    if verbose:
        print(f"Step 7: Deduplicated remaining duplicate URLs ({duplicate_urls_remaining:,} removed) -> Remaining: {len(data):,}")

    # Step 8: Standardize column names to lowercase 'url' and 'label'
    if url_col != 'url' or label_col != 'label':
        data = data.rename(columns={url_col: 'url', label_col: 'label'})
    data = data.reset_index(drop=True)

    # Step 9: Final validation of missing or invalid values
    assert data['url'].isnull().sum() == 0, "Cleaned dataset contains null URLs!"
    assert data['label'].isnull().sum() == 0, "Cleaned dataset contains null labels!"
    assert set(data['label'].unique()).issubset({0, 1}), "Cleaned dataset contains non-binary labels!"

    final_count = len(data)
    legit_count = int((data['label'] == 0).sum())
    phish_count = int((data['label'] == 1).sum())

    stats['final_count'] = final_count
    stats['final_distribution'] = {
        'legitimate_0': legit_count,
        'phishing_1': phish_count,
        'legitimate_percentage': round((legit_count / final_count) * 100, 2) if final_count > 0 else 0,
        'phishing_percentage': round((phish_count / final_count) * 100, 2) if final_count > 0 else 0,
    }

    if verbose:
        print("-" * 70)
        print("PREPROCESSING SUMMARY:")
        print(f"  - Initial Records:  {initial_count:,}")
        print(f"  - Final Records:    {final_count:,} ({final_count / initial_count * 100:.2f}% retained)")
        print(f"  - Legitimate (0):   {legit_count:,} ({stats['final_distribution']['legitimate_percentage']}%)")
        print(f"  - Phishing (1):     {phish_count:,} ({stats['final_distribution']['phishing_percentage']}%)")
        print("=" * 70)

    return data, stats


def save_cleaned_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "dataset/processed/cleaned_dataset.csv"
) -> Path:
    """
    Save the cleaned dataset to CSV.

    Args:
        df: Cleaned pandas DataFrame.
        output_path: Destination path for cleaned CSV.

    Returns:
        Path object of saved CSV.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Cleaned dataset saved successfully to: '{out_p.resolve()}' ({len(df):,} records).")
    return out_p
