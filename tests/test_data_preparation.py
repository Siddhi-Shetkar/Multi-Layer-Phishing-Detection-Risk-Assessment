"""
Unit and integration tests for src/data_preparation.py and pipeline reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_preparation import (
    generate_feature_descriptions,
    prepare_features_and_splits,
    validate_features,
)
from src.feature_extraction import FEATURE_COLUMNS


@pytest.fixture
def mock_cleaned_data():
    urls = [
        'https://legit-site-1.org/home',
        'https://legit-site-2.edu/research',
        'https://legit-site-3.gov/service',
        'https://legit-site-4.com/products',
        'https://legit-site-5.com/docs',
        'http://phish-site-1.tk/login/update',
        'http://phish-site-2.ml/verify/account',
        'http://192.168.1.50/secure/bank',
        'http://phish-site-4.cf/paypal/confirm',
        'http://phish-site-5.ga/signin'
    ]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5 legit, 5 phish
    return pd.DataFrame({'url': urls, 'label': labels})


def test_validate_features_success():
    df_valid = pd.DataFrame(
        np.ones((5, len(FEATURE_COLUMNS)), dtype=float),
        columns=FEATURE_COLUMNS
    )
    validate_features(df_valid)  # Should not raise


def test_validate_features_with_nans():
    df_bad = pd.DataFrame(
        np.ones((5, len(FEATURE_COLUMNS)), dtype=float),
        columns=FEATURE_COLUMNS
    )
    df_bad.loc[0, 'url_length'] = np.nan
    with pytest.raises(ValueError, match="null values"):
        validate_features(df_bad)


def test_validate_features_with_infs():
    df_bad = pd.DataFrame(
        np.ones((5, len(FEATURE_COLUMNS)), dtype=float),
        columns=FEATURE_COLUMNS
    )
    df_bad.loc[1, 'hostname_length'] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        validate_features(df_bad)


def test_prepare_features_and_splits_workflow(tmp_path, mock_cleaned_data):
    summary = prepare_features_and_splits(
        cleaned_df=mock_cleaned_data,
        output_dir=tmp_path,
        test_size=0.2,
        random_state=42,
        verbose=False
    )

    assert summary['total_records'] == 10
    assert summary['train_records'] == 8
    assert summary['test_records'] == 2

    # Verify split files exist
    x_train_p = Path(summary['x_train_path'])
    x_test_p = Path(summary['x_test_path'])
    y_train_p = Path(summary['y_train_path'])
    y_test_p = Path(summary['y_test_path'])

    assert x_train_p.exists()
    assert x_test_p.exists()
    assert y_train_p.exists()
    assert y_test_p.exists()

    X_train = pd.read_csv(x_train_p)
    X_test = pd.read_csv(x_test_p)
    y_train = pd.read_csv(y_train_p)
    y_test = pd.read_csv(y_test_p)

    # 1. Feature column order consistency
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS

    # 2. Prevention of target leakage: 'label' must NOT be in feature matrices
    assert 'label' not in X_train.columns
    assert 'label' not in X_test.columns
    assert 'url' not in X_train.columns
    assert 'url' not in X_test.columns

    # 3. Target label preservation
    assert set(y_train['label'].unique()).issubset({0, 1})
    assert set(y_test['label'].unique()).issubset({0, 1})


def test_split_reproducibility(tmp_path, mock_cleaned_data):
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    prepare_features_and_splits(mock_cleaned_data, output_dir=dir1, random_state=42, verbose=False)
    prepare_features_and_splits(mock_cleaned_data, output_dir=dir2, random_state=42, verbose=False)

    X_train_1 = pd.read_csv(dir1 / "splits" / "X_train.csv")
    X_train_2 = pd.read_csv(dir2 / "splits" / "X_train.csv")
    pd.testing.assert_frame_equal(X_train_1, X_train_2)

    y_train_1 = pd.read_csv(dir1 / "splits" / "y_train.csv")
    y_train_2 = pd.read_csv(dir2 / "splits" / "y_train.csv")
    pd.testing.assert_frame_equal(y_train_1, y_train_2)


def test_generate_feature_descriptions(tmp_path):
    desc_file = tmp_path / "desc.md"
    saved = generate_feature_descriptions(desc_file)
    assert saved.exists()
    content = saved.read_text(encoding='utf-8')
    assert "url_length" in content
    assert "has_ip_address" in content
    assert "has_suspicious_keyword" in content
