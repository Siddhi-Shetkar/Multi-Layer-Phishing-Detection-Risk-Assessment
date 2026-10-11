"""
Unit tests for src/data_loader.py.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    find_column_name,
    get_verified_label_mapping,
    inspect_dataset,
    load_dataset,
    validate_columns,
)


@pytest.fixture
def sample_raw_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        'URL': [
            'https://www.google.com',
            'http://paypal-verify-fake.com/login',
            'https://github.com/torvalds',
            'http://192.168.0.1/admin'
        ],
        'Label': [1, 0, 1, 0]  # Raw format: 1=Legit, 0=Phish
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_find_column_name():
    df = pd.DataFrame({'Url': ['a'], 'LABEL': [1]})
    assert find_column_name(df, 'url') == 'Url'
    assert find_column_name(df, 'label') == 'LABEL'
    assert find_column_name(df, 'non_existent') is None


def test_validate_columns_success():
    df = pd.DataFrame({'url': ['https://a.com'], 'label': [0]})
    url_col, label_col = validate_columns(df)
    assert url_col == 'url'
    assert label_col == 'label'


def test_validate_columns_missing_url():
    df = pd.DataFrame({'wrong_col': ['https://a.com'], 'label': [0]})
    with pytest.raises(ValueError, match="missing required column"):
        validate_columns(df)


def test_validate_columns_missing_label():
    df = pd.DataFrame({'url': ['https://a.com'], 'target': [0]})
    with pytest.raises(ValueError, match="missing required column"):
        validate_columns(df)


def test_load_dataset_success(sample_raw_csv):
    df = load_dataset(sample_raw_csv, standardize_col_names=True)
    assert 'url' in df.columns
    assert 'label' in df.columns
    assert len(df) == 4


def test_load_dataset_file_not_found(tmp_path):
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(non_existent)


def test_load_dataset_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty|contains 0 rows"):
        load_dataset(empty_file)


def test_inspect_dataset(sample_raw_csv):
    df = pd.read_csv(sample_raw_csv)
    summary = inspect_dataset(df, verbose=False)
    assert summary['num_rows'] == 4
    assert summary['num_cols'] == 2
    assert summary['duplicate_rows'] == 0
    assert summary['conflicting_urls'] == 0


def test_get_verified_label_mapping():
    # Numeric 0/1 mapping (Raw 0=Phishing -> 1, Raw 1=Legitimate -> 0)
    mapping = get_verified_label_mapping([0, 1])
    assert mapping[0] == 1
    assert mapping[1] == 0

    # String mapping
    str_mapping = get_verified_label_mapping(['phishing', 'legitimate'])
    assert str_mapping['phishing'] == 1
    assert str_mapping['legitimate'] == 0
