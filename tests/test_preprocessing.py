"""
Unit tests for src/preprocessing.py.
"""

import pandas as pd
import pytest

from src.preprocessing import clean_dataset, is_malformed_url, save_cleaned_dataset


def test_is_malformed_url():
    # Valid URLs
    assert not is_malformed_url("https://www.google.com")
    assert not is_malformed_url("google.com/test")
    assert not is_malformed_url("http://192.168.1.1:8080")

    # Malformed URLs
    assert is_malformed_url("")
    assert is_malformed_url("   ")
    assert is_malformed_url(None)
    assert is_malformed_url("\x00\x01malformed")


def test_clean_dataset_whitespace_and_nulls():
    raw_df = pd.DataFrame({
        'url': [
            '  https://www.google.com  ',  # whitespace
            None,                           # null
            '',                             # empty
            'https://legitimate.org',
            'http://phishing-site.xyz'
        ],
        'label': [1, 1, 0, 1, 0]  # Raw format: 1=Legit, 0=Phish
    })
    cleaned_df, stats = clean_dataset(raw_df, verbose=False)

    assert len(cleaned_df) == 3
    assert '  https://www.google.com  ' not in cleaned_df['url'].values
    assert 'https://www.google.com' in cleaned_df['url'].values
    # Check normalized labels (Raw 1 -> Standardized 0, Raw 0 -> Standardized 1)
    legit_entry = cleaned_df[cleaned_df['url'] == 'https://legitimate.org']
    assert legit_entry['label'].values[0] == 0  # Legitimate
    phish_entry = cleaned_df[cleaned_df['url'] == 'http://phishing-site.xyz']
    assert phish_entry['label'].values[0] == 1  # Phishing


def test_clean_dataset_duplicates():
    raw_df = pd.DataFrame({
        'url': [
            'https://repeat.com',
            'https://repeat.com',  # exact duplicate
            'https://unique.com'
        ],
        'label': [1, 1, 1]
    })
    cleaned_df, stats = clean_dataset(raw_df, verbose=False)
    assert len(cleaned_df) == 2
    assert stats['removed_counts']['exact_duplicates'] == 1


def test_clean_dataset_conflicting_labels():
    raw_df = pd.DataFrame({
        'url': [
            'https://conflict.com',
            'https://conflict.com',  # conflicting label
            'https://safe.com'
        ],
        'label': [1, 0, 1]  # Conflicting!
    })
    cleaned_df, stats = clean_dataset(raw_df, verbose=False)
    # The conflicting URL should be removed
    assert 'https://conflict.com' not in cleaned_df['url'].values
    assert len(cleaned_df) == 1
    assert stats['removed_counts']['conflicting_urls'] == 1


def test_save_cleaned_dataset(tmp_path):
    df = pd.DataFrame({'url': ['https://example.com'], 'label': [0]})
    out_file = tmp_path / "cleaned.csv"
    saved = save_cleaned_dataset(df, out_file)
    assert saved.exists()
    reloaded = pd.read_csv(saved)
    assert len(reloaded) == 1
    assert reloaded['url'].iloc[0] == 'https://example.com'
