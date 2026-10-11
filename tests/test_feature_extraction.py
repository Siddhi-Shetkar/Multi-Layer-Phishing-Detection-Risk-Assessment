"""
Unit tests for src/feature_extraction.py.
"""

import pandas as pd
import pytest

from src.feature_extraction import (
    FEATURE_COLUMNS,
    check_is_ip_address,
    extract_features_from_dataset,
    extract_url_features,
    parse_url_components,
)


def test_feature_columns_completeness_and_order():
    url = "https://example.com"
    feats = extract_url_features(url)
    assert list(feats.keys()) == FEATURE_COLUMNS
    assert len(feats) == len(FEATURE_COLUMNS)
    # Check all are numeric
    for k, v in feats.items():
        assert isinstance(v, (int, float)), f"Feature {k} is not numeric: {type(v)}"


def test_ordinary_url():
    url = "https://www.example.org/about/us"
    feats = extract_url_features(url)
    assert feats['is_https'] == 1
    assert feats['has_ip_address'] == 0
    assert feats['has_explicit_port'] == 0
    assert feats['num_query_parameters'] == 0
    assert feats['url_length'] == len(url)
    assert feats['num_dots'] == 2  # www.example.org


def test_phishing_like_url():
    url = "http://login.secure-bank.update.com.attacker.net/verify/account.php?id=99&token=xyz"
    feats = extract_url_features(url)
    assert feats['is_https'] == 0
    assert feats['has_suspicious_keyword'] == 1  # login, update, verify, account
    assert feats['num_subdomains'] >= 2
    assert feats['has_multiple_subdomains'] == 1
    assert feats['num_query_parameters'] == 2
    assert feats['num_dots'] >= 4


def test_url_containing_ip_address():
    url_ipv4 = "http://192.168.1.100:8080/admin/login"
    feats_v4 = extract_url_features(url_ipv4)
    assert feats_v4['has_ip_address'] == 1
    assert feats_v4['has_explicit_port'] == 1
    assert feats_v4['has_unusual_port'] == 1  # 8080 is outside 80 and 443
    assert feats_v4['hostname_has_digits'] == 1

    url_ipv6 = "http://[2001:db8::1]:80/index.html"
    feats_v6 = extract_url_features(url_ipv6)
    assert feats_v6['has_ip_address'] == 1
    assert feats_v6['has_explicit_port'] == 1
    assert feats_v6['has_unusual_port'] == 0  # 80 is standard web port


def test_url_with_query_parameters():
    url = "https://store.com/search?category=books&sort=desc&limit=25"
    feats = extract_url_features(url)
    assert feats['num_query_parameters'] == 3
    assert feats['query_length'] > 0
    assert feats['num_equal_signs'] == 3
    assert feats['num_ampersands'] == 2
    assert feats['num_question_marks'] == 1


def test_url_with_at_symbol():
    url = "http://legitimate-bank.com@evil-phishing.com/login"
    feats = extract_url_features(url)
    assert feats['num_at_symbols'] == 1
    assert feats['has_at_symbol'] == 1
    assert feats['hostname_length'] == len("evil-phishing.com")


def test_url_without_scheme():
    url = "www.mysite.com/profile/user?view=full"
    feats = extract_url_features(url)
    assert feats['is_https'] == 0
    assert feats['hostname_length'] == len("www.mysite.com")
    assert feats['path_length'] == len("/profile/user")
    assert feats['num_query_parameters'] == 1


def test_malformed_and_edge_case_urls():
    # Empty string
    feats_empty = extract_url_features("")
    assert feats_empty['url_length'] == 0
    assert feats_empty['num_dots'] == 0
    assert feats_empty['has_ip_address'] == 0

    # Malformed port
    feats_bad_port = extract_url_features("http://example.com:notaport/test")
    assert feats_bad_port['has_explicit_port'] == 1
    assert feats_bad_port['has_unusual_port'] == 1

    # Percent encoded characters
    feats_encoded = extract_url_features("https://site.com/test%20space%21")
    assert feats_encoded['has_encoded_characters'] == 1
    assert feats_encoded['num_percent_symbols'] == 2


def test_url_shortener_detection():
    url_bitly = "https://bit.ly/3xYz123"
    feats = extract_url_features(url_bitly)
    assert feats['has_url_shortener'] == 1

    url_normal = "https://www.standarddomain.com/page"
    feats_normal = extract_url_features(url_normal)
    assert feats_normal['has_url_shortener'] == 0


def test_extract_features_from_dataset():
    df = pd.DataFrame({
        'url': [
            'https://site-a.com',
            'http://10.0.0.1/bad',
            'https://bit.ly/test'
        ],
        'label': [0, 1, 1]
    })
    features_df = extract_features_from_dataset(df, url_col='url', verbose=False)
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) == 3
    assert list(features_df.columns) == FEATURE_COLUMNS
