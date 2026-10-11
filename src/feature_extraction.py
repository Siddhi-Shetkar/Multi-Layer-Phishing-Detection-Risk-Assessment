"""
Module for extracting structural, lexical, host, and suspicious pattern features from URLs.

Part of Part 1: Multi-Layer Phishing Detection & Risk Assessment.
"""

import ipaddress
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import urllib.parse
import pandas as pd
import tldextract


# Configurable list of known URL shortening domains
DEFAULT_SHORTENER_DOMAINS: Set[str] = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'short.to',
    'cutt.ly', 'rb.gy', 'tiny.cc', 'lnkd.in', 'rebrand.ly', 'shorte.st',
    'trib.al', 'v.gd', 'qr.net', '1url.com', 'clck.ru', 's.id', 'vzturl.com',
    'tr.im', 'u.to', 'wp.me', 'post.ly'
}

# Configurable list of suspicious terms common in phishing lures
DEFAULT_SUSPICIOUS_KEYWORDS: List[str] = [
    'login', 'verify', 'update', 'account', 'password', 'secure',
    'banking', 'confirm', 'signin', 'security', 'wallet', 'authenticate',
    'credential', 'billing', 'support', 'recover'
]

# Standardized feature column names in consistent fixed order
FEATURE_COLUMNS: List[str] = [
    # A. Length and structure features
    'url_length',
    'hostname_length',
    'path_length',
    'query_length',
    'num_dots',
    'num_hyphens',
    'num_underscores',
    'num_slashes',
    'num_question_marks',
    'num_equal_signs',
    'num_ampersands',
    'num_at_symbols',
    'num_percent_symbols',
    'num_digits',
    'num_special_characters',
    # B. Hostname and domain features
    'has_ip_address',
    'hostname_has_digits',
    'hostname_has_hyphen',
    'num_subdomains',
    'has_multiple_subdomains',
    'is_https',
    'has_explicit_port',
    'has_unusual_port',
    'num_query_parameters',
    # C. Suspicious-pattern features
    'has_url_shortener',
    'has_suspicious_keyword',
    'has_encoded_characters',
    'has_at_symbol',
]

# Offline Public Suffix List extractor (avoids internet calls)
_offline_tld_extractor = tldextract.TLDExtract(cache_dir=False, suffix_list_urls=())


def parse_url_components(url: str) -> Tuple[str, str, str, str, str, Optional[int]]:
    """
    Safely parse URL components without throwing unhandled exceptions.

    Args:
        url: Raw URL string.

    Returns:
        Tuple of (scheme, hostname, path, query, fragment, port)
    """
    url_str = str(url).strip() if url is not None else ""
    if not url_str:
        return "", "", "", "", "", None

    # Handle scheme-less URLs
    if "://" in url_str:
        candidate_url = url_str
        has_scheme = True
    else:
        candidate_url = "//" + url_str
        has_scheme = False

    try:
        split_res = urllib.parse.urlsplit(candidate_url)
        scheme = split_res.scheme.lower() if has_scheme else ""
        raw_netloc = split_res.netloc
        hostname = (split_res.hostname or "").lower()
        path = split_res.path or ""
        query = split_res.query or ""
        fragment = split_res.fragment or ""

        # Extract port safely
        try:
            port = split_res.port
        except ValueError:
            # e.g., invalid port representation in netloc like ':999999' or ':abc'
            port = -1
    except Exception:
        # Fallback for unparseable URLs
        scheme = ""
        hostname = ""
        path = url_str
        query = ""
        fragment = ""
        port = None

    return scheme, hostname, path, query, fragment, port


def check_is_ip_address(hostname: str) -> int:
    """
    Determine if hostname is an IPv4 or IPv6 address.

    Args:
        hostname: Lowercase parsed hostname.

    Returns:
        1 if valid IP address, 0 otherwise.
    """
    if not hostname:
        return 0
    # Strip brackets from IPv6 representations e.g. [2001:db8::1]
    clean_host = hostname.strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return 1
    except ValueError:
        return 0


def extract_url_features(
    url: str,
    shortener_domains: Optional[Set[str]] = None,
    suspicious_keywords: Optional[List[str]] = None
) -> Dict[str, Union[int, float]]:
    """
    Extract a dictionary of consistently named numeric features from a single URL string.

    Args:
        url: Input URL string.
        shortener_domains: Optional custom set of shortening domains.
        suspicious_keywords: Optional custom list of suspicious terms.

    Returns:
        Dictionary of numeric feature values adhering to FEATURE_COLUMNS.
    """
    url_str = str(url).strip() if url is not None else ""
    shorteners = shortener_domains if shortener_domains is not None else DEFAULT_SHORTENER_DOMAINS
    keywords = suspicious_keywords if suspicious_keywords is not None else DEFAULT_SUSPICIOUS_KEYWORDS

    # 1. Parse components
    scheme, hostname, path, query, fragment, port = parse_url_components(url_str)

    # A. Length and structure features
    url_length = len(url_str)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    num_dots = url_str.count('.')
    num_hyphens = url_str.count('-')
    num_underscores = url_str.count('_')
    num_slashes = url_str.count('/')
    num_question_marks = url_str.count('?')
    num_equal_signs = url_str.count('=')
    num_ampersands = url_str.count('&')
    num_at_symbols = url_str.count('@')
    num_percent_symbols = url_str.count('%')
    num_digits = sum(1 for c in url_str if c.isdigit())
    num_special_characters = sum(1 for c in url_str if not c.isalnum())

    # B. Hostname and domain features
    has_ip = check_is_ip_address(hostname)
    hostname_has_digits = 1 if any(c.isdigit() for c in hostname) else 0
    hostname_has_hyphen = 1 if '-' in hostname else 0

    # Subdomains extraction via PSL-aware tldextract
    # Note on limitation: For local networks or unregistered TLDs, PSL may fall back to root label.
    if hostname and not has_ip:
        try:
            ext = _offline_tld_extractor(url_str)
            if ext.subdomain:
                sub_parts = [s for s in ext.subdomain.split('.') if s]
                num_subdomains = len(sub_parts)
            else:
                num_subdomains = 0
        except Exception:
            # Heuristic fallback if PSL extraction fails
            host_dots = hostname.count('.')
            num_subdomains = max(0, host_dots - 1)
    else:
        num_subdomains = 0

    has_multiple_subdomains = 1 if num_subdomains > 1 else 0
    is_https = 1 if scheme == 'https' else 0

    # Port features
    if port is not None:
        has_explicit_port = 1
        # Common web ports are standard HTTP (80) and HTTPS (443)
        has_unusual_port = 1 if port not in (80, 443) else 0
    else:
        has_explicit_port = 0
        has_unusual_port = 0

    # Query parameters
    if query:
        try:
            parsed_params = urllib.parse.parse_qsl(query, keep_blank_values=True)
            num_query_parameters = len(parsed_params)
        except Exception:
            num_query_parameters = query.count('&') + 1
    else:
        num_query_parameters = 0

    # C. Suspicious-pattern features
    # Check if registered domain or host matches known shorteners
    has_url_shortener = 0
    if hostname:
        if hostname in shorteners:
            has_url_shortener = 1
        else:
            # Check suffix / domain
            for short_dom in shorteners:
                if hostname.endswith('.' + short_dom) or hostname == short_dom:
                    has_url_shortener = 1
                    break

    # Check suspicious keywords in full URL (case-insensitive)
    url_lower = url_str.lower()
    has_suspicious_keyword = 1 if any(kw in url_lower for kw in keywords) else 0

    # Percent encoded
    has_encoded_characters = 1 if '%' in url_str else 0

    # @ symbol
    has_at_symbol = 1 if '@' in url_str else 0

    # As instructed: Reuse the hostname-based IP address indicator rather than creating duplicate columns
    # 'has_ip_address' is captured above and included once.

    features: Dict[str, Union[int, float]] = {
        'url_length': int(url_length),
        'hostname_length': int(hostname_length),
        'path_length': int(path_length),
        'query_length': int(query_length),
        'num_dots': int(num_dots),
        'num_hyphens': int(num_hyphens),
        'num_underscores': int(num_underscores),
        'num_slashes': int(num_slashes),
        'num_question_marks': int(num_question_marks),
        'num_equal_signs': int(num_equal_signs),
        'num_ampersands': int(num_ampersands),
        'num_at_symbols': int(num_at_symbols),
        'num_percent_symbols': int(num_percent_symbols),
        'num_digits': int(num_digits),
        'num_special_characters': int(num_special_characters),
        'has_ip_address': int(has_ip),
        'hostname_has_digits': int(hostname_has_digits),
        'hostname_has_hyphen': int(hostname_has_hyphen),
        'num_subdomains': int(num_subdomains),
        'has_multiple_subdomains': int(has_multiple_subdomains),
        'is_https': int(is_https),
        'has_explicit_port': int(has_explicit_port),
        'has_unusual_port': int(has_unusual_port),
        'num_query_parameters': int(num_query_parameters),
        'has_url_shortener': int(has_url_shortener),
        'has_suspicious_keyword': int(has_suspicious_keyword),
        'has_encoded_characters': int(has_encoded_characters),
        'has_at_symbol': int(has_at_symbol),
    }

    return features


def extract_features_from_dataset(
    df: pd.DataFrame,
    url_col: str = 'url',
    batch_size: int = 10000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract features for all URLs in the supplied DataFrame and return a DataFrame
    containing the features in a consistent column order.

    Args:
        df: Input DataFrame containing URLs.
        url_col: Column name containing URL strings.
        batch_size: Size of batches for memory efficiency and progress logging.
        verbose: Whether to log extraction progress.

    Returns:
        DataFrame containing extracted numeric features in FEATURE_COLUMNS order.
    """
    if url_col not in df.columns:
        # Fallback to case-insensitive lookup
        found_col = None
        for c in df.columns:
            if c.lower() == url_col.lower():
                found_col = c
                break
        if found_col is None:
            raise ValueError(f"Column '{url_col}' not found in DataFrame columns: {list(df.columns)}")
        url_col = found_col

    urls = df[url_col].tolist()
    total_urls = len(urls)

    if verbose:
        print(f"Extracting features for {total_urls:,} URLs (batch size: {batch_size:,})...")

    feature_records: List[Dict[str, Union[int, float]]] = []

    for idx, u in enumerate(urls):
        feat = extract_url_features(u)
        feature_records.append(feat)

        if verbose and (idx + 1) % batch_size == 0:
            print(f"  Processed {idx + 1:,} / {total_urls:,} URLs ({(idx + 1) / total_urls * 100:.1f}%)")

    if verbose and total_urls % batch_size != 0:
        print(f"  Processed {total_urls:,} / {total_urls:,} URLs (100.0%)")

    features_df = pd.DataFrame(feature_records, columns=FEATURE_COLUMNS)
    return features_df
