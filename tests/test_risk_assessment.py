"""
Unit tests for the Risk Assessment System.
Multi-Layer Phishing Detection & Risk Assessment.
"""

import pytest
from src.risk_assessment import DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, RiskAssessmentEngine, assess_url_risk


def test_engine_threshold_validation():
    """Verify threshold boundary validation upon initialization."""
    # Valid initializations
    engine = RiskAssessmentEngine(0.35, 0.70)
    assert engine.low_threshold == 0.35
    assert engine.high_threshold == 0.70

    # Invalid: low >= high
    with pytest.raises(ValueError):
        RiskAssessmentEngine(0.70, 0.35)

    with pytest.raises(ValueError):
        RiskAssessmentEngine(0.50, 0.50)

    # Invalid: out of bounds [0, 1]
    with pytest.raises(ValueError):
        RiskAssessmentEngine(-0.1, 0.7)

    with pytest.raises(ValueError):
        RiskAssessmentEngine(0.2, 1.2)


def test_risk_levels_across_threshold_boundaries():
    """Verify LOW, MEDIUM, and HIGH risk assignment based on model probability."""
    engine = RiskAssessmentEngine(low_threshold=0.35, high_threshold=0.70)
    clean_features = {
        "has_ip_address": 0,
        "has_at_symbol": 0,
        "has_url_shortener": 0,
        "has_suspicious_keyword": 0,
        "num_subdomains": 1,
        "url_length": 25,
        "is_https": 1
    }

    # Low risk (prob < 0.35)
    pred_low = {"predicted_class": 0, "phishing_probability": 0.15, "confidence": 0.85}
    res_low = engine.assess_risk("https://example.com", clean_features, pred_low)
    assert res_low["risk_level"] == "LOW"
    assert res_low["classification"] == "LEGITIMATE"
    assert res_low["phishing_probability"] == 0.15

    # Medium risk (0.35 <= prob < 0.70)
    pred_med = {"predicted_class": 1, "phishing_probability": 0.55, "confidence": 0.55}
    res_med = engine.assess_risk("https://example-borderline.com", clean_features, pred_med)
    assert res_med["risk_level"] == "MEDIUM"

    # High risk (prob >= 0.70)
    pred_high = {"predicted_class": 1, "phishing_probability": 0.88, "confidence": 0.88}
    res_high = engine.assess_risk("https://phishing-threat.com", clean_features, pred_high)
    assert res_high["risk_level"] == "HIGH"
    assert res_high["classification"] == "PHISHING"


def test_heuristic_risk_escalation():
    """Verify risk escalates when low probability model meets critical security anomalies."""
    engine = RiskAssessmentEngine(low_threshold=0.35, high_threshold=0.70)

    # Model gives low probability (e.g., 0.20), but URL uses raw IP address
    ip_features = {
        "has_ip_address": 1,
        "has_at_symbol": 0,
        "has_url_shortener": 0,
        "num_subdomains": 0,
        "url_length": 25,
        "is_https": 0
    }
    pred = {"predicted_class": 0, "phishing_probability": 0.20, "confidence": 0.80}
    res = engine.assess_risk("http://192.168.1.1/index.html", ip_features, pred)

    assert res["risk_level"] == "MEDIUM"
    assert res["escalated_by_heuristics"] is True
    assert "Direct IP address hosting" in res["critical_flags"]


def test_warning_signals_and_explanations():
    """Verify specific warning indicators trigger appropriate explanatory messages."""
    engine = RiskAssessmentEngine()
    features = {
        "has_ip_address": 1,
        "has_url_shortener": 1,
        "has_suspicious_keyword": 1,
        "has_at_symbol": 1,
        "has_encoded_characters": 1,
        "num_percent_symbols": 3,
        "num_subdomains": 4,
        "hostname_has_hyphen": 1,
        "url_length": 120,
        "num_query_parameters": 4,
        "is_https": 0
    }
    warnings, safe_signals = engine.analyze_url_features(features)

    assert any("Raw IP Address Hostname" in w for w in warnings)
    assert any("URL Shortening Service" in w for w in warnings)
    assert any("Sensitive Authentication Lure" in w for w in warnings)
    assert any("Authority Override Symbol" in w for w in warnings)
    assert any("Excessive Subdomain Nesting" in w for w in warnings)
    assert any("Unencrypted HTTP Protocol" in w for w in warnings)


def test_convenience_function():
    """Test assess_url_risk convenience wrapper."""
    res = assess_url_risk(
        url="https://legit-site.org",
        features={"has_ip_address": 0, "has_at_symbol": 0, "has_url_shortener": 0, "num_subdomains": 1},
        prediction_result={"predicted_class": 0, "phishing_probability": 0.05, "confidence": 0.95}
    )
    assert res["risk_level"] == "LOW"
    assert "disclaimer" in res
    assert "DISCLAIMER" in res["disclaimer"]
