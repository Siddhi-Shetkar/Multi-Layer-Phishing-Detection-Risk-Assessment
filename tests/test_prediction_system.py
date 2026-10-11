"""
End-to-end integration tests for the URL Prediction System.
Multi-Layer Phishing Detection & Risk Assessment.
"""

from pathlib import Path
import pytest

from main import predict_and_assess_url
from src.feature_extraction import FEATURE_COLUMNS, extract_url_features
from src.model import load_model
from src.risk_assessment import RiskAssessmentEngine


@pytest.fixture(scope="module")
def loaded_system():
    """Load the trained model and risk assessment engine once for integration tests."""
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        pytest.skip("Model artifact 'models/best_model.joblib' not yet generated.")
    model, meta = load_model(model_path)
    engine = RiskAssessmentEngine()
    return model, meta, engine


def test_predict_legitimate_url(loaded_system):
    """Verify prediction workflow on a legitimate URL without any network requests."""
    model, meta, engine = loaded_system
    url = "https://www.google.com/search?q=cybersecurity"

    res = predict_and_assess_url(url, model, engine, verbose=False)

    assert res["url"] == url
    assert res["classification"] in ("LEGITIMATE", "PHISHING")
    assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0.0 <= res["phishing_probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["risk_score"] <= 100.0
    assert isinstance(res["warnings"], list)
    assert isinstance(res["safe_signals"], list)
    assert "DISCLAIMER" in res["disclaimer"]


def test_predict_phishing_url(loaded_system):
    """Verify prediction workflow flags obvious malicious threat patterns."""
    model, meta, engine = loaded_system
    url = "http://192.168.1.50/secure-bank-login/account-verification?token=admin"

    res = predict_and_assess_url(url, model, engine, verbose=False)

    assert res["url"] == url
    assert res["classification"] == "PHISHING"
    assert res["risk_level"] in ("MEDIUM", "HIGH")
    assert res["phishing_probability"] > 0.5
    assert len(res["warnings"]) > 0
    assert any("Raw IP Address Hostname" in w for w in res["warnings"])


def test_feature_alignment_matches_schema():
    """Verify that extract_url_features produces exact FEATURE_COLUMNS in exact order."""
    url = "https://test.example.com/path?arg=1#fragment"
    features = extract_url_features(url)

    assert len(features) == len(FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        assert col in features
        assert isinstance(features[col], (int, float))


def test_empty_or_whitespace_url_raises_value_error(loaded_system):
    """Ensure empty or blank URL input is rejected gracefully with clear error."""
    model, meta, engine = loaded_system
    with pytest.raises(ValueError, match="Cannot predict on an empty or blank URL"):
        predict_and_assess_url("", model, engine, verbose=False)

    with pytest.raises(ValueError, match="Cannot predict on an empty or blank URL"):
        predict_and_assess_url("   ", model, engine, verbose=False)


def test_malformed_url_safety(loaded_system):
    """Ensure extremely malformed URLs are analyzed safely without crashes."""
    model, meta, engine = loaded_system
    malformed_urls = [
        "htp:/invalid-domain",
        "://no-protocol",
        "https://" + ("a" * 500) + ".com",
        "http://@@@attacker.com",
    ]

    for m_url in malformed_urls:
        res = predict_and_assess_url(m_url, model, engine, verbose=False)
        assert res["url"] == m_url
        assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert 0.0 <= res["phishing_probability"] <= 1.0
