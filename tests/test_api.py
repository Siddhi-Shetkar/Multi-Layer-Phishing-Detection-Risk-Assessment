"""
Unit tests for the SENTINEL API Server and Endpoints.
Multi-Layer Phishing Detection & Risk Assessment.
"""

import json
from pathlib import Path
import pytest
from server import get_model_metrics, initialize_model
from src.feature_extraction import FEATURE_COLUMNS
from src.model import load_model, predict_single_url_features
from src.risk_assessment import RiskAssessmentEngine


def test_server_initialize_model():
    """Verify that initialize_model successfully locates and loads model artifacts."""
    loaded = initialize_model()
    assert loaded is True, "initialize_model should return True when best_model.joblib exists"


def test_get_model_metrics():
    """Verify that get_model_metrics retrieves genuine evaluation report and tabular comparison."""
    metrics = get_model_metrics()
    assert metrics["status"] == "success"
    assert "report" in metrics
    assert "comparison_table" in metrics
    assert isinstance(metrics["comparison_table"], list)
    assert len(metrics["comparison_table"]) >= 5, "Should have metrics for at least 5 evaluated models"

    # Verify best model details
    report = metrics["report"]
    assert report is not None
    assert "best_model" in report
    assert "all_models_ranked" in report
    assert report["feature_count"] == 28


def test_predict_endpoint_logic():
    """Test the complete prediction pipeline invoked by the API predict handler."""
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        pytest.skip("Model artifact missing")

    model, meta = load_model(model_path)
    engine = RiskAssessmentEngine()

    from src.feature_extraction import extract_url_features

    # Test legitimate URL
    legit_url = "https://www.google.com"
    legit_features = extract_url_features(legit_url)
    assert len(legit_features) == 28
    legit_pred = predict_single_url_features(model, legit_features, FEATURE_COLUMNS)
    legit_risk = engine.assess_risk(legit_url, legit_features, legit_pred)

    assert legit_risk["risk_level"] in ("LOW", "MEDIUM")
    assert 0.0 <= legit_risk["phishing_probability"] <= 1.0

    # Test phishing lure
    phish_url = "http://192.168.1.100/paypal-verify/login.php"
    phish_features = extract_url_features(phish_url)
    phish_pred = predict_single_url_features(model, phish_features, FEATURE_COLUMNS)
    phish_risk = engine.assess_risk(phish_url, phish_features, phish_pred)

    assert phish_risk["classification"] == "PHISHING"
    assert phish_risk["risk_level"] in ("MEDIUM", "HIGH")
    assert any("Raw IP Address Hostname" in w for w in phish_risk["warnings"])
