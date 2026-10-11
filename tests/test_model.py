"""
Unit tests for Machine Learning Model Training, Evaluation, and Serialization.
Multi-Layer Phishing Detection & Risk Assessment.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.feature_extraction import FEATURE_COLUMNS
from src.model import (
    evaluate_model,
    get_model_candidates,
    load_model,
    predict_single_url_features,
    save_model_artifacts,
    train_model,
)


@pytest.fixture
def sample_dataset():
    """Create a minimal synthetic dataset matching the exact 28 feature schema."""
    np.random.seed(42)
    n_samples = 40
    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = np.random.randint(0, 10, size=n_samples)

    X = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    y = pd.Series([0 if i < 20 else 1 for i in range(n_samples)], name="label")
    return X, y


def test_get_model_candidates():
    """Verify all 5 required model architectures are registered and instantiable."""
    candidates = get_model_candidates(random_state=42)
    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "Gradient Boosting",
    ]
    for model_name in expected_models:
        assert model_name in candidates, f"Model '{model_name}' missing from candidates"
        assert hasattr(candidates[model_name], "fit"), f"{model_name} must have fit method"


def test_train_and_evaluate_model(sample_dataset):
    """Test model training and comprehensive metric computation."""
    X, y = sample_dataset
    candidates = get_model_candidates(random_state=42)
    dt_model = candidates["Decision Tree"]

    fitted_model, elapsed = train_model("Decision Tree", dt_model, X, y, verbose=False)
    assert elapsed >= 0.0
    assert hasattr(fitted_model, "classes_")

    eval_result = evaluate_model("Decision Tree", fitted_model, X, y)
    assert eval_result["model_name"] == "Decision Tree"
    assert 0.0 <= eval_result["accuracy"] <= 1.0
    assert 0.0 <= eval_result["precision"] <= 1.0
    assert 0.0 <= eval_result["recall"] <= 1.0
    assert 0.0 <= eval_result["f1_score"] <= 1.0
    assert not np.isnan(eval_result["roc_auc"])
    assert "confusion_matrix" in eval_result
    assert len(eval_result["confusion_matrix"]) == 2


def test_model_serialization_and_loading(sample_dataset):
    """Test model saving, metadata generation, and artifact loading."""
    X, y = sample_dataset
    candidates = get_model_candidates(random_state=42)
    model = candidates["Logistic Regression"]
    model, _ = train_model("Logistic Regression", model, X, y, verbose=False)
    eval_res = evaluate_model("Logistic Regression", model, X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_p, meta_p = save_model_artifacts(
            model=model,
            model_name="Logistic Regression",
            feature_names=FEATURE_COLUMNS,
            metrics=eval_res,
            output_dir=tmp_dir,
            filename_prefix="test_model"
        )

        assert model_p.exists()
        assert meta_p.exists()

        # Check metadata contents
        with open(meta_p, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        assert meta_json["model_name"] == "Logistic Regression"
        assert meta_json["num_features"] == len(FEATURE_COLUMNS)
        assert meta_json["feature_columns"] == FEATURE_COLUMNS
        assert "evaluation_metrics" in meta_json

        # Check load_model
        loaded_model, loaded_meta = load_model(model_p, meta_p)
        assert loaded_meta["model_name"] == "Logistic Regression"
        pred = loaded_model.predict(X[:2])
        assert len(pred) == 2


def test_load_model_missing_file():
    """Verify load_model raises FileNotFoundError when model path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model_dir/missing_model.joblib")


def test_predict_single_url_features(sample_dataset):
    """Test inference helper on a single feature dictionary."""
    X, y = sample_dataset
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(X, y)

    dummy_features = {col: 1 for col in FEATURE_COLUMNS}
    res = predict_single_url_features(model, dummy_features, FEATURE_COLUMNS)

    assert res["predicted_class"] == 1
    assert res["classification"] == "PHISHING"
    assert 0.0 <= res["phishing_probability"] <= 1.0
    assert 0.0 <= res["confidence"] <= 1.0


def test_predict_features_without_proba():
    """Test predict_single_url_features handles estimators without predict_proba."""
    class CustomEstimator:
        def predict(self, df):
            return np.array([0])

    model = CustomEstimator()
    dummy_features = {col: 0 for col in FEATURE_COLUMNS}
    res = predict_single_url_features(model, dummy_features, FEATURE_COLUMNS)

    assert res["predicted_class"] == 0
    assert res["classification"] == "LEGITIMATE"
    assert res["phishing_probability"] == 0.0
