"""
Multi-Layer Phishing Detection & Risk Assessment - Core Source Package.
Contributors: Siddhi Shetkar & Sankeerthana Thoviti
"""

from src import (
    data_loader,
    data_preparation,
    eda,
    feature_extraction,
    preprocessing,
    model,
    risk_assessment,
)

__all__ = [
    "data_loader",
    "preprocessing",
    "feature_extraction",
    "eda",
    "data_preparation",
    "model",
    "risk_assessment",
]
