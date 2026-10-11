"""
SENTINEL API Server — Backend for Multi-Layer Phishing Detection & Risk Assessment.

Provides REST API endpoints and serves the static frontend:
- POST /api/predict: Extract 28 features, infer ML class, and evaluate risk score
- GET /api/model-metrics: Return real model evaluation metrics and comparison table
- GET /api/health: Return server health, loaded model, and feature schema
- Static files: Serves index.html, styles.css, app.js, and results/visualizations/*
"""

import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import FEATURE_COLUMNS, extract_url_features
from src.model import load_model, predict_single_url_features
from src.risk_assessment import DEFAULT_HIGH_THRESHOLD, DEFAULT_LOW_THRESHOLD, RiskAssessmentEngine

# Global singleton model and risk engine
_LOADED_MODEL = None
_LOADED_METADATA: Dict[str, Any] = {}
_RISK_ENGINE = RiskAssessmentEngine(low_threshold=DEFAULT_LOW_THRESHOLD, high_threshold=DEFAULT_HIGH_THRESHOLD)


def initialize_model(model_path: Optional[Path] = None) -> bool:
    """Load the trained machine learning model and metadata into memory."""
    global _LOADED_MODEL, _LOADED_METADATA
    if model_path is None:
        model_path = PROJECT_ROOT / "models" / "best_model.joblib"

    metadata_path = PROJECT_ROOT / "models" / "best_model_metadata.json"

    if not model_path.exists():
        print(f"[Warning] Model file not found at '{model_path}'. Attempting backup artifact...")
        backup_path = PROJECT_ROOT / "models" / "phishing_model.joblib"
        if backup_path.exists():
            model_path = backup_path
        else:
            print("[Error] No model artifact available. Run 'python main.py --train' first.")
            return False

    try:
        _LOADED_MODEL, _LOADED_METADATA = load_model(
            model_path=model_path,
            metadata_path=metadata_path if metadata_path.exists() else None
        )
        model_name = _LOADED_METADATA.get("model_name", "Trained Classifier")
        print(f"[SENTINEL] Model successfully loaded: '{model_name}' ({len(FEATURE_COLUMNS)} features).")
        return True
    except Exception as exc:
        print(f"[Error] Failed to load model: {exc}")
        return False


def get_model_metrics() -> Dict[str, Any]:
    """Load model evaluation report and comparison CSV from results/reports/."""
    report_path = PROJECT_ROOT / "results" / "reports" / "model_evaluation_report.json"
    csv_path = PROJECT_ROOT / "results" / "reports" / "model_comparison.csv"

    response_data: Dict[str, Any] = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report": None,
        "comparison_table": []
    }

    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                response_data["report"] = json.load(f)
        except Exception as e:
            response_data["report_error"] = str(e)

    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            response_data["comparison_table"] = df.to_dict(orient="records")
        except Exception:
            # Fallback csv reader without pandas if needed
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                response_data["comparison_table"] = list(reader)

    return response_data


class SentinelRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler supporting both REST API endpoints and static assets."""

    def __init__(self, *args, **kwargs):
        # Set directory to project root for static file resolution
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self):
        """Append security and CORS headers to all responses."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.end_headers()

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Helper to serialize and transmit JSON payload."""
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        """Route GET requests: APIs vs Static Files."""
        parsed_path = urlparse(self.path).path

        if parsed_path == "/api/health":
            self.handle_health()
        elif parsed_path == "/api/model-metrics":
            self.handle_metrics()
        elif parsed_path.startswith("/api/"):
            self._send_json_response({"error": "Endpoint not found"}, status_code=404)
        else:
            # Serve static files (index.html, styles.css, app.js, results/visualizations/*)
            super().do_GET()

    def do_POST(self):
        """Route POST requests."""
        parsed_path = urlparse(self.path).path

        if parsed_path == "/api/predict":
            self.handle_predict()
        else:
            self._send_json_response({"error": "Endpoint not found"}, status_code=404)

    def handle_health(self):
        """Return server readiness and model metadata."""
        is_ready = _LOADED_MODEL is not None
        model_name = _LOADED_METADATA.get("model_name", "Not Loaded") if is_ready else None

        self._send_json_response({
            "status": "online" if is_ready else "degraded",
            "model_ready": is_ready,
            "model_name": model_name,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_schema": FEATURE_COLUMNS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": "SENTINEL Security Engine v2.0"
        })

    def handle_metrics(self):
        """Return genuine evaluation metrics for all 5 algorithms."""
        metrics_data = get_model_metrics()
        self._send_json_response(metrics_data)

    def handle_predict(self):
        """Process URL prediction request via ML model and Risk Engine."""
        global _LOADED_MODEL, _LOADED_METADATA, _RISK_ENGINE

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 or content_length > 65536:  # Max 64KB for safety
            self._send_json_response({"error": "Invalid request payload size"}, status_code=400)
            return

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(raw_body)
        except Exception:
            self._send_json_response({"error": "Invalid JSON format"}, status_code=400)
            return

        url = body.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            self._send_json_response({"error": "A non-empty URL string is required."}, status_code=400)
            return

        url = url.strip()

        # Check model readiness
        if _LOADED_MODEL is None:
            if not initialize_model():
                self._send_json_response(
                    {"error": "ML model artifact is currently not available on server."},
                    status_code=503
                )
                return

        start_time = time.perf_counter()

        try:
            # 1. Offline Lexical Feature Extraction (28 features)
            features = extract_url_features(url)

            # 2. Machine Learning Model Inference
            pred_res = predict_single_url_features(
                model=_LOADED_MODEL,
                features_dict=features,
                feature_names=FEATURE_COLUMNS
            )

            # 3. Multi-Layer Risk Assessment
            threshold_low = float(body.get("low_threshold", _RISK_ENGINE.low_threshold))
            threshold_high = float(body.get("high_threshold", _RISK_ENGINE.high_threshold))

            engine = _RISK_ENGINE
            if threshold_low != engine.low_threshold or threshold_high != engine.high_threshold:
                try:
                    engine = RiskAssessmentEngine(low_threshold=threshold_low, high_threshold=threshold_high)
                except ValueError:
                    engine = _RISK_ENGINE

            risk_assessment = engine.assess_risk(
                url=url,
                features=features,
                prediction_result=pred_res
            )

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Build sanitized response payload
            response = {
                "success": True,
                "url": url,
                "predicted_class": risk_assessment["predicted_class"],
                "classification": risk_assessment["classification"],
                "risk_level": risk_assessment["risk_level"],
                "risk_score": risk_assessment["risk_score"],
                "phishing_probability": risk_assessment["phishing_probability"],
                "confidence": risk_assessment["confidence"],
                "thresholds": risk_assessment["thresholds"],
                "reason_summary": risk_assessment["reason_summary"],
                "escalated_by_heuristics": risk_assessment["escalated_by_heuristics"],
                "critical_flags": risk_assessment["critical_flags"],
                "warnings": risk_assessment["warnings"],
                "safe_signals": risk_assessment["safe_signals"],
                "disclaimer": risk_assessment["disclaimer"],
                "features": features,
                "model_name": _LOADED_METADATA.get("model_name", "Gradient Boosting"),
                "scan_time_ms": elapsed_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self._send_json_response(response, status_code=200)

        except Exception as exc:
            # Do not leak internal stack traces
            self._send_json_response(
                {"error": f"Failed to analyze URL: {str(exc)}"},
                status_code=500
            )


def run_server(port: int = 8000, host: str = "127.0.0.1") -> None:
    """Start the SENTINEL API and web application server."""
    initialize_model()

    server_address = (host, port)
    try:
        httpd = ThreadingHTTPServer(server_address, SentinelRequestHandler)
    except OSError as err:
        if "address already in use" in str(err).lower() or err.errno == 10048 or err.errno == 98:
            print(f"[Warning] Port {port} is busy. Trying port {port + 1}...")
            server_address = (host, port + 1)
            httpd = ThreadingHTTPServer(server_address, SentinelRequestHandler)
            port = port + 1
        else:
            raise

    url = f"http://{host}:{port}/"
    print("\n" + "=" * 70)
    print("  🛡️  SENTINEL — Phishing Detection & Risk Assessment Server")
    print("=" * 70)
    print(f"  • Web Dashboard & API: {url}")
    print(f"  • Health Check:        {url}api/health")
    print(f"  • Model Metrics:       {url}api/model-metrics")
    print(f"  • Live Prediction:     POST {url}api/predict")
    print("-" * 70)
    print("  Press CTRL+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SENTINEL] Shutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start SENTINEL Security Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)
