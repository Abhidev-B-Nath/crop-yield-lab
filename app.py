"""Local web app for demonstrating crop yield prediction."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from crop_yield_model import load_and_clean_data, train_model

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
ROOT = Path(__file__).parent
DATASET = ROOT / "crop_yield.csv"
STATIC_FILE = ROOT / "index.html"

MODEL, METRICS, _EVALUATION, TRAINING_SAMPLES = train_model(DATASET)
DATA = load_and_clean_data(DATASET)
FEATURES = [column for column in DATA.columns if column != "Yield"]
NUMERICAL_FEATURES = DATA[FEATURES].select_dtypes(include=["number"]).columns.tolist()
CATEGORICAL_FEATURES = [column for column in FEATURES if column not in NUMERICAL_FEATURES]
CATEGORY_OPTIONS = {
    column: sorted(DATA[column].dropna().astype(str).unique().tolist())
    for column in CATEGORICAL_FEATURES
}


def json_response(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def predict(sample: dict[str, Any]) -> float:
    values = {feature: sample.get(feature) for feature in FEATURES}
    frame = pd.DataFrame([values])
    for feature in NUMERICAL_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return float(MODEL.predict(frame)[0])


class CropYieldHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json_response(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(
                {
                    "training_samples": TRAINING_SAMPLES,
                    "testing_samples": len(_EVALUATION),
                    "metrics": METRICS,
                    "features": FEATURES,
                    "numerical_features": NUMERICAL_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "category_options": CATEGORY_OPTIONS,
                }
            )
            return
        if path in {"/", "/index.html"}:
            body = STATIC_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json({"error": "Not found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            sample = json.loads(self.rfile.read(length))
            if not isinstance(sample, dict):
                raise ValueError("Request must be a JSON object.")
            missing = [feature for feature in FEATURES if sample.get(feature) in (None, "")]
            if missing:
                raise ValueError(f"Enter values for: {', '.join(missing)}")
            result = predict(sample)
            self.send_json({"prediction": result})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, status=400)
        except Exception:
            self.send_json({"error": "The sample could not be scored."}, status=500)


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CropYieldHandler)
    print(f"Crop Yield Lab running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
