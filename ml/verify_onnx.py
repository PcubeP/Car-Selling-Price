"""Sanity-check that the exported ONNX model matches the sklearn predictions.

Loads reference rows saved by train.py, runs them through onnxruntime with the
same per-column string/float inputs the browser will use, and compares.
"""
from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "public", "model"))
CATEGORICAL = ["brand", "model", "body_type", "fuel", "transmission", "owner"]
NUMERIC = ["year", "km_driven"]


def main() -> None:
    with open(os.path.join(MODEL_DIR, "reference_predictions.json")) as f:
        refs = json.load(f)
    with open(os.path.join(MODEL_DIR, "schema.json")) as f:
        out_name = json.load(f)["outputName"]

    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "car_price.onnx"))

    max_diff = 0.0
    for row in refs:
        feeds = {c: np.array([[row[c]]], dtype=object) for c in CATEGORICAL}
        for c in NUMERIC:
            feeds[c] = np.array([[row[c]]], dtype=np.float32)
        onnx_pred = float(sess.run([out_name], feeds)[0][0][0])
        diff = abs(onnx_pred - row["pred"])
        max_diff = max(max_diff, diff)
        print(f"sklearn={row['pred']:>12.2f}  onnx={onnx_pred:>12.2f}  diff={diff:.4f}")

    print(f"\nMax abs diff across {len(refs)} rows: {max_diff:.4f}")
    assert max_diff < 1.0, "ONNX and sklearn predictions diverge!"
    print("PARITY OK")


if __name__ == "__main__":
    main()
