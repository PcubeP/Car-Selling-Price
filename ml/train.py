"""Train a used-car price regressor and export it to ONNX for the browser.

Pipeline: OneHotEncoder (categoricals) + raw numerics -> GradientBoostingRegressor.
Exports:
  ../public/model/car_price.onnx   (one named input per column)
  ../public/model/schema.json      (UI form spec + metrics, derived from the data)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# skl2onnx emits Python bools for TreeEnsemble node flags, which newer protobuf
# rejects in an int field. Coerce bools to ints inside make_attribute.
import onnx.helper as _oh

_orig_make_attribute = _oh.make_attribute


def _make_attribute(key, value, *args, **kwargs):
    if isinstance(value, (list, tuple)):
        value = [int(v) if isinstance(v, bool) else v for v in value]
    return _orig_make_attribute(key, value, *args, **kwargs)


_oh.make_attribute = _make_attribute

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "used_cars.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "public", "model"))

CATEGORICAL = ["brand", "model", "body_type", "fuel", "transmission", "owner"]
NUMERIC = ["year", "km_driven"]
TARGET = "price"


def build_model() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    reg = GradientBoostingRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def to_onnx(model: Pipeline) -> bytes:
    initial_types = [(c, StringTensorType([None, 1])) for c in CATEGORICAL]
    initial_types += [(c, FloatTensorType([None, 1])) for c in NUMERIC]
    onx = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=15,
    )
    return onx.SerializeToString()


def build_schema(df: pd.DataFrame, metrics: dict, output_name: str) -> dict:
    models = []
    for name, g in df.groupby("model"):
        models.append(
            {
                "model": name,
                "brand": g["brand"].iloc[0],
                "body_type": g["body_type"].iloc[0],
                "fuels": sorted(g["fuel"].unique().tolist()),
            }
        )
    models.sort(key=lambda m: (m["brand"], m["model"]))
    return {
        "priceUnit": "INR",
        "refYear": 2024,
        "outputName": output_name,
        "inputOrder": CATEGORICAL + NUMERIC,
        "numeric": {
            "year": {"min": int(df.year.min()), "max": int(df.year.max()), "default": 2019, "step": 1},
            "km_driven": {
                "min": 500,
                "max": int(df.km_driven.max()),
                "default": 45000,
                "step": 1000,
            },
        },
        "categorical": {
            "fuel": sorted(df.fuel.unique().tolist()),
            "transmission": sorted(df.transmission.unique().tolist()),
            "owner": ["First", "Second", "Third"],
        },
        "models": models,
        "metrics": metrics,
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X = df[CATEGORICAL + NUMERIC].copy()
    for c in NUMERIC:
        X[c] = X[c].astype(np.float32)
    y = df[TARGET].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    mape = float(np.mean(np.abs((y_test - pred) / y_test)) * 100)
    metrics = {"mae": round(mae), "r2": round(r2, 4), "mape": round(mape, 2), "n_test": int(len(y_test))}
    print("Metrics:", metrics)

    os.makedirs(OUT_DIR, exist_ok=True)
    onnx_bytes = to_onnx(model)
    onnx_path = os.path.join(OUT_DIR, "car_price.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_bytes)

    # Read the graph's single output name so the front-end can fetch it.
    import onnx

    output_name = onnx.load_model_from_string(onnx_bytes).graph.output[0].name

    schema = build_schema(df, metrics, output_name)
    with open(os.path.join(OUT_DIR, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    # Save a few reference predictions for parity checking against onnxruntime-web.
    sample = X_test.head(5).reset_index(drop=True)
    refs = [
        {**{k: (float(v) if k in NUMERIC else v) for k, v in row.items()}, "pred": float(p)}
        for row, p in zip(sample.to_dict("records"), model.predict(sample))
    ]
    with open(os.path.join(OUT_DIR, "reference_predictions.json"), "w") as f:
        json.dump(refs, f, indent=2)

    print(f"Wrote {onnx_path} ({len(onnx_bytes) // 1024} KB)")
    print(f"Output tensor name: {output_name}")


if __name__ == "__main__":
    main()
