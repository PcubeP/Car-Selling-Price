# India Used-Car Price Predictor

A machine-learning price estimator for India-market used cars. A gradient-boosting model is trained
in Python, exported to **ONNX**, and run **in the browser** with `onnxruntime-web` — so predictions
happen on-device with no backend.

## Features

- Live price estimate as you change inputs (model, fuel, transmission, owner, year, odometer).
- Model quality surfaced in the UI (R², mean error, test-set size).
- A "likely range" band around the estimate (± the model's mean absolute percentage error).
- Brand and body-type are derived automatically from the chosen model.
- Fully reproducible ML pipeline: dataset generation → training → ONNX export → parity check.

## Tech stack

- **ML:** Python 3.12, scikit-learn (`GradientBoostingRegressor`), `skl2onnx`, `onnxruntime`
- **Web:** React 18 + TypeScript + Vite, Tailwind CSS, framer-motion
- **Inference:** `onnxruntime-web` (WASM backend loaded from CDN)

## How it works

```
ml/generate_dataset.py ─▶ ml/data/used_cars.csv
ml/train.py            ─▶ public/model/car_price.onnx  +  public/model/schema.json
ml/verify_onnx.py      ─▶ checks ONNX == sklearn predictions
web (src/) ── fetch schema.json + car_price.onnx ──▶ onnxruntime-web ──▶ price
```

- **Dataset (`ml/generate_dataset.py`)** — real used-car listings sit behind logins, so this builds a
  transparent **synthetic** dataset (6,000 rows, fixed seed) from a documented generative model:
  a base on-road price per model, then age depreciation, odometer wear, and fuel/transmission/owner
  adjustments plus noise. 20 popular India-market models.
- **Training (`ml/train.py`)** — a scikit-learn `Pipeline` (one-hot categoricals + passthrough
  numerics → `GradientBoostingRegressor`) exported to ONNX with one named input per column
  (6 string + 2 float). It also writes `schema.json` (the UI form spec + metrics) and reference
  predictions.
- **Inference (`src/hooks/usePricePredictor.ts`)** — loads the ONNX model and schema, builds one
  tensor per input column, and runs `onnxruntime-web`. Calls are **serialized** because
  onnxruntime-web forbids concurrent `run()` on one session.

Model quality on the held-out test set: **R² ≈ 0.99, mean error ≈ 7%**.

## Getting started

### Web app

```bash
npm install
npm run dev        # http://localhost:5174
npm run build
```

### Retrain the model (optional)

```bash
cd ml
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
python generate_dataset.py     # -> ml/data/used_cars.csv
python train.py                # -> ../public/model/car_price.onnx + schema.json
python verify_onnx.py          # parity check (ONNX vs sklearn)
```

Pinned versions in `ml/requirements.txt` are the tested-working set
(numpy 1.26.4 / scipy 1.13.1 / onnx 1.16.2 / skl2onnx 1.17.0 / onnxruntime 1.20.1).

## Project structure

```
ml/
  generate_dataset.py   # synthetic dataset generator (documented assumptions)
  train.py              # train GBR + export ONNX + schema.json
  verify_onnx.py        # ONNX/sklearn parity check
  requirements.txt
public/model/
  car_price.onnx        # exported model (served by Vite)
  schema.json           # form spec + metrics
src/
  App.tsx               # live form + result card + metrics
  hooks/usePricePredictor.ts
  lib/schema.ts, lib/format.ts
```

## Notes / caveats

- The dataset is **synthetic and illustrative** — estimates demonstrate the pipeline, not real market prices.
- `onnxruntime-web` must be pinned to the exact version used for the CDN WASM path, or the wasm/JS
  glue mismatches. See `usePricePredictor.ts`.
- `skl2onnx` emits Python bools for tree-node flags that newer protobuf rejects; `train.py` includes
  a small `make_attribute` coercion to handle this during export.
