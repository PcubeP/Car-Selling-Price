"""Generate a realistic *synthetic* India used-car dataset.

Real used-car listings sit behind Kaggle/marketplace logins, so this builds a
reproducible dataset from a transparent generative model: a base on-road price
per model, then age depreciation, odometer wear, and fuel/transmission/owner
adjustments plus noise. Prices are in Indian Rupees (INR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REF_YEAR = 2024
RNG = np.random.default_rng(42)
N_ROWS = 6000

LAKH = 100_000

# model -> (brand, base on-road price in lakh, plausible fuels, body type)
MODELS: dict[str, tuple[str, float, list[str], str]] = {
    "Alto": ("Maruti", 5.0, ["Petrol", "CNG"], "Hatchback"),
    "WagonR": ("Maruti", 6.5, ["Petrol", "CNG"], "Hatchback"),
    "Swift": ("Maruti", 8.0, ["Petrol", "Diesel"], "Hatchback"),
    "Baleno": ("Maruti", 9.0, ["Petrol"], "Hatchback"),
    "Dzire": ("Maruti", 8.5, ["Petrol", "CNG"], "Sedan"),
    "Ertiga": ("Maruti", 12.0, ["Petrol", "CNG"], "MPV"),
    "i20": ("Hyundai", 9.5, ["Petrol", "Diesel"], "Hatchback"),
    "Venue": ("Hyundai", 11.5, ["Petrol", "Diesel"], "SUV"),
    "Creta": ("Hyundai", 16.0, ["Petrol", "Diesel"], "SUV"),
    "Amaze": ("Honda", 8.5, ["Petrol", "Diesel"], "Sedan"),
    "City": ("Honda", 14.0, ["Petrol", "Diesel"], "Sedan"),
    "Punch": ("Tata", 8.0, ["Petrol", "CNG"], "SUV"),
    "Nexon": ("Tata", 12.0, ["Petrol", "Diesel"], "SUV"),
    "Thar": ("Mahindra", 15.5, ["Petrol", "Diesel"], "SUV"),
    "Scorpio": ("Mahindra", 17.0, ["Diesel"], "SUV"),
    "XUV700": ("Mahindra", 22.0, ["Petrol", "Diesel"], "SUV"),
    "Innova Crysta": ("Toyota", 22.0, ["Diesel"], "MPV"),
    "Fortuner": ("Toyota", 40.0, ["Diesel"], "SUV"),
    "Sonet": ("Kia", 11.5, ["Petrol", "Diesel"], "SUV"),
    "Seltos": ("Kia", 16.5, ["Petrol", "Diesel"], "SUV"),
}

FUEL_ADJ = {"Petrol": 1.0, "Diesel": 1.05, "CNG": 0.96}
TRANS_ADJ = {"Manual": 1.0, "Automatic": 1.07}
OWNER_ADJ = {"First": 1.0, "Second": 0.92, "Third": 0.84}
OWNER_P = {"First": 0.6, "Second": 0.3, "Third": 0.1}


def depreciation(age: int) -> float:
    # ~15% off in the first year, then ~11%/yr, with a resale floor.
    if age <= 0:
        return 0.96
    return max(0.15, 0.85 * (0.89 ** (age - 1)))


def sample_rows(n: int) -> pd.DataFrame:
    names = list(MODELS)
    rows = []
    for _ in range(n):
        model = names[RNG.integers(len(names))]
        brand, base_lakh, fuels, body = MODELS[model]

        age = int(RNG.integers(0, 15))  # 0..14 years
        year = REF_YEAR - age

        # Older cars tend to have more km; ~12k km/yr with spread, plus a base.
        km = int(max(500, RNG.normal(12_000 * max(age, 0.4), 18_000)))
        km = min(km, 260_000)

        fuel = fuels[RNG.integers(len(fuels))]
        transmission = "Automatic" if RNG.random() < (0.15 + 0.02 * (base_lakh > 12)) else "Manual"
        owner = RNG.choice(list(OWNER_P), p=list(OWNER_P.values()))

        km_factor = 1.0 - min(0.28, (km / 260_000) * 0.28)
        noise = RNG.normal(1.0, 0.06)

        price = (
            base_lakh * LAKH
            * depreciation(age)
            * km_factor
            * FUEL_ADJ[fuel]
            * TRANS_ADJ[transmission]
            * OWNER_ADJ[owner]
            * noise
        )
        price = int(round(max(60_000, price) / 1000) * 1000)  # round to nearest 1k

        rows.append(
            dict(
                brand=brand,
                model=model,
                body_type=body,
                year=year,
                km_driven=km,
                fuel=fuel,
                transmission=transmission,
                owner=owner,
                price=price,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    import os

    df = sample_rows(N_ROWS)
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "used_cars.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    print(df.describe(include="all").T[["count", "unique", "top", "freq"]])


if __name__ == "__main__":
    main()
