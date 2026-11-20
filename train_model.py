"""Train the HIV prevalence XGBoost regression model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hiv_prevalence.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["Adult_Prevalence", "Annual_Deaths", "Year_of_estimate"]
TARGET = "Num_People"


def load_and_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the semicolon-delimited source CSV and normalize its columns."""
    df = pd.read_csv(path, sep=";", encoding="latin1")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    rename_map = {
        "Country/Region": "Country_Region",
        "Adult prevalence of HIV/AIDS": "Adult_Prevalence",
        "Number of people with HIV/AIDS": "Num_People",
        "Annual deaths from HIV/AIDS": "Annual_Deaths",
        "Year of estimate": "Year_of_estimate",
    }
    df = df.rename(columns=rename_map)

    required = ["Country_Region", "Adult_Prevalence", TARGET, "Annual_Deaths", "Year_of_estimate"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di CSV: {', '.join(missing)}")

    df["Country_Region"] = df["Country_Region"].astype("string").str.strip()
    for col in ["Adult_Prevalence", TARGET, "Annual_Deaths"]:
        values = (
            df[col].astype("string").str.strip()
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[col] = pd.to_numeric(values, errors="coerce")

    df["Year_of_estimate"] = pd.to_numeric(df["Year_of_estimate"], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df = df[(df[FEATURES] >= 0).all(axis=1)]
    if len(df) < 10:
        raise ValueError(f"Data valid terlalu sedikit: {len(df)} baris.")
    return df


def main() -> None:
    df = load_and_clean_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler dipertahankan agar alur sesuai notebook awal.
    # Untuk model berbasis tree, scaling sebenarnya tidak wajib.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "parameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
            "test_size": 0.2,
        },
    }

    joblib.dump(pipeline, MODEL_DIR / "xgboost_pipeline.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training selesai.")
    print(f"Data valid : {len(df)} baris")
    print(f"Train/Test : {len(X_train)}/{len(X_test)}")
    print(f"MAE        : {metrics['mae']:,.2f}")
    print(f"RMSE       : {metrics['rmse']:,.2f}")
    print(f"R²         : {metrics['r2']:.4f}")
    print(f"Model      : {MODEL_DIR / 'xgboost_pipeline.joblib'}")


if __name__ == "__main__":
    main()
