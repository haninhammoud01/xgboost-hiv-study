from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hiv_prevalence.csv"
MODEL_PATH = ROOT / "model" / "xgboost_pipeline.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"
FEATURES = ["Adult_Prevalence", "Annual_Deaths", "Year_of_estimate"]

st.set_page_config(page_title="HIV Prevalence | XGBoost", page_icon="📊", layout="wide")

st.title("HIV Prevalence Prediction")
st.caption("XGBoost Regression · Exploratory machine learning project")
st.markdown(
    "A small interactive interface for exploring the relationship between adult HIV prevalence, "
    "annual deaths, estimation year, and the estimated number of people living with HIV/AIDS."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    rename_map = {
        "Country/Region": "Country_Region",
        "Adult prevalence of HIV/AIDS": "Adult_Prevalence",
        "Number of people with HIV/AIDS": "Num_People",
        "Annual deaths from HIV/AIDS": "Annual_Deaths",
        "Year of estimate": "Year_of_estimate",
    }
    df = df.rename(columns=rename_map)
    for col in ["Adult_Prevalence", "Num_People", "Annual_Deaths"]:
        values = (df[col].astype("string").str.strip()
                  .str.replace("%", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.replace(" ", "", regex=False))
        df[col] = pd.to_numeric(values, errors="coerce")
    df["Year_of_estimate"] = pd.to_numeric(df["Year_of_estimate"], errors="coerce")
    return df

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
    st.stop()

model = load_model()
df = load_data()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Prediction input")
    prevalence = st.number_input("Adult prevalence (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
    annual_deaths = st.number_input("Annual deaths", min_value=0, max_value=10_000_000, value=5_000, step=100)
    year = st.number_input("Year of estimate", min_value=1900, max_value=2100, value=2024, step=1)
    predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

st.subheader("Model performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{metrics['mae']:,.0f}")
c2.metric("RMSE", f"{metrics['rmse']:,.0f}")
c3.metric("R² score", f"{metrics['r2']:.3f}")
c4.metric("Complete records", f"{metrics['rows_total']}")

if predict_clicked:
    input_df = pd.DataFrame([{
        "Adult_Prevalence": prevalence,
        "Annual_Deaths": annual_deaths,
        "Year_of_estimate": year,
    }], columns=FEATURES)
    prediction = float(model.predict(input_df)[0])
    st.subheader("Prediction result")
    st.metric("Estimated number of people living with HIV/AIDS", f"{max(0, prediction):,.0f}")
    st.caption("This is a model estimate, not an official statistic or medical assessment.")

st.divider()
st.subheader("Feature importance")
try:
    importance = model.named_steps["xgb"].feature_importances_
    importance_df = pd.DataFrame({"Feature": FEATURES, "Importance": importance}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost feature importance")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
except Exception as exc:
    st.info(f"Feature importance belum dapat ditampilkan: {exc}")

st.subheader("Dataset overview")
valid = df.dropna(subset=FEATURES + ["Num_People"])
st.write(f"Dataset source file: **hiv_prevalence.csv** · {len(df)} records loaded · {len(valid)} complete records for model features.")
st.dataframe(valid.rename(columns={
    "Country_Region": "Country / Region",
    "Adult_Prevalence": "Adult prevalence (%)",
    "Num_People": "Number of people with HIV/AIDS",
    "Annual_Deaths": "Annual deaths",
    "Year_of_estimate": "Year of estimate",
}), use_container_width=True, hide_index=True)

with st.expander("About this model and its limitations"):
    st.markdown(
        "- The model follows the feature set and XGBoost hyperparameters in the original notebook.\n"
        "- The uploaded CSV currently provides 67 complete rows for the four model variables.\n"
        "- Evaluation uses a single 80:20 train-test split (`random_state=42`); results may vary with another split.\n"
        "- StandardScaler is retained to mirror the notebook workflow, although scaling is generally not required by tree-based XGBoost.\n"
        "- The dataset combines country-level estimates from different years. Predictions should be treated as exploratory and should not be used for clinical, policy, or public-health decisions."
    )

st.caption("Built with Python, Streamlit, scikit-learn, and XGBoost.")
