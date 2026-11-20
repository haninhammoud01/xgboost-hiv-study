# HIV Prevalence Prediction with XGBoost

## Application Preview

![Dashboard](assets/dashboard.png)

An exploratory regression project that estimates the number of people living with HIV/AIDS using adult prevalence, annual deaths, and the year of estimate.

## Project structure

```text
xgboost-hiv-study/
├── app.py
├── train_model.py
├── hiv_prevalence.csv
├── XGBoost_Regression.ipynb
├── requirements.txt
└── model/
    ├── xgboost_pipeline.joblib
    └── metrics.json
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

## Model

- Algorithm: `XGBRegressor`
- Features: `Adult_Prevalence`, `Annual_Deaths`, `Year_of_estimate`
- Target: `Num_People`
- Parameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`
- Evaluation: MAE, RMSE, and R² using an 80:20 train-test split.

The preprocessing and model are stored together in a scikit-learn pipeline. The scaler is retained to stay close to the original notebook workflow; feature scaling is not generally necessary for tree-based XGBoost models.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Deploy. Keep `hiv_prevalence.csv`, `model/`, and `requirements.txt` in the repository.

## Limitations

The uploaded dataset contains 187 country records, but only 67 rows have complete values for all model features and target. Estimates are drawn from different years. The model is intended for learning and exploration, not clinical diagnosis, official reporting, or public-health decision-making. A single train-test split on a small dataset can produce unstable evaluation results.
