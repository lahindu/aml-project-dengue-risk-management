# FastAPI backend for the dengue dashboard and prediction page
#
# Put this file next to data_prep.py, the CSV and srilanka_dengue_weekly_by_district.pkl, then run:
#     uvicorn app:app --reload
# and open http://127.0.0.1:8000

import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from data_prep import load_dataset, fill_missing_weeks

DATASET_PATH = "srilanka_dengue_weekly_by_district.csv"
MODEL_PATH = "srilanka_dengue_ebm_model.pkl"
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Dengue Risk Sri Lanka")

# ------------------------------------------------------------------
# load the model and the data once, when the server starts
# ------------------------------------------------------------------
with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)

ebm = model_data["model"]
features = model_data["features"]
district_categories = model_data["district_categories"]
district_thresholds = model_data["district_thresholds"]

# same loading and gap filling as in training (no charts)
history = fill_missing_weeks(load_dataset(DATASET_PATH))
history = history.sort_values(["district", "start.date"]).reset_index(drop=True)

FIRST_WEEK = history["start.date"].min()
LAST_WEEK = history["start.date"].max()


# ------------------------------------------------------------------
# dashboard data
# ------------------------------------------------------------------
@app.get("/api/districts")
def districts():
    return {
        "districts": district_categories,
        "thresholds": {d: round(float(t), 1) for d, t in district_thresholds.items()},
        "first_week": str(FIRST_WEEK.date()),
        "last_week": str(LAST_WEEK.date()),
    }


@app.get("/api/summary")
def summary():
    national = history.groupby("start.date")["cases"].sum()
    latest = float(national.iloc[-1])
    previous = float(national.iloc[-2])

    # totals per year (the first and last years are not complete)
    yearly = history.groupby(history["start.date"].dt.year)["cases"].sum()

    # seasonality: average national cases for each week of the year (full years only)
    full_years = history[(history["start.date"].dt.year > FIRST_WEEK.year)
                         & (history["start.date"].dt.year < LAST_WEEK.year)]
    by_week = full_years.groupby(["start.date"])["cases"].sum().reset_index()
    by_week["week"] = by_week["start.date"].dt.isocalendar().week.astype(int)
    by_week = by_week[by_week["week"] <= 52]
    seasonality = by_week.groupby("week")["cases"].mean()

    # districts: total since 2007 and the latest week
    totals = history.groupby("district")["cases"].sum().sort_values(ascending=False)
    latest_rows = history[history["start.date"] == LAST_WEEK].set_index("district")["cases"]

    latest_table = []
    for district in district_categories:
        cases = float(latest_rows.get(district, 0))
        threshold = float(district_thresholds[district])
        latest_table.append({
            "district": district,
            "cases": int(cases),
            "threshold": round(threshold, 1),
            "above_threshold": bool(cases >= threshold),
        })
    latest_table.sort(key=lambda r: r["cases"], reverse=True)

    return {
        "total_cases": int(history["cases"].sum()),
        "weeks": int(history["start.date"].nunique()),
        "districts": len(district_categories),
        "first_week": str(FIRST_WEEK.date()),
        "last_week": str(LAST_WEEK.date()),
        "latest_week_cases": int(latest),
        "previous_week_cases": int(previous),
        "peak_week": str(national.idxmax().date()),
        "peak_week_cases": int(national.max()),
        "districts_above_threshold": sum(r["above_threshold"] for r in latest_table),
        "yearly": {"years": [int(y) for y in yearly.index], "cases": [int(c) for c in yearly.values]},
        "seasonality": {"weeks": [int(w) for w in seasonality.index],
                        "cases": [round(float(c), 1) for c in seasonality.values]},
        "district_totals": {"districts": list(totals.index), "cases": [int(c) for c in totals.values]},
        "latest_by_district": latest_table,
    }


@app.get("/api/series")
def series(district: str = "All"):
    """Weekly cases for one district, or the whole country with district=All."""
    if district == "All":
        data = history.groupby("start.date")["cases"].sum()
        threshold = None
    elif district in district_categories:
        data = history[history["district"] == district].set_index("start.date")["cases"]
        threshold = round(float(district_thresholds[district]), 1)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown district: {district}")

    return {
        "district": district,
        "dates": [str(d.date()) for d in data.index],
        "cases": [int(c) for c in data.values],
        "threshold": threshold,
    }


# ------------------------------------------------------------------
# prediction
# ------------------------------------------------------------------
def build_features(district, week_start):
    """Build one feature row from the 4 weeks of cases up to week_start."""
    week_start = pd.Timestamp(week_start)
    rows = history[(history["district"] == district) & (history["start.date"] <= week_start)]

    if rows.empty or week_start - rows["start.date"].max() > pd.Timedelta(days=6):
        raise HTTPException(
            status_code=400,
            detail=f"There is no case data for the week of {week_start.date()}. "
                   f"Choose a date between {FIRST_WEEK.date()} and {LAST_WEEK.date()}.")

    last_4 = rows.tail(4)
    if len(last_4) < 4:
        raise HTTPException(status_code=400,
                            detail="The model needs 4 weeks of history. Choose a later date.")

    cases = last_4["cases"].tolist()          # oldest -> newest
    date = last_4.iloc[-1]["start.date"]       # the data week used

    # same seasonal features as data_prep.create_features
    days_in_year = 366 if date.is_leap_year else 365
    year_phase = (date.dayofyear - 1) / days_in_year

    row = pd.DataFrame([{
        "district_code": district_categories.index(district),
        "week_sin": np.sin(2 * np.pi * year_phase),
        "week_cos": np.cos(2 * np.pi * year_phase),
        "cases_lag_1": cases[3],     # this week
        "cases_lag_2": cases[2],     # 1 week ago
        "cases_lag_3": cases[1],     # 2 weeks ago
        "cases_lag_4": cases[0],     # 3 weeks ago
        "cases_rolling_avg_4": np.mean(cases),
    }])
    return row[features], date, last_4


@app.get("/api/predict")
def predict(district: str, week_start: str):
    if district not in district_categories:
        raise HTTPException(status_code=404, detail=f"Unknown district: {district}")
    try:
        pd.Timestamp(week_start)
    except ValueError:
        raise HTTPException(status_code=400, detail="Use a date like 2026-05-04.")

    X, date, last_4 = build_features(district, week_start)
    predicted = float(ebm.predict(X)[0])
    threshold = float(district_thresholds[district])
    high_risk = predicted >= threshold

    # if the next week is in the data (a past date), send the real value too
    next_date = date + pd.Timedelta(days=7)
    next_row = history[(history["district"] == district) & (history["start.date"] == next_date)]
    actual = int(next_row["cases"].iloc[0]) if not next_row.empty else None

    # local explanation: how much each term pushed the prediction up or down.
    # With the Poisson loss the scores are on the log scale, exp(score) = multiplier.
    scores = pd.Series(ebm.eval_terms(X)[0], index=ebm.term_names_)
    scores = scores[scores.abs() > 0.01].sort_values(key=abs, ascending=False).head(5)
    reasons = [{"term": term, "score": round(float(s), 3), "multiplier": round(float(np.exp(s)), 2)}
               for term, s in scores.items()]

    return {
        "district": district,
        "data_week": str(date.date()),
        "predicted_week": str(next_date.date()),
        "last_4_weeks": [{"week": str(d.date()), "cases": int(c)}
                         for d, c in zip(last_4["start.date"], last_4["cases"])],
        "predicted_cases": round(predicted, 1),
        "predicted_cases_rounded": int(round(predicted)),   # same rounding as Python prints
        "threshold": round(threshold, 1),
        "high_risk": bool(high_risk),
        "actual_cases": actual,
        "actual_high_risk": None if actual is None else bool(actual >= threshold),
        "reasons": reasons,
    }


# ------------------------------------------------------------------
# frontend
# ------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_FOLDER, "index.html"))
