import numpy as np

history = pd.read_csv("srilanka_dengue_weekly_by_district.csv")
history.columns = history.columns.str.strip()
history["start.date"] = pd.to_datetime(history["start.date"])
history = history.sort_values(["district", "start.date"]).reset_index(drop=True)


def build_features(district, week_start):
    """Build one feature row from the 4 weeks of cases up to week_start."""
    if district not in district_categories:
        raise ValueError(f"Unknown district '{district}'. Choose from: {district_categories}")

    week_start = pd.Timestamp(week_start)
    rows = history[(history["district"] == district) & (history["start.date"] <= week_start)]

    if rows.empty:
        raise ValueError(f"No data for {district} on or before {week_start.date()}.")
    if week_start - rows["start.date"].max() > pd.Timedelta(days=6):
        last = history["start.date"].max().date()
        raise ValueError(f"No case data for the week of {week_start.date()}. "
                         f"The dataset ends on {last}; add newer weekly cases to the CSV first.")

    last_4 = rows.tail(4)
    if len(last_4) < 4:
        raise ValueError("Need at least 4 weeks of history before this date.")

    this_week = last_4.iloc[-1]          # the week the user asked for
    cases = last_4["cases"].tolist()     # oldest -> newest

    # same seasonal features as data_prep.create_features
    date = this_week["start.date"]
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
    return row[features], date


def predict_next_week(district, week_start):
    """Predict next week's dengue cases from a district and a week start date."""
    X, date = build_features(district, week_start)
    predicted = float(ebm.predict(X)[0])
    threshold = float(district_thresholds[district])
    risk = "HIGH RISK" if predicted >= threshold else "Normal"

    print(f"\n{district} - data week starting {date.date()}")
    print("Last 4 weeks (oldest -> newest):",
          [int(X[c].iloc[0]) for c in ["cases_lag_4", "cases_lag_3", "cases_lag_2", "cases_lag_1"]])
    print(f"Predicted cases next week: {predicted:.0f}")
    print(f"Risk threshold: {threshold:.1f} -> {risk}")

    # if next week is also in the data, show the real value
    next_week = history[(history["district"] == district)
                        & (history["start.date"] == date + pd.Timedelta(days=7))]
    if not next_week.empty:
        print(f"Actual cases next week: {next_week['cases'].iloc[0]:.0f}")

    return {"district": district, "week_start": date.date(),
            "predicted_cases": round(predicted), "threshold": threshold, "risk": risk}


result = predict_next_week("Colombo", "2026-05-04")
