# Dengue web app (FastAPI backend + HTML frontend)

## Where to put the files

Copy `app.py` and the `static` folder into your project folder, next to these existing files:

```
your-project/
├── app.py                                  <- new (backend)
├── static/index.html                       <- new (frontend)
├── data_prep.py
├── srilanka_dengue_weekly_by_district.csv
└── srilanka_dengue_ebm_model.pkl           <- created by python main.py
```

## Run

```
pip install fastapi uvicorn
python main.py                 # only if the .pkl file does not exist yet
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## Pages

- **Dashboard**: headline numbers, weekly cases (all of Sri Lanka or one district with its
  threshold), cases per year, a typical year (seasonality), total cases by district, and the
  latest week by district compared with each threshold.
- **Predict next week**: choose a district and a week start date. Shows the predicted cases for
  the next week, the risk threshold and HIGH RISK / Normal. For past weeks it also shows the
  actual cases next week.

## API (open http://127.0.0.1:8000/docs to try it)

| Endpoint | What it returns |
|---|---|
| `GET /api/districts` | district names, thresholds, first and last data week |
| `GET /api/summary` | dashboard numbers and chart data |
| `GET /api/series?district=Colombo` | weekly cases (use `All` for the whole country) |
| `GET /api/predict?district=Colombo&week_start=2026-05-04` | prediction, threshold, risk, actual (if known), reasons |

The page loads Chart.js and the Public Sans font from the internet, so the computer needs an
internet connection for the charts.
