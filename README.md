# CarSure — Complete B.Tech Data Science Project

Used Car Valuation & Vehicle History Platform.

## Features
- Vehicle search and comparison selection
- AI/development used-car price estimate
- Fair buying range
- Dealer asking price vs market listing average
- Accident history
- Service history
- Insurance and claims
- Vehicle condition
- KM verification
- Transparent Vehicle History Score
- Data confidence/evidence transparency
- Saved valuation history
- Clickable saved valuation details
- SQLite database
- Scikit-learn Random Forest model

## Important
The included records are development/sample data. The ML accuracy metrics are development-only and must not be presented as real-world market accuracy.

## First-time setup
```powershell
cd C:\Users\dell\Downloads\CarSure
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ml\train_improved_model.py
.\venv\Scripts\python.exe database\load_data.py
.\venv\Scripts\python.exe app.py
```

Open: http://127.0.0.1:5000/

Do not run `database/load_data.py` repeatedly after adding saved valuations because it recreates the base tables. Saved valuations are stored in `valuation_requests` by Flask.
