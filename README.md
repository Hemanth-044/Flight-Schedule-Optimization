
# Flight Schedule Optimization — Real Data Only (Hackathon MVP)

**Goal:** Analyze one real week of flights for a busy airport, detect peak-time congestion, and retime departures within ±10 minutes under Mumbai/Delhi-like runway capacity profiles. Provide an NLP-ish interface for queries.

## 0) Environment
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pandas numpy scikit-learn joblib ortools streamlit
```

## 1) Get Real Data (BTS On-Time Performance)
Export a single busy airport (e.g., JFK, LAX, ATL) for **one week** with columns:
`FlightDate, Carrier, TailNum, Origin, Dest, CRSDepTime, DepTime, CRSArrTime, ArrTime`

Save as: `data/raw_bts_week.csv`

## 2) Prepare & Feature Engineer
```bash
python scripts/0_prepare_bts.py --input_csv data/raw_bts_week.csv --airport JFK --output_csv data/normalized_week.csv
```
This:
- Builds `sched_dep/act_dep/sched_arr/act_arr` timestamps
- Computes `dep_delay, arr_delay, dow, minute`
- Adds **rolling demand** per 5-min bucket: `count, roll_15, roll_60`

## 3) Train Delay Model
```bash
python scripts/1_train_model.py --input_csv data/normalized_week.csv
```
- Trains a GradientBoostingRegressor on time-of-day + demand features
- Prints **Test MAE (minutes)** and saves `models/delay_model.pkl`

## 4) Optimize Schedule (Capacity-Constrained Retiming)
Choose a profile from `configs/airports.json`:
- `mumbai` (44/hr), `mumbai_hiro` (46/hr), `delhi_westerly` (~84/hr), `delhi_easterly` (~74/hr)

Run:
```bash
python scripts/2_optimize_schedule.py --input_csv data/normalized_week.csv --airport_profile mumbai --output_csv data/optimized_schedule.csv
```
- Optimizes **first day** of your week for demo speed
- Retimes each flight up to ±10 minutes to keep **<= capacity per 5-min bucket**
- Falls back to a **greedy heuristic** if OR-Tools isn't available

## 5) NLP-ish App (Streamlit)
```bash
streamlit run app.py
```
Upload `data/normalized_week.csv` and try:
- `busiest 30-min window`
- `top congested slots`
- `high-impact flights`

## Notes & Limitations (be transparent to judges)
- Real data: BTS (USA). We **map capacity profiles** to mimic Mumbai/Delhi runway limits.
- MVP does **not** model wake-pair separations or gate/turn-time constraints.
- Arrival/departure cross-midnight handling is minimal; pick a week with fewer overnight flights if possible.
- Optimization currently uses **combined movements** (arr + dep) in buckets (simple but effective).

## Deliverables for the hackathon
- `data/normalized_week.csv` (real, no mock)
- `data/optimized_schedule.csv`
- `models/delay_model.pkl`
- Screenshots from Streamlit: busiest windows, top buckets, high-impact candidates
