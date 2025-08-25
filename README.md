# Flight Schedule Optimization for Congested Airports

## Overview

This project optimizes flight schedules at high-traffic airports such as New York JFK, Mumbai (BOM), and Delhi (DEL) to reduce congestion and minimize delays. Leveraging real-world flight data and FAA capacity profiles, combined with machine learning delay predictions and constraint programming-based scheduling optimization, the project provides a scalable and adaptive solution.

An interactive dashboard built with Streamlit offers detailed visualizations and an NLP-powered interface to explore schedules, congested time windows, delay causes, and optimization impact.

---

## Features

- **Data preprocessing:** Clean and enrich flight data focusing on congested airports.
- **Delay prediction model:** Gradient Boosting Regressor predicting departure delays with strong accuracy.
- **Schedule optimization:** Assign flights to 5-minute departure buckets within capacity & weather-dependent constraints.
- **Capacity modes:** Model variable airport capacities based on operational modes and weather conditions.
- **Interactive dashboard:** Explore optimized schedules, congestion trends, delays, and cancellations.
- **NLP interface:** Query busiest windows, high-impact flights, and receive scheduling insights easily.

---

## Project Structure
```
flight_sched_realdata_only/
├── configs/
│ └── airports.json # Airport capacity configurations
├── data/
│ ├── raw_bts_week_clean.csv # Raw flight data (excluded from repo)
│ ├── normalized_week.csv # Processed flight data
│ └── optimized_schedule.csv # Output from optimizer
├── scripts/
│ ├── 0_prepare_bts.py # Data preprocessing script
│ ├── 1_train_model.py # Delay prediction model training
│ ├── 2_optimize_schedule.py # Flight schedule optimizer
│ └── utils.py # Helper functions
├── app.py # Streamlit dashboard app
├── requirements.txt # Dependencies
└── README.md # This file
```


---

## Setup & Installation

1. Clone the repository:
```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```


2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows

pip install -r requirements.txt
```

```
3. Prepare or acquire flight data for your target airport(s). **Note:**  
Don’t commit raw data or virtual environments to the repo; keep datasets locally in the `data/` folder.
```
---

## Usage

### Data preparation
```
python -m scripts.0_prepare_bts --input_csv data/raw_bts_week_clean.csv --output_csv data/normalized_week.csv
```


### Train delay prediction model
```
python -m scripts.1_train_model --input_csv data/normalized_week.csv
```

### Optimize flight schedule
```
Specify airport and capacity mode as needed:

python -m scripts.2_optimize_schedule --input_csv data/normalized_week.csv --airport_config configs/airports.json --airport_profile JFK --capacity_mode Visual_DeparturePriority --output_csv data/optimized_schedule.csv
```


### Launch the dashboard
```
streamlit run app.py

```

Upload the normalized and optimized CSV files in the Streamlit app sidebar and explore interactive visualizations and NLP queries.

---

## Key Results
```
- Reduction of peak departure congestion within configured capacity modes.
- Improved distribution of departure delays across daily operational windows.
- Interactive insights on delay reasons and cancellations support operational decision making.
- Adaptability to Indian airports like Mumbai and Delhi through configuration updates.
```
---

## Model Performance
```
We trained a machine learning model to predict departure delays using extensive historical flight data. The model achieved a test mean absolute error (MAE) of **6.80 minutes**, demonstrating accurate delay prediction supporting effective scheduling optimization. These predictions enable prioritization of flights with higher delay risk, improving overall schedule robustness.
```
---

## Contribution Guidelines
```
Contributions, issues, and feature requests are welcome! Feel free to submit pull requests or open issues.
```

---

## Acknowledgments

- Bureau of Transportation Statistics (BTS) for flight data  
- FAA for airport capacity profiles  
- Google OR-Tools team for constraint programming tools  
- Streamlit for enabling rapid interactive dashboards
