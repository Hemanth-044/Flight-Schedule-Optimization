import argparse
import pandas as pd
from scripts.utils import combine_date_time, add_demand_features, label_delays

def main(args):
    df = pd.read_csv(args.input_csv, low_memory=False)

    # Rename key columns to consistent internal names
    rename_map = {
        "FL_DATE": "FlightDate",
        "OP_UNIQUE_CARRIER": "Carrier",
        "TAIL_NUM": "TailNum",
        "ORIGIN": "Origin",
        "DEST": "Dest",
        "CRS_DEP_TIME": "CRSDepTime",
        "DEP_TIME": "DepTime",
        "CRS_ARR_TIME": "CRSArrTime",
        "ARR_TIME": "ArrTime",
        "DEP_DELAY": "DepDelay",
        "DEP_DEL15": "DepDel15",
        "ARR_DELAY": "ArrDelay",
        "ARR_DEL15": "ArrDel15",
        "CANCELLED": "Cancelled",
        "CANCELLATION_CODE": "CancellationCode",
        "CARRIER_DELAY": "CarrierDelay",
        "WEATHER_DELAY": "WeatherDelay",
        "NAS_DELAY": "NASDelay",
        "SECURITY_DELAY": "SecurityDelay",
        "LATE_AIRCRAFT_DELAY": "LateAircraftDelay"
    }
    df = df.rename(columns=rename_map)

    # Create combined datetime columns
    df["sched_dep"] = combine_date_time(df["FlightDate"], df["CRSDepTime"])
    df["act_dep"] = combine_date_time(df["FlightDate"], df["DepTime"])
    df["sched_arr"] = combine_date_time(df["FlightDate"], df["CRSArrTime"])
    df["act_arr"] = combine_date_time(df["FlightDate"], df["ArrTime"])

    # Drop rows missing scheduled departure datetime (mandatory)
    df = df.dropna(subset=["sched_dep"])

    # Convert delay and cancellation related columns to numeric & fill missing
    numeric_cols = [
        "DepDelay", "ArrDelay", "DepDel15", "ArrDel15",
        "Cancelled", "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Add delay flags and demand features
    df = label_delays(df)
    df = add_demand_features(df, bucket_minutes=args.bucket_minutes)

    # Save full dataframe with all columns + features
    df.to_csv(args.output_csv, index=False)
    print(f"Prepared {len(df)} rows with extended features → {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--airport", required=False)
    parser.add_argument("--output_csv", default="data/normalized_week.csv")
    parser.add_argument("--bucket_minutes", type=int, default=5)
    args = parser.parse_args()
    main(args)
