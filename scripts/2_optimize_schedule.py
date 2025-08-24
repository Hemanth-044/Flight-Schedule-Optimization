import argparse
import json
import pandas as pd

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except Exception:
    ORTOOLS_AVAILABLE = False

def main(args):
    df = pd.read_csv(args.input_csv)

    with open(args.airport_config) as f:
        cfgs = json.load(f)

    profile_key = args.airport_profile.upper()
    if profile_key not in cfgs:
        raise KeyError(f"Airport profile '{args.airport_profile}' (normalized as '{profile_key}') not found in config. Available keys: {list(cfgs.keys())}")

    cfg = cfgs[profile_key]

    capacity_modes = cfg.get("capacity_modes", {})
    default_mode = cfg.get("default_capacity_mode", "")
    capacity_mode = args.capacity_mode or default_mode
    if capacity_mode not in capacity_modes:
        raise KeyError(f"Capacity mode '{capacity_mode}' not found in config capacity_modes keys: {list(capacity_modes.keys())}")

    departures_per_hr = capacity_modes[capacity_mode]["departures"]
    bucket = cfg.get("bucket_minutes", 5)
    cap_per_bucket = max(1, int(departures_per_hr * bucket / 60))
    window = cfg.get("window_plus_minus_min", 15)

    print(f"Using capacity mode: {capacity_mode} with {departures_per_hr} departures per hour")
    print(f"Capacity per {bucket}-minute bucket: {cap_per_bucket}")

    df.columns = df.columns.str.lower()
    df["sched_dep"] = pd.to_datetime(df["sched_dep"])
    df["minute"] = df["sched_dep"].dt.hour * 60 + df["sched_dep"].dt.minute
    df["min_lb"] = (df["minute"] - window).clip(lower=0)
    df["min_ub"] = (df["minute"] + window).clip(upper=1439)
    df["bucket_lb"] = (df["min_lb"] // bucket).astype(int)
    df["bucket_ub"] = (df["min_ub"] // bucket).astype(int)

    dfd = df.copy().reset_index(drop=True)
    if dfd.empty:
        raise SystemExit("No flights found in dataset.")

    max_bucket = 1440 // bucket

    if not ORTOOLS_AVAILABLE:
        dfd["bucket_sched"] = (dfd["minute"] // bucket).astype(int)
        alloc = {b: 0 for b in range(max_bucket)}
        assign = []
        for _, row in dfd.sort_values("bucket_sched").iterrows():
            lb, ub = int(row["bucket_lb"]), int(row["bucket_ub"])
            best_b, best_dist = None, 1e9
            for b in range(lb, ub + 1):
                dist = abs(b - row["bucket_sched"])
                if alloc.get(b, 0) < cap_per_bucket and dist < best_dist:
                    best_b, best_dist = b, dist
            if best_b is None:
                best_b = int(row["bucket_sched"])
            alloc[best_b] = alloc.get(best_b, 0) + 1
            assign.append(best_b)
        dfd["bucket_opt"] = assign
    else:
        model = cp_model.CpModel()
        X = {}
        for i, row in dfd.iterrows():
            for b in range(int(row["bucket_lb"]), int(row["bucket_ub"]) + 1):
                X[(i, b)] = model.NewBoolVar(f"x_{i}_{b}")

        for i, row in dfd.iterrows():
            model.Add(sum(X[(i, b)] for b in range(int(row["bucket_lb"]), int(row["bucket_ub"]) + 1)) == 1)

        dfd['date'] = dfd['sched_dep'].dt.date
        dates = dfd['date'].unique()

        for date in dates:
            flights_on_date = dfd[dfd['date'] == date]
            for b in range(max_bucket):
                in_b = [X[(i, b)] for i, row in flights_on_date.iterrows()
                        if (int(row["bucket_lb"]) <= b <= int(row["bucket_ub"])) and ((i, b) in X)]
                if in_b:
                    model.Add(sum(in_b) <= cap_per_bucket)

        dev_terms = []
        for i, row in dfd.iterrows():
            sched_b = int(row["minute"] // bucket)
            for b in range(int(row["bucket_lb"]), int(row["bucket_ub"]) + 1):
                dev_terms.append(abs(b - sched_b) * X[(i, b)])
        model.Minimize(sum(dev_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 8
        solver.Solve(model)

        chosen = []
        for i, row in dfd.iterrows():
            pick = None
            for b in range(int(row["bucket_lb"]), int(row["bucket_ub"]) + 1):
                if solver.Value(X[(i, b)]) == 1:
                    pick = b
                    break
            if pick is None:
                pick = int(row["minute"] // bucket)
            chosen.append(pick)
        dfd["bucket_opt"] = chosen

    dfd["opt_minute"] = dfd["bucket_opt"] * bucket + bucket // 2
    dfd["opt_dep"] = pd.to_datetime(dfd["sched_dep"].dt.date.astype(str)) + pd.to_timedelta(dfd["opt_minute"], unit="m")

    out_cols = ["carrier", "tailnum", "origin", "dest", "sched_dep", "act_dep", "dep_delay", "count", "roll_15", "roll_60", "opt_dep"]
    out = dfd[out_cols].copy()
    out.to_csv(args.output_csv, index=False)

    print(f"Optimized schedule saved → {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/normalized_week.csv")
    parser.add_argument("--airport_config", default="configs/airports.json")
    parser.add_argument("--airport_profile", default="JFK")
    parser.add_argument("--capacity_mode", default=None, help="Optional capacity mode to select")
    parser.add_argument("--output_csv", default="data/optimized_schedule.csv")
    args = parser.parse_args()
    main(args)
