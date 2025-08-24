
import pandas as pd
import numpy as np

def parse_hhmm(series):
    """Convert HHMM or integer times (e.g., 945) into '%H%M' strings; return pandas Series of zero-padded 'HHMM'."""
    s = series.astype(str).str.replace(r'\.0$', '', regex=True).str.replace(r'\D', '', regex=True).str.strip()
    s = s.replace({'': pd.NA, 'nan': pd.NA})
    s = s.fillna('')
    s = s.apply(lambda x: x.zfill(4) if len(x) in (1,2,3) else x)
    # Some BTS exports encode midnight as '2400' -> map to '0000'
    s = s.replace({'2400':'0000'})
    return s

def combine_date_time(date_series, hhmm_series, tz=None):
    """Combine date ('YYYY-MM-DD' or similar) and HHMM into pandas datetime (timezone-naive)."""
    d = pd.to_datetime(date_series, errors='coerce')
    t = parse_hhmm(hhmm_series)
    # if time missing, leave NaT
    mask = t.eq('') | t.isna()
    dt_str = d.dt.strftime('%Y-%m-%d') + t
    out = pd.to_datetime(dt_str, format='%Y-%m-%d%H%M', errors='coerce')
    out[mask] = pd.NaT
    return out

def minute_of_day(ts):
    ts = pd.to_datetime(ts)
    return ts.dt.hour * 60 + ts.dt.minute

def add_demand_features(df, bucket_minutes=5, horizon_60_min=True):
    d = df.copy()
    d["dep_ts"] = pd.to_datetime(d["sched_dep"])
    d["dep_minute"] = minute_of_day(d["dep_ts"])
    d["bucket"] = (d["dep_minute"] // bucket_minutes).astype('Int64')
    counts = d.groupby("bucket").size().rename("count").to_frame().reset_index()
    # Ensure full day buckets exist
    full = pd.DataFrame({"bucket": range(0, 1440 // bucket_minutes)})
    counts = full.merge(counts, on="bucket", how="left").fillna({"count":0})
    counts["roll_15"] = counts["count"].rolling(3, min_periods=1, center=True).sum()  # 3*5=15
    counts["roll_60"] = counts["count"].rolling(12, min_periods=1, center=True).sum() # 12*5=60
    d = d.merge(counts, on="bucket", how="left")
    return d

def label_delays(df):
    d = df.copy()
    d["dep_delay"] = (pd.to_datetime(d["act_dep"]) - pd.to_datetime(d["sched_dep"])).dt.total_seconds()/60.0
    d["dep_delay"] = d["dep_delay"].fillna(0).clip(lower=0)
    d["arr_delay"] = (pd.to_datetime(d["act_arr"]) - pd.to_datetime(d["sched_arr"])).dt.total_seconds()/60.0
    d["arr_delay"] = d["arr_delay"].fillna(0).clip(lower=0)
    d["dow"] = pd.to_datetime(d["sched_dep"]).dt.dayofweek
    d["minute"] = pd.to_datetime(d["sched_dep"]).dt.hour * 60 + pd.to_datetime(d["sched_dep"]).dt.minute
    return d

def busiest_windows(df, bucket_minutes=5, window_buckets=6):
    d = df.copy()
    d["dep_minute"] = minute_of_day(pd.to_datetime(d["sched_dep"]))
    d["bucket"] = (d["dep_minute"] // bucket_minutes).astype(int)
    counts = d.groupby("bucket").size().rename("count").to_frame().reset_index()
    counts = counts.set_index("bucket")
    if counts.empty:
        return counts.reset_index()
    roll = counts["count"].rolling(window_buckets, min_periods=1).sum()
    out = roll.sort_values(ascending=False).head(12).to_frame("window_count").reset_index()
    out["window_start_minute"] = out["bucket"] * bucket_minutes
    return out.reset_index(drop=True)
