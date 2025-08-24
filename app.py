import streamlit as st
import pandas as pd
import json
from scripts.utils import busiest_windows
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Schedule Optimizer — Extended Data", layout="wide")
st.title("✈️ Flight Schedule Optimizer — Extended BTS Data")

with st.sidebar:
    st.header("Upload Files")
    raw_up = st.file_uploader("Upload normalized_week.csv", type=["csv"])
    opt_up = st.file_uploader("Upload optimized_schedule.csv", type=["csv"])
    st.markdown("---")
    st.header("Select Date for Analysis")
    selected_date = st.date_input("Choose flight date", value=pd.to_datetime("2025-01-01"), 
                                 min_value=pd.to_datetime("2025-01-01"), max_value=pd.to_datetime("2025-01-31"))
    st.markdown("---")
    st.header("Capacity Mode Selection")
    capacity_mode_choice = st.selectbox(
        "Select Airport Capacity Mode",
        options=["Visual_ArrivalPriority", "Visual_DeparturePriority", "Marginal", "Instrument"],
        index=1
    )
    st.markdown("---")
    st.header("NLP-ish query")
    q = st.text_input("Try: 'busiest 30-min window', 'top congested slots', 'high-impact flights'")

def load_df(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    return df

raw_df = load_df(raw_up)
opt_df = load_df(opt_up)

def get_bucket(dt_col):
    dt_col = pd.to_datetime(dt_col)
    return dt_col.dt.hour * 60 + dt_col.dt.minute // 5

def bucket_to_time(bucket):
    minutes = bucket * 5
    h = minutes // 60
    m = minutes % 60
    suffix = "AM"
    if h >= 12:
        suffix = "PM"
        if h > 12:
            h -= 12
    if h == 0:
        h = 12
    return f"{h:02d}:{m:02d} {suffix}"

try:
    with open("configs/airports.json") as f:
        cfgs = json.load(f)
    jfk_cfg = cfgs.get("JFK", {})
    usage_perc = jfk_cfg.get("mode_usage_percent", {})
    capacity_modes = jfk_cfg.get("capacity_modes", {})
    capacity_limit = capacity_modes.get(capacity_mode_choice, {}).get("departures", 60)
except Exception:
    usage_perc = {}
    capacity_limit = 60

if raw_df is not None and len(raw_df):
    raw_df['sched_dep'] = pd.to_datetime(raw_df['sched_dep'])
    filtered_raw = raw_df[raw_df['sched_dep'].dt.date == selected_date].copy()
    st.success(f"Loaded {len(filtered_raw):,} rows for {selected_date} from normalized_week.csv.")

    bw = busiest_windows(filtered_raw, bucket_minutes=5, window_buckets=6)
    st.subheader("Busiest 30-min Windows (Filtered Date)")
    st.dataframe(bw)

    filtered_raw['sched_bucket'] = get_bucket(filtered_raw['sched_dep'])
    sched_counts = filtered_raw.groupby('sched_bucket').size()
    sched_counts.index = sched_counts.index.map(bucket_to_time)

    if opt_df is not None and len(opt_df):
        opt_df['opt_dep'] = pd.to_datetime(opt_df['opt_dep'])
        opt_df['sched_dep'] = pd.to_datetime(opt_df['sched_dep'])
        filtered_opt = opt_df[opt_df['opt_dep'].dt.date == selected_date].copy()
        st.success(f"Loaded {len(filtered_opt):,} rows for {selected_date} from optimized_schedule.csv.")

        filtered_opt['opt_bucket'] = get_bucket(filtered_opt['opt_dep'])
        opt_counts = filtered_opt.groupby('opt_bucket').size()
        opt_counts.index = opt_counts.index.map(bucket_to_time)
    else:
        filtered_opt = None
        opt_counts = None

    # Congestion Over Time Chart
    st.subheader("Congestion Over Time")
    df_comp = pd.DataFrame({"Before": sched_counts})
    if opt_counts is not None:
        df_comp["After"] = opt_counts
    df_comp.fillna(0, inplace=True)
    st.line_chart(df_comp)

    st.write(f"Capacity mode '{capacity_mode_choice}' limit: {capacity_limit} departures per hour")
    capacity_per_bucket = max(1, int(capacity_limit * 5 / 60))
    st.write(f"Estimated capacity per 5-minute bucket: {capacity_per_bucket}")

    # Additional congestion plot with capacity line
    fig, ax = plt.subplots()
    sched_counts.plot(ax=ax, label='Before Optimization')
    if opt_counts is not None:
        opt_counts.plot(ax=ax, label='After Optimization')
    ax.axhline(y=capacity_per_bucket, color='r', linestyle='--', label='Capacity Limit')
    ax.set_ylabel('Number of Flights')
    ax.set_xlabel('Time Bucket')
    ax.legend()
    st.pyplot(fig)

    # Delay distribution before optimization
    st.subheader("Departure Delay Distribution Before Optimization")
    fig, ax = plt.subplots()
    filtered_raw["dep_delay"].hist(bins=50, ax=ax, color="steelblue", alpha=0.7)
    ax.set_xlabel("Departure Delay (minutes)")
    ax.set_ylabel("Number of Flights")
    st.pyplot(fig)

    # Optimized Departure Time vs Original Delay scatter
    if filtered_opt is not None and len(filtered_opt) > 0:
        st.subheader("Original Departure Delay vs Optimized Departure Hour")
        fig, ax = plt.subplots()
        ax.scatter(filtered_opt["dep_delay"],
                   filtered_opt["opt_dep"].dt.hour + filtered_opt["opt_dep"].dt.minute / 60,
                   alpha=0.3, color="orange")
        ax.set_xlabel("Original Departure Delay (minutes)")
        ax.set_ylabel("Optimized Departure Time (Hour of Day)")
        st.pyplot(fig)

    # Distribution of timing changes after optimization
    if filtered_opt is not None and len(filtered_opt) > 0:
        st.subheader("Distribution of Departure Time Shifts After Optimization")
        timing_diff = (filtered_opt["opt_dep"] - filtered_opt["sched_dep"]).dt.total_seconds() / 60
        fig, ax = plt.subplots()
        timing_diff.hist(bins=50, color="green", alpha=0.7, ax=ax)
        ax.set_xlabel("Time Shift in Minutes (Optimized - Original)")
        ax.set_ylabel("Number of Flights")
        st.pyplot(fig)

    # Display optimized schedule
    if filtered_opt is not None and len(filtered_opt) > 0:
        st.subheader(f"Optimized Schedule for {selected_date}")
        cols_to_show = ["carrier", "tailnum", "origin", "dest", "sched_dep", "opt_dep", "dep_delay"]
        existing_cols = [c for c in cols_to_show if c in filtered_opt.columns]
        st.dataframe(filtered_opt[existing_cols].sort_values("opt_dep"))

    # Delay reasons summary
    delay_reasons_cols = ["carrierdelay", "weatherdelay", "nasdelay", "securitydelay", "lateaircraftdelay"]
    if all(col in filtered_raw.columns for col in delay_reasons_cols):
        delay_summary = filtered_raw[delay_reasons_cols].describe().T
        st.subheader("Delay Reasons Summary")
        st.dataframe(delay_summary)
    else:
        st.info("Delay reason data not available.")

    # Cancellations count
    if "cancelled" in filtered_raw.columns:
        st.subheader("Cancellations Count")
        st.bar_chart(filtered_raw['cancelled'].value_counts())
    else:
        st.info("Cancellation data not available.")

    # NLP Query handling
    st.subheader("NLP Query Result")
    if q:
        ql = q.lower()
        if "busiest" in ql and ("30" in ql or "thirty" in ql):
            st.dataframe(bw.head(8))
        elif "congested" in ql or "top" in ql:
            st.dataframe(bw.head(12))
        elif "high-impact" in ql or "retime" in ql:
            st.write("Heuristic: flights inside top-3 busiest buckets.")
            top_buckets = bw["bucket"].head(3).tolist() if "bucket" in bw.columns else []
            d = filtered_raw.copy()
            d["dep_minute"] = pd.to_datetime(d["sched_dep"]).dt.hour * 60 + pd.to_datetime(d["sched_dep"]).dt.minute
            d["bucket"] = (d["dep_minute"] // 5).astype(int)
            out = d[d["bucket"].isin(top_buckets)].sort_values("depdelay", ascending=False).head(25)
            st.dataframe(out[["carrier", "tailnum", "origin", "dest", "sched_dep", "depdelay", "count", "roll_15", "roll_60"]])
        else:
            st.info("No matching intent; try examples like 'busiest 30-min window'.")

else:
    st.warning("Upload the normalized_week.csv produced by the preparation script.")

st.markdown("---")
st.caption("Tip: After running optimizer, upload the optimized CSV to compare congestion.")
