import argparse
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def main(args):
    df = pd.read_csv(args.input_csv)

    # Convert columns to lowercase for consistent naming
    df.columns = df.columns.str.lower()

    # Feature list including delays and cancellation features
    features = [
        "minute", "dow", "count", "roll_15", "roll_60",
        "cancelled", "carrierdelay", "weatherdelay", "nasdelay",
        "securitydelay", "lateaircraftdelay"
    ]
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df["depdelay"].fillna(0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)

    joblib.dump(model, "models/delay_model.pkl")
    print(f"Trained model saved to models/delay_model.pkl with Test MAE: {mae:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/normalized_week.csv")
    args = parser.parse_args()
    main(args)
