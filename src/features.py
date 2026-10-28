"""Cleaning, feature engineering and time-ordered validation for the traffic data.

Columns f1..f14 are anonymised: some are integer category codes, some are measurements.
What the EDA showed (see README): f7 is an hourly time index (f7 % 24 == f14, the hour),
f8 is a row id, f2 is pure noise, f3 is a period code whose values do not repeat in later data,
f13 flags holidays only at midnight, and y < 100 are sensor outages.
"""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

FEATURES = [f"f{i}" for i in range(1, 15)]
OUTAGE = 100                       # volumes below this are sensor outages (lowest normal night-time volume is 123)
DROP = ["f2", "f3", "f7", "f8"]    # noise, period code, time index, row id


def clean_train(reg_df, clf_df):
    """Drop outage rows from the training data (both task files, kept aligned)."""
    keep = (reg_df["y"] >= OUTAGE).to_numpy()
    return reg_df[keep].reset_index(drop=True), clf_df[keep].reset_index(drop=True)


def make_features(df, fix=True, hol=True, drop=True):
    X = df[FEATURES].copy()
    if fix:
        X.loc[X["f5"] > 1000, "f5"] = np.nan        # impossible rain-like value
        X.loc[X["f12"] < -100, "f12"] = np.nan      # impossible sentinel value
    if hol:
        day = X["f7"] // 24                          # f7 counts hours
        X["hol_day"] = day.isin(day[X["f13"] > 0]).astype(int)   # holiday is only flagged at midnight: spread it to the whole day
    return X.drop(columns=DROP) if drop else X


def ts_cv(m, X, y, t, scoring="r2"):
    """Sort rows by time, then train on the past and test on the next block (5 splits)."""
    o = np.argsort(t.to_numpy(), kind="stable")
    return cross_val_score(m, X.iloc[o], y.iloc[o], cv=TimeSeriesSplit(5), scoring=scoring).mean()
