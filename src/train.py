"""Train the final regressor (volume) and classifier (traffic level) and score them on the later period.

Usage: python src/train.py --data data
"""
import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from features import clean_train, make_features
from models import CLF_MODEL, CLF_PARAMS, REG_MODEL, REG_PARAMS, make_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()

reg_train = pd.read_csv(os.path.join(a.data, "partB_reg_train.csv"))
reg_val = pd.read_csv(os.path.join(a.data, "partB_reg_val.csv"))
clf_train = pd.read_csv(os.path.join(a.data, "partB_clf_train.csv"))
clf_val = pd.read_csv(os.path.join(a.data, "partB_clf_val.csv"))

tr_reg, tr_clf = clean_train(reg_train, clf_train)
reg = make_model(REG_MODEL, "reg", REG_PARAMS).fit(make_features(tr_reg), tr_reg["y"])
clf = make_model(CLF_MODEL, "clf", CLF_PARAMS).fit(make_features(tr_clf), tr_clf["y"])
p_reg = reg.predict(make_features(reg_val))
p_clf = clf.predict(make_features(clf_val))

print("volume  | R2:", round(r2_score(reg_val["y"], p_reg), 4), "| MAE:", round(mean_absolute_error(reg_val["y"], p_reg)))
print("level   | macro-F1:", round(f1_score(clf_val["y"], p_clf, average="macro"), 4))
pd.DataFrame({"volume_pred": p_reg, "level_pred": p_clf}).to_csv("predictions.csv", index=False)
