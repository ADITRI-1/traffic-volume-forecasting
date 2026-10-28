"""All experiments behind the README.

Usage: python src/experiments.py --data data     (about 5 minutes)
Figures go to results/, key numbers to results/metrics.json.
"""
import argparse
import json
import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score
from sklearn.tree import plot_tree
from features import FEATURES, DROP, clean_train, make_features, ts_cv
from models import SEED, REG_PARAMS, make_model

warnings.filterwarnings("ignore")
OUT = "results"


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=130)
    plt.close()


def r(x):
    return round(float(x), 4)


def main(data):
    os.makedirs(OUT, exist_ok=True)
    reg_train = pd.read_csv(os.path.join(data, "partB_reg_train.csv"))
    reg_val = pd.read_csv(os.path.join(data, "partB_reg_val.csv"))
    clf_train = pd.read_csv(os.path.join(data, "partB_clf_train.csv"))
    clf_val = pd.read_csv(os.path.join(data, "partB_clf_val.csv"))
    tr, va = reg_train, reg_val
    yr, yrv = tr.y, va.y
    t = tr.f7
    res = {}

    # 1. EDA: impossible values, outages, columns that do not carry over to the future
    out = tr.y < 100
    print("largest f5:", tr.f5.nlargest(2).tolist(), "| smallest f12:", sorted(tr.f12.dropna().unique().tolist())[:2])
    print("y < 100:", out.sum(), "rows | f7 train range:", tr.f7.min(), tr.f7.max(), "| val range:", va.f7.min(), va.f7.max())
    print("f3 in train:", sorted(tr.f3.unique().tolist()), "| in val:", sorted(va.f3.unique().tolist()))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))
    ax[0].scatter(tr.f7, tr.y, s=1, alpha=0.2)
    ax[0].scatter(tr.f7[out], tr.y[out], s=4, c="r")
    ax[0].set_title("y over time (red: outages, y < 100)")
    ax[1].hist(tr.f12.dropna(), bins=100, log=True)
    ax[1].set_title("f12 (sentinel at -363.57)")
    ax[2].hist(tr.f7, bins=60, label="train")
    ax[2].hist(va.f7, bins=20, label="val")
    ax[2].legend()
    ax[2].set_title("f7 (time index): val is later")
    save("eda.png")

    # 2. linear baseline and why it fails
    med = tr[FEATURES].median()
    Xi, Xvi = tr[FEATURES].fillna(med), va[FEATURES].fillna(med)
    res["linear_val_r2"] = r(r2_score(yrv, make_model("linear", "reg").fit(Xi, yr).predict(Xvi)))
    print("linear val R2:", res["linear_val_r2"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    hour = tr.groupby("f14").y.mean()
    a, b = np.polyfit(tr.f14, yr, 1)
    ax[0].scatter(tr.f14, yr, s=1, alpha=0.05)
    ax[0].plot(hour.index, hour.values, "r-o", label="mean per hour")
    ax[0].plot(hour.index, a * hour.index + b, "k--", label="linear fit")
    ax[0].set_xlabel("f14 (hour)")
    ax[0].legend()
    for d in sorted(tr.f11.unique()):
        h = tr[tr.f11 == d].groupby("f14").y.mean()
        ax[1].plot(h.index, h.values, label="f11 = " + str(d))
    ax[1].set_xlabel("f14 (hour)")
    ax[1].set_title("mean volume per hour for each day code")
    ax[1].legend()
    save("hour_profile.png")

    # 3. random vs time-ordered validation
    kf = KFold(5, shuffle=True, random_state=SEED)
    res["validation"] = {}
    for name, cols in [("all columns", FEATURES), ("without f2 f3 f7 f8", [c for c in FEATURES if c not in DROP])]:
        m = make_model("hist_gradient_boosting", "reg")
        X, Xv = tr[cols], va[cols]
        res["validation"][name] = {"random_kfold": r(cross_val_score(m, X, yr, cv=kf, scoring="r2").mean()),
                                   "time_ordered": r(ts_cv(m, X, yr, t)),
                                   "val": r(r2_score(yrv, m.fit(X, yr).predict(Xv)))}
        print(name, res["validation"][name])

    # 4. single tree: depth curve and first splits
    XK, XvK = Xi.drop(columns=DROP), Xvi.drop(columns=DROP)
    depths = list(range(1, 21))
    r_tr, r_cv, r_va = [], [], []
    for d in depths:
        m = make_model("decision_tree", "reg", dict(max_depth=d)).fit(XK, yr)
        r_tr.append(r2_score(yr, m.predict(XK)))
        r_va.append(r2_score(yrv, m.predict(XvK)))
        r_cv.append(ts_cv(m, XK, yr, t))
    plt.plot(depths, r_tr, "o-", label="train")
    plt.plot(depths, r_cv, "o-", label="time-ordered CV")
    plt.plot(depths, r_va, "o-", label="later period (val)")
    plt.xlabel("max_depth")
    plt.ylabel("R2")
    plt.legend()
    save("tree_depth.png")
    best_d = depths[int(np.argmax(r_cv))]
    print("best depth:", best_d)
    m = make_model("decision_tree", "reg", dict(max_depth=best_d)).fit(XK, yr)
    plt.figure(figsize=(18, 7))
    plot_tree(m, max_depth=2, feature_names=list(XK.columns), filled=True, fontsize=9)
    save("tree_top.png")

    # 5. bagging vs random forest vs boosting
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    trees = [10, 25, 50, 100, 200]
    for name in ["bagging", "random_forest"]:
        sc = [ts_cv(make_model(name, "reg", dict(n_estimators=n, n_jobs=-1)), XK, yr, t) for n in trees]
        ax[0].plot(trees, sc, "o-", label=name)
    iters = [50, 100, 200, 400]
    for lr in [0.03, 0.1, 0.3]:
        sc = [ts_cv(make_model("hist_gradient_boosting", "reg", dict(max_iter=n, learning_rate=lr)), XK, yr, t) for n in iters]
        ax[1].plot(iters, sc, "o-", label="lr = " + str(lr))
    ax[0].set_xlabel("number of trees")
    ax[0].set_ylabel("time-ordered CV R2")
    ax[0].legend()
    ax[1].set_xlabel("number of trees (boosting)")
    ax[1].legend()
    save("ensembles.png")
    models = [("decision_tree", dict(max_depth=best_d)),
              ("bagging", dict(n_estimators=100, n_jobs=-1)),
              ("random_forest", dict(n_estimators=100, min_samples_leaf=5, n_jobs=-1)),
              ("gradient_boosting", dict(n_estimators=300, max_depth=5)),
              ("hist_gradient_boosting", dict(max_iter=200, learning_rate=0.1))]
    res["models"] = {}
    for name, p in models:
        m = make_model(name, "reg", p)
        res["models"][name] = {"time_cv": r(ts_cv(m, XK, yr, t)), "val": r(r2_score(yrv, m.fit(XK, yr).predict(XvK)))}
        print(name, res["models"][name])

    # 6. impurity vs permutation vs drop-column importance (random forest, all columns)
    p = dict(n_estimators=100, min_samples_leaf=5, n_jobs=-1)
    rf = make_model("random_forest", "reg", p).fit(Xi, yr)
    base = r2_score(yrv, rf.predict(Xvi))
    perm = permutation_importance(rf, Xvi, yrv, scoring="r2", n_repeats=5, random_state=SEED, n_jobs=-1)
    drop = []
    for c in FEATURES:
        m = make_model("random_forest", "reg", p).fit(Xi.drop(columns=c), yr)
        drop.append(base - r2_score(yrv, m.predict(Xvi.drop(columns=c))))
    imp = pd.DataFrame({"impurity": rf.feature_importances_, "permutation": perm.importances_mean, "drop_column": drop}, index=FEATURES)
    print(imp.sort_values("impurity", ascending=False).round(4))
    imp.plot.bar(subplots=True, layout=(1, 3), figsize=(16, 3.5), legend=False)
    save("importance.png")

    # 7. ablation of cleaning and feature steps
    trc, trcl = clean_train(reg_train.copy(), clf_train.copy())
    steps = [("raw columns", reg_train, dict(fix=False, hol=False, drop=False)),
             ("+ drop f2 f3 f7 f8", reg_train, dict(fix=False, hol=False)),
             ("+ impossible values -> NaN", reg_train, dict(hol=False)),
             ("+ drop outage rows", trc, dict(hol=False)),
             ("+ holiday-day flag (final)", trc, dict())]
    res["ablation"] = {}
    for name, d, k in steps:
        X, Xv = make_features(d, **k), make_features(reg_val, **k)
        m = make_model("hist_gradient_boosting", "reg", REG_PARAMS)
        res["ablation"][name] = {"time_cv": r(ts_cv(m, X, d.y, d.f7)), "val": r(r2_score(yrv, m.fit(X, d.y).predict(Xv)))}
        print(name, res["ablation"][name])
    names = list(res["ablation"])
    plt.figure(figsize=(8, 3.5))
    plt.plot(names, [res["ablation"][n]["time_cv"] for n in names], "o-", label="time-ordered CV")
    plt.plot(names, [res["ablation"][n]["val"] for n in names], "o-", label="later period (val)")
    plt.xticks(rotation=15, fontsize=8)
    plt.ylabel("R2")
    plt.legend()
    save("ablation.png")

    # 8. decision tree vs SVM on the traffic level (4 classes)
    sets = [("raw features", clf_train[FEATURES], clf_train.y, clf_train.f7, clf_val[FEATURES]),
            ("cleaned features", make_features(trcl), trcl.y, trcl.f7, make_features(clf_val))]
    tree_grid = {"max_depth": [4, 6, 8, 10, 12, 16, None], "min_samples_leaf": [1, 5, 20, 50]}
    svm_grid = {"svc__C": [1, 3, 10, 30, 100], "svc__gamma": ["scale", 0.03, 0.1, 0.3]}
    res["tree_vs_svm"] = {}
    for name, X, y, t_, Xv in sets:
        o = np.argsort(t_.to_numpy(), kind="stable")
        tg = GridSearchCV(make_model("decision_tree", "clf"), tree_grid, cv=TimeSeriesSplit(5), scoring="f1_macro", n_jobs=-1)
        tg.fit(X.iloc[o], y.iloc[o])
        s = o[np.sort(np.random.RandomState(SEED).choice(len(o), 8000, replace=False))]   # SVM is slow: 8000 rows, still in time order
        sg = GridSearchCV(make_model("svm", "clf"), svm_grid, cv=TimeSeriesSplit(5), scoring="f1_macro", n_jobs=-1)
        sg.fit(X.iloc[s], y.iloc[s])
        sp = dict(kernel="rbf", C=sg.best_params_["svc__C"], gamma=sg.best_params_["svc__gamma"])
        tree = make_model("decision_tree", "clf", tg.best_params_).fit(np.asarray(X, float), y)
        svm = make_model("svm", "clf", sp).fit(np.asarray(X, float), y)
        res["tree_vs_svm"][name] = {"tree_val_f1": r(f1_score(clf_val.y, tree.predict(np.asarray(Xv, float)), average="macro")),
                                    "svm_val_f1": r(f1_score(clf_val.y, svm.predict(np.asarray(Xv, float)), average="macro"))}
        print(name, tg.best_params_, sp, res["tree_vs_svm"][name])

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    main(ap.parse_args().data)
