"""Model factory (every model seeded the same way) and the final choices."""
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 2024027

TREE_MODELS = {
    "decision_tree": (DecisionTreeRegressor, DecisionTreeClassifier),
    "bagging": (BaggingRegressor, BaggingClassifier),
    "random_forest": (RandomForestRegressor, RandomForestClassifier),
    "gradient_boosting": (GradientBoostingRegressor, GradientBoostingClassifier),
    "hist_gradient_boosting": (HistGradientBoostingRegressor, HistGradientBoostingClassifier),
}

# final models, chosen with time-ordered CV
REG_MODEL = "hist_gradient_boosting"
REG_PARAMS = dict(max_iter=200, learning_rate=0.05)
CLF_MODEL = "hist_gradient_boosting"
CLF_PARAMS = dict(max_iter=200, learning_rate=0.05)
BAND = (0.1, 0.9)       # quantiles of the 80% prediction band


def make_model(name, task, params=None):
    """name: a key of TREE_MODELS, "linear" or "svm"; task: "reg" or "clf"."""
    params = dict(params or {})
    if name == "linear":
        return LinearRegression(**params)
    if name == "svm":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(**{**params, "random_state": SEED}))
    cls = TREE_MODELS[name][0 if task == "reg" else 1]
    return cls(**{**params, "random_state": SEED})


def make_quantile_model(q):
    """Final regressor trained with the pinball loss: predicts the q-th quantile of the volume."""
    return HistGradientBoostingRegressor(**{**REG_PARAMS, "loss": "quantile", "quantile": q, "random_state": SEED})
