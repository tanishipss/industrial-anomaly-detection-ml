import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_curve
from src.config import RANDOM_STATE

def train_base_model(X, y):
    model = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.zeros(len(X))

    for tr, val in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[val] = model.predict_proba(X.iloc[val])[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    best_threshold = thresholds[np.argmax(f1)]

    model.fit(X, y)
    return model, best_threshold
