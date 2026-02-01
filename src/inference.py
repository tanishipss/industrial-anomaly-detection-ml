import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from src.thresholding import dual_threshold_decision
from src.config import HIGH_POS, HIGH_NEG, RANDOM_STATE

def pseudo_label_and_predict(model, X, X_test, y):
    probs = model.predict_proba(X_test)[:, 1]

    confident_idx = np.where(
        (probs > HIGH_POS) | (probs < HIGH_NEG)
    )[0]

    pseudo_X = X_test.iloc[confident_idx]
    pseudo_y = (probs[confident_idx] > 0.5).astype(int)

    X_aug = pd.concat([X, pseudo_X])
    y_aug = np.concatenate([y, pseudo_y])

    model_pl = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.015,
        num_leaves=128,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=(y_aug == 0).sum() / (y_aug == 1).sum(),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model_pl.fit(X_aug, y_aug)

    final_probs = model_pl.predict_proba(X_test)[:, 1]
    preds = dual_threshold_decision(final_probs, y_aug.mean() * 1.2)

    return preds
