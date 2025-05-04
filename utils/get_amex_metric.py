import numpy as np

def get_amex_metric_calculated(y_true: np.array, y_pred: np.array) -> float:
    n_pos = y_true.sum()
    n_neg = y_true.size - n_pos
    idx   = np.argsort(y_pred)[::-1]
    preds, target = y_pred[idx], y_true[idx]

    weight = 20.0 - target * 19.0
    cum_w  = (weight / weight.sum()).cumsum()
    d      = target[cum_w <= 0.04].sum() / n_pos

    lor = (target / n_pos).cumsum()
    g   = ((lor - cum_w) * weight).sum()
    g_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))
    g_norm = g / g_max
    return 0.5 * (g_norm + d)

def lgb_amex_metric(y_true, y_pred):
    return ('amex', get_amex_metric_calculated(y_true, y_pred), True)