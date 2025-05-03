
import pandas as pd
import numpy as np

def calc_psi(df_train: pd.DataFrame,
             df_test: pd.DataFrame,
             n_bins: int = 10,
             eps: float = 1e-6) -> pd.DataFrame:
    """
    Compute Population Stability Index (PSI) for every column present in both data‑frames.

    Parameters
    ----------
    df_train : pd.DataFrame
        Reference (historic) data.
    df_test  : pd.DataFrame
        New / out‑of‑time data.
    n_bins   : int, optional
        Number of quantile bins (default = 10).
    eps      : float, optional
        Small constant to avoid log(0).

    Returns
    -------
    pd.DataFrame with columns:
        feature   – column name
        psi       – PSI value
        category  – 'stable', 'watch', or 'remove' (0.10 / 0.20 thresholds)
    """
    common_cols = df_train.columns.intersection(df_test.columns)
    psi_values = []

    def _psi(col_tr, col_te):
        # quantile breakpoints on train
        breaks = np.unique(np.nanquantile(col_tr.dropna(), 
                                          q=np.linspace(0, 1, n_bins + 1)))
        if len(breaks) < 3:                       # constant / too few uniques
            return np.nan
        tr_bins = pd.cut(col_tr, breaks, include_lowest=True)
        te_bins = pd.cut(col_te, breaks, include_lowest=True)

        tr_perc = tr_bins.value_counts(normalize=True, sort=False)
        te_perc = te_bins.value_counts(normalize=True, sort=False)

        tr_perc, te_perc = tr_perc.align(te_perc, fill_value=0)
        tr_perc = tr_perc.replace(0, eps)
        te_perc = te_perc.replace(0, eps)

        return ((tr_perc - te_perc) * np.log(tr_perc / te_perc)).sum()

    for col in common_cols:
        psi = _psi(df_train[col], df_test[col])
        psi_values.append(psi)

    psi_df = (pd.DataFrame({'feature': common_cols, 'psi': psi_values})
              .assign(category=lambda d: pd.cut(
                  d['psi'],
                  bins=[-np.inf, 0.10, 0.20, np.inf],
                  labels=['stable', 'watch', 'remove']))
              .sort_values('psi', ascending=False)
              .reset_index(drop=True))

    return psi_df