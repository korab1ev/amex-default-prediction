import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_group_corr_pair(df_train: pd.DataFrame,
                          df_test: pd.DataFrame,
                          feats: list,
                          title: str,
                          cmap: str = "coolwarm"):
    """
    Side‑by‑side correlation heat‑maps (train vs test) with numbers.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
    feats             : list[str]  – feature names to plot
    title             : str        – figure title prefix
    cmap              : str        – matplotlib colormap
    """
    feats = [f for f in feats if f in df_train.columns and f in df_test.columns]
    n = len(feats)

    # correlations
    corr_tr = df_train[feats].corr().round(2)
    corr_te = df_test [feats].corr().round(2)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)   # hide strictly upper

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.45*n*2), max(8, 0.45*n)),
                             sharex=True, sharey=True)
    for ax, corr, lbl in zip(axes, [corr_tr, corr_te], ['train', 'test']):
        corr_masked = corr.mask(mask)

        im = ax.imshow(corr_masked, cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(i+1):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        color="black", fontsize=8)

        ax.set_xticks(range(n), feats, rotation=90, ha="center", fontsize=8)
        ax.set_yticks(range(n), feats, fontsize=8)
        ax.set_title(f"{title} ({lbl})")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Correlation")
    # fig.tight_layout()
    plt.show()