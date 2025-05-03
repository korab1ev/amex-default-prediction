import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_cat_hist(train: pd.DataFrame,
                  test: pd.DataFrame,
                  cat_cols: list,
                  normalize: bool = True):
    """
    For each categorical feature plot train vs test bar chart.

    Parameters
    ----------
    train, test : pd.DataFrame
    cat_cols    : list[str] – categorical columns to iterate over
    normalize   : bool      – True → plot shares (0‑1); False → absolute counts
    """
    for col in cat_cols:
        if col not in train.columns or col not in test.columns:
            print(f"{col} missing in one of the DataFrames, skipped.")
            continue

        tr = train[col].astype('category')
        te = test[col].astype('category')

        # Make sure both have same category order
        all_cats = pd.Index(tr.cat.categories).union(te.cat.categories)
        tr = tr.cat.set_categories(all_cats)
        te = te.cat.set_categories(all_cats)

        tr_counts = tr.value_counts(normalize=normalize).reindex(all_cats, fill_value=0)
        te_counts = te.value_counts(normalize=normalize).reindex(all_cats, fill_value=0)

        idx = np.arange(len(all_cats))
        w = 0.4

        plt.figure(figsize=(max(6, 0.5 * len(all_cats)), 4))
        plt.bar(idx - w/2, tr_counts, width=w, label="train")
        plt.bar(idx + w/2, te_counts, width=w, label="test")

        plt.xticks(idx, all_cats, rotation=90, fontsize=8)
        plt.ylabel("share" if normalize else "count")
        plt.title(f"{col}: train vs test distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()
