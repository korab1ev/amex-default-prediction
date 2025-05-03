
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_distr(df: pd.DataFrame, feature: str, n_bins: int = 10):
    """
    Строит помесячное распределение признака (share в каждом квантильном бине)
    отдельно для target = 0 и target = 1. Легенда показывает номера бин‑квантилей
    и их числовые границы.
    """
    # копируем, чтобы не портить оригинал
    df = df.copy()
    df['generation'] = pd.to_datetime(df['S_2']).dt.to_period('M').astype(str)
    
    # формируем квантильные бины на полной выборке
    df['bin'] = pd.qcut(df[feature], q=n_bins, duplicates='drop')
    
    # получаем упорядоченный список интервалов
    bin_intervals = df['bin'].cat.categories
    
    def share_matrix(subframe):
        """возвращает DataFrame generation × bin с долями внутри месяца"""
        counts = (
            subframe.groupby(['generation', 'bin'], observed=True)
                    .size()
                    .unstack(fill_value=0)
        )
        # нормируем на 1 в каждом поколении
        return counts.div(counts.sum(axis=1), axis=0).sort_index()
    
    share_0 = share_matrix(df[df['target'] == 0])
    share_1 = share_matrix(df[df['target'] == 1])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, data, title in zip(axes, [share_0, share_1], ['target = 0', 'target = 1']):
        
        bottom = np.zeros(len(data))
        # столбики в порядке интервалов
        for i, interval in enumerate(bin_intervals):
            
            values = data.get(interval, pd.Series(0, index=data.index))
            ax.bar(data.index, values, bottom=bottom,
                   label=f"bin {i}: {interval.left:.4f} – {interval.right:.4f}")
            bottom += values.values
            
        ax.set_title(title)
        ax.set_xlabel("generation (YYYY-MM)")
        ax.set_ylabel("Share in month")
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylim(0, 1.01)
    
    # единая легенда справа
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    fig.suptitle(f"Monthly share of '{feature}' bins (q={n_bins})", y=1.02)
    plt.show()