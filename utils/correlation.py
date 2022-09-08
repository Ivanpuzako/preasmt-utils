import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from scipy.stats import chi2_contingency
from collections import defaultdict
import numpy as np


def calc_correlation(
    df, method: str, min_threshold=0.6, max_n: int = 0
) -> pd.DataFrame:
    correlation = df.corr(method=method)
    cols_corr = {}
    for col_name, row in correlation.iterrows():
        ## remove correlation with itself
        r = row.copy().drop(col_name)
        max_corr = r.max()
        if max_corr >= min_threshold:
            cols_corr[col_name] = max_corr
    if max_n > 0:
        cols_corr = sorted(correlation.items(), key=lambda x: x[1], reverse=True)[
            :max_n
        ]
        cols_corr = [r[0] for r in cols_corr]
    else:
        cols_corr = list(cols_corr.keys())

    return round(df[cols_corr].corr(method=method), 2)


def cramers_V(var1: np.array, var2: np.array):
    crosstab = np.array(
        pd.crosstab(var1, var2, rownames=None, colnames=None)
    )  # Cross table building
    stat = chi2_contingency(crosstab)[
        0
    ]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = (
        min(crosstab.shape) - 1
    )  # Take the minimum value between the columns and the rows of the cross table
    return stat / (obs * mini)


def plot_corr_matrix(corr: pd.DataFrame, title: str = ""):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap="cool", mask=mask, center=0, vmin=-1, vmax=1)
    plt.title(title)
