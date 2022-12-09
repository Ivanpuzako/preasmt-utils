import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from scipy.stats import chi2_contingency
from collections import defaultdict
import numpy as np


def calc_correlation(
    df, method: str = "spearman", min_threshold=0.6, max_n: int = 0
) -> pd.DataFrame:
    correlation = df.corr(method=method)
    cols_corr = {}
    for col_name, row in correlation.iterrows():
        ## remove correlation with itself
        r = row.copy().drop(col_name)
        max_corr = np.abs(r).max()
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


def get_most_correlated_to(
    corr_df, target, min_abs_corr=0.1, max_features=100
) -> pd.DataFrame:
    target_corr = corr_df[target]
    most_correlated = np.abs(target_corr).sort_values()[-max_features:]
    cols = list(most_correlated[np.abs(most_correlated) >= min_abs_corr].index)
    return corr_df.loc[cols, cols]


def cramers_V(var1: np.array, var2: np.array) -> float:
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
