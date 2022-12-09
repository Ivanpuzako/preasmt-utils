import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List


def plot_num_features_cat_target(
    features: pd.DataFrame, target: pd.Series, legends: List[str] = None
):
    feature_names = features.columns
    fig, axes = plt.subplots(len(feature_names) // 2 + len(feature_names) % 2, 2)
    fig.set_figheight(len(feature_names) // 2 * 5)
    fig.set_figwidth(17)
    axes = axes.flatten()
    for i, feature in enumerate(feature_names):
        ax = sns.boxplot(data=features, x=target, y=feature, ax=axes[i])
        if legends is not None:
            ax.set_title(legends[i])


def plot_cat_features_cat_target(
    features: pd.DataFrame, target: pd.Series, legends=None
):
    feature_names = features.columns
    fig, axes = plt.subplots(len(feature_names))
    fig.set_figheight(len(feature_names) * 5)
    fig.set_figwidth(17)
    axes = axes.flatten()
    for i, feature in enumerate(feature_names):
        ax = sns.countplot(data=features, x=feature, hue=target, ax=axes[i])
        ax.grid()
        if legends is not None:
            ax.set_title(legends[i])


def plot_linear_regression_importance(
    estimator, is_pipeline=True, feature_names=None, max_num=-1
):
    if feature_names is None and is_pipeline:
        last_step_name = estimator.steps[-2][0]
        feature_names = estimator[last_step_name].get_feature_names_out()
        model = estimator[-1]

    else:
        model = estimator
    coefs = pd.DataFrame(
        model.coef_.reshape(-1, 1),
        columns=["Coefficients"],
        index=feature_names,
    )
    coefs["abs_value"] = coefs["Coefficients"].map(np.abs)
    coefs = coefs.sort_values("abs_value", ascending=True)
    if max_num > 0:
        coefs = coefs[:max_num]
    coefs.loc[:, ["Coefficients"]].plot.barh(figsize=(9, 7))
    plt.title("Regression model importance")
    plt.axvline(x=0, color=".5")
    plt.xlabel("Raw coefficient values")
    plt.subplots_adjust(left=0.3)
    return coefs


def plot_corr_matrix(corr: pd.DataFrame, title: str = "", annot=True):
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap="cool", mask=mask, center=0, vmin=-1, vmax=1, annot=annot)
    plt.title(title)
