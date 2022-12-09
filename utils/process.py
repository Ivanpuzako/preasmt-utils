import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Callable, Any, Union
from collections import defaultdict
from sklearn.model_selection._split import BaseCrossValidator
from copy import deepcopy
from utils.features import Features
from scipy.stats import kruskal, chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class DataProcessor:
    def __init__(self, df: pd.DataFrame, target: str):
        self.target_name = target
        self.initial_df = deepcopy(df)
        self.df, self.target = df.drop([target], axis=1), df[target]
        self.features = Features.from_dataframe(self.df)

    def copy(self) -> "DataProcessor":
        return deepcopy(self)

    def drop_features(self, to_drop: List[str]):
        self.features.drop(to_drop)
        self.df.drop(to_drop, axis=1, inplace=True)

    def get_constant_features(self, distinct_thr: float = 0.98, verbose=True):
        constants = []
        for col in self.df.columns:
            col_dist = self.df[col].value_counts(normalize=True).to_dict()
            col_dist = sorted(col_dist.items(), key=lambda item: item[1])
            most_freq_element, frequency = col_dist[-1]
            if frequency >= distinct_thr:
                constants.append(col)
                if verbose:
                    print(
                        f"constant column '{col}' with value {most_freq_element} at {round(frequency*100,2)}%"
                    )
        return constants

    def get_nan_features(self, nan_thr: float = 0.5, verbose: bool = True):
        nan_features = []
        nans = dict(self.df.isna().sum() / len(self.df))

        for col, nan_ratio in nans.items():
            if nan_ratio >= nan_thr:
                nan_features.append(col)
                if verbose:
                    print(f"NAN column '{col}' with {round(nan_ratio*100,3)}% misses")
        return nan_features

    def get_nan_rows(self, nan_thr: Union[int, float] = 0.5):
        row_nans_stat = self.df.isna().sum(1)
        if nan_thr <= 1 and isinstance(nan_thr, float):
            nan_thr = int(self.df.shape[1] * nan_thr)
        nan_rows = row_nans_stat > nan_thr
        print(f"found {nan_rows.sum()} itmes with large number of nans")
        return nan_rows

    def cross_validate(
        self,
        estimator,
        metrics: List[Tuple[str, Callable]],
        cv: BaseCrossValidator = StratifiedKFold(n_splits=3),
    ):
        folds = [fold for fold in cv.split(self.x_train, self.y_train)]
        result = pd.DataFrame()
        for fold_n, (train_idx, test_idx) in enumerate(folds):
            x_train, y_train = (
                self.x_train.iloc[train_idx],
                self.y_train.iloc[train_idx],
            )
            x_test, y_test = self.x_train.iloc[test_idx], self.y_train.iloc[test_idx]
            estimator_cp = deepcopy(estimator)
            estimator_cp.fit(x_train, y_train)
            pred_train = estimator_cp.predict(x_train)
            pred_test = estimator_cp.predict(x_test)
            fold_result = {"fold_#": fold_n}
            for metric_name, metric in metrics:
                if "prob" in metric_name:
                    pred_train_proba = estimator_cp.predict_proba(x_train)[:, -1]
                    pred_test_proba = estimator_cp.predict_proba(x_test)[:, -1]
                    fold_result["train_" + metric_name] = metric(
                        y_train, pred_train_proba
                    )
                    fold_result["test_" + metric_name] = metric(y_test, pred_test_proba)
                else:
                    fold_result["train_" + metric_name] = metric(y_train, pred_train)
                    fold_result["test_" + metric_name] = metric(y_test, pred_test)
            result = result.append(fold_result, ignore_index=True)
        train_cols = [c for c in result.columns if c.startswith("train")]
        test_cols = [c for c in result.columns if c.startswith("test")]
        result = result[["fold_#"] + train_cols + test_cols]
        avg_result = result.mean().to_dict()
        avg_result["fold_#"] = "AVERAGE:"
        result = result.append(avg_result, ignore_index=True).set_index("fold_#")
        return result

    def features_target_analysys(
        self,
        data,
        features=None,
        criterion=kruskal,
        p_value=0.05,
        target=None,
        verbose=True,
    ) -> Dict[str, bool]:
        features_to_analyze = (
            features if features is not None else self.features["numerical"]
        )
        target_values = target if target is not None else self.target
        feature_importance = {}
        print(f"check feature importance by {criterion} criterion")
        print("H0 - feature does not affect Target")
        for feature_name in features_to_analyze:
            stat, p = criterion(data[feature_name], target_values)
            is_important = p <= p_value
            feature_importance[feature_name] = is_important
            if verbose:
                if is_important:
                    #  print(f"feature {feature_name} is important with p={round(p,2)}")
                    pass
                else:
                    print(
                        f"feature {feature_name} probably is NOT important with p_value={round(p,2)}"
                    )
        return feature_importance

    ## TODO


class ClfProcessor(DataProcessor):
    def num_feature_importance(self, verbose=True):
        def criterion(feature, target):
            samples = []
            for value in np.unique(target):
                samples.append(feature[target == value])
            stat, p = kruskal(*samples)
            return stat, p

        return self.features_target_analysys(
            self.df,
            features=self.features["numerical"],
            criterion=criterion,
            target=self.target,
            verbose=verbose,
        )

    def cat_feature_importance(self, verbose=True):
        def criterion(f1, f2):
            chi2, p, dof, expected = chi2_contingency(pd.crosstab(f1, f2))
            return chi2, p

        return self.features_target_analysys(
            self.df,
            features=self.features["categorical"],
            criterion=criterion,
            target=self.target,
            verbose=verbose,
        )

    def evaluate(
        self,
        estimator,
        target: np.array = None,
        labels: List[Any] = None,
        title: str = "Confusion matrix",
    ):
        y_pred = estimator.predict(self.x_test)
        y_test = target if target is not None else self.y_test
        print(classification_report(y_test, y_pred, target_names=labels))
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=None, index=None)
        sns.heatmap(cm, annot=True, cmap="cool", fmt="d")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title(title)

    def train_test_split(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df,
            self.target,
            stratify=self.target,
            test_size=test_size,
            random_state=2022,
        )
        print(f"stratified split with test size {test_size}")
        print("train target dist")
        print(self.y_train.value_counts())
        print("test target dist")
        print(self.y_test.value_counts())


class RegProcessor(DataProcessor):
    def train_test_split(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df,
            self.target,
            test_size=test_size,
            random_state=2022,
        )
        print(f"split with test size {test_size}")
