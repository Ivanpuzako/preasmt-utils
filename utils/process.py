import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
from sklearn.model_selection._split import BaseCrossValidator
from copy import deepcopy
from sklearn.model_selection import train_test_split, StratifiedKFold


class Features:
    def __init__(self, features: Dict[str, List[str]]):
        self.features = defaultdict(list, **features)

    def drop(self, to_drop: List[str]):
        for to_drop_name in to_drop:
            for category, features in self.features.items():
                if to_drop_name in features:
                    self.features[category].remove(to_drop_name)

    def add(self, to_add: List[str], category: str):
        self.drop(to_add)
        self.features[category] += to_add

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        features = defaultdict(list)
        for feature in df.columns:
            if df[feature].dtype == "object":
                features["categorical"].append(feature)
            else:
                features["numerical"].append(feature)
        return cls(features)

    def __repr__(self):
        return str(self.features)

    def __getitem__(self, key):
        return self.features[key]


class DataProcessor:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df, self.target = df.drop([target], axis=1), df[target]
        self.features = Features.from_dataframe(self.df)

    def copy(self) -> "DataProcessor":
        return deepcopy(self)

    def drop_features(self, to_drop: List[str]):
        self.features.drop(to_drop)
        self.df.drop(to_drop, axis=1, inplace=True)

    def get_constant_features(self, distinct_thr=0.98, verbose=True):
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

    def get_nan_features(self, nan_thr=0.5, verbose=True):
        nan_features = []
        nans = dict(self.df.isna().sum() / len(self.df))

        for col, nan_ratio in nans.items():
            if nan_ratio >= nan_thr:
                nan_features.append(nan_thr)
                if verbose:
                    print(f"NAN column '{col}' with {round(nan_ratio*100,2)}% misses")
        return nan_features

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
            estimator.fit(x_train, y_train)
            pred_train = estimator.predict(x_train)
            pred_test = estimator.predict(x_test)
            fold_result = {"fold_#": fold_n}
            for metric_name, metric in metrics:
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

    def show_target_dist(self):
        pass

    def train_test_split(self):
        pass


class ClfProcessor(DataProcessor):
    def show_target_dist(self):
        pass

    def train_test_split(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.df,
            self.target,
            stratify=self.target,
            test_size=test_size,
            random_state=2022,
        )
        print("stratified split with test size {test_size}")
        print("train target dist")
        print(self.y_train.value_counts())
        print("test target dist")
        print(self.y_test.value_counts())


class RegProcessor(DataProcessor):
    pass
