import pandas as pd
from collections import defaultdict
from typing import Dict, List


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
            elif df[feature].nunique() == 2:
                features["binary"].append(feature)
            else:
                features["numerical"].append(feature)
        return cls(features)

    def __repr__(self):
        return str(self.features)

    def __getitem__(self, key):
        return self.features[key]


def encode_with_frequency(
    features: pd.DataFrame, max_features: int = 10, min_samples_category: float = 100
):
    """having high cardinality features, save only most popular features and drop less popular"""
    result = features.copy()
    for col in result.columns:
        feature = result[col]
        if isinstance(min_samples_category, float):
            min_samples = int(len(feature) * min_samples)
        else:
            min_samples = min_samples_category
        stat = feature.value_counts()
        stat = stat[stat >= min_samples]
        stat = stat[:max_features]
        feature = pd.Series(feature).map(
            lambda x: "other" if x not in stat.index else x
        )
        result[col] = feature
    return result


def quantize(feature: pd.Series, borders):
    def quantize_(value, borders):
        for i, _ in enumerate(borders[:-1]):
            if value > borders[i] and value <= borders[i + 1]:
                return i

    return feature.map(lambda x: quantize_(x, borders))
