import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from collections import defaultdict
from copy import deepcopy

class Features:
    def __init__(self, features: Dict[List[str]]):
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
            if df[feature].dtype == 'object':
                features['categorical'].append(feature)
            else:
                features['numerical'].append(feature)
        return cls(features)
    
    def __repr__(self):
        return str(self.features)
        
        
    

class DataProcessor:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df, self.target = df.drop([target], axis=1), df[target]
        self.features = Features.from_dataframe(self.df)        
                
    def copy(self) -> "DataProcessor":
        return deepcopy(self)
        
    def drop_features(self, to_drop: List[str]):
        self.features.drop(to_drop)
        self.df.drop(to_drop, axis=1, inplace=True)
    
    def get_constant_features(self, distinct_thr = 0.98, verbose=True):
        constants = []
        for col in self.df.columns:
            col_dist = self.df[col].value_counts(normalize=True).to_dict()
            col_dist = sorted(col_dist.items(), key=lambda item: item[1])
            most_freq_element, frequency = col_dist[-1]
            if frequency>= distinct_thr:
                constants.append(col)
                if verbose:
                    print(f"constant column '{col}' with value {most_freq_element} at {round(frequency*100,2)}%")
        return constants

    def get_nan_features(self, nan_thr=0.5, verbose=True):
        nan_features = []
        nans = dict(self.df.isna().sum()/len(self.df))
        
        for col, nan_ratio in nans.items():
            if nan_ratio >= nan_thr:
                nan_features.append(nan_thr)
                if verbose:
                    print(f"NAN column '{col}' with {round(nan_ratio*100,2)}% misses")
        return nan_features
    
    def numeric_corr(self):
        pass

            
    
    
        
    

def ClfProcessor(DataProcessor):
    pass

def RegProcessor(DataProcessor):
    pass
        