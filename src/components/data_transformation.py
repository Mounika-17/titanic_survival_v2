import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

# -----------------------------
# Custom Transformers
# -----------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import yeojohnson

class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fill_values_ = {}

    def fit(self, X, y=None):
        X_ = X.copy()
        for col in X_.columns:
            if X_[col].dtype == "object" or X_[col].dtype.name == 'category':
                if X_[col].isnull().sum() > 0:
                    self.fill_values_[col] = X_[col].mode()[0]
            else:
                if X_[col].isnull().sum() > 0:
                    skewness = X_[col].skew()
                    if skewness > 1 or skewness < -1:
                        self.fill_values_[col] = X_[col].median()
                    else:
                        self.fill_values_[col] = X_[col].mean()
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, fill_value in self.fill_values_.items():
            if col in X_.columns:
                X_[col] = X_[col].fillna(fill_value)

        # handle unseen missing values (test set only)
        for col in X_.columns:
            if X_[col].isnull().sum() > 0 and col not in self.fill_values_:
                if X_[col].dtype == "object":
                    fill_value = X_[col].mode()[0]
                else:
                    skewness = X_[col].skew()
                    if skewness > 1 or skewness < -1:
                        fill_value = X_[col].median()
                    else:
                        fill_value = X_[col].mean()
                X_[col] = X_[col].fillna(fill_value)
        return X_


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_threshold=10, skew_threshold=0.75):
        self.discrete_threshold = discrete_threshold
        self.skew_threshold = skew_threshold
        self.params_ = {}
        self.outlier_columns = []

    def fit(self, X, y=None):
        X_ = X.copy()
        numeric_columns = X_.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_[col].dropna().empty:
                continue
            unique_values = X_[col].nunique()
            col_type = 'discrete' if unique_values <= self.discrete_threshold else 'continuous'
            Q1 = X_[col].quantile(0.25)
            Q3 = X_[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            extreme_outliers = ((X_[col] < lower_bound) | (X_[col] > upper_bound)).sum()
            if extreme_outliers > 0:
                self.outlier_columns.append(col)
                if col_type == 'discrete':
                    self.params_[col] = ('cap', lower_bound, upper_bound)
                else:
                    skewness = X_[col].skew()
                    if abs(skewness) > self.skew_threshold:
                        if (X_[col] >= 0).all():
                            self.params_[col] = ('log',)
                        else:
                            self.params_[col] = ('yeojohnson',)
                    else:
                        self.params_[col] = ('cap', lower_bound, upper_bound)
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, params in self.params_.items():
            if params[0] == 'cap':
                _, lower, upper = params
                X_[col] = np.where(X_[col] < lower, lower,
                                   np.where(X_[col] > upper, upper, X_[col]))
            elif params[0] == 'log':
                X_[col] = np.log1p(X_[col])
            elif params[0] == 'yeojohnson':
                X_[col], _ = yeojohnson(X_[col])
        return X_


class Binner(BaseEstimator, TransformerMixin):
    def __init__(self, column, bins, labels):
        self.column = column
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        # pd.cut() — a pandas function that divides continuous data into bins.include_lowest=True: ensures the lowest value is included in the first bin.
        # This Binner class is used in data preprocessing pipelines to bin continuous numerical features into categorical ranges.
        X_[self.column + "Bin"] = pd.cut(
            X_[self.column], bins=self.bins, labels=self.labels, include_lowest=True
        )
        return X_


# -----------------------------
# Config
# -----------------------------


class DataTransformation:
    def build_preprocessor(self, ordinal_columns, nominal_columns, numeric_features):
        preprocessor = ColumnTransformer(
            transformers=[
                ('ord', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_columns),
                ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_columns),
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='drop'
        )
        return preprocessor

    def get_data_transformer_object(self, ordinal_columns, nominal_columns, numeric_features):
        try:
            pipeline = Pipeline([
                ('custom_imputer', CustomImputer()),
                ('outliers', OutlierHandler()),
                ('age_binner', Binner("Age",
                                      bins=[0, 12, 18, 35, 50, 80],
                                      labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"])),
                ('fare_binner', Binner("Fare",
                                       bins=[-0.1, 7.91, 14.45, 31, 600],
                                       labels=["Low", "Mid", "High", "VeryHigh"])),
                ('encode_scale', self.build_preprocessor(ordinal_columns, nominal_columns, numeric_features))
            ])
            return pipeline
            logger.info("Preprocessing object created")
        except Exception as e:
            raise CustomException(e, sys)