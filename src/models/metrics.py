# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import metrics
import json
import ast
import time
from sklearn import linear_model
import eli5
from eli5.sklearn import PermutationImportance
import shap
from tqdm import tqdm_notebook
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from src.util.log_util import set_logger

logger = set_logger(__name__)

def calculate_metrics(model, X_train: pd.DataFrame() = None, y_train: pd.DataFrame() = None, X_valid: pd.DataFrame() = None,
                      y_valid: pd.DataFrame() = None, columns: list = []) -> pd.DataFrame():
    columns = columns if len(columns) > 0 else list(X_train.columns)
    train_pred = model.predict_proba(X_train[columns])
    valid_pred = model.predict_proba(X_valid[columns])
    f1 = 0
    best_t = 0
    for t in np.arange(0.1, 1, 0.05):
        valid_pr = (valid_pred[:, 1] > t).astype(int)
        valid_f1 = metrics.f1_score(y_valid, valid_pr)
        if valid_f1 > f1:
            f1 = valid_f1
            best_t = t

    t = best_t
    train_pr = (train_pred[:, 1] > t).astype(int)
    valid_pr = (valid_pred[:, 1] > t).astype(int)
    train_f1 = metrics.f1_score(y_train, train_pr)
    valid_f1 = metrics.f1_score(y_valid, valid_pr)
    score_df = []
    print(f'Best threshold: {t:.2f}. Train f1: {train_f1:.4f}. Valid f1: {valid_f1:.4f}.')
    score_df.append(['F1', np.round(train_f1, 4), np.round(valid_f1, 4)])
    train_r = metrics.recall_score(y_train, train_pr)
    valid_r = metrics.recall_score(y_valid, valid_pr)

    score_df.append(['Recall', np.round(train_r, 4), np.round(valid_r, 4)])
    train_p = metrics.precision_score(y_train, train_pr)
    valid_p = metrics.precision_score(y_valid, valid_pr)

    score_df.append(['Precision', np.round(train_p, 4), np.round(valid_p, 4)])
    train_roc = metrics.roc_auc_score(y_train, train_pred[:, 1])
    valid_roc = metrics.roc_auc_score(y_valid, valid_pred[:, 1])

    score_df.append(['ROCAUC', np.round(train_roc, 4), np.round(valid_roc, 4)])
    train_apc = metrics.average_precision_score(y_train, train_pred[:, 1])
    valid_apc = metrics.average_precision_score(y_valid, valid_pred[:, 1])

    score_df.append(['APC', np.round(train_apc, 4), np.round(valid_apc, 4)])
    print(metrics.confusion_matrix(y_valid, valid_pr))
    score_df = pd.DataFrame(score_df, columns=['Metric', 'Train', 'Valid'])
    print(score_df)

    return score_df, t

if __name__ == '__main__':
    pass