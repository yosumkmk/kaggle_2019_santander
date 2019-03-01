# -*- coding: utf-8 -*-
import gc
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
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
from catboost import CatBoostClassifier
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
from logging.handlers import RotatingFileHandler
import logging
from pytz import utc, timezone

def get_logger(name):
    BASE_DIR = os.path.realpath(os.path.dirname(__file__))
    LOG_DIR = os.path.join(BASE_DIR, 'logs')  # ログファイルディレクトリ

    # ログファイルディレクトリがなければ作成する
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if name is None:
        name = __name__

    #################
    # logger 設定
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(module)s][%(funcName)s] %(message)s', datefmt='%m-%d %H:%M:%S.%s,%03d')

    def customTime(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Tokyo")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    log_fmt.converter = customTime

    strm_handler = StreamHandler()
    strm_handler.setLevel(DEBUG)
    strm_handler.setFormatter(log_fmt)
    logger.addHandler(strm_handler)

    rotate_filehandler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, 'log_util.log'),
        maxBytes=1024 * 1024 * 5,
        backupCount=10
    )
    rotate_filehandler.setLevel(INFO)
    rotate_filehandler.setFormatter(log_fmt)
    logger.addHandler(rotate_filehandler)

    logger.info('start')
    return logger

logger = get_logger(__name__)


def read_data(nrows=None):
    logger.info('Input data')
    train_df = pd.read_csv('../input/train.csv', nrows=nrows)
    test_df = pd.read_csv('../input/test.csv')
    return train_df, test_df


def process_data(train_df, test_df):
    logger.info('Features engineering')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    for df in [test_df, train_df]:
        for feat in idx:
            df['r2_' + feat] = np.round(df[feat], 2)
            df['r2_' + feat] = np.round(df[feat], 2)
        df['sum'] = df[idx].sum(axis=1)
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df

def train_model(X, X_test, y, params, n_fold=5, shuffle_folds=True,  model_type='lgb', plot_feature_importance=False,
                averaging='usual', model=None, folds_random_state=42):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    folds = StratifiedKFold(n_splits=n_fold, shuffle=shuffle_folds, random_state=folds_random_state)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=20000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=1000,
                              early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')

            y_pred = model.predict_proba(X_test)[:, 1]

        if model_type == 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)

            y_pred = model_results.predict(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            return oof, prediction, feature_importance
        return oof, prediction, scores

    else:
        return oof, prediction, scores


def run_model(train_df, test_df):
    logger.info('Prepare the model')
    features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    target = train_df['target']
    logger.info('Run model')
    param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
    num_round = 15000
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=100)
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    return predictions


def submit(test_df, predictions):
    logger.info('Prepare submission')
    sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
    sub["target"] = predictions
    sub.to_csv("submission.csv", index=False)


def main(nrows=None):
    train_df, test_df = read_data(nrows)
    train_df, test_df = process_data(train_df, test_df)
    predictions = run_model(train_df, test_df)
    submit(test_df, predictions)


if __name__ == "__main__":
    main()

if __name__ == '__main__':
    pass