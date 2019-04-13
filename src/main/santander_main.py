# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
from src.util.log_util import set_logger
from src.data.make_dataset import read_train_data, read_test_data
from src.features.build_features import process_data, add_shuffle_data
from src.models.train_model import train_model
from src.submission.submit_data import submit
from src.result.summarize_result import storage_src
import numpy as np
import time

logger = set_logger(__name__)
metric = 'auc'
params = dict(objective = "binary",
               boost="gbdt",
               metric="auc",
               boost_from_average="false",
               num_threads=10,
               learning_rate = 0.01,
               num_leaves = 13,
               max_depth=-1,
               tree_learner = "serial",
               feature_fraction = 0.05,
               bagging_freq = 5,
               bagging_fraction = 0.4,
               min_data_in_leaf = 80,
               min_sum_hessian_in_leaf = 10.0,
               verbosity = -1,
              device_type = 'gpu'
              )

def main_submit():
    start_time = time.time()
    train = read_train_data(nrows=None)
    test = read_test_data()
    idx = [c for c in train.columns if c not in ['ID_code', 'target']]

    train, test = process_data(train, test)
    X = train.drop(['ID_code', 'target'], axis=1)
    y = train['target']
    X_test = test.drop(['ID_code'], axis=1)
    oof, predictions, scores, feature_importance = train_model(X, X_test, y, idx, params, n_fold=10,
                                                               plot_feature_importance=True, model_type='lgb_sklearn')
    str_metric_score = metric + '_0' + str(int(scores['auc_score'].iloc[0] * 10000))
    submit(test, predictions, str_metric_score)
    comment = 'count feature'
    storage_src(str_metric_score, scores, feature_importance, comment)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

if __name__ == "__main__":
    main_submit()
    # main_test()
