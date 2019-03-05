# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from src.util.log_util import set_logger
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
from sklearn.preprocessing import StandardScaler
from src.data.make_dataset import read_permutation_importance

logger = set_logger(__name__)


def process_data(train_df, test_df):
    logger.info('Features engineering')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    remove_columns = []
    scaler = StandardScaler()
    for df in [test_df, train_df]:
        for feat in idx:
            df[feat] = np.round(df[feat], 3)
            df[feat] = np.round(df[feat], 3)
    train_df[idx] = scaler.fit_transform(train_df[idx])
    test_df[idx] = scaler.transform(test_df[idx])
    perm_imp = read_permutation_importance()
    # remove_features_weight = 0
    remove_features = perm_imp[perm_imp.weight < -0.0002]
    remove_columns.extend(remove_features.feature.tolist())
    train_df.drop(columns=remove_columns, inplace=True)
    test_df.drop(columns=remove_columns, inplace=True)


    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df

if __name__ == '__main__':
    pass