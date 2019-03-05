# -*- coding: utf-8 -*-
import warnings
from src.util.log_util import set_logger
import pandas as pd
warnings.filterwarnings('ignore')
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
import numpy as np
import os
import data.permutation_importance
import data.input

logger = set_logger(__name__)


def read_train_data(nrows=None):
    logger.info('Input train_data')
    train_df = pd.read_csv(os.path.join(data.input.__path__[0], 'train.csv'), nrows=nrows)
    return train_df

def read_test_data():
    logger.info('Input test_data')
    test_df = pd.read_csv(os.path.join(data.input.__path__[0], 'test.csv'))
    return test_df

def read_permutation_importance(name='0303_base_feature.csv'):
    return pd.read_csv(os.path.join(data.permutation_importance.__path__[0], name))

def split_train_data(train, split_rate):
    split_index = len(train) // split_rate
    return train[0:split_index], train[split_index]

def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)

if __name__ == '__main__':
    pass