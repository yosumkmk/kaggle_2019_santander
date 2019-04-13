# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.util.log_util import set_logger
from logging import StreamHandler, Formatter, getLogger, FileHandler, DEBUG, INFO, ERROR
from sklearn.preprocessing import StandardScaler
from src.data.make_dataset import read_permutation_importance
import random
from sklearn.decomposition import TruncatedSVD
from src.features.calculate_pdf import calculate_pdf
from joblib import Parallel, delayed

logger = set_logger(__name__)

def svd_feature(train, test, n_components=100, random_state=1337):
    svd_ = TruncatedSVD(n_components, random_state=random_state)
    train = svd_.fit_transform(train)
    test = svd_.transform(test)
    train = pd.DataFrame(train)
    train.columns = ['svd_' + str(i) for i in range(n_components)]
    test = pd.DataFrame(test)
    test.columns = ['svd_' + str(i) for i in range(n_components)]

    return train, test

def add_shuffle_data(train_df):
    repetition = 2
    new_data = pd.DataFrame()
    for i in range(repetition):
        new_train = random_sample_data(train_df, (i + 1) * 1337)
        new_data = pd.concat([new_data, new_train], axis=0)
    train_df = pd.concat([train_df, new_data], axis=0)
    train_df.reset_index(inplace=True, drop=True)
    return train_df

def random_sample_data(data, seed_num):
    idx = [c for c in data.columns if c not in ['ID_code', 'target']]
    new_df0 = pd.DataFrame()
    new_df1 = pd.DataFrame()
    data0 = data[data['target']==0]
    data1 = data[data['target']==1]
    for i, x in enumerate(idx):
        random.seed(seed_num + i)
        new_df0[x] = random.sample(data0[x].values.tolist(), len(data0))
        new_df1[x] = random.sample(data1[x].values.tolist(), len(data1))
    new_df0['target'] = 0
    new_df1['target'] = 1
    new_df = pd.concat([new_df0, new_df1], axis=0)
    new_df.reset_index(inplace=True, drop=True)
    new_df['ID_code'] = ['train_' + str(i) for i in range(len(data), len(new_df) + len(data))]
    return new_df

def max_min_feature(df, idx):
    num_list = np.arange(5)
    for num in num_list:
        use_values = df[idx].values
        df['sum_max_top' + str(num)] = 0
        df['sum_min_top' + str(num)] = 0
        sum_max_top = df['sum_max_top' + str(num)].values
        sum_min_top = df['sum_min_top' + str(num)].values
        for i in range(len(df)):
            sort_values = np.sort(use_values[i])
            sum_max_top[i] = sort_values[-(num+1)]
            sum_min_top[i] = sort_values[num]
        df['sum_max_top' + str(num)] = sum_max_top
        df['sum_min_top' + str(num)] = sum_min_top
    return df

def id_match_feature(train_df, test_df, idx):
    train_df['ID_num'] = [int(train_df['ID_code'][x][6::]) for x in range(len(train_df))]
    test_df['ID_num'] = [int(test_df['ID_code'][x][5::]) for x in range(len(test_df))]
    for df in [test_df, train_df]:
        df['ID_match'] = np.sum(df[idx] == np.tile((df['ID_num'].values / 10 ** 4).reshape(-1, 1), (1, len(idx))), axis=1)
        df.drop('ID_num', inplace=True)

def round_feature(train_df, test_df, idx):
    for df in [test_df, train_df]:
        for feat in idx:
            df[feat] = np.round(df[feat], 3)
            df[feat] = np.round(df[feat], 3)

def outlier_distribution_categorize(train_df, test_df, idx):
    outlier_dict = {}
    for c in idx:
        count_df = np.round(train_df[c], 2).value_counts().sort_index()
        outlier = count_df[(
            #         ((count_df.diff().abs() > 50) & (count_df < 200)) |
            (count_df.diff().abs() > 130)
        )].index.tolist()
        outlier_dict[c] = outlier
    for k, l in outlier_dict.items():
        o_l = []
        if len(l) >= 1:
            min_o = min(l)
            max_o = max(l)
            if min_o < 0:
                o_l.append(min_o)
            if max_o > 0:
                o_l.append(max_o)
            for o in o_l:
                train_df[k + '_outlier_' + str(o)] = (np.round(train_df[k], 2) == o).astype(np.int64)
                test_df[k + '_outlier_' + str(o)] = (np.round(test_df[k], 2) == o).astype(np.int64)

    return train_df, test_df

def print_corr(corr_mat,col,bar=0.95):
    #print(col)
    cols = corr_mat.loc[corr_mat[col]>bar,col].index.values
    return cols

def feature_group(df_pdf):
    corr_mat = df_pdf.corr(method='pearson')
    groups = []
    skip_list = []
    for i in range(0, 200):
        if (i not in skip_list):
            cols = print_corr(corr_mat, 'var_prob_' + str(i))
            if (len(cols) > 1):
                groups.append(cols)
                for e, v in enumerate(cols):
                    skip_list.append(int(v[9:]))
    return groups

def groups_feature_mean(train, test, f_groups):
    for i, g_list in enumerate(f_groups):
        train['group_' + str(i)] = 0
        test['group_' + str(i)] = 0
        col_list = []
        for g in g_list:
            col = 'var_' + g[9::]
            col_list.append(col)
            train['group_' + str(i)] += train[col]
            test['group_' + str(i)] += test[col]
        train['group_' + str(i)] /= len(g_list)
        test['group_' + str(i)] /= len(g_list)
        # train['group_skew_' + str(i)] = train[col_list].skew()
        # test['group_skew_' + str(i)] = test[col_list].skew()
        # train['group_std_' + str(i)] = train[col_list].std()
        # test['group_std_' + str(i)] = test[col_list].std()
    return train, test

def convert_pdf(train, test):
    x_query = np.linspace(-5, 5, 501)
    prob_df, train, test = calculate_pdf(train, test)
    idx = [c for c in train.columns if c not in ['ID_code', 'target', 'new_var_162']]
    idx_prob = [c for c in prob_df.columns if c not in ['ID_code', 'target', 'new_var_162']]

    for i, i_p in zip(idx, idx_prob):
        train[i] = prob_df[i_p].values[np.searchsorted(x_query, train[i].values)]
        test[i] = prob_df[i_p].values[np.searchsorted(x_query, test[i].values)]
    return train, test, prob_df

def var_162_features(train, test):
    train['new_var_162'] = [int(float(str(x * 10000000000)[2::])) for x in train['var_162']]
    test['new_var_162'] = [int(float(str(x * 10000000000)[2::])) for x in test['var_162']]
    return train, test

def count_sampla(train, test):
    idx = ['var_' + str(x) for x in range(200)]
    df = pd.concat([train[idx], test[idx]], axis=0)
    sample_df = df[idx].values
    # count_df = np.zeros((sample_df.shape))

    def calc_sample(i):
        count_sample = sample_df == sample_df[i, :]
        return np.sum(count_sample, axis=0), i

    processed = Parallel(n_jobs=-1)([delayed(calc_sample)(i) for i in range(len(df))])
    processed.sort(key=lambda x: x[1])
    count_df = [t[0] for t in processed]

    # for i in tqdm(range(len(df))):
    #     count_sample = sample_df == sample_df[i, :]
    #     count_df[i, :] = np.sum(count_sample, axis=0)

    count_idx = ['count_var_' + str(x) for x in range(200)]
    count_df = pd.DataFrame(count_df, columns=count_idx, index=df.index)
    df.merge(count_df, left_index=True, right_index=True)
    train = df.iloc[0:200000, :]
    test = df.iloc[200000::, :]
    test.drop(columns=['target'], inplace=True, axis=1)
    return train, test


def process_data(train_df, test_df):
    logger.info('Features engineering')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    train_df, test_df, prob_df = convert_pdf(train_df, test_df)
    f_groups = feature_group(prob_df)
    train_df, test_df = groups_feature_mean(train_df, test_df, f_groups)
    train_df, test_df = count_sampla(train_df, test_df)
    # train, test = svd_feature(train_df[idx], test_df[idx], n_components=180)

    # train_df = pd.concat([train_df, train_svd], axis=1)
    # test_df = pd.concat([test_df, test_svd], axis=1)
    # perm_imp = read_permutation_importance()
    # remove_features_weight = 0
    # remove_features = perm_imp[perm_imp.weight < -0.0002]
    # remove_columns.extend(remove_features.feature.tolist())
    # train_df.drop(columns=remove_columns, inplace=True)
    # test_df.drop(columns=remove_columns, inplace=True)
    # train_df, test_df = outlier_distribution_categorize(train_df, test_df, idx)
    print('Train and test shape:', train_df.shape, test_df.shape)
    return train_df, test_df


if __name__ == '__main__':
    pass
