# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

def calculate_pdf(train, test):
    train, test = reverse(train, test)
    train, test = scale(train, test)
    pdfs = get_pdfs(train)
    df_pdf = pd.DataFrame(pdfs.T, columns=['var_prob_%d' % i for i in range(200)])
    return df_pdf, train, test

def logloss(y, yp):
    yp = np.clip(yp, 1e-5, 1 - 1e-5)
    return -y * np.log(yp) - (1 - y) * np.log(1 - yp)


def reverse(tr, te):
    reverse_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 18, 19,
                    22, 24, 25, 26, 27, 41, 29,
                    32, 35, 37, 40, 48, 49, 47,
                    55, 51, 52, 53, 60, 61, 62, 65, 66, 67, 69,
                    70, 71, 74, 78, 79,
                    82, 84, 89, 90, 91, 94, 95, 96, 97, 99, 103,
                    105, 106, 110, 111, 112, 118, 119, 125, 128,
                    130, 133, 134, 135, 137, 138,
                    140, 144, 145, 147, 151, 155, 157, 159,
                    161, 162, 163, 164, 167, 168,
                    170, 171, 173, 175, 176, 179,
                    180, 181, 184, 185, 187, 189,
                    190, 191, 195, 196, 199]
    reverse_list = ['var_%d' % i for i in reverse_list]
    for col in reverse_list:
        tr[col] = tr[col] * (-1)
        te[col] = te[col] * (-1)
    return tr, te


def scale(tr, te):
    trte = pd.concat([tr, te], axis=0)
    for col in tr.columns:
        if col.startswith('var_'):
            mean, std = tr[col].mean(), tr[col].std()
            tr[col] = (tr[col] - mean) / std
            te[col] = (te[col] - mean) / std
    return tr, te


def getp_vec_sum(x, x_sort, y, std, c=0.5):
    # x is sorted
    left = x - std / c
    right = x + std / c
    p_left = np.searchsorted(x_sort, left)
    p_right = np.searchsorted(x_sort, right)
    p_right[p_right >= y.shape[0]] = y.shape[0] - 1
    p_left[p_left >= y.shape[0]] = y.shape[0] - 1
    return (y[p_right] - y[p_left])


def get_pdf(tr, col, x_query=None, smooth=3):
    std = tr[col].std()
    df = tr.groupby(col).agg({'target': ['sum', 'count']})
    cols = ['sum_y', 'count_y']
    df.columns = cols
    df = df.reset_index()
    df = df.sort_values(col)
    y, c = cols

    df[y] = df[y].cumsum()
    df[c] = df[c].cumsum()

    if x_query is None:
        rmin, rmax, res = -5.0, 5.0, 501
        x_query = np.linspace(rmin, rmax, res)

    dg = pd.DataFrame()
    tm = getp_vec_sum(x_query, df[col].values, df[y].values, std, c=smooth)
    cm = getp_vec_sum(x_query, df[col].values, df[c].values, std, c=smooth) + 1
    dg['res'] = tm / cm
    dg.loc[cm < 500, 'res'] = 0.1
    return dg['res'].values


def get_pdfs(tr):
    y = []
    for i in range(200):
        name = 'var_%d' % i
        res = get_pdf(tr, name)
        y.append(res)
    return np.vstack(y)


def print_corr(corr_mat, col, bar=0.97):
    print(col)
    cols = corr_mat.loc[corr_mat[col] > bar, col].index.values
    cols_ = ['var_%s' % (i.split('_')[-1]) for i in cols]
    print(cols)
    return cols


if __name__ == '__main__':
    pass