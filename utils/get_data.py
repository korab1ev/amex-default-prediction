
import pandas as pd
import numpy as np
import gc
from typing import List


def get_train_data(TRAIN_PATH: str) -> pd.DataFrame:
    '''Returns train dataset'''
    df_train = pd.read_parquet(TRAIN_PATH)
    return df_train


def get_test_data(TEST_PATH: str) -> pd.DataFrame:
    '''Returns test dataset'''
    df_test = pd.read_parquet(TEST_PATH)
    return df_test


def get_target(TARGET_PATH: str) -> pd.DataFrame:
    '''Retruns dataset with train targets'''
    df_train_target = pd.read_csv(TARGET_PATH)
    return df_train_target


def get_train_data_with_target_merged(df_train: pd.DataFrame, df_train_target: pd.DataFrame) -> pd.DataFrame:
    '''Retruns train dataset with target variable merged'''
    df_train_w_target = (
        df_train
        .merge(df_train_target,
            on='customer_ID',
            how='left'
        )
    )
    # df_train_w_target.groupby('target', dropna=False).count()['customer_ID']
    '''
    target
    0    4153582
    1    1377869
    Name: customer_ID, dtype: int64    
    '''
    return df_train_w_target


def get_all_features(df: pd.DataFrame) -> List:
    '''Returns list of all features from the dataset'''
    return list(df)


def get_cat_features() -> List:
    '''Returns list of categorical features from the dataset'''
    cat_features = ['B_30', 'B_38', 'D_114', 
                    'D_116', 'D_117', 'D_120', 
                    'D_126', 'D_63', 'D_64', 
                    'D_66', 'D_68']
    
    return cat_features


def get_num_features(all_features: List, cat_features: List) -> List:
    '''Returns list of all numerical features from the dataset'''
    num_feats = [col for col in all_features if col not in cat_features + ['customer_ID', 'S_2', 'target']]

    return num_feats

# features
def get_df_w_aggrs(df: pd.DataFrame, feats: List) ->  pd.DataFrame:
    '''Returns dataframe with generated aggregates based on numerical and categorical features'''

    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer

    # тех долг: hard coded last agg on categorical features
    cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    
    df_min = (df
        .groupby(cid, observed=True)
        .min()[feats]
        .rename(columns={f: f"{f}_min" for f in feats})
    )
    print(df_min.shape)

    df_max = (df
        .groupby(cid, observed=True)
        .max()[feats]
        .rename(columns={f: f"{f}_max" for f in feats})
    )
    print(df_max.shape)

    df_avg = (df
        .drop('S_2', axis='columns')
        .groupby(cid, observed=True)
        .mean()[feats]
        .rename(columns={f: f"{f}_avg" for f in feats})
    )
    print(df_avg.shape)

    df_last = (df
        .loc[last, feats + cat_features] # hard coded this, тех. долг
        .rename(columns={f: f"{f}_last" for f in feats + cat_features})
        .set_index(np.asarray(cid[last]))
    )
    print(df_last.shape)

    df_aggrs = (pd.concat([df_min, df_max, df_avg, df_last], axis=1)
        .reset_index()
        .rename(columns={'index': 'customer_ID'})
    )
    print(df_aggrs.shape)

    '''
    del df, df_min, df_max, df_avg, cid, last
    gc.collect()
    '''
    return df_aggrs