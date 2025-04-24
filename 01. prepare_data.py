
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
    '''Returns dataframe with generated aggregates based on numerical features'''

    cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
    last = (cid != np.roll(cid, -1)) # mask for last statement of every customer

    df_min = (df
        .groupby(cid)
        .min()[feats]
        .rename(columns={f: f"{f}_min" for f in feats})
    )
    print(df_min.shape)

    df_max = (df
        .groupby(cid)
        .max()[feats]
        .rename(columns={f: f"{f}_max" for f in feats})
    )
    print(df_max.shape)

    df_avg = (df
        .drop('S_2', axis='columns')
        .groupby(cid)
        .mean()[feats]
        .rename(columns={f: f"{f}_avg" for f in feats})
    )
    print(df_avg.shape)

    df_last = (df
        .loc[last, feats]
        .rename(columns={f: f"{f}_last" for f in feats})
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



def check_zapolnenie(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns pd.DataFrame with isNotNullShare of each column of given df'''
    # Calculate percent of not null share each column 
    col_pct_notNull = [] 
    for col in df.columns: 
        percent_notNull = np.mean(~df[col].isnull())*100 
        col_pct_notNull.append([col, percent_notNull]) 
        
    col_pct_notNull_df = pd.DataFrame(col_pct_notNull, columns = ['column_name','isNotNullShare']).sort_values(by = 'isNotNullShare', ascending = False) 
    #print(col_pct_notNull_df)
    return col_pct_notNull_df





def main() -> int:
    # kaggle paths
    #df_train = get_train_data(TRAIN_PATH=f'../input/amex-data-integer-dtypes-parquet-format/train.parquet')
    #df_train_target = get_target(TARGET_PATH='/kaggle/input/train-labels-amex/train_labels.csv')
    #df_test = get_test_data(TEST_PATH=f'../input/amex-data-integer-dtypes-parquet-format/test.parquet')
    
    df_train = get_train_data(TRAIN_PATH='./data/train.parquet')

    all_features = get_all_features(df_train)
    cat_features = get_cat_features()
    num_features = get_num_features(all_features, cat_features)
    # len(all_features), len(cat_features), len(num_features) -> (190, 11, 178)

    df_train_agg = get_df_w_aggrs(df=df_train, feats=all_features)
    df_train_target = get_target(TARGET_PATH='./data/train_labels.csv')
    df_train = get_train_data_with_target_merged(df_train=df_train_agg, df_train_target=df_train_target)
    
    '''
    df_train.target.value_counts()
    target
    0    340085
    1    118828
    Name: count, dtype: int64
    '''

    df_test = get_test_data(TEST_PATH='./data/test.parquet')
    df_test = get_df_w_aggrs(df=df_test, feats=all_features)

    # zapolnenie_train = check_zapolnenie(df_train)
    # zapolnenie_test = check_zapolnenie(df_test)


    return 0



