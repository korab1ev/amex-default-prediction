
import pandas as pd
import numpy as np
import gc

def get_train_data(TRAIN_PATH: str) -> pd.DataFrame:
    '''Returns train dataset'''
    df_train = pd.read_parquet(TRAIN_PATH)
    return df_train


def get_test_data(TEST_PATH: str) -> pd.DataFrame:
    '''Returns test dataset'''
    df_test = pd.read_parquet(TEST_PATH)
    return df_test


def get_target(TARGET_PATH) -> pd.DataFrame:
    '''Retruns dataset with train targets'''
    df_train_target = pd.read_csv(TARGET_PATH)
    return df_train_target


def get_train_data_with_target_merged(df_train, df_train_target) -> pd.DataFrame:
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


def generate_agg_feats(df) ->  pd.DataFrame:

    df_aggrs = (

    )

    '''
    features_avg = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_50', 'D_51', 'D_53', 'D_54', 'D_55', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_65', 'D_66', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_75', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_86', 'D_91', 'D_92', 'D_94', 'D_96', 'D_103', 'D_104', 'D_108', 'D_112', 'D_113', 'D_114', 'D_115', 'D_117', 'D_118', 'D_119', 'D_120', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_129', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_14', 'R_15', 'R_16', 'R_17', 'R_20', 'R_21', 'R_22', 'R_24', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_9', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_18', 'S_22', 'S_23', 'S_25', 'S_26']
    features_min = ['B_2', 'B_4', 'B_5', 'B_9', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_19', 'B_20', 'B_28', 'B_29', 'B_33', 'B_36', 'B_42', 'D_39', 'D_41', 'D_42', 'D_45', 'D_46', 'D_48', 'D_50', 'D_51', 'D_53', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_62', 'D_70', 'D_71', 'D_74', 'D_75', 'D_78', 'D_83', 'D_102', 'D_112', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_128', 'D_132', 'D_140', 'D_141', 'D_144', 'D_145', 'P_2', 'P_3', 'R_1', 'R_27', 'S_3', 'S_5', 'S_7', 'S_9', 'S_11', 'S_12', 'S_23', 'S_25']
    features_max = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_21', 'B_23', 'B_24', 'B_25', 'B_29', 'B_30', 'B_33', 'B_37', 'B_38', 'B_39', 'B_40', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_52', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_63', 'D_64', 'D_65', 'D_70', 'D_71', 'D_72', 'D_73', 'D_74', 'D_76', 'D_77', 'D_78', 'D_80', 'D_82', 'D_84', 'D_91', 'D_102', 'D_105', 'D_107', 'D_110', 'D_111', 'D_112', 'D_115', 'D_116', 'D_117', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_126', 'D_128', 'D_131', 'D_132', 'D_133', 'D_134', 'D_135', 'D_136', 'D_138', 'D_140', 'D_141', 'D_142', 'D_144', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_3', 'R_5', 'R_6', 'R_7', 'R_8', 'R_10', 'R_11', 'R_14', 'R_17', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_7', 'S_8', 'S_11', 'S_12', 'S_13', 'S_15', 'S_16', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']
    features_last = ['B_1', 'B_2', 'B_3', 'B_4', 'B_5', 'B_6', 'B_7', 'B_8', 'B_9', 'B_10', 'B_11', 'B_12', 'B_13', 'B_14', 'B_15', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'B_21', 'B_22', 'B_23', 'B_24', 'B_25', 'B_26', 'B_28', 'B_29', 'B_30', 'B_32', 'B_33', 'B_36', 'B_37', 'B_38', 'B_39', 'B_40', 'B_41', 'B_42', 'D_39', 'D_41', 'D_42', 'D_43', 'D_44', 'D_45', 'D_46', 'D_47', 'D_48', 'D_49', 'D_50', 'D_51', 'D_52', 'D_53', 'D_54', 'D_55', 'D_56', 'D_58', 'D_59', 'D_60', 'D_61', 'D_62', 'D_63', 'D_64', 'D_65', 'D_69', 'D_70', 'D_71', 'D_72', 'D_73', 'D_75', 'D_76', 'D_77', 'D_78', 'D_79', 'D_80', 'D_81', 'D_82', 'D_83', 'D_86', 'D_91', 'D_96', 'D_105', 'D_106', 'D_112', 'D_114', 'D_119', 'D_120', 'D_121', 'D_122', 'D_124', 'D_125', 'D_126', 'D_127', 'D_130', 'D_131', 'D_132', 'D_133', 'D_134', 'D_138', 'D_140', 'D_141', 'D_142', 'D_145', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7', 'R_8', 'R_9', 'R_10', 'R_11', 'R_12', 'R_13', 'R_14', 'R_15', 'R_19', 'R_20', 'R_26', 'R_27', 'S_3', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_11', 'S_12', 'S_13', 'S_16', 'S_19', 'S_20', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'S_27']

    for i in ['test', 'train'] if INFERENCE else ['train']:
        df = pd.read_parquet(f'../input/amex-data-integer-dtypes-parquet-format/{i}.parquet')
        cid = pd.Categorical(df.pop('customer_ID'), ordered=True)
        last = (cid != np.roll(cid, -1)) # mask for last statement of every customer
        if 'target' in df.columns:
            df.drop(columns=['target'], inplace=True)
        gc.collect()
        print('Read', i)
        df_avg = (df
                .groupby(cid)
                .mean()[features_avg]
                .rename(columns={f: f"{f}_avg" for f in features_avg})
                )
        gc.collect()
        print('Computed avg', i)
        df_min = (df
                .groupby(cid)
                .min()[features_min]
                .rename(columns={f: f"{f}_min" for f in features_min})
                )
        gc.collect()
        print('Computed min', i)
        df_max = (df
                .groupby(cid)
                .max()[features_max]
                .rename(columns={f: f"{f}_max" for f in features_max})
                )
        gc.collect()
        print('Computed max', i)
        df = (df.loc[last, features_last]
            .rename(columns={f: f"{f}_last" for f in features_last})
            .set_index(np.asarray(cid[last]))
            )
        gc.collect()
        print('Computed last', i)
        df = pd.concat([df, df_min, df_max, df_avg], axis=1)
        if i == 'train': train = df
        else: test = df
        print(f"{i} shape: {df.shape}")
        del df, df_avg, df_min, df_max, cid, last

    target = pd.read_csv('../input/amex-default-prediction/train_labels.csv').target.values
    print(f"target shape: {target.shape}")
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

    df_train = get_train_data(TRAIN_PATH=f'../input/amex-data-integer-dtypes-parquet-format/train.parquet')
    df_train_target = get_target(TARGET_PATH='/kaggle/input/train-labels-amex/train_labels.csv')
    df_train_w_target = get_train_data_with_target_merged(df_train=df_train, df_train_target=df_train_target)

    df_test = get_test_data(TEST_PATH=f'../input/amex-data-integer-dtypes-parquet-format/test.parquet')


    df_train_agg = generate_agg_feats(df=df_train)


    # zapolnenie_train = check_zapolnenie(df_train)
    # zapolnenie_test = check_zapolnenie(df_test)


    return 0



