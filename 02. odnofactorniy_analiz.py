
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation


def get_amex_metric_calculated(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)



'''
def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)
'''


def main() -> int:
    
    '''
    df_train = get_train_data(TRAIN_PATH='./data/train.parquet')

    all_features = get_all_features(df_train)
    cat_features = get_cat_features()
    num_features = get_num_features(all_features, cat_features)
    # len(all_features), len(cat_features), len(num_features) -> (190, 11, 178)

    df_train_agg = get_df_w_aggrs(df=df_train, numerical_features=num_features)
    df_train_target = get_target(TARGET_PATH='./data/train_labels.csv')
    df_train = get_train_data_with_target_merged(df_train=df_train_agg, df_train_target=df_train_target)

    df_test = get_test_data(TEST_PATH='./data/test.parquet')
    df_test = get_df_w_aggrs(df=df_test, numerical_features=num_features)

    # zapolnenie_train = check_zapolnenie(df_train)
    # zapolnenie_test = check_zapolnenie(df_test)
    '''

    # print(get_amex_metric_calculated(y_true=df_train.target, y_pred=df_train.target)) # 1.0000000000015785

    clf = LGBMClassifier()

    '''
    n_estimators=n_estimators,
                          learning_rate=0.03, reg_lambda=50,
                          min_child_samples=2400,
                          num_leaves=95,
                          colsample_bytree=0.19,
                          max_bins=511, random_state=random_state
    '''

    # clf.fit(X_tr, y_tr, eval_set = [(X_va, y_va)], eval_metric=[lgb_amex_metric], callbacks=[log_evaluation(100)])

    return 0