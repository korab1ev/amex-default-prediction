import numpy as np, pandas as pd, optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats
from utils.get_amex_metric import get_amex_metric_calculated, lgb_amex_metric

# ---------- backward selection w/ Optuna every 5th step ---------------------
def run_backward_selection(df_train,
                           df_test,
                           target_col,
                           num_feats,
                           cat_feats,
                           group_col,
                           min_feats: int = 10,
                           random_state: int = 42):

    num, cat = num_feats.copy(), cat_feats.copy()
    history, list_imp_dfs, step = [], [], 0
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=random_state)
    test_preds_df = pd.DataFrame(index=df_test.customer_ID)

    y, groups = df_train[target_col].values, df_train[group_col].values

    # --- start with baseline params
    tuned_max_depth, tuned_estimators = 6, 400

    def n_to_drop(n):
        if n > 250: return max(1, int(n*.10))
        if n > 50: return max(1, int(n*.05))
        if n >  25: return 2
        return 1

    while True:
        feats = num + cat
        if len(feats) <= min_feats: break

        # -------- Optuna every 2, 7, 12, ... iteration -------------------------
        if step % 5 == 2:
            def objective(trial):
                md = trial.suggest_int('max_depth', 2, 8)
                nt = trial.suggest_int('n_estimators', 200, 500)
                params = dict(
                    learning_rate=.05, subsample=.9, subsample_freq=1,
                    colsample_bytree=.8, objective='binary',
                    random_state=random_state,
                    max_depth=md, n_estimators=nt,
                    n_jobs=-1, verbosity=-1
                )
                cv_scores=[]
                for tr_idx, va_idx in sgkf.split(
                        df_train[[group_col, target_col]], y, groups):
                    model = LGBMClassifier(**params)
                    model.fit(df_train.iloc[tr_idx][feats], y[tr_idx],
                              categorical_feature=cat)
                    pred = model.predict_proba(df_train.iloc[va_idx][feats])[:,1]
                    cv_scores.append(get_amex_metric_calculated(y[va_idx], pred))
                return -np.mean(cv_scores)
            study = optuna.create_study(direction='minimize',
                                         sampler=optuna.samplers.TPESampler(seed=random_state))
            study.optimize(objective, n_trials=15, n_jobs=-1, show_progress_bar=True) #  15
            tuned_max_depth  = study.best_params['max_depth']
            tuned_estimators = study.best_params['n_estimators']

        params = dict(
            learning_rate=.05, subsample=.9, subsample_freq=1,
            colsample_bytree=.8, objective='binary',
            random_state=random_state,
            max_depth=tuned_max_depth, n_estimators=tuned_estimators,
            n_jobs=-1, verbosity=-1
        )

        # -------- CV and logging ------------------------------------
        fold_scores, fold_test_raw = [], []
        print(f"\n=== Step {step} | feats {len(feats)} "
              f"(num {len(num)}, cat {len(cat)}) | depth {tuned_max_depth} trees {tuned_estimators}")
        for fold, (tr_idx, va_idx) in enumerate(
                sgkf.split(df_train[[group_col, target_col]], y, groups)):
            print(f"Fold {fold}")
            X_tr, X_va = df_train.iloc[tr_idx][feats], df_train.iloc[va_idx][feats]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      eval_metric=lgb_amex_metric,
                      categorical_feature=cat)
            pred_va = model.predict_proba(X_va)[:,1]
            sc = get_amex_metric_calculated(y_va, pred_va)
            print(f" Amex = {sc:.5f}")
            fold_scores.append(sc)
            fold_test_raw.append(model.predict_proba(df_test[feats], raw_score=True))
            # fold_test_raw.append(model.predict_proba(df_test[feats])[:,1]) # опять тут pd, а не скоры. глаз да глаз за этой темой

        mean_cv = float(np.mean(fold_scores))

        ci = stats.t.interval(.95, len(fold_scores)-1,
                              loc=mean_cv, scale=stats.sem(fold_scores))
        print(f"Mean CV Amex = {mean_cv:.5f}  CI95 = [{ci[0]:.5f}, {ci[1]:.5f}]")
        # print(f"Mean CV Amex: {mean_cv:.5f} ± {np.std(fold_scores):.5f}")
        test_preds_df[f'iter_{step}'] = np.mean(fold_test_raw, axis=0)

        # -------- importance & dropping ------------------------------
        model_full = LGBMClassifier(**params).fit(
            df_train[feats], df_train[target_col], categorical_feature=cat)
        
        gains = model_full.booster_.feature_importance('gain')
        imp_df = (pd.DataFrame({'feature': feats, 'gain': gains})
                    .sort_values('gain', ascending=False))
        imp_df['share'] = imp_df['gain'] / imp_df['gain'].sum()
        list_imp_dfs.append({'step': step, 'importance': imp_df})

        drop_list = (imp_df.query('gain > 0')
                           .sort_values('gain') # по дефолту оно ascending=True, поэтому тут сверху будут с маленькими imp_gain
                           .head(n_to_drop(len(feats)))['feature']
                           .tolist())
                          
        num = [f for f in num if f not in drop_list]
        cat = [f for f in cat if f not in drop_list]

        history.append({'step': step,
                        'n_feats_left': len(feats),
                        'n_num_left': len(num),
                        'n_cat_left': len(cat),
                        'max_depth': tuned_max_depth,
                        'n_estimators': tuned_estimators,
                        'cv_mean': mean_cv,
                        'cv_ci_lo': ci[0], 
                        'cv_ci_hi': ci[1]})
        step += 1

    return (pd.DataFrame(history), list_imp_dfs,
            {'num': num, 'cat': cat}, test_preds_df)
