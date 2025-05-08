# Data
import pandas as pd
import numpy as np
import gc
from typing import List
from tqdm import tqdm
from utils.get_data import *
from utils.get_not_null_share_stats import get_not_null_share_df
from utils.get_psi_calculated import calc_psi
from utils.get_plot_distr import plot_distr
from utils.get_pair_correlations import plot_group_corr_pair
from utils.get_cat_feats_histograms import plot_cat_hist
from utils.get_amex_metric import get_amex_metric_calculated, lgb_amex_metric
from utils.run_backward_selection import run_backward_selection

# Models
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# PCA
from sklearn.decomposition import PCA

# KFold
from sklearn.model_selection import StratifiedGroupKFold

# Preprocessing
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.impute import SimpleImputer

# Logistic regression
from optbinning import BinningProcess, OptimalBinning
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# PyTorch
import torch

# Visualisation
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import shap

# Hyperparameter tuning
import optuna

# Statistics
from scipy import stats

# Warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# Save files
import pickle