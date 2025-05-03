# Data
import pandas as pd
import numpy as np
import gc
from typing import List
from tqdm import tqdm
from utils.get_data import *
from utils.get_not_null_share_stats import get_not_null_share_df


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

# Visualisation
import seaborn as sns
import shap

# Warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)