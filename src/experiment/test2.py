import sys
from pathlib import Path
import logging
from data.peak_hour_analyze import peak_hour_frequency
import matplotlib.pyplot as plt
from algorithms.TIDAL import TIDAL
from data.loader import DiskDataLoader
from config.settings import DataConfig, DirConfig, WarehouseConfig, TestConfig
from visualization.plotter import TelaPlotter
from catboost import CatBoostRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
    r2_score
)
from scipy.stats import loguniform, randint, uniform
import contextlib
import joblib
from tqdm import tqdm
from datetime import datetime
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, RandomizedSearchCV
import pickle
import numpy as np
from cluster_info_init.cluster_info_initializer import ClusterInfoInitializer
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,File "%(pathname)s", line %(lineno)d, %(message)s',
    handlers=[
        logging.FileHandler('test2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# CatBoost Regression best params: {'depth': 10, 'iterations': 2000, 'l2_leaf_reg': 5, 'learning_rate': 0.1} Best RMSE: 1.5086317804084781


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    这是一个上下文管理器，用于给 joblib 的并行任务打补丁，
    使其能够通过 tqdm 显示进度条。
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def plot_prediction_comparison(real_data, predicted_data, model_name, axis):
    df = pd.DataFrame(
        {"real_data": real_data, "predicted_data": predicted_data})
    df = df.sort_values(by="real_data", ascending=True)
    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(df)), df["real_data"],
                color="red", s=1, alpha=0.3)
    plt.scatter(range(len(df)), df["predicted_data"],
                color="blue", s=1, alpha=0.3)
    plt.title(f"{model_name} Prediction Comparison")
    plt.grid(True, linestyle='--', alpha=0.5)
    time_str = datetime.now().strftime("%d%H%M")
    save_dir = os.path.join(DirConfig.TEMP_DIR, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f'{model_name}_{axis}_prediction_comparison_{time_str}.png'))
    logger.info(
        f"{model_name}_{axis}_prediction_comparison.png saved to {os.path.join(save_dir, f'{model_name}_{axis}_prediction_comparison_{time_str}.png')}")
    plt.close()


def search_catboost_params(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP",
                     "disk_type", "volume_type", "business_type"]
    X_train = train_items[features + cat_type_list].copy()
    for field in cat_type_list:
        X_train[field] = X_train[field].astype(str)
    y_train = train_items["avg_bandwidth"]
    X_test = test_items[features + cat_type_list].copy()
    for field in cat_type_list:
        X_test[field] = X_test[field].astype(str)
    y_test = test_items["avg_bandwidth"]
    params_distributions = {
        'learning_rate': loguniform(0.01, 0.2),
        'depth': randint(4, 15),
        'l2_leaf_reg': randint(1, 12),
        'loss_function': ['RMSE', 'MAE', 'Huber:delta=1.0'],
        'iterations': randint(1000, 10000),
        'one_hot_max_size': [2, 10, 20, 50],
        'random_strength': [1, 2, 5, 10],
        'subsample': uniform(0.6, 0.35),
        'bootstrap_type': ['Bernoulli', 'MVS']
    }
    cb = CatBoostRegressor(
        cat_features=cat_type_list,
        verbose=0,
        thread_count=6,
        eval_metric='MAE'
    )
    model_regressor = RandomizedSearchCV(
        estimator=cb,
        param_distributions=params_distributions,
        n_iter=1000,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=12,
        verbose=1,
        random_state=42
    )
    model_regressor.fit(X_train, y_train)
    best_catboost = model_regressor.best_estimator_
    feature_names = X_train.columns.tolist()
    importances = best_catboost.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    logger.info(feature_imp_df)
    logger.info(f"best_params: {model_regressor.best_params_}")
    logger.info(f"best_score: {model_regressor.best_score_:.6f}")
    best_params = model_regressor.best_params_
    best_catboost = CatBoostRegressor(
        cat_features=cat_type_list,
        verbose=0,
        thread_count=80,
        **best_params,
        eval_metric='MAE'
    )
    best_catboost.fit(X_train, y_train, eval_set=(X_test, y_test))
    results = best_catboost.evals_result_
    y_pred = best_catboost.predict(X_test)
    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    return y_test, y_pred, results


def train_catboost(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP",
                     "disk_type", "volume_type", "business_type"]
    X_train = train_items[features + cat_type_list].copy()
    for field in cat_type_list:
        X_train[field] = X_train[field].astype(str)
    y_train = train_items["avg_bandwidth"]
    X_test = test_items[features + cat_type_list].copy()
    for field in cat_type_list:
        X_test[field] = X_test[field].astype(str)
    y_test = test_items["avg_bandwidth"]
    model_regressor = CatBoostRegressor(
        cat_features=cat_type_list,
        verbose=0,
        thread_count=100,
        depth=10,
        iterations=10000,
        learning_rate=0.05,
        loss_function='RMSE',
        eval_metric='RMSE',
        one_hot_max_size=10,
        bagging_temperature=0.5391612766752545,
        # early_stopping_rounds=50,
        subsample=0.95,
        l2_leaf_reg=5
    )
    model_regressor.fit(X_train, y_train, eval_set=(X_test, y_test))
    logger.info(model_regressor.get_feature_importance(prettified=True))
    y_pred = model_regressor.predict(X_test)
    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    results = model_regressor.evals_result_
    return y_test, y_pred, results


def train_lightgbm(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP",
                     "disk_type", "volume_type", "business_type"]
    cat_dtypes_dict = {}
    X_train = train_items[features + cat_type_list].copy()
    for field in cat_type_list:
        X_train[field] = X_train[field].astype("category")
        cat_dtypes_dict[field] = X_train[field].dtype
    y_train = train_items["avg_bandwidth"]
    X_test = test_items[features + cat_type_list].copy()
    y_test = test_items["avg_bandwidth"]
    for field in cat_type_list:
        if field in cat_dtypes_dict:
            target_type = cat_dtypes_dict[field]
            X_test[field] = X_test[field].astype(target_type)
        else:
            X_test[field] = X_test[field].astype("category")
    params_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                   'num_leaves': [31, 63, 95],
                   'max_depth': [-1, 8, 10, 15, 20],
                   'min_child_samples': [20, 50, 80, 100],
                   'subsample': [0.6, 0.7, 0.8, 0.9],
                   'objective': ['regression', 'regression_l1', 'huber', 'tweedie', 'poisson']}
    lgbm = lgb.LGBMRegressor(
        n_estimators=3000,
        metric='rmse',
        random_state=42,
        n_jobs=2,
        verbose=-1
    )
    model_regressor = GridSearchCV(estimator=lgbm,
                                   param_grid=params_grid,
                                   cv=5,
                                   scoring='neg_root_mean_squared_error',
                                   n_jobs=48,
                                   verbose=1
                                   )
    model_regressor.fit(X_train, y_train)
    best_lgbm = model_regressor.best_estimator_
    feature_names = X_train.columns.tolist()
    importances = best_lgbm.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    logger.info(feature_imp_df)
    logger.info(f"best_params: {model_regressor.best_params_}")
    logger.info(f"best_score: {model_regressor.best_score_:.6f}")

    # GridSearchCV 训练的模型没有 evals_result_，需要用最佳参数重新训练以获取训练历史
    best_params = model_regressor.best_params_
    best_lgbm = lgb.LGBMRegressor(
        metric='rmse',
        random_state=42,
        n_jobs=24,
        verbose=-1,
        **best_params
    )
    best_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        callbacks=[lgb.early_stopping(
            stopping_rounds=50), lgb.log_evaluation(period=200)]
    )
    results = best_lgbm.evals_result_
    y_pred = best_lgbm.predict(X_test)
    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    return y_test, y_pred, results


def search_lightgbm_params(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP",
                     "disk_type", "volume_type", "business_type"]
    cat_dtypes_dict = {}
    X_train = train_items[features + cat_type_list].copy()

    for field in cat_type_list:
        X_train[field] = X_train[field].astype("category")
        cat_dtypes_dict[field] = X_train[field].dtype
    y_train = (train_items["avg_bandwidth"])
    X_test = test_items[features + cat_type_list].copy()
    y_test = (test_items["avg_bandwidth"])
    for field in cat_type_list:
        if field in cat_dtypes_dict:
            target_type = cat_dtypes_dict[field]
            X_test[field] = X_test[field].astype(target_type)
        else:
            X_test[field] = X_test[field].astype("category")
    params_distributions = {
        'learning_rate': loguniform(0.001, 0.2),
        'num_leaves': randint(20, 150),
        'max_depth': randint(-1, 20),
        'min_child_samples': randint(20, 200),
        'subsample': uniform(0.6,  0.35),
        'subsample_freq': randint(1, 10),
        'objective': ['regression', 'regression_l1', 'huber'],
        'n_estimators': randint(100, 3000)
    }
    lgbm = lgb.LGBMRegressor(
        metric='mae', random_state=42, n_jobs=6, verbose=-1)
    model_regressor = RandomizedSearchCV(estimator=lgbm, param_distributions=params_distributions,
                                         n_iter=1000, cv=5, scoring='neg_mean_absolute_error', n_jobs=12, verbose=1)
    model_regressor.fit(X_train, y_train)
    best_lgbm = model_regressor.best_estimator_
    feature_names = X_train.columns.tolist()
    importances = best_lgbm.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    logger.info(feature_imp_df)
    logger.info(f"best_params: {model_regressor.best_params_}")
    logger.info(f"best_score: {model_regressor.best_score_:.6f}")
    best_params = model_regressor.best_params_.copy()
    best_lgbm = lgb.LGBMRegressor(
        metric='mae', random_state=4, n_jobs=80, verbose=-1, **best_params)
    best_lgbm.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        callbacks=[lgb.early_stopping(
            stopping_rounds=50), lgb.log_evaluation(period=200)],
        categorical_feature=cat_type_list
    )
    results = best_lgbm.evals_result_
    y_pred = best_lgbm.predict(X_test)
    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    return y_test, y_pred, results


def search_xgboost_params(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP",
                     "disk_type", "volume_type", "business_type"]
    cat_dtypes_dict = {}
    X_train = train_items[features + cat_type_list].copy()

    for field in cat_type_list:
        X_train[field] = X_train[field].astype("category")
        cat_dtypes_dict[field] = X_train[field].dtype
    y_train = (train_items["avg_bandwidth"])
    X_test = test_items[features + cat_type_list].copy()
    y_test = (test_items["avg_bandwidth"])
    for field in cat_type_list:
        if field in cat_dtypes_dict:
            target_type = cat_dtypes_dict[field]
            X_test[field] = X_test[field].astype(target_type)
        else:
            X_test[field] = X_test[field].astype("category")
    params_distributions = {
        'learning_rate': loguniform(0.001, 0.2),
        'max_depth': randint(3, 15),
        'min_child_weight': randint(1, 100),
        'subsample': uniform(0.6, 0.35),
        'colsample_bytree': uniform(0.6, 0.35),
        'objective': ['reg:squarederror', 'reg:absoluteerror', 'reg:pseudohubererror'],
        'reg_alpha': loguniform(1e-3, 10.0),
        'reg_lambda': loguniform(1e-3, 10.0),
        'n_estimators': randint(100, 3000)
    }
    xgb_reg = XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        n_jobs=6,
        random_state=42,
        eval_metric='mae'  # 默认监控指标
    )
    model_regressor = RandomizedSearchCV(estimator=xgb_reg, param_distributions=params_distributions,
                                         n_iter=1000, cv=5, scoring='neg_mean_absolute_error', n_jobs=12, verbose=1)
    model_regressor.fit(X_train, y_train)
    best_xgb_search = model_regressor.best_estimator_
    feature_names = X_train.columns.tolist()
    importances = best_xgb_search.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    logger.info(feature_imp_df)
    logger.info(f"best_params: {model_regressor.best_params_}")
    logger.info(f"best_score: {model_regressor.best_score_:.6f}")

    best_params = model_regressor.best_params_
    best_xgb = XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        n_jobs=80,
        random_state=42,
        eval_metric='mae',
        early_stopping_rounds=50,
        **best_params,
    )

    best_xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=200
    )

    results = best_xgb.evals_result()
    y_pred = best_xgb.predict(X_test)
    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    return y_test, y_pred, results


def search_random_forest_params(train_items, test_items, features):
    cat_type_list = ["disk_if_VIP", "disk_type",
                     "volume_type", "business_type"]
    X_train = train_items[features + cat_type_list].copy()
    X_test = test_items[features + cat_type_list].copy()

    y_train = train_items["avg_bandwidth"]
    y_test = test_items["avg_bandwidth"]

    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)

    X_train[cat_type_list] = encoder.fit_transform(X_train[cat_type_list])
    X_test[cat_type_list] = encoder.transform(X_test[cat_type_list])

    params_distributions = {
        'n_estimators': randint(100, 500),      # 树的数量
        'max_depth': [None, 10, 20, 30, 50],    # 树的深度
        'min_samples_split': randint(2, 20),    # 裂变所需的最小样本数
        'min_samples_leaf': randint(1, 10),     # 叶子节点最小样本数
        'max_features': ['sqrt', 'log2', 1.0],  # 每次裂变考虑的特征数
        'bootstrap': [True, False]              # 是否有放回抽样
    }

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=6,
        verbose=0
    )

    model_regressor = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params_distributions,
        n_iter=1000,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=12,
        verbose=1,
        random_state=42
    )

    model_regressor.fit(X_train, y_train)

    best_rf = model_regressor.best_estimator_

    feature_names = X_train.columns.tolist()
    importances = best_rf.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    logger.info("RandomForest Feature Importances:")
    logger.info(feature_imp_df)
    logger.info(f"best_params: {model_regressor.best_params_}")
    logger.info(f"best_score: {model_regressor.best_score_:.6f}")

    y_pred = best_rf.predict(X_test)

    logger.info(f"y_pred_min: {np.min(y_pred)}")
    y_pred = np.maximum(y_pred, 0)
    results = {}
    return y_test, y_pred, results


def evaluate_models(y_test, y_pred, model_name):
    logger.info(f"Evaluating {model_name} model")
    logger.info(
        f"RMSE : {root_mean_squared_error(y_test, y_pred):.6f}")
    logger.info(f"MAE : {mean_absolute_error(y_test, y_pred):.6f}")
    logger.info(f"Max Error : {max_error(y_test, y_pred):.6f}")
    logger.info(f"R² Score : {r2_score(y_test, y_pred):.6f}")
    logger.info(
        f"log R² Score : {r2_score(np.log1p(y_test), np.log1p(y_pred)):.6f}")
    logger.info(f"mean bandwidth : {np.mean(y_test):.6f}")
    plot_prediction_comparison(y_test, y_pred, model_name, "normal")
    plot_prediction_comparison(
        np.log1p(y_test), np.log1p(y_pred), model_name, "log")


def plot_loss_curve(results, model_name):
    """
    绘制训练损失曲线
    支持 CatBoost 和 LightGBM 的结果结构
    """
    if results is None or len(results) == 0:
        logger.warning(
            f"{model_name}: No evaluation results available for plotting")
        return

    # 检查可用的键
    available_keys = list(results.keys())
    logger.info(
        f"{model_name}: Available keys in evals_result_: {available_keys}")

    # 尝试不同的键名组合（CatBoost vs LightGBM）
    train_key = None
    val_key = None

    # CatBoost 使用 'learn' 和 'validation'
    if 'learn' in results:
        train_key = 'learn'
    elif 'training' in results:
        train_key = 'training'
    elif 'train' in results:
        train_key = 'train'

    if 'validation' in results:
        val_key = 'validation'
    elif 'valid_1' in results:
        val_key = 'valid_1'
    elif 'valid_0' in results:
        val_key = 'valid_0'
    elif 'valid' in results:
        val_key = 'valid'

    train_rmse = None
    val_rmse = None

    if train_key:
        train_metrics = results[train_key]
        train_rmse = train_metrics.get('RMSE', train_metrics.get('rmse', []))
        epochs = len(train_rmse) if train_rmse else 0
    elif val_key:
        val_metrics = results[val_key]
        val_rmse = val_metrics.get('RMSE', val_metrics.get('rmse', []))
        epochs = len(val_rmse) if val_rmse else 0
    else:
        logger.warning(
            f"{model_name}: No valid train or validation key found in results")
        return

    if epochs == 0:
        logger.warning(f"{model_name}: No data points found in results")
        return

    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))

    if train_rmse and len(train_rmse) == epochs:
        plt.plot(x_axis, train_rmse, label='Train RMSE')

    if val_key:
        val_metrics = results[val_key]
        val_rmse = val_metrics.get('RMSE', val_metrics.get('rmse', []))
        if val_rmse and len(val_rmse) == epochs:
            plt.plot(x_axis, val_rmse, label='Validation RMSE')
        elif val_rmse:
            logger.warning(
                f"{model_name}: Validation RMSE length ({len(val_rmse)}) doesn't match epochs ({epochs})")

    plt.legend()
    plt.ylabel('RMSE (Bandwidth Mbps)')
    plt.xlabel('Iterations')
    plt.title(f'{model_name} Training Loss Curve')
    plt.grid(True)
    time_str = datetime.now().strftime("%d%H%M")
    save_dir = os.path.join(DirConfig.TEMP_DIR, f'{model_name}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f'{model_name}_loss_curve_{time_str}.png'))
    logger.info(
        f"{model_name}_loss_curve.png saved to {os.path.join(save_dir, f'{model_name}_loss_curve_{time_str}.png')}")
    plt.close()


def train_models(log, history, model_name, history_transform=False):
    cluster_index_list = DataConfig.CLUSTER_DIR_LIST
    logger.info(f"cluster_index_list: {cluster_index_list}")
    loader = DiskDataLoader()
    items = loader.load_items(cluster_index_list)
    items["cpu_muti_memory"] = items["vm_cpu"] * items["vm_memory"]
    if history_transform:
        items["has_history_bandwidth"] = (
            ~items["recent_history_bandwidth_memory"].isna()).astype(int)
        avg_bandwidth_mean = items["avg_bandwidth"].mean()
        items["recent_history_bandwidth_memory"] = items["recent_history_bandwidth_memory"].fillna(
            avg_bandwidth_mean)
    if log:
        items["avg_bandwidth"] = np.log1p(items["avg_bandwidth"])
    train_items, test_items = train_test_split(
        items, test_size=0.2, random_state=42)
    if history:
        if history_transform:
            features = ["disk_capacity", "vm_cpu", "vm_memory",
                        "has_history_bandwidth", "recent_history_bandwidth_memory"]
        else:
            features = ["disk_capacity", "vm_cpu",
                        "vm_memory", "recent_history_bandwidth_memory"]
    else:
        features = ["disk_capacity", "vm_cpu", "vm_memory"]
    if model_name == "CatBoost":
        y_test, y_pred, results = search_catboost_params(
            train_items, test_items, features)
    elif model_name == "LightGBM":
        y_test, y_pred, results = search_lightgbm_params(
            train_items, test_items, features)
    elif model_name == "XGBoost":
        y_test, y_pred, results = search_xgboost_params(
            train_items, test_items, features)
    elif model_name == "RandomForest":
        y_test, y_pred, results = search_random_forest_params(
            train_items, test_items, features)
    if log:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
    evaluate_models(y_test, y_pred, model_name)
    plot_loss_curve(results, model_name)


if __name__ == "__main__":
    for log in [False, True]:
        for history in [False, True]:
            for model_name in ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]:
                logger.info(
                    f"--------log: {log}, history: {history}, model_name: {model_name}--------")
                train_models(log, history,  model_name)
                logger.info(
                    "-----------------------------\n")
