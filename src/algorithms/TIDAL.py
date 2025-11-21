from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import os
import pickle
from transformers import pipeline
from .base_algorithm import BaseAlgorithm
from config.settings import DataConfig, ModelConfig, DirConfig
from data.utils import get_circular_trace as get_circular_trace_util
from data.utils import iterate_first_day as iterate_first_day_util
from visualization.plotter import TelaPlotter
import joblib
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)


class TIDAL(BaseAlgorithm):

    def __init__(self, model_name: str):
        super().__init__("TIDAL")
        self.model_name = model_name
        self.business_type_list = DataConfig.BUSINESS_TYPE_LIST
        self.scaler = None
        self.warehouse_trace_vector = np.zeros(
            (int(24/DataConfig.TRACE_VECTOR_INTERVAL), self.warehouse_number))
        self.cat_dtypes_dict = {}
        self.cat_type_list = ["disk_if_VIP", "ins_type",
                              "disk_type", "volume_type", "business_type"]

    def load_and_preprocess_items(self):
        predicted_items = self.predict(self.test_items.copy())
        return predicted_items

    def select_warehouse(self, item: pd.Series) -> int:
        disk_pre_trace_vector = item["pre_trace_vector"]
        overload_mask = self.check_warehouse_overload_after_placement(item)
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+item["disk_capacity"] <=
                         self.warehouses_max[:, 0])
        after_placed_bandwidth = self.warehouse_trace_vector + \
            disk_pre_trace_vector[:, np.newaxis]
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]
        original_bandwidth_util = self.warehouse_trace_vector / \
            self.warehouses_max[:, 1]
        trace_vector_mask = (after_placed_bandwidth_util <= 1).all(axis=0)
        while True:
            monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
            combined_mask = capacity_mask & monitor_mask
            eligible_warehouses_indices = np.where(combined_mask)[0]
            if len(eligible_warehouses_indices) == 0:
                return -1
            original_bandwidth_util_deviation = np.var(
                original_bandwidth_util[:, eligible_warehouses_indices], axis=0)
            absolute_deviation = np.var(
                after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
            delta_deviation = absolute_deviation-original_bandwidth_util_deviation
            min_delta_deviation_index = np.argmin(delta_deviation)
            selected_warehouse = eligible_warehouses_indices[min_delta_deviation_index]
            if not overload_mask[selected_warehouse]:
                self.warehouses_cannot_use_by_monitor[selected_warehouse] = 1
                continue
            else:
                break
        return selected_warehouse

    def train_models(self, model_name: str = "catboost"):
        items = self.train_items.copy()
        grouped_vectors = self.make_business_type_vector(items)
        grouped_vectors.to_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, 'business_type_vector.csv'))
        plotter = TelaPlotter()
        plotter.plot_business_type_vector(grouped_vectors, os.path.join(
            DirConfig.BUSINESS_TYPE_DIR))
        if model_name == "catboost":
            model_regressor = self.train_catboost(items)
        elif model_name == "lightgbm":
            model_regressor = self.train_lightgbm(items)
        elif model_name == "xgboost":
            model_regressor = self.train_xgboost(items)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        os.makedirs(os.path.join(DirConfig.MODEL_DIR,
                    self.algorithm_name), exist_ok=True)
        model_regressor_dir = os.path.join(
            DirConfig.MODEL_DIR, self.algorithm_name, f"{model_name}_regressor.pkl")
        with open(model_regressor_dir, "wb") as f:
            pickle.dump(model_regressor, f)
        with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{model_name}_cat_dtypes_dict.pkl"), "wb") as f:
            pickle.dump(self.cat_dtypes_dict, f)

    def predict(self, items: pd.DataFrame):
        # if self.model_name == "distill_bert":
        # predict_business_type = self.predict_distill_bert(items)
        # else:
        #     raise ValueError(f"不支持的模型: {self.model_name}")
        predict_business_type = items['business_type']
        items['pre_business_type'] = predict_business_type
        return self.predict_disk_trace_vector(items)

    def predict_distill_bert(self, items: pd.DataFrame):
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, "distill_bert_model")
        text_classifier = pipeline(
            "text-classification", tokenizer=model_path, model=model_path, device=0)
        predictions = text_classifier(items['description'])
        return predictions

    def predict_disk_trace_vector(self, items: pd.DataFrame):
        trace_vector = pd.read_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, 'business_type_vector.csv'), index_col=0)
        bandwidth_scale = self.predict_catboost(items)
        pre_trace_vector = bandwidth_scale[:, np.newaxis] * \
            trace_vector.loc[items['pre_business_type']].values
        items['pre_trace_vector'] = [pre_trace_vector[i]
                                     for i in range(len(items))]
        return items

    def predict_lightgbm(self, items: pd.DataFrame):
        model_regressor = self.load_model(
            model_name="lightgbm", train_models=True)
        data_predict = items[["disk_capacity",
                             "vm_cpu", "vm_memory"] + self.cat_type_list].copy()
        for field in self.cat_type_list:
            if field in self.cat_dtypes_dict:
                target_type = self.cat_dtypes_dict[field]
                data_predict[field] = data_predict[field].astype(target_type)
            else:
                data_predict[field] = data_predict[field].astype("category")
        bandwidth_scale = np.exp(model_regressor.predict(data_predict))
        return bandwidth_scale

    def predict_catboost(self, items: pd.DataFrame):
        model_regressor = self.load_model(
            model_name="catboost", train_models=False)
        data_predict = items[["disk_capacity",
                             "vm_cpu", "vm_memory", "recent_history_bandwidth_memory"] + self.cat_type_list].copy()
        for field in self.cat_type_list:
            data_predict[field] = data_predict[field].astype(str)
        data_predict.to_csv(os.path.join(
            DirConfig.TEMPLE_DIR, 'trash', 'data_predict.csv'))
        bandwidth_scale = model_regressor.predict(data_predict)
        return bandwidth_scale

    def load_model(self, model_name: str = "catboost", train_models: bool = True) -> lgb.LGBMRegressor:
        if train_models:
            self.train_models(model_name=model_name)
            return self.load_model(model_name=model_name, train_models=False)
        model_regressor_dir = os.path.join(
            DirConfig.MODEL_DIR, self.algorithm_name, f"{model_name}_regressor.pkl")
        if not os.path.exists(model_regressor_dir):
            self.train_models(model_name=model_name)
            return self.load_model(model_name=model_name, train_models=False)
        with open(model_regressor_dir, "rb") as f:
            model_regressor = pickle.load(f)
        with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{model_name}_cat_dtypes_dict.pkl"), "rb") as f:
            self.cat_dtypes_dict = pickle.load(f)
        return model_regressor

    def convert_trace_to_vector(self, trace_df: pd.DataFrame):
        week_bandwidth = get_circular_trace_util(
            trace_df, iterate_first_day_util(trace_df['timestamp']), 288*7)
        shifted_bandwidth = np.roll(week_bandwidth, -12)
        week_df = pd.DataFrame({
            'datetime': pd.date_range(start='2023-05-09 00:00:00', periods=len(week_bandwidth), freq='5min'),
            'bandwidth': shifted_bandwidth
        }).set_index('datetime')
        bandwidth_trace_resamples = week_df["bandwidth"].resample(
            f'{DataConfig.TRACE_VECTOR_INTERVAL}h').mean()
        trace_vector = bandwidth_trace_resamples.groupby(
            bandwidth_trace_resamples.index.hour).mean()
        trace_vector = trace_vector/trace_vector.sum()
        return trace_vector

    def make_certain_business_type_vector(self, items):
        trace_vectors = items.apply(lambda x: self.convert_trace_to_vector(
            self.disks_trace[x['cluster_index']][x['disk_ID']]), axis=1)
        return trace_vectors.mean(axis=0)

    def make_business_type_vector(self, items):
        grouped_vectors = items.groupby('business_type').apply(
            self.make_certain_business_type_vector)
        return grouped_vectors

    def train_catboost(self, items: pd.DataFrame):
        train_data = items[["disk_capacity",
                            "vm_cpu", "vm_memory", "recent_history_bandwidth_memory"] + self.cat_type_list].copy()
        for field in self.cat_type_list:
            train_data[field] = train_data[field].astype(str)
        model_regressor = CatBoostRegressor(
            cat_features=self.cat_type_list,
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
        model_regressor.fit(train_data, items["avg_bandwidth"])
        logger.info(model_regressor.get_feature_importance(prettified=True))
        # params_distributions = {
        #     'learning_rate': loguniform(0.01, 0.2),
        #     'depth': randint(4, 15),
        #     'l2_leaf_reg': randint(1, 12),
        #     'loss_function': ['RMSE', 'MAE', 'Huber'],
        #     'iterations': randint(1000, 4000),
        #     'one_hot_max_size': [2, 10, 20, 50],
        #     'random_strength': [1, 2, 5, 10],
        #     'bagging_temperature': uniform(0, 1)
        # }
        # cb = CatBoostRegressor(
        #     cat_features=self.cat_type_list,
        #     verbose=0,
        #     early_stopping_rounds=50,
        #     thread_count=18
        # )
        # model_regressor = RandomizedSearchCV(
        #     estimator=cb,
        #     param_distributions=params_distributions,
        #     n_iter=300,
        #     cv=5,
        #     scoring='neg_root_mean_squared_error',
        #     n_jobs=4,
        #     verbose=2,
        #     random_state=42
        # )
        # y_train = np.log(items["avg_bandwidth"])
        # model_regressor.fit(train_data, y_train)
        # best_catboost = model_regressor.best_estimator_
        # feature_names = train_data.columns.tolist()
        # importances = best_catboost.feature_importances_
        # feature_imp_df = pd.DataFrame({
        #     'Feature': feature_names,
        #     'Importance': importances
        # }).sort_values(by='Importance', ascending=False)
        # feature_imp_df.to_csv(os.path.join(
        #     DirConfig.TIDAL_DIR, 'catboost_feature_importance.csv'))
        # results_df = pd.DataFrame(model_regressor.cv_results_)
        # results_df.to_csv(os.path.join(
        #     DirConfig.TIDAL_DIR, 'catboost_cv_search_results.csv'))
        return model_regressor

    def train_lightgbm(self, items: pd.DataFrame):
        train_data = items[["disk_capacity",
                            "vm_cpu", "vm_memory"] + self.cat_type_list].copy()
        for field in self.cat_type_list:
            train_data[field] = train_data[field].astype("category")
            self.cat_dtypes_dict[field] = train_data[field].dtype

        params_distributions = {
            'objective': ['regression', 'regression_l1', 'huber', 'tweedie', 'poisson'],
            'alpha': uniform(0.5, 0.45),
            'tweedie_variance_power': uniform(1.1, 0.8),
            'learning_rate': loguniform(0.005, 0.1),
            'n_estimators': randint(50, 3000),
            'num_leaves': randint(10, 150),
            'max_depth': randint(5, 15),
            'min_child_samples': randint(20, 200),
            'subsample': uniform(0.6, 0.4),
            'subsample_freq': [1],
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }

        lgbm = lgb.LGBMRegressor(
            metric='rmse',
            random_state=42,
            n_jobs=2,
            verbose=1)
        model_regressor = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=params_distributions,
            n_iter=50,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=4,
            verbose=1,
            random_state=42,
        )
        y_train = np.log(items["avg_bandwidth"])
        model_regressor.fit(train_data, y_train)
        best_lgbm = model_regressor.best_estimator_
        feature_names = train_data.columns.tolist()
        importances = best_lgbm.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        feature_imp_df.to_csv(os.path.join(
            DirConfig.TIDAL_DIR, 'lightgbm_feature_importance.csv'))
        results_df = pd.DataFrame(model_regressor.cv_results_)
        results_df.to_csv(os.path.join(
            DirConfig.TIDAL_DIR, 'lightgbm_cv_search_results.csv'))
        return best_lgbm

    def train_xgboost(self, items: pd.DataFrame):
        train_data = items[["disk_capacity", "vm_cpu",
                            "vm_memory"] + self.cat_type_list].copy()

        for field in self.cat_type_list:
            train_data[field] = train_data[field].astype("category")
            self.cat_dtypes_dict[field] = train_data[field].dtype

        params_distributions = {
            'objective': ['reg:squarederror', 'reg:absoluteerror', 'reg:tweedie'],
            'tweedie_variance_power': uniform(1.1, 0.8),
            'n_estimators': randint(500, 3000),
            'learning_rate': loguniform(0.005, 0.15),
            'max_depth': randint(3, 20),
            'min_child_weight': randint(10, 200),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'tree_method': ['hist'],
        }

        xgb = XGBRegressor(
            enable_categorical=True,
            n_jobs=18,
            verbosity=0
        )

        model_regressor = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=params_distributions,
            n_iter=300,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=4,
            random_state=42,
            verbose=1
        )
        y_train = np.log(items["avg_bandwidth"])
        model_regressor.fit(train_data, y_train)
        best_xgb = model_regressor.best_estimator_
        feature_names = train_data.columns.tolist()
        importances = best_xgb.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        feature_imp_df.to_csv(os.path.join(
            DirConfig.TIDAL_DIR, 'xgboost_feature_importance.csv'))
        results_df = pd.DataFrame(model_regressor.cv_results_)
        results_df.to_csv(os.path.join(
            DirConfig.TIDAL_DIR, 'xgboost_cv_search_results.csv'))
        return best_xgb

    def train_Kmeans(self, items: pd.DataFrame):
        train_data = items[["avg_bandwidth", "peak_bandwidth"]]
        cluster_K = ModelConfig.SCDA_CLUSTER_K

        self.scaler = MinMaxScaler()
        train_data_scaled = self.scaler.fit_transform(train_data)
        model_cluster = KMeans(n_clusters=cluster_K)
        model_cluster.fit(train_data_scaled)
        the_silhouette_score = silhouette_score(
            train_data_scaled, model_cluster.labels_)
        logger.info(f"Silhouette score: {the_silhouette_score}")

        return model_cluster

    def train_DecisionTree(self, items: pd.DataFrame, labels_cluster: List[int]):
        params_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        items['labels'] = items['business_type'].apply(
            lambda x: self.business_type_list.index(
                x) if x in self.business_type_list else -1
        )
        train_data = items[["disk_capacity",
                            "labels", "vm_cpu", "vm_memory", "volume_type"]]
        model_classify = GridSearchCV(
            estimator=DecisionTreeClassifier(criterion="gini"),
            param_grid=params_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0
        )
        model_classify.fit(train_data, labels_cluster)

        logger.info(f"DecisionTree model: {model_classify.best_params_}")
        logger.info(f"准确率: {model_classify.best_score_}")

        return model_classify.best_estimator_

    def additional_state_update(self, selected_warehouse: int, item: pd.Series):
        self.warehouse_trace_vector[:,
                                    selected_warehouse] += item["pre_trace_vector"]
        return
