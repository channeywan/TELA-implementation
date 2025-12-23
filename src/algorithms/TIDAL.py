from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import os
import pickle
from .base_algorithm import BaseAlgorithm
from config.settings import DataConfig, ModelConfig, DirConfig, WarehouseConfig
from data.utils import get_circular_trace as get_circular_trace_util
from data.utils import iterate_first_day as iterate_first_day_util
from visualization.plotter import TelaPlotter
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.stats import randint, uniform, loguniform
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
logger = logging.getLogger(__name__)


class TIDAL(BaseAlgorithm):

    def __init__(self, ml_model: str = "random_forest", distill_model: str = "distilbert-base-multilingual-cased"):
        super().__init__("TIDAL")
        self.ml_model = ml_model
        self.distill_model = distill_model
        self.business_type_list = DataConfig.BUSINESS_TYPE_LIST
        self.trace_vector_interval = 12
        self.warehouse_trace_vector = np.zeros(
            (int(24/self.trace_vector_interval), self.warehouse_number))
        self.cat_dtypes_dict = {}
        self.cat_type_list = ["disk_if_VIP", "ins_type",
                              "disk_type", "volume_type", "business_type"]
        self.unknown_threshold = 0.6
        self.general_warehouse_capacity = np.zeros(self.warehouse_number)
        self.features = ["disk_capacity", "vm_cpu",
                         "vm_memory", "recent_history_bandwidth_memory"]
        self.ordinal_encoder = None
        self.min_bandwidth_warehouse_num = 4
        self.num_unknown = 0
        self.noise_ratio = 0
        self.target="min_delta_util_var"
        self.overload_duration=None
        self.time_imbalance=None
        self.space_imbalance=None
        self.overload_percentage=None
        self.random_state=2

    def load_and_preprocess_items(self):
        predicted_items = self.predict(self.test_items.copy())
        predicted_items.to_csv(os.path.join(
            DirConfig.INTERMEDIATE_DIR, f'predicted_items_TIDAL_{self.trace_vector_interval}h.csv'))
        self.num_unknown = len(
            predicted_items[predicted_items['pre_business_type'] == 'generic-unknown'])
        return predicted_items

    def select_warehouse(self, item: pd.Series) -> int:
        # overload_mask = self.check_warehouse_overload_after_placement(item)
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+item["disk_capacity"] <=
                         self.warehouses_max[:, 0])

        disk_pre_trace_vector = item["pre_trace_vector"]
        after_placed_bandwidth = self.warehouse_trace_vector + \
            disk_pre_trace_vector[:, np.newaxis]
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]
        if item["pre_business_type"] == "generic-unknown":
            while True:
                # monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
                combined_mask = capacity_mask
                if not combined_mask.any():
                    combined_mask = np.ones_like(capacity_mask, dtype=bool)
                eligible_warehouses_indices = np.where(combined_mask)[0]
                min_utilization_index = np.argmin(
                    self.general_warehouse_capacity[eligible_warehouses_indices]/self.warehouses_max[eligible_warehouses_indices, 0])
                selected_warehouse = eligible_warehouses_indices[min_utilization_index]
                # if not overload_mask[selected_warehouse]:
                #     if self.warehouses_cannot_use_by_monitor[selected_warehouse] == 1:
                #         return selected_warehouse
                #     else:
                #         self.warehouses_cannot_use_by_monitor[selected_warehouse] = 1
                #         continue
                # else:
                #     return selected_warehouse
                with open(os.path.join(DirConfig.INTERMEDIATE_DIR, f'delta_deviation_{self.algorithm_name}_{self.trace_vector_interval}h.csv'), 'a') as f:
                    f.write("selected_warehouse:" + str(selected_warehouse))
                    f.write("  warehouses_indices:" +
                            str(eligible_warehouses_indices.tolist()))
                    f.write("  unknown_warehouse_capacity:" +
                            str((self.general_warehouse_capacity[eligible_warehouses_indices]/self.warehouses_max[eligible_warehouses_indices, 0]).tolist()))
                    f.write('\n')
                return selected_warehouse
        original_bandwidth = self.warehouse_trace_vector
        original_bandwidth_util = original_bandwidth / \
            self.warehouses_max[:, 1]
        # trace_vector_mask = (after_placed_bandwidth_util <= 1).all(axis=0)
        while True:
            # monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
            combined_mask = capacity_mask
            if not combined_mask.any():
                combined_mask = np.ones_like(capacity_mask, dtype=bool)
                # return -1
            warehouse_scores = np.sum(original_bandwidth_util, axis=0)
            warehouse_scores[~combined_mask] = np.inf
            candidate_indices = np.argsort(warehouse_scores)[
                :self.min_bandwidth_warehouse_num]
            eligible_warehouses_indices = candidate_indices[warehouse_scores[candidate_indices] != np.inf]
            # eligible_warehouses_indices = np.where(combined_mask)[0]
            if self.target == "min_delta_util_var":
                original_bandwidth_util_deviation = np.var(
                original_bandwidth_util[:, eligible_warehouses_indices], axis=0)
                absolute_deviation = np.var(
                    after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
                delta_deviation = absolute_deviation-original_bandwidth_util_deviation
                target_index = np.argmin(delta_deviation)
            if self.target == "min_util_var":
                absolute_deviation = np.var(
                    after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
                target_index = np.argmin(absolute_deviation)
            if self.target == "min_load_var":
                target_index = np.argmin(np.var(
                    after_placed_bandwidth[:, eligible_warehouses_indices], axis=0))
            if self.target == "min_delta_load_var":
                original_bandwidth_load_deviation = np.var(
                    original_bandwidth[:, eligible_warehouses_indices], axis=0)
                absolute_deviation = np.var(
                    after_placed_bandwidth[:, eligible_warehouses_indices], axis=0)
                delta_deviation = absolute_deviation-original_bandwidth_load_deviation
                target_index = np.argmin(delta_deviation)
            if self.target == "min_peak_load_before_placement":
                target_index = np.argmin(np.max(original_bandwidth_util, axis=0))
            if self.target == "min_peak_load_after_placement":
                target_index = np.argmin(np.max(after_placed_bandwidth_util, axis=0))
            selected_warehouse = eligible_warehouses_indices[target_index]
            # with open(os.path.join(DirConfig.INTERMEDIATE_DIR, f'delta_deviation_{self.algorithm_name}_{self.trace_vector_interval}h.csv'), 'a') as f:
            #     f.write("selected_warehouse:" + str(selected_warehouse))
            #     f.write("  warehouses_indices:" +
            #             str(eligible_warehouses_indices.tolist()))
            #     f.write("  original_bandwidth_util_deviation:" +
            #             str(original_bandwidth_util_deviation.tolist()))
            #     f.write("  after_placed_bandwidth_util_deviation:" +
            #             str(absolute_deviation.tolist()))
            #     f.write("  delta_deviation:" +
            #             str(delta_deviation.tolist()))
            #     f.write("  original_bandwidth_util:" +
            #             str((np.mean(self.warehouse_trace_vector, axis=0)/self.warehouses_max[:, 1]).tolist()))

            #     f.write("  warehouse_capacity:" +
            #             str((self.warehouses_resource_allocated[:, 0]/self.warehouses_max[:, 0]).tolist()))
            #     f.write('\n')
            # if not overload_mask[selected_warehouse]:
            #     if self.warehouses_cannot_use_by_monitor[selected_warehouse] == 1:
            #         break
            #     else:
            #         self.warehouses_cannot_use_by_monitor[selected_warehouse] = 1
            #         continue
            # else:
            #     break
            return selected_warehouse
        return selected_warehouse

    def generate_business_type_vector(self):
        items = self.train_items.copy()
        grouped_vectors = self.make_business_type_vector(items)
        grouped_vectors.to_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, f'business_type_vector_{self.trace_vector_interval}h.csv'))
        # TelaPlotter().plot_business_type_vector(grouped_vectors, os.path.join(
        #     DirConfig.BUSINESS_TYPE_DIR))

    def train_ml_models(self, model_name: str):
        items = self.train_items.copy()
        if model_name == "catboost":
            model_regressor = self.train_catboost(items)
        elif model_name == "random_forest":
            model_regressor = self.train_random_forest(items)
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
        if model_name == "random_forest":
            with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{model_name}_ordinal_encoder.pkl"), "wb") as f:
                pickle.dump(self.ordinal_encoder, f)

    def predict(self, items: pd.DataFrame):
        predict_business_type = self.predict_business_type(items)
        items['pre_business_type'] = predict_business_type
        return self.predict_disk_trace_vector(items)

    def predict_business_type(self, items: pd.DataFrame):
        batch_size = 1024
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, self.distill_model)
        Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path).to(Device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        results = []
        descriptions = items['description'].tolist()
        for i in range(0, len(items), batch_size):
            batch_text = descriptions[i:i+batch_size]
            inputs = tokenizer(batch_text, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=128).to(Device)
            inputs = {k: v.to(Device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = F.softmax(model(**inputs).logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)
            batch_confs = max_probs.cpu().numpy()
            batch_ids = pred_ids.cpu().numpy()
            for conf, p_id in zip(batch_confs, batch_ids):
                if conf < self.unknown_threshold:
                    results.append("generic-unknown")
                else:
                    results.append(model.config.id2label[p_id])
        # 手动加入noise
        if self.noise_ratio > 0:
            results = pd.Series(results)
            replacement_pool = pd.Series(
                list(model.config.id2label.values())+['generic-unknown'])
            # noise_indices=np.random.choice(results[results=='infra-node'].index, size=noise_num*0.5, replace=False, random_state=42)
            # results[noise_indices]='generic-unknown'
            # noise_indices=np.random.choice(results[results=='generic-unknown'].index, size=noise_num*0., replace=False, random_state=42)
            # results[noise_indices]='infra-node'
            noise_indices = results.sample(
                frac=self.noise_ratio, replace=False, random_state=self.random_state).index
            new_values = replacement_pool.sample(
                n=len(noise_indices), replace=True, random_state=self.random_state)
            results.loc[noise_indices] = new_values.values
            results = results.tolist()
        return results

    def predict_disk_trace_vector(self, items: pd.DataFrame, retrain_model: bool = False, generate_business_type_vector: bool = False):
        if retrain_model:
            self.train_ml_models(model_name=self.ml_model)
        if generate_business_type_vector:
            self.generate_business_type_vector()
        trace_vector = pd.read_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, f'business_type_vector_{self.trace_vector_interval}h.csv'), index_col=0)
        bandwidth_scale = self.predict_ml_model(items)
        bandwidth_scale = np.array(bandwidth_scale)
        pre_trace_vector = bandwidth_scale[:, np.newaxis] * \
            trace_vector.loc[items['pre_business_type']].values
        items['pre_trace_vector'] = list(pre_trace_vector)
        return items

    def predict_ml_model(self, items: pd.DataFrame):
        if self.ml_model == "catboost":
            return self.predict_catboost(items)
        elif self.ml_model == "lightgbm":
            return self.predict_lightgbm(items)
        elif self.ml_model == "xgboost":
            return self.predict_xgboost(items)
        elif self.ml_model == "random_forest":
            return self.predict_random_forest(items)
        else:
            raise ValueError(f"不支持的模型: {self.ml_model}")

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
        ml_model = self.load_model(ml_model_name=self.ml_model)
        data_predict = items[self.features + self.cat_type_list].copy()
        for field in self.cat_type_list:
            data_predict[field] = data_predict[field].astype(str)
        bandwidth_scale = ml_model.predict(data_predict)
        return bandwidth_scale

    def predict_random_forest(self, items: pd.DataFrame):
        ml_model = self.load_model(ml_model_name=self.ml_model)
        data_predict = items[self.features + self.cat_type_list].copy()
        data_predict[self.cat_type_list] = self.ordinal_encoder.transform(
            data_predict[self.cat_type_list])
        bandwidth_scale = ml_model.predict(data_predict)
        bandwidth_scale = np.expm1(bandwidth_scale)
        return bandwidth_scale

    def load_model(self, ml_model_name: str) -> CatBoostRegressor | RandomForestRegressor:
        model_regressor_dir = os.path.join(
            DirConfig.MODEL_DIR, self.algorithm_name, f"{ml_model_name}_regressor.pkl")
        if not os.path.exists(model_regressor_dir):
            self.train_ml_models(model_name=ml_model_name)
            return self.load_model(model_name=ml_model_name)
        with open(model_regressor_dir, "rb") as f:
            ml_model = pickle.load(f)
        with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{ml_model_name}_cat_dtypes_dict.pkl"), "rb") as f:
            self.cat_dtypes_dict = pickle.load(f)
        if ml_model_name == "random_forest":
            with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{ml_model_name}_ordinal_encoder.pkl"), "rb") as f:
                self.ordinal_encoder = pickle.load(f)
        return ml_model

    def convert_trace_to_vector(self, trace_df: pd.DataFrame):
        week_bandwidth = get_circular_trace_util(
            trace_df, iterate_first_day_util(trace_df['timestamp']), 288*7)
        shifted_bandwidth = np.roll(week_bandwidth, -60)
        week_df = pd.DataFrame({
            'datetime': pd.date_range(start='2023-05-09 00:00:00', periods=len(week_bandwidth), freq='5min'),
            'bandwidth': shifted_bandwidth
        }).set_index('datetime')
        bandwidth_trace_resamples = week_df["bandwidth"].resample(
            f'{self.trace_vector_interval}h').mean()
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

    def train_catboost(self, train_items: pd.DataFrame):
        X_train = train_items[self.features + self.cat_type_list].copy()
        for field in self.cat_type_list:
            X_train[field] = X_train[field].astype(str)
        y_train = train_items["avg_bandwidth"]
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
        model_regressor.fit(X_train, y_train)
        return model_regressor

    def train_random_forest(self, train_items: pd.DataFrame):
        X_train = train_items[self.features + self.cat_type_list].copy()
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[self.cat_type_list] = self.ordinal_encoder.fit_transform(
            X_train[self.cat_type_list])
        y_train = np.log1p(train_items["avg_bandwidth"])
        model_regressor = RandomForestRegressor(
            bootstrap=False,
            max_depth=20,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=4,
            n_estimators=307,
            random_state=42,
            n_jobs=100,
            verbose=0)
        model_regressor.fit(X_train, y_train)
        return model_regressor

    def additional_state_update(self, selected_warehouse: int, item: pd.Series) -> None:
        if item["pre_business_type"] == "generic-unknown":
            self.general_warehouse_capacity[selected_warehouse] += item["disk_capacity"]
        else:
            self.warehouse_trace_vector[:,
                                        selected_warehouse] += item["pre_trace_vector"]
        return
