from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import os
import pickle
from .base_algorithm_cpu import BaseAlgorithm
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
import time

class TIDAL(BaseAlgorithm):

    def __init__(self, ml_model: str = "random_forest", distill_model: str = "distilbert-base-multilingual-cased"):
        super().__init__("TIDAL")
        self.ml_model = ml_model
        self.distill_model = distill_model
        self.model_regressor = None
        self.ordinal_encoder = None
        self.model_bert = None
        self.tokenizer = None
        self.business_type_list = DataConfig.BUSINESS_TYPE_LIST
        self.trace_vector_interval = DataConfig.TRACE_VECTOR_INTERVAL
        self.warehouse_trace_vector = np.zeros(
            (int(24/self.trace_vector_interval), self.warehouse_number))
        self.unknown_threshold = 0.6
        self.general_warehouse_capacity = np.zeros(self.warehouse_number)
        self.cat_type_list = ["disk_if_VIP", "ins_type",
                              "disk_type", "volume_type", "business_type"]
        self.features = ["disk_capacity", "vm_cpu",
                         "vm_memory", "recent_history_bandwidth_memory"]
        self.device = torch.device("cpu") # 既然确定只用 CPU，直接写死
        self._init_bert_model()
    def _init_bert_model(self):
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, self.distill_model)
        self.model_bert = AutoModelForSequenceClassification.from_pretrained(
            model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_bert.eval()
        self.id2label = self.model_bert.config.id2label
        self.model_regressor = self.load_model(ml_model_name=self.ml_model)
    def load_and_preprocess_items(self):
        self.start_time = time.perf_counter()
        predicted_items = self.predict(self.test_items.copy())
        return predicted_items

    def select_warehouse(self, item: tuple) -> int:
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+item.disk_capacity <=
                         self.warehouses_max[:, 0])

        disk_pre_trace_vector = item.pre_trace_vector
        after_placed_bandwidth = self.warehouse_trace_vector + \
            disk_pre_trace_vector[:, np.newaxis]
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]
        if item.pre_business_type == "generic-unknown":
            combined_mask = capacity_mask
            if not combined_mask.any():
                combined_mask = np.ones_like(capacity_mask, dtype=bool)
            eligible_warehouses_indices = np.where(combined_mask)[0]
            min_utilization_index = np.argmin(
                self.general_warehouse_capacity[eligible_warehouses_indices]/self.warehouses_max[eligible_warehouses_indices, 0])
            selected_warehouse = eligible_warehouses_indices[min_utilization_index]
            return selected_warehouse
        original_bandwidth = self.warehouse_trace_vector
        original_bandwidth_util = original_bandwidth / \
            self.warehouses_max[:, 1]
        combined_mask = capacity_mask
        if not combined_mask.any():
            combined_mask = np.ones_like(capacity_mask, dtype=bool)
        warehouse_scores = np.sum(original_bandwidth_util, axis=0)
        warehouse_scores[~combined_mask] = np.inf
        candidate_indices = np.argsort(warehouse_scores)[:3]
        eligible_warehouses_indices = candidate_indices[warehouse_scores[candidate_indices] != np.inf]
        original_bandwidth_util_deviation = np.var(
            original_bandwidth_util[:, eligible_warehouses_indices], axis=0)
        absolute_deviation = np.var(
            after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
        delta_deviation = absolute_deviation-original_bandwidth_util_deviation
        min_delta_deviation_index = np.argmin(delta_deviation)
        selected_warehouse = eligible_warehouses_indices[min_delta_deviation_index]
        return selected_warehouse

    def predict(self, items: pd.DataFrame):
        items = self.predict_business_type(items)
        return self.predict_disk_trace_vector(items)

    def predict_business_type(self, items: pd.DataFrame):
        items["disk_name"]=items["disk_name"].astype(str)
        items["50percent_disk_name"]=items["disk_name"].apply(lambda x: x[:int(len(x) * 0.2)])
        items["cache_key"]=items["project_name"]+"_"+items["vm_name"]+"_"+items["50percent_disk_name"]
        cache_items=items.groupby("cache_key")["description"].first().reset_index()
        logging.info(f"cache_hitten_ratio: {1-len(cache_items)/len(items)}")
        bert_begin_time = time.perf_counter()
        batch_size = 32
        results = []
        descriptions = cache_items["description"].tolist()
        with torch.no_grad():
            for i in range(0, len(cache_items), batch_size):
                batch_text = descriptions[i:i+batch_size]
                inputs = self.tokenizer(batch_text, return_tensors="pt",
                                padding="longest", truncation=True, max_length=128).to(self.device)
                logits = self.model_bert(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                max_probs, pred_ids = torch.max(probs, dim=-1)
                batch_confs = max_probs.detach().numpy() 
                batch_ids = pred_ids.detach().numpy()
                unknown_mask = batch_confs < self.unknown_threshold
                batch_labels = np.array([self.id2label[pid] for pid in batch_ids],dtype=object)
                batch_labels[unknown_mask] = "generic-unknown"
                results.extend(batch_labels)
        bert_end_time = time.perf_counter()
        logger.info(f"BERT predict business type time: {bert_end_time - bert_begin_time} seconds")
        cache_items["pre_business_type"] = results
        return pd.merge(items, cache_items, on="cache_key", how="left")


    def predict_disk_trace_vector(self, items: pd.DataFrame):
        trace_vector = pd.read_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, 'business_type_vector.csv'), index_col=0)
        bandwidth_scale = self.predict_random_forest(items)
        bandwidth_scale = np.array(bandwidth_scale)
        pre_trace_vector = bandwidth_scale[:, np.newaxis] * \
            trace_vector.loc[items['pre_business_type']].values

        
        items['pre_trace_vector'] = list(pre_trace_vector)
        return items

    def predict_random_forest(self, items: pd.DataFrame):
        ml_model = self.model_regressor
        data_predict = items[self.features + self.cat_type_list].copy()
        data_predict[self.cat_type_list] = self.ordinal_encoder.transform(
            data_predict[self.cat_type_list])
        bandwidth_scale = ml_model.predict(data_predict)
        bandwidth_scale = np.expm1(bandwidth_scale)
        return bandwidth_scale

    def load_model(self, ml_model_name: str) -> CatBoostRegressor | RandomForestRegressor:
        model_regressor_dir = os.path.join(
            DirConfig.MODEL_DIR, self.algorithm_name, f"{ml_model_name}_regressor.pkl")
        with open(model_regressor_dir, "rb") as f:
            ml_model = pickle.load(f)
        with open(os.path.join(DirConfig.MODEL_DIR, self.algorithm_name, f"{ml_model_name}_ordinal_encoder.pkl"), "rb") as f:
            self.ordinal_encoder = pickle.load(f)
        return ml_model

    def additional_state_update(self, selected_warehouse: int, item: pd.Series) -> None:
        if item.pre_business_type == "generic-unknown":
            self.general_warehouse_capacity[selected_warehouse] += item.disk_capacity
        else:
            self.warehouse_trace_vector[:,
                                        selected_warehouse] += item.pre_trace_vector
        return
