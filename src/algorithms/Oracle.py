from .base_algorithm import BaseAlgorithm
from config.settings import DirConfig, ModelConfig, DataConfig, WarehouseConfig
import logging
import pandas as pd
import numpy as np
from typing import Any
import os
import joblib
logger = logging.getLogger(__name__)


class Oracle(BaseAlgorithm):
    def __init__(self):
        super().__init__("Oracle")

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        logger.info(f"Loading cluster trace data for Oracle")
        for cluster_index in DataConfig.CLUSTER_INDEX_LIST_ORACLE:
            trace_dir = os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl")
            self.disks_trace[cluster_index] = joblib.load(trace_dir)
        logger.info(f"Loaded cluster trace data for Oracle")
        return self.loader.load_items(type="both", cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_ORACLE, purpose="train")

    def select_warehouse(self, item: pd.Series) -> int:
        if self.current_time == 0:
            return 0
        reverse_time_window = min(12*24*7, self.current_time)
        disk_trace_bandwidth = self.disks_trace[item['cluster_index']
                                                ][item['disk_ID']]
        future_bandwidth = self._get_circular_trace(
            disk_trace_bandwidth, item["disk_capacity"], self.current_time, 12*24*7)
        selected_warehouse = -1
        monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+item["disk_capacity"] <=
                         self.warehouses_max[:, 0]*ModelConfig.RESERVATION_RATE_FOR_MONITOR)
        after_placed_bandwidth = self.warehouses_trace[self.current_time-reverse_time_window:
                                                       self.current_time, :, 1]+np.tile(future_bandwidth[len(future_bandwidth)-reverse_time_window:, 1].reshape(reverse_time_window, 1), (1, self.warehouse_number))
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]

        combined_mask = monitor_mask & capacity_mask & self.get_overload_mask(
            after_placed_bandwidth_util)
        eligible_warehouses_indices = np.where(combined_mask)[0]
        if len(eligible_warehouses_indices) == 0:
            return -1
        max_bandwidth_util = np.max(
            after_placed_bandwidth[:, eligible_warehouses_indices]/self.warehouses_max[eligible_warehouses_indices, 1], axis=0)
        min_peak_bandwidth_index = np.argmin(max_bandwidth_util)
        selected_warehouse = eligible_warehouses_indices[min_peak_bandwidth_index]
        return selected_warehouse

    def get_overload_mask(self, after_placed_bandwidth_util: np.ndarray) -> np.ndarray:
        overload_mask = np.ones(self.warehouse_number, dtype=bool)
        for i in range(self.warehouse_number):
            warehouse_overload_time = np.where(
                after_placed_bandwidth_util[:, i] > 1)[0]
            if len(warehouse_overload_time) < 2:
                continue
            overload_last_time = np.diff(warehouse_overload_time)
            if np.min(overload_last_time) > DataConfig.VIOLATION_WINDOW_SIZE:
                overload_mask[i] = True
        return overload_mask
