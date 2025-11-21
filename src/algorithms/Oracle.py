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
        return self.test_items.copy()

    def select_warehouse(self, item: pd.Series) -> int:
        selected_warehouse = -1
        if self.current_time == 0:
            return 0
        disk_trace_bandwidth = self.disks_trace[item['cluster_index']
                                                ][item['disk_ID']]
        first_day_line = self._iterate_first_day(
            disk_trace_bandwidth["timestamp"])
        future_bandwidth = self._get_circular_trace(
            disk_trace_bandwidth, item["disk_capacity"], first_day_line, DataConfig.DISK_NUMBER)
        monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+item["disk_capacity"] <=
                         self.warehouses_max[:, 0])
        after_placed_bandwidth = self.warehouses_trace[:DataConfig.DISK_NUMBER,
                                                       :, 1] + future_bandwidth[:, 1][:, np.newaxis]
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]
        overload_mask = self.check_warehouse_overload_after_placement(item)
        combined_mask = capacity_mask & overload_mask
        eligible_warehouses_indices = np.where(combined_mask)[0]
        if len(eligible_warehouses_indices) == 0:
            return -1

        # absolute_deviation = np.sum(np.abs(after_placed_bandwidth_util[:, eligible_warehouses_indices]-np.mean(
        #     after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)), axis=0)/np.mean(after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
        absolute_deviation = np.std(after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)/np.mean(
            after_placed_bandwidth_util[:, eligible_warehouses_indices], axis=0)
        min_absolute_deviation_index = np.argmin(absolute_deviation)
        selected_warehouse = eligible_warehouses_indices[min_absolute_deviation_index]
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
