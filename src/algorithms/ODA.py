import logging
from visualization.plotter import DiskPlotter
from data.processor import DiskDataProcessor
from data.loader import DiskDataLoader
from config.settings import DirConfig, ModelConfig, DataConfig, WarehouseConfig
import matplotlib.pyplot as plt
import os
import linecache
import random
import numpy as np
import matplotlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import pandas as pd
import joblib
from .base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)


class ODA(BaseAlgorithm):
    """ODA算法实现"""

    def __init__(self):
        super().__init__("ODA")

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        for cluster_index in DataConfig.CLUSTER_INDEX_LIST_ODA:
            trace_dir = os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl")
            self.disks_trace[cluster_index] = joblib.load(trace_dir)
        return self.loader.load_items(cluster_index_list=DataConfig.CLUSTER_INDEX_LIST_ODA)

    def select_warehouse(self, item) -> int:
        """
        选择一个仓库
        选择当前容量利用率最低的仓库
        """
        disk_capacity = item["disk_capacity"]
        capacity_mask = (self.warehouses_resource_allocated[:, 0]+disk_capacity <=
                         self.warehouses_max[:, 0])
        monitor_mask = (self.warehouses_cannot_use_by_monitor == 0)
        combined_mask = monitor_mask & capacity_mask
        eligible_warehouses_indices = np.where(combined_mask)[0]
        if len(eligible_warehouses_indices) == 0:
            return -1
        allocated_capacity = self.warehouses_resource_allocated[eligible_warehouses_indices, 0]
        min_utilization_index = np.argmin(
            allocated_capacity/WarehouseConfig.WAREHOUSE_MAX[eligible_warehouses_indices, 0])
        selected_warehouse = eligible_warehouses_indices[min_utilization_index]
        return selected_warehouse
