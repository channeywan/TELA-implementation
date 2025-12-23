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
import time
logger = logging.getLogger(__name__)


class RoundRobin(BaseAlgorithm):
    """RoundRobin算法实现"""

    def __init__(self):
        super().__init__("RoundRobin")
        self.round_robin_index = 0

    def load_and_preprocess_items(self):
        """加载和预处理数据"""
        self.start_time = time.perf_counter()
        return self.test_items.copy()

    def select_warehouse(self, item) -> int:
        """
        选择一个仓库
        轮询选择一个仓库
        """
        selected_warehouse = self.round_robin_index % self.warehouse_number
        self.round_robin_index += 1
        return selected_warehouse
