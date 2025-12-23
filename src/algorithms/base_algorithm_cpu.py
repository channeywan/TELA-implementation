from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from collections import deque
import numpy as np
import pandas as pd
import logging
import os
from tqdm import tqdm
from data.loader import DiskDataLoader
from data.processor import DiskDataProcessor
from config.settings import WarehouseConfig, DataConfig, ModelConfig, DirConfig
from visualization.plotter import TelaPlotter
from sklearn.model_selection import train_test_split
import pickle
import time
logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """云盘放置算法基类"""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.loader = DiskDataLoader()
        self.processor = DiskDataProcessor()
        self.plotter = TelaPlotter()
        self.output_dir = getattr(DirConfig, f'{self.algorithm_name}_DIR')
        self.warehouses_max = WarehouseConfig.WAREHOUSE_MAX
        self.warehouse_number = WarehouseConfig.WAREHOUSE_NUMBER
        self.resource_dimension_number = 2
        self.current_time = -1
        self.warehouses_trace = np.zeros(
            (DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER, self.warehouse_number, self.resource_dimension_number))
        self.warehouses_resource_allocated = np.zeros(
            (self.warehouse_number, self.resource_dimension_number))
        self.warehouses_cannot_use_by_monitor = np.zeros(
            self.warehouse_number)
        self.violation_time_queues = [deque(maxlen=DataConfig.MAX_VIOLATION_OCCURRENCE)
                                      for _ in range(self.warehouse_number)]
        self.disks_trace = self.loader.load_all_trace()
        self.test_items = None
        self.place_items_number_list = []
        self.start_time = None
        self.end_time = None
    def run(self) -> None:
        """运行算法的主流程"""
        for episode in range(ModelConfig.EPISODES):
            logger.info(f"Episode {episode+1}/{ModelConfig.EPISODES}")
            # 加载和预处理数据（子类实现）
            items = self.load_and_preprocess_items()

            # 放置云盘
            self.place_item(items)
            self.end_time = time.perf_counter()
        logger.info(f"Algorithm {self.algorithm_name} running time: {self.end_time - self.start_time} seconds")
    @abstractmethod
    def load_and_preprocess_items(self):
        """加载和预处理数据（子类实现）"""
        pass

    @abstractmethod
    def select_warehouse(self, item) -> int:
        """选择仓库的策略（子类实现）"""
        pass

    def additional_state_update(self, selected_warehouse: int, item: pd.Series) -> None:
        """额外状态更新（子类可重写）"""
        pass

    def place_item(self, items: pd.DataFrame):
        """放置云盘的通用流程"""
        disk_number = DataConfig.DISK_NUMBER
        if disk_number > len(items):
            disk_number = len(items)
        items_placed = 0
        available_items = items.sample(
            n=disk_number, replace=False, random_state=42)
        for item in available_items.itertuples(index=False):
        # for _, item in tqdm(available_items.iterrows(), total=len(available_items), desc="Placing items"):
            self.current_time += 1
            items_placed += 1
            selected_warehouse = self.select_warehouse(
                item)
            self.update_warehouse_state_tidal(
                selected_warehouse, item)
            if self.algorithm_name == "TELA":
                self.monitor_warning_violation(
                )
        return 


    def update_warehouse_state(self, selected_warehouse: int, item: pd.Series) -> None:
        """
        更新仓库状态
        由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        input:
            selected_warehouse: 选择的仓库编号
            item: 要放置的云盘
            warehouses_cannot_use_by_monitor: 不能使用的仓库列表
            warehouses_resource_allocated: 目前仓库已经分配的资源
            warehouses_trace: 全生命周期内每个时间戳的仓库资源使用情况
            current_time: 当前时间戳，由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        """
        # 提取物品信息
        cluster_index = item["cluster_index"]
        disk_ID = item["disk_ID"]
        disk_capacity = item["disk_capacity"]
        # 更新资源分配
        self.warehouses_resource_allocated[selected_warehouse][0] += item["disk_capacity"]

        disk_trace_bandwidth = self.disks_trace[cluster_index][disk_ID]
        first_day_line = self._iterate_first_day(
            disk_trace_bandwidth["timestamp"])
        circular_trace = self._get_circular_trace(
            disk_trace_bandwidth, disk_capacity, first_day_line, DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER)
        if circular_trace is None:
            logger.error(
                f"the disk trace is not support circular trace, cluster_index: {cluster_index}, disk_ID: {disk_ID}")
            return
        self.warehouses_trace[:, selected_warehouse] += circular_trace
        self.additional_state_update(
            selected_warehouse, item)
        return
    def update_warehouse_state_tidal(self, selected_warehouse: int, item: tuple) -> None:
        """
        更新仓库状态
        由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        input:
            selected_warehouse: 选择的仓库编号
            item: 要放置的云盘
            warehouses_cannot_use_by_monitor: 不能使用的仓库列表
            warehouses_resource_allocated: 目前仓库已经分配的资源
            warehouses_trace: 全生命周期内每个时间戳的仓库资源使用情况
            current_time: 当前时间戳，由于进行云盘放置模拟时是每个时间戳放置一个云盘，所以当前时间戳等于已经放置的云盘数量
        """
        # 提取物品信息
        cluster_index = item.cluster_index
        disk_ID = item.disk_ID
        disk_capacity = item.disk_capacity
        # 更新资源分配
        self.warehouses_resource_allocated[selected_warehouse][0] += item.disk_capacity

        disk_trace_bandwidth = self.disks_trace[cluster_index][disk_ID]
        first_day_line = self._iterate_first_day(
            disk_trace_bandwidth["timestamp"])
        circular_trace = self._get_circular_trace(
            disk_trace_bandwidth, disk_capacity, first_day_line, DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER)
        if circular_trace is None:
            logger.error(
                f"the disk trace is not support circular trace, cluster_index: {cluster_index}, disk_ID: {disk_ID}")
            return
        self.warehouses_trace[:, selected_warehouse] += circular_trace
        self.additional_state_update(
            selected_warehouse, item)
        return
    def check_warehouse_overload_after_placement(self,  item: pd.Series) -> np.ndarray:
        """
        检查仓库放置当前盘后的负载超限
        """
        disk_trace_bandwidth = self.disks_trace[item['cluster_index']
                                                ][item['disk_ID']]
        first_day_line = self._iterate_first_day(
            disk_trace_bandwidth["timestamp"])
        future_bandwidth = self._get_circular_trace(
            disk_trace_bandwidth, item["disk_capacity"], first_day_line, DataConfig.DISK_NUMBER)
        after_placed_bandwidth = self.warehouses_trace[:DataConfig.DISK_NUMBER,
                                                       :, 1] + future_bandwidth[:, 1][:, np.newaxis]
        after_placed_bandwidth_util = after_placed_bandwidth / \
            self.warehouses_max[:, 1]
        overload_mask = np.sum(after_placed_bandwidth_util >=
                               1, axis=0) <= DataConfig.MAX_OVERLOAD_LIFETIME_OCCURRENCE
        return overload_mask
    def monitor_warning_violation(self) -> None:
        """监控SLA违反"""
        self.warehouses_cannot_use_by_monitor = np.sum(
            self.warehouses_trace[:DataConfig.DISK_NUMBER, :, 1] > self.warehouses_max[:, 1], axis=0) >= DataConfig.MAX_OVERLOAD_LIFETIME_OCCURRENCE
        for warehouse in range(self.warehouse_number):
            if self.warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue

            if np.any(
                self.warehouses_trace[self.current_time][warehouse] >
                self.warehouses_max[warehouse]
            ):
                self.violation_time_queues[warehouse].append(
                    self.current_time)
                while len(self.violation_time_queues[warehouse]) >= 2 and self.violation_time_queues[warehouse][-1] - self.violation_time_queues[warehouse][0] >= DataConfig.VIOLATION_WINDOW_SIZE:
                    self.violation_time_queues[warehouse].popleft()
                if len(self.violation_time_queues[warehouse]) >= DataConfig.MAX_VIOLATION_OCCURRENCE:
                    self.warehouses_cannot_use_by_monitor[warehouse] = 1

                self.violation_time_queues[warehouse].append(
                    self.current_time)

    def encode_item(self, items: pd.DataFrame) -> pd.DataFrame:
        """
        对物品进行编码
        input:
            item: 物品列表
        output:
            item_encoded: 编码后的物品列表
        """
        type_mapping = {"root": 0, "data": 1}
        items = items.copy()
        items["disk_type"] = items["disk_type"].map(type_mapping)
        return items

    def _iterate_first_day(self, timestamps) -> int:
        """
        计算第一周第一天的行数
        """
        target_time = "2023-05-09 00:00:00"
        target_time = pd.to_datetime(target_time)
        line_len = -1
        for timestamp in timestamps:
            line_len += 1
            if timestamp.weekday() == target_time.weekday() and timestamp.time() == target_time.time():
                break
        return line_len

    def _get_circular_trace(self, disk_trace_bandwidth: pd.DataFrame, disk_capacity: int, begin_line: int, trace_len: int) -> np.ndarray:
        """
        return :DataFrame, shape(trace_len, [disk_capacity, bandwidth])
        """
        timestamps = disk_trace_bandwidth["timestamp"]
        last_timestamp = timestamps.iloc[-1]
        target_timestamp = last_timestamp + pd.Timedelta("5 min")
        target_weekday = target_timestamp.weekday()
        target_time = target_timestamp.time()
        all_weekdays = timestamps.dt.weekday
        all_times = timestamps.dt.time
        mask_match = (all_weekdays == target_weekday) & (
            all_times == target_time)
        matches = np.where(mask_match)[0]
        if len(matches) == 0:
            logger.error(
                "the disk trace is not support circular trace (no matches found)")
            return None
        start_index = matches[0]
        base_trace = disk_trace_bandwidth.iloc[begin_line:]["bandwidth"].to_numpy(
        )
        append_trace = disk_trace_bandwidth.iloc[start_index:]["bandwidth"].to_numpy(
        )
        if len(append_trace) == 0:
            logger.error(
                "append_trace is empty, cannot tile for circular trace")
            return None
        len_base = len(base_trace)

        if len_base >= trace_len:
            circular_bandwidth_trace = base_trace[:trace_len]
        else:
            len_needed = trace_len - len_base
            num_repeats = int(np.ceil(len_needed / len(append_trace)))
            padding = np.tile(append_trace, num_repeats)[:len_needed]
            circular_bandwidth_trace = np.concatenate([base_trace, padding])
        circular_trace = np.column_stack([np.full(
            len(circular_bandwidth_trace), disk_capacity), circular_bandwidth_trace])

        return circular_trace
