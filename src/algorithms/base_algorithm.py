from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from collections import deque
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from tqdm import tqdm
from data.loader import DiskDataLoader
from data.processor import DiskDataProcessor
from config.settings import WarehouseConfig, DataConfig, ModelConfig, DirConfig
from visualization.plotter import TelaPlotter
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
        self.disks_trace = {}

    def run(self) -> None:
        """运行算法的主流程"""
        # 运行多个episode
        for episode in range(ModelConfig.EPISODES):
            logger.info(f"Episode {episode+1}/{ModelConfig.EPISODES}")
            # 加载和预处理数据（子类实现）
            items = self.load_and_preprocess_items()

            # 放置云盘
            _, _ = self.place_item(items)

            # 评估结果
            utilize_mean_on_time, violation_count_warehouse, max_violation_duration, all_violation_durations, warehouse_load_imb, imbalance_on_timewindow = self.evaluate_warehouses()

        self.write_result(utilize_mean_on_time, violation_count_warehouse, max_violation_duration,
                          all_violation_durations, warehouse_load_imb, imbalance_on_timewindow)

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
        logger.info(f"{self.algorithm_name}开始模拟放置云盘")
        disk_number = DataConfig.DISK_NUMBER
        if disk_number > len(items):
            disk_number = len(items)
        items_placed = 0
        # 用于可视化调试
        warehouses_allocation_history = []
        cannot_use_by_monitor_history = []
        # 可视化
        available_items = items.sample(
            n=disk_number, replace=False, random_state=42)
        placed_warehouses = []
        for _, item in tqdm(available_items.iterrows(), total=len(available_items), desc="Placing items"):
            self.current_time += 1
            items_placed += 1

            selected_warehouse = self.select_warehouse(
                item)
            placed_warehouses.append(selected_warehouse)
            if selected_warehouse == -1:
                if 0 not in self.warehouses_cannot_use_by_monitor:
                    logger.warning(f"所有仓库均不可用，时间: {self.current_time} 。")
                    items_placed -= 1
                    continue
                else:
                    cluster_index = item["cluster_index"]
                    disk_ID = item["disk_ID"]
                    logger.warning(
                        f"No warehouse can place cluster{cluster_index} item {disk_ID} at time {self.current_time}.")
                    items_placed -= 1
                    continue
            self.update_warehouse_state(
                selected_warehouse, item)

            warehouses_allocation_history.append(
                self.warehouses_resource_allocated.copy())
            cannot_use_by_monitor_history.append(
                self.warehouses_cannot_use_by_monitor.copy())

            # 监控违规
            self.monitor_warning_violation(
            )

        logger.info(f"Placed {items_placed} items. Total items: {len(items)}, "
                    f"Disk number: {DataConfig.DISK_NUMBER}, "
                    f"Warehouse number: {self.warehouse_number}")

        # 以下内容为可视化调试
        # self.plotter.plot_resource_allocation_animation(
        #     np.array(warehouses_allocation_history),
        #     self.output_dir, "resource_allocation_animation"
        # )
        # self.plotter.plot_resource_allocation_animation(np.array(warehouses_trace[:DataConfig.DISK_NUMBER]),
        #                                                 self.output_dir, "warehouses_trace_when_placing")
        # with open(os.path.join(self.output_dir, 'cannot_use_by_monitor_history'), 'w') as f:
        #     for time, cannot_use_by_monitor in enumerate(cannot_use_by_monitor_history):
        #         f.write(
        #             f"{time}:"+','.join(map(str, cannot_use_by_monitor)) + '\n')
        # np.savetxt(os.path.join(self.output_dir, 'placed_warehouses'),
        #            np.array(placed_warehouses), delimiter=',', fmt='%d')
        # available_items.to_csv(os.path.join(
        #     self.output_dir, 'available_items'), index=False, header=True)
        # 可视化调试
        return np.array(placed_warehouses), available_items

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

    def monitor_warning_violation(self) -> None:
        """监控SLA违反"""
        for warehouse in range(self.warehouse_number):
            if self.warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue

            if np.any(
                self.warehouses_trace[self.current_time][warehouse] >
                self.warehouses_max[warehouse] *
                    ModelConfig.RESERVATION_RATE_FOR_MONITOR
            ):
                self.violation_time_queues[warehouse].append(
                    self.current_time)
                while len(self.violation_time_queues[warehouse]) >= 2 and self.violation_time_queues[warehouse][-1] - self.violation_time_queues[warehouse][0] >= DataConfig.VIOLATION_WINDOW_SIZE:
                    self.violation_time_queues[warehouse].popleft()
                if len(self.violation_time_queues[warehouse]) >= DataConfig.MAX_VIOLATION_OCCURRENCE:
                    self.warehouses_cannot_use_by_monitor[warehouse] = 1

                self.violation_time_queues[warehouse].append(
                    self.current_time)

    def evaluate_warehouses(self):
        """
        评估仓库
        utilize_trace:每个时间戳的仓库利用率[timestamp][warehouse][dimension]
        is_warehouse_violated:每个仓库当前时刻是否违规[warehouse]
        violation_count_per_warehouse:每个仓库累计违规次数[warehouse]
        current_violation_duration:每个仓库当前连续违规时长[warehouse]
        max_violation_duration:每个仓库历史最大连续违规时长[warehouse]
        all_violation_durations:每个仓库所有连续违规时长的列表[warehouse][n]
        """
        evaluate_warehouses_trace = self.warehouses_trace[DataConfig.DISK_NUMBER:]
        max_violation_duration = np.zeros(
            self.warehouse_number, dtype=int)
        all_violation_durations = [[]
                                   for _ in range(self.warehouse_number)]
        utilize_trace = evaluate_warehouses_trace/self.warehouses_max
        self.plotter.plot_warehouse_trace(
            utilize_trace[:, :, 1], self.output_dir, f"{self.algorithm_name}_warehouse_trace")
        utilize_mean_on_time = np.mean(utilize_trace, axis=0)
        violation_count_warehouse = (
            utilize_trace > 1).any(axis=2).sum(axis=0)
        warehouse_load_std = np.mean(np.std(utilize_trace, axis=1), axis=0)
        warehouse_load_mean = np.mean(np.mean(utilize_trace, axis=1), axis=0)
        warehouse_load_imb = np.where(
            warehouse_load_mean == 0, 0, warehouse_load_std/warehouse_load_mean)
        for i in range(self.warehouse_number):
            series = (utilize_trace > 1).any(axis=2)[:, i]
            padded_series = np.concatenate(([False], series, [False]))
            int_series = padded_series.astype(int)
            diff_series = np.diff(int_series)
            starts = np.where(diff_series == 1)[0]
            ends = np.where(diff_series == -1)[0]
            durations = ends - starts
            all_violation_durations[i].append(durations.tolist())
            if len(durations) > 0:
                max_violation_duration[i] = np.max(durations)
            else:
                max_violation_duration[i] = 0
        imbalance_on_timewindow = self.evaluate_imbalance_in_one_day(
            utilize_trace)
        return (utilize_mean_on_time, violation_count_warehouse, max_violation_duration,
                all_violation_durations, warehouse_load_imb, imbalance_on_timewindow)

    def evaluate_imbalance_in_one_day(self, utilize_trace: np.ndarray) -> float:
        bandwidth_trace = utilize_trace[:, :, 1]
        time_index = pd.date_range(
            start='2025-01-01 00:00:00', periods=len(utilize_trace), freq='5min')
        trace_avg_across_resource = pd.DataFrame(bandwidth_trace, columns=[
            f'warehouse{i}' for i in range(self.warehouse_number)], index=time_index)
        imbalance_on_timewindow = {}
        for windows_length_in_one_day in DataConfig.WINDOWS_LENGTH_IN_ONE_DAY:
            window_means_among_warehouses = trace_avg_across_resource.resample(
                f'{windows_length_in_one_day}').mean()
            daily_cv_among_warehouses = window_means_among_warehouses.resample(
                '1D').std()/window_means_among_warehouses.resample(
                '1D').mean()
            imbalance_among_warehouses = daily_cv_among_warehouses.to_numpy().mean()
            imbalance_on_timewindow[windows_length_in_one_day] = imbalance_among_warehouses
        return imbalance_on_timewindow

    def write_result(self, utilize_mean_on_time, violation_count_warehouse, max_violation_duration, all_violation_durations, warehouse_load_imb, imbalance_on_timewindow):
        """写入结果"""
        result_dir = os.path.join(
            self.output_dir, f"{self.algorithm_name}_result.txt")

        with open(result_dir, "w") as f:
            # 写入配置信息
            # f.write("----------------------Config------------------------\n")
            # f.write(f"warehouse_number:{self.warehouse_number}\n")
            # f.write(f"episodes:{ModelConfig.EPISODES}\n")
            # f.write(
            #     f"evaluate_time_number:{DataConfig.EVALUATE_TIME_NUMBER}\n")
            # f.write(
            #     f"violation_window_size:{DataConfig.VIOLATION_WINDOW_SIZE}\n")
            # f.write(
            #     f"max_violation_occurrence:{DataConfig.MAX_VIOLATION_OCCURRENCE}\n")
            # f.write(
            #     f"reservation_rate_for_monitor:{ModelConfig.RESERVATION_RATE_FOR_MONITOR}\n")
            # f.write(f"disk_number:{DataConfig.DISK_NUMBER}\n")
            # f.write("---------------------------------------------------\n")

            # 写入各项结果
            f.write(
                f"-----------------------utilize_mean_on_time--------------------------\n")
            for warehouse in range(self.warehouse_number):
                f.write(f"warehouse{warehouse}:" + ",".join(map(str,
                        utilize_mean_on_time[warehouse])) + "\n")
            f.write(
                f"-----------------------utilize_mean----------------------------------\n")
            f.write(f"average_utilize_mean:" +
                    ','.join(map(str, np.mean(utilize_mean_on_time, axis=0))) + "\n")
            f.write(
                f"-----------------------warehouse_load------------------------------\n")
            warehouse_load = utilize_mean_on_time*self.warehouses_max
            for warehouse in range(self.warehouse_number):
                f.write(
                    f"warehouse{warehouse}:" + ",".join(map(str, warehouse_load[warehouse])) + "\n")
            f.write(f"average_warehouse_load:" +
                    ','.join(map(str, np.mean(warehouse_load, axis=0))) + "\n")
            f.write(
                f"-----------------------violation_count_warehouse--------------------------\n")
            f.write(",".join(map(str, violation_count_warehouse)) + "\n")
            f.write("average_violation_count_warehouse:" +
                    str(np.mean(violation_count_warehouse)) + "\n")
            f.write(
                "--------------------------max_violation_duration-------------------------\n")
            f.write(",".join(map(str, max_violation_duration)) + "\n")
            f.write("average_max_violation_duration:" +
                    str(np.mean(max_violation_duration)) + "\n")
            f.write(
                "--------------------------all_violation_durations-------------------------\n")
            for warehouse in range(self.warehouse_number):
                f.write(f"warehouse{warehouse}:" + ",".join(map(str,
                        all_violation_durations[warehouse])) + "\n")
            f.write(
                "--------------------------warehouse_load_imb-------------------------\n")
            f.write(
                "capacity:"+str(warehouse_load_imb[0]) + "\nbandwidth:"+str(warehouse_load_imb[1]) + "\n")
            f.write(
                "--------------------------imbalance_on_timewindow-------------------------\n")
            for windows_length_in_one_day in DataConfig.WINDOWS_LENGTH_IN_ONE_DAY:
                f.write(f"{windows_length_in_one_day}:" +
                        str(imbalance_on_timewindow[windows_length_in_one_day]) + "\n")
        logger.info(f"结果已写入 {result_dir}")

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
        line_len = -1
        last_weekday = -1
        current_weekday = -1
        for timestamp in timestamps:
            line_len += 1
            current_weekday = datetime.fromtimestamp(timestamp).weekday()
            if current_weekday == 0 and last_weekday == 6:
                break
            last_weekday = current_weekday
        return line_len

    def _get_circular_trace(self, disk_trace_bandwidth: pd.DataFrame, disk_capacity: int, begin_line: int, trace_len: int) -> np.ndarray:
        """
        return :DataFrame, shape(trace_len, [disk_capacity, bandwidth])
        """
        timestamps = disk_trace_bandwidth["timestamp"].to_numpy()
        last_timestamp = timestamps[-1]
        start_index = -1
        last_weekday = datetime.fromtimestamp(last_timestamp).weekday()
        last_time = datetime.fromtimestamp(last_timestamp).time()
        for index, current_timestamp in enumerate(timestamps):
            if datetime.fromtimestamp(current_timestamp).weekday() == last_weekday and datetime.fromtimestamp(current_timestamp).time() == last_time:
                start_index = index
                break
        circular_bandwidth_trace = disk_trace_bandwidth[begin_line:]["bandwidth"].to_numpy(
        )
        append_trace = disk_trace_bandwidth[start_index:]["bandwidth"].to_numpy(
        )
        if start_index == -1:
            logger.error("the disk trace is not support circular trace")
            return None
        while (len(circular_bandwidth_trace) < trace_len):
            circular_bandwidth_trace = np.concatenate(
                [circular_bandwidth_trace, append_trace], axis=0)
        circular_bandwidth_trace = circular_bandwidth_trace[:trace_len]
        circular_trace = np.column_stack([np.full(
            len(circular_bandwidth_trace), disk_capacity), circular_bandwidth_trace])
        return circular_trace
