from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Dict, Optional
from collections import deque
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import pickle
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

    def run(self) -> None:
        """运行算法的主流程"""
        # 初始化统计变量
        utilize_trace_episodes = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, 3))
        violation_count_per_warehouse_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        max_violation_duration_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        episode_all_violation_durations = []
        average_dimension_load_imb_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        average_warehouse_load_imb_episodes = np.zeros(3)
        warehouses_dimension_used_avg_episodes = np.zeros(3)
        # 运行多个episode
        for episode in range(ModelConfig.EPISODES):
            logger.info(f"Episode {episode+1}/{ModelConfig.EPISODES}")

            # 加载和预处理数据（子类实现）
            items = self.load_and_preprocess_items()

            # 放置云盘
            warehouses_trace, _, _ = self.place_item(items)

            # 评估结果
            utilize_mean_on_time, violation_count_per_warehouse, max_violation_duration, \
                all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb, warehouses_dimension_used_avg = \
                self.evaluate_warehouses(
                    warehouses_trace[DataConfig.DISK_NUMBER:])

            utilize_trace_episodes += utilize_mean_on_time
            violation_count_per_warehouse_episodes += violation_count_per_warehouse
            max_violation_duration_episodes += max_violation_duration
            episode_all_violation_durations.append(all_violation_durations)
            average_dimension_load_imb_episodes += average_dimension_load_imb
            average_warehouse_load_imb_episodes += average_warehouse_load_imb
            warehouses_dimension_used_avg_episodes += warehouses_dimension_used_avg
        # 计算平均值
        utilize_trace_episodes /= ModelConfig.EPISODES
        violation_count_per_warehouse_episodes /= ModelConfig.EPISODES
        max_violation_duration_episodes /= ModelConfig.EPISODES
        average_dimension_load_imb_episodes /= ModelConfig.EPISODES
        average_warehouse_load_imb_episodes /= ModelConfig.EPISODES
        warehouses_dimension_used_avg_episodes /= ModelConfig.EPISODES
        # 写入结果
        self.write_result(
            utilize_trace_episodes, violation_count_per_warehouse_episodes,
            max_violation_duration_episodes, episode_all_violation_durations,
            average_dimension_load_imb_episodes, average_warehouse_load_imb_episodes,
            warehouses_dimension_used_avg_episodes
        )

    @abstractmethod
    def load_and_preprocess_items(self):
        """加载和预处理数据（子类实现）"""
        pass

    @abstractmethod
    def select_warehouse(self, item, warehouses_resource_allocated: np.ndarray,
                         warehouses_cannot_use_by_monitor: np.ndarray,
                         additional_state: Any) -> int:
        """选择仓库的策略（子类实现）"""
        pass

    def initialize_additional_state(self):
        """初始化额外的状态数据（子类可重写）"""
        return None

    def place_item(self, items: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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
        warehouses_trace = np.zeros(
            (DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER, WarehouseConfig.WAREHOUSE_NUMBER, 3))
        warehouses_resource_allocated = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, 3))

        warehouses_cannot_use_by_monitor = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        violation_time_queues = [deque(maxlen=DataConfig.MAX_VIOLATION_OCCURRENCE)
                                 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        additional_state = self.initialize_additional_state()
        available_items = items.sample(
            n=disk_number, replace=False, random_state=42)
        placed_warehouses = []
        for current_time, (_, item) in enumerate(available_items.iterrows()):
            items_placed += 1

            selected_warehouse = self.select_warehouse(
                item, warehouses_resource_allocated, warehouses_cannot_use_by_monitor, additional_state
            )
            placed_warehouses.append(selected_warehouse)
            if selected_warehouse == -1:
                if 0 not in warehouses_cannot_use_by_monitor:
                    logger.warning(f"所有仓库均不可用，时间: {current_time} 。")
                    items_placed -= 1
                    continue
                else:
                    cluster_index = item["cluster_index"]
                    disk_ID = item["disk_ID"]
                    logger.warning(
                        f"No warehouse can place cluster{cluster_index} item {disk_ID} at time {current_time}.")
                    items_placed -= 1
                    continue
            self.update_warehouse_state(
                selected_warehouse, item, warehouses_resource_allocated,
                warehouses_trace, items_placed-1, additional_state
            )

            warehouses_allocation_history.append(
                warehouses_resource_allocated.copy())
            cannot_use_by_monitor_history.append(
                warehouses_cannot_use_by_monitor.copy())

            # 监控违规
            self.monitor_warning_violation(
                warehouses_trace, warehouses_cannot_use_by_monitor,
                violation_time_queues, items_placed-1
            )

            if items_placed % 1000 == 0:
                logger.info(f"已模拟放置 {items_placed} 云盘于时间 {current_time}.")

        logger.info(f"Placed {items_placed} items. Total items: {len(items)}, "
                    f"Disk number: {DataConfig.DISK_NUMBER}, "
                    f"Warehouse number: {WarehouseConfig.WAREHOUSE_NUMBER}")

        # 以下内容为可视化调试
        # self.plotter.plot_resource_allocation_animation(
        #     np.array(warehouses_allocation_history),
        #     self.output_dir, "resource_allocation_animation"
        # )
        # self.plotter.plot_resource_allocation_animation(np.array(warehouses_trace[:DataConfig.DISK_NUMBER]),
        #                                                 self.output_dir, "warehouses_trace_when_placing")
        with open(os.path.join(self.output_dir, 'cannot_use_by_monitor_history'), 'w') as f:
            for time, cannot_use_by_monitor in enumerate(cannot_use_by_monitor_history):
                f.write(
                    f"{time}:"+','.join(map(str, cannot_use_by_monitor)) + '\n')
        np.savetxt(os.path.join(self.output_dir, 'placed_warehouses'),
                   np.array(placed_warehouses), delimiter=',', fmt='%d')
        available_items.to_csv(os.path.join(
            self.output_dir, 'available_items'), index=False, header=True)
        # 可视化调试
        return warehouses_trace, placed_warehouses, available_items

    def _select_item(self, available_items: pd.DataFrame) -> pd.Series:
        """选择一个物品（可以是随机或其他策略）"""
        return available_items.sample().iloc[0]

    def additional_state_update(self, selected_warehouse: int, item: pd.Series,
                                warehouses_resource_allocated: np.ndarray,
                                warehouses_trace: np.ndarray,
                                current_time: int,
                                additional_state: Any) -> None:
        """额外状态更新（子类可重写）"""
        pass

    def update_warehouse_state(self, selected_warehouse: int, item: pd.Series,
                               warehouses_resource_allocated: np.ndarray,
                               warehouses_trace: np.ndarray,
                               current_time: int,
                               additional_state: Any) -> None:
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
        warehouses_resource_allocated[selected_warehouse][0] += item["disk_capacity"]

        # 读取trace文件并更新trace
        trace_dir = os.path.join(DirConfig.TRACE_ROOT,
                                 f"20_136090{cluster_index}", f"{disk_ID}")
        disk_trace_bandwidth = np.loadtxt(
            trace_dir, delimiter=',', usecols=(0, 2, 4), dtype=float)
        circular_trace = self._get_circular_trace(
            disk_trace_bandwidth, disk_capacity, DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER)
        if circular_trace is None:
            logger.error(
                f"the disk trace is not support circular trace, cluster_index: {cluster_index}, disk_ID: {disk_ID}")
            return
        warehouses_trace[current_time:DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER, selected_warehouse,
                         :] += circular_trace[current_time:DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER, :]
        self.additional_state_update(
            selected_warehouse, item, warehouses_resource_allocated, warehouses_trace, current_time, additional_state)
        return

    def monitor_warning_violation(self, warehouses_trace: np.ndarray,
                                  warehouses_cannot_use_by_monitor: np.ndarray,
                                  warehouse_violation_time_queues: List[deque],
                                  current_time: int) -> None:
        """监控SLA违反"""
        for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
            if warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue

            # 检查是否超过阈值
            if (warehouses_trace[current_time][warehouse][0] >
                WarehouseConfig.WAREHOUSE_MAX[0][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                warehouses_trace[current_time][warehouse][1] >
                WarehouseConfig.WAREHOUSE_MAX[1][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR or
                warehouses_trace[current_time][warehouse][2] >
                    WarehouseConfig.WAREHOUSE_MAX[2][warehouse] * ModelConfig.RESERVATION_RATE_FOR_MONITOR):
                warehouse_violation_time_queues[warehouse].append(current_time)
                while len(warehouse_violation_time_queues[warehouse]) >= 2 and warehouse_violation_time_queues[warehouse][-1] - warehouse_violation_time_queues[warehouse][0] >= DataConfig.VIOLATION_WINDOW_SIZE:
                    warehouse_violation_time_queues[warehouse].popleft()
                if len(warehouse_violation_time_queues[warehouse]) >= DataConfig.MAX_VIOLATION_OCCURRENCE:
                    warehouses_cannot_use_by_monitor[warehouse] = 1

                warehouse_violation_time_queues[warehouse].append(current_time)

    def evaluate_warehouses(self, warehouses_trace: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        评估仓库
        utilize_trace:每个时间戳的仓库利用率[timestamp][warehouse][dimension]
        is_warehouse_violated:每个仓库当前时刻是否违规[warehouse]
        violation_count_per_warehouse:每个仓库累计违规次数[warehouse]
        current_violation_duration:每个仓库当前连续违规时长[warehouse]
        max_violation_duration:每个仓库历史最大连续违规时长[warehouse]
        all_violation_durations:每个仓库所有连续违规时长的列表[warehouse][n]
        """

        max_violation_duration = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER, dtype=int)
        all_violation_durations = [[]
                                   for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        warehouses_max_ndarray = np.array(WarehouseConfig.WAREHOUSE_MAX).T
        utilize_trace = warehouses_trace/warehouses_max_ndarray
        utilize_mean_on_time = np.mean(utilize_trace, axis=0)
        warehouses_dimension_used_avg = np.mean(
            np.mean(warehouses_trace, axis=1), axis=0)
        violation_count_per_warehouse = (
            utilize_trace > 1).any(axis=2).sum(axis=0)
        dimension_load_std = np.mean(np.std(utilize_trace, axis=2), axis=0)
        warehouse_load_std = np.mean(np.std(utilize_trace, axis=1), axis=0)
        dimension_load_mean = np.mean(np.mean(utilize_trace, axis=2), axis=0)
        warehouse_load_mean = np.mean(np.mean(utilize_trace, axis=1), axis=0)
        average_dimension_load_imb = np.where(
            dimension_load_mean == 0, 0, dimension_load_std/dimension_load_mean)
        average_warehouse_load_imb = np.where(
            warehouse_load_mean == 0, 0, warehouse_load_std/warehouse_load_mean)
        for i in range(WarehouseConfig.WAREHOUSE_NUMBER):
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

        return (utilize_mean_on_time, violation_count_per_warehouse, max_violation_duration,
                all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb, warehouses_dimension_used_avg)

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

    def write_result(self, utilization_mean_on_time, violation_count_per_warehouse, max_violation_duration,
                     all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb, warehouses_dimension_used_avg):
        """写入结果"""
        DirConfig.ensure_dirs()
        result_dir = os.path.join(
            self.output_dir, f"{self.algorithm_name}_result.txt")

        with open(result_dir, "w") as f:
            # 写入配置信息
            f.write("----------------------Config------------------------\n")
            f.write(f"warehouse_number:{WarehouseConfig.WAREHOUSE_NUMBER}\n")
            f.write(f"episodes:{ModelConfig.EPISODES}\n")
            f.write(
                f"evaluate_time_number:{DataConfig.EVALUATE_TIME_NUMBER}\n")
            f.write(
                f"violation_window_size:{DataConfig.VIOLATION_WINDOW_SIZE}\n")
            f.write(
                f"max_violation_occurrence:{DataConfig.MAX_VIOLATION_OCCURRENCE}\n")
            f.write(
                f"reservation_rate_for_monitor:{ModelConfig.RESERVATION_RATE_FOR_MONITOR}\n")
            f.write(f"disk_number:{DataConfig.DISK_NUMBER}\n")
            f.write("---------------------------------------------------\n")

            # 写入各项结果
            sections = [
                ("violation_count_per_warehouse", violation_count_per_warehouse),
                ("max_violation_duration", max_violation_duration),
                ("average_dimension_load_imb", average_dimension_load_imb),
                ("average_warehouse_load_imb", average_warehouse_load_imb),
                ("warehouses_dimension_used_avg", warehouses_dimension_used_avg)
            ]

            for section_name, data in sections:
                f.write(
                    f"----------------------{section_name}------------------------\n")
                f.write(",".join(map(str, data)) + "\n")
                f.write("---------------------------------------------------\n")
            f.write("------------------warehouses_utilization--------------------\n")
            for warehouse, warehouse_util in enumerate(utilization_mean_on_time):
                f.write(f"warehouse{warehouse}: " +
                        ",".join(map(str, warehouse_util))+"\n")
            f.write("-----------------------------------------------------------\n")
            # 写入所有违规持续时间
            f.write(
                "----------------------all_violation_durations------------------------\n")
            for episode in range(len(all_violation_durations)):
                for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                    f.write(f"episode{episode+1},warehouse{warehouse}:" +
                            ",".join(map(str, all_violation_durations[episode][warehouse])) + "\n")
            f.write("---------------------------------------------------\n")

        logger.info(f"结果已写入 {result_dir}")

    def _iterate_first_day(self, timestamps: np.ndarray) -> int:
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

    def _get_circular_trace(self, disk_trace_bandwidth: np.ndarray, disk_capacity: int, trace_len: int) -> np.ndarray:
        """
        计算循环trace
        """
        first_day_line = self._iterate_first_day(disk_trace_bandwidth[:, 0])
        timestamps = disk_trace_bandwidth[:, 0]
        last_timestamp = timestamps[-1]
        start_index = -1
        last_weekday = datetime.fromtimestamp(last_timestamp).weekday()
        last_time = datetime.fromtimestamp(last_timestamp).time()
        for index, current_timestamp in enumerate(timestamps):
            if datetime.fromtimestamp(current_timestamp).weekday() == last_weekday and datetime.fromtimestamp(current_timestamp).time() == last_time:
                start_index = index
                break
        circular_bandwidth_trace = disk_trace_bandwidth[first_day_line:, 1:]
        if start_index == -1:
            logger.error("the disk trace is not support circular trace")
            return None
        while (len(circular_bandwidth_trace) < trace_len):
            circular_bandwidth_trace = np.concatenate(
                (circular_bandwidth_trace, disk_trace_bandwidth[start_index:, 1:]), axis=0)
        circular_trace = np.column_stack(
            (np.full(len(circular_bandwidth_trace), disk_capacity), circular_bandwidth_trace))
        return circular_trace

    def evaluate_imbalance_in_one_day(self, warehouses_trace: np.ndarray, windows_length_in_one_day: int) -> float:
        utilize_trace = warehouses_trace / \
            np.array(WarehouseConfig.WAREHOUSE_MAX).T
        trace_avg_across_warehouses = np.mean(utilize_trace, axis=1)
        trace_avg_across_resource = np.mean(utilize_trace, axis=2)
        time_index = pd.date_range(
            start='2025-01-01 00:00:00', periods=len(utilize_trace), freq='5T')
        trace_avg_across_warehouses = pd.DataFrame(
            trace_avg_across_warehouses, columns=['capacity', 'RBW', 'WBW'], index=time_index)
        trace_avg_across_resource = pd.DataFrame(trace_avg_across_resource, columns=[
            f'warehouse{i}' for i in range(WarehouseConfig.WAREHOUSE_NUMBER)], index=time_index)
        window_means_among_resource_dimension = trace_avg_across_warehouses.resample(
            f'{windows_length_in_one_day}H').mean()
        window_means_among_warehouses = trace_avg_across_resource.resample(
            f'{windows_length_in_one_day}H').mean()
        daily_cv_among_resource_dimension = window_means_among_resource_dimension.resample(
            '1D').std()/window_means_among_resource_dimension.resample(
            '1D').mean()
        daily_cv_among_warehouses = window_means_among_warehouses.resample(
            '1D').std()/window_means_among_warehouses.resample(
            '1D').mean()
        imbalance_among_resource_dimension = daily_cv_among_resource_dimension.mean()
        imbalance_among_warehouses = daily_cv_among_warehouses.mean()
        return imbalance_among_resource_dimension, imbalance_among_warehouses
