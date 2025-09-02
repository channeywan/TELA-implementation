from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Dict, Optional
from collections import deque
import numpy as np
import pandas as pd
import logging
import os
import linecache

from data.loader import DiskDataLoader
from data.processor import DiskDataProcessor
from visualization.plotter import DiskPlotter
from config.settings import WarehouseConfig, DataConfig, ModelConfig, DirConfig

logger = logging.getLogger(__name__)


class BaseAlgorithm(ABC):
    """云盘放置算法基类"""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.loader = DiskDataLoader()
        self.processor = DiskDataProcessor()
        self.plotter = DiskPlotter()

    def run(self) -> None:
        """运行算法的主流程"""
        # 初始化统计变量
        utilize_trace_episodes = np.zeros((DataConfig.EVALUATE_TIME_NUMBER, 3))
        violation_count_per_warehouse_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        max_violation_duration_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        episode_all_violation_durations = []
        average_dimension_load_imb_episodes = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        average_warehouse_load_imb_episodes = np.zeros(3)

        # 运行多个episode
        for episode in range(ModelConfig.EPISODES):
            logger.info(f"Episode {episode+1}/{ModelConfig.EPISODES}")

            # 加载和预处理数据（子类实现）
            items = self.load_and_preprocess_items()

            # 放置云盘
            warehouses_trace, _ = self.place_item(items)

            # 评估结果
            utilize_trace, violation_count_per_warehouse, max_violation_duration, \
                all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb = \
                self.evaluate_warehouses(warehouses_trace)

            utilize_trace_episodes += utilize_trace
            violation_count_per_warehouse_episodes += violation_count_per_warehouse
            max_violation_duration_episodes += max_violation_duration
            episode_all_violation_durations.append(all_violation_durations)
            average_dimension_load_imb_episodes += average_dimension_load_imb
            average_warehouse_load_imb_episodes += average_warehouse_load_imb

        # 计算平均值
        utilize_trace_episodes /= ModelConfig.EPISODES
        violation_count_per_warehouse_episodes /= ModelConfig.EPISODES
        max_violation_duration_episodes /= ModelConfig.EPISODES
        average_dimension_load_imb_episodes /= ModelConfig.EPISODES
        average_warehouse_load_imb_episodes /= ModelConfig.EPISODES

        # 写入结果
        self.write_result(
            utilize_trace_episodes, violation_count_per_warehouse_episodes,
            max_violation_duration_episodes, episode_all_violation_durations,
            average_dimension_load_imb_episodes, average_warehouse_load_imb_episodes
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

    def place_item(self, items: pd.DataFrame) -> np.ndarray:
        """放置云盘的通用流程"""
        logger.info("开始模拟放置云盘")
        disk_number = DataConfig.DISK_NUMBER
        if disk_number > len(items):
            disk_number = len(items)
        items_placed = 0
        warehouses_trace = np.zeros(
            (DataConfig.EVALUATE_TIME_NUMBER, WarehouseConfig.WAREHOUSE_NUMBER, 3))
        warehouses_resource_allocated = np.zeros(
            (WarehouseConfig.WAREHOUSE_NUMBER, 3))
        warehouses_cannot_use_by_monitor = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER)
        violation_time_queues = [deque(maxlen=DataConfig.VIOLATION_QUEUE_LENGTH)
                                 for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        additional_state = self.initialize_additional_state()
        available_items = items.sample(n=disk_number, replace=False)
        placed_warehouses = np.zeros(disk_number, dtype=int)
        for _, item in available_items.iterrows():
            # 选择云盘
            items_placed += 1

            # 选择仓库
            selected_warehouse = self.select_warehouse(
                item, warehouses_resource_allocated, warehouses_cannot_use_by_monitor, additional_state
            )
            placed_warehouses[items_placed-1] = selected_warehouse
            if selected_warehouse == -1:
                if 0 not in warehouses_cannot_use_by_monitor:
                    break
                else:
                    cluster_index = item["cluster_index"]
                    disk_ID = item["disk_ID"]
                    logger.warning(
                        f"No warehouse can place cluster{cluster_index} item {disk_ID} at No.{items_placed}.")
                    continue

            # 更新仓库状态
            self.update_warehouse_state(
                selected_warehouse, item, warehouses_resource_allocated,
                warehouses_trace, items_placed, additional_state
            )

            # 监控违规
            self.monitor_warning_violation(
                warehouses_trace, warehouses_cannot_use_by_monitor,
                violation_time_queues, items_placed
            )

            if items_placed % 1000 == 0:
                logger.info(f"已模拟放置 {items_placed} 云盘.")

        logger.info(f"Placed {items_placed} items. Total items: {len(items)}, "
                    f"Disk number: {DataConfig.DISK_NUMBER}, "
                    f"Warehouse number: {WarehouseConfig.WAREHOUSE_NUMBER}")

        return warehouses_trace, placed_warehouses

    def _select_item(self, available_items: pd.DataFrame) -> pd.Series:
        """选择一个物品（可以是随机或其他策略）"""
        return available_items.sample().iloc[0]

    def additional_state_update(self, selected_warehouse: int, item: pd.Series,
                                warehouses_resource_allocated: np.ndarray,
                                warehouses_trace: np.ndarray,
                                current_time: int,
                                additional_state: Any):
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
            trace_dir, delimiter=',', usecols=(2, 4), dtype=int)
        read_bandwidth_trace = disk_trace_bandwidth[:, 0]
        write_bandwidth_trace = disk_trace_bandwidth[:, 1]
        capacity_trace = np.full(len(read_bandwidth_trace), disk_capacity)
        update_pack = np.column_stack(
            (capacity_trace, read_bandwidth_trace, write_bandwidth_trace))

        warehouses_trace[current_time:DataConfig.EVALUATE_TIME_NUMBER, selected_warehouse,
                         :] += update_pack[current_time:DataConfig.EVALUATE_TIME_NUMBER, :]
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

                if len(warehouse_violation_time_queues[warehouse]) == DataConfig.VIOLATION_QUEUE_LENGTH:
                    oldest_violation_time = warehouse_violation_time_queues[warehouse].popleft(
                    )
                    if current_time - oldest_violation_time <= DataConfig.MIN_VIOLATION_TIME_WINDOW:
                        warehouses_cannot_use_by_monitor[warehouse] = 1

                warehouse_violation_time_queues[warehouse].append(current_time)

    def evaluate_warehouses(self, warehouses_trace: np.ndarray) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        评估仓库
        utilize_trace:每个时间戳的平均仓库利用率[timestamp][dimension]
        is_warehouse_violated:每个仓库当前时刻是否违规[warehouse]
        violation_count_per_warehouse:每个仓库累计违规次数[warehouse]
        current_violation_duration:每个仓库当前连续违规时长[warehouse]
        max_violation_duration:每个仓库历史最大连续违规时长[warehouse]
        all_violation_durations:每个仓库所有连续违规时长的列表[warehouse][n]
        """
        utilize_trace = np.zeros((DataConfig.EVALUATE_TIME_NUMBER, 3))
        violation_count_per_warehouse = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER, dtype=int)
        current_violation_duration = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER, dtype=int)
        max_violation_duration = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER, dtype=int)
        all_violation_durations = [[]
                                   for _ in range(WarehouseConfig.WAREHOUSE_NUMBER)]
        average_dimension_load_imb = np.zeros(
            WarehouseConfig.WAREHOUSE_NUMBER, dtype=float)
        average_warehouse_load_imb = np.zeros(3, dtype=float)
        warehouses_max_ndarray = np.array(WarehouseConfig.WAREHOUSE_MAX).T

        for time_idx in range(DataConfig.EVALUATE_TIME_NUMBER):
            # 计算利用率
            warehouses_trace_now = warehouses_trace[time_idx]

            warehouses_utilize_now = warehouses_trace_now / warehouses_max_ndarray
            utilize_trace[time_idx] = np.mean(warehouses_utilize_now, axis=0)

            # 计算SLA违反
            is_warehouse_violated = np.any(
                warehouses_utilize_now >= 1, axis=1).astype(int)

            violation_mask = (is_warehouse_violated == 1)
            violation_count_per_warehouse[violation_mask] += 1
            current_violation_duration[violation_mask] += 1

            end_of_vioation_mask = (~violation_mask) & (
                current_violation_duration > 0)
            for warehouse in np.where(end_of_vioation_mask)[0]:
                duration = current_violation_duration[warehouse]
                all_violation_durations[warehouse].append(duration)
                if duration > max_violation_duration[warehouse]:
                    max_violation_duration[warehouse] = duration
                current_violation_duration[warehouse] = 0

            # 计算负载不均衡度
            warehouse_utilize_var = np.var(warehouses_utilize_now, axis=0)
            warehouse_utilize_mean = np.mean(warehouses_utilize_now, axis=0)
            warehouse_utilize_std = np.std(warehouses_utilize_now, axis=0)
            warehouse_load_imb = np.where(
                warehouse_utilize_mean == 0, 0, warehouse_utilize_std / warehouse_utilize_mean)

            dimension_utilize_var = np.var(warehouses_utilize_now, axis=1)
            dimension_utilize_mean = np.mean(warehouses_utilize_now, axis=1)
            dimension_utilize_std = np.std(warehouses_utilize_now, axis=1)
            dimension_load_imb = np.where(
                dimension_utilize_mean == 0, 0, dimension_utilize_std / dimension_utilize_mean)

            # 累加负载不均衡度
            average_dimension_load_imb += dimension_load_imb
            average_warehouse_load_imb += warehouse_load_imb

        # 计算平均值
        average_dimension_load_imb /= DataConfig.EVALUATE_TIME_NUMBER
        average_warehouse_load_imb /= DataConfig.EVALUATE_TIME_NUMBER

        final_violation_mask = (current_violation_duration > 0)
        for warehouse in np.where(final_violation_mask)[0]:
            duration = current_violation_duration[warehouse]
            all_violation_durations[warehouse].append(duration)
            if duration > max_violation_duration[warehouse]:
                max_violation_duration[warehouse] = duration

        return (utilize_trace, violation_count_per_warehouse, max_violation_duration,
                all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb)

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

    def write_result(self, utilize_trace, violation_count_per_warehouse, max_violation_duration,
                     all_violation_durations, average_dimension_load_imb, average_warehouse_load_imb):
        """写入结果"""
        DirConfig.ensure_dirs()
        result_dir = os.path.join(getattr(
            DirConfig, f"{self.algorithm_name}_DIR"), f"{self.algorithm_name}_result.txt")

        with open(result_dir, "w") as f:
            # 写入配置信息
            f.write("----------------------Config------------------------\n")
            f.write(f"warehouse_number:{WarehouseConfig.WAREHOUSE_NUMBER}\n")
            f.write(f"episodes:{ModelConfig.EPISODES}\n")
            f.write(
                f"evaluate_time_number:{DataConfig.EVALUATE_TIME_NUMBER}\n")
            f.write(
                f"violation_queue_length:{DataConfig.VIOLATION_QUEUE_LENGTH}\n")
            f.write(
                f"min_violation_time_window:{DataConfig.MIN_VIOLATION_TIME_WINDOW}\n")
            f.write(f"cluster_index_list:{DataConfig.CLUSTER_INDEX_LIST}\n")
            f.write(
                f"reservation_rate_for_monitor:{ModelConfig.RESERVATION_RATE_FOR_MONITOR}\n")
            f.write(f"episode_remain:{DataConfig.DISK_NUMBER}\n")
            f.write("---------------------------------------------------\n")

            # 写入各项结果
            sections = [
                ("utilize_trace", utilize_trace),
                ("violation_count_per_warehouse", violation_count_per_warehouse),
                ("max_violation_duration", max_violation_duration),
                ("average_dimension_load_imb", average_dimension_load_imb),
                ("average_warehouse_load_imb", average_warehouse_load_imb)
            ]

            for section_name, data in sections:
                f.write(
                    f"----------------------{section_name}------------------------\n")
                if section_name == "utilize_trace":
                    for timestamp in range(len(data)):
                        f.write(",".join(map(str, data[timestamp])) + "\n")
                else:
                    f.write(",".join(map(str, data)) + "\n")
                f.write("---------------------------------------------------\n")

            # 写入所有违规持续时间
            f.write(
                "----------------------all_violation_durations------------------------\n")
            for episode in range(len(all_violation_durations)):
                for warehouse in range(WarehouseConfig.WAREHOUSE_NUMBER):
                    f.write(f"episode{episode+1},warehouse{warehouse}," +
                            ",".join(map(str, all_violation_durations[episode][warehouse])) + "\n")
            f.write("---------------------------------------------------\n")

        logger.info(f"结果已写入 {result_dir}")
