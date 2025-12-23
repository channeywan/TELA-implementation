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
from data.utils import sort_test_items
import pickle
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
        self.violation_window_size = 1600
        self.max_violation_occurrence = 650
        self.warehouses_trace = np.zeros(
            (DataConfig.EVALUATE_TIME_NUMBER+DataConfig.DISK_NUMBER, self.warehouse_number, self.resource_dimension_number))
        self.warehouses_resource_allocated = np.zeros(
            (self.warehouse_number, self.resource_dimension_number))
        self.warehouses_cannot_use_by_monitor = np.zeros(
            self.warehouse_number)
        self.violation_time_queues = [deque(maxlen=self.max_violation_occurrence)
                                      for _ in range(self.warehouse_number)]
        self.disks_trace = self.loader.load_all_trace()
        # self.selected_items = self.loader.load_selected_items()
        # self.train_items, self.test_items = train_test_split(
        #     self.selected_items, test_size=0.3, random_state=42)
        # self.train_items = self.loader.load_items(DataConfig.CLUSTER_DIR_LIST)
        # self.test_items = self.train_items.copy()
        self.train_items = None
        self.test_items = None
        self.place_items_number_list = []
        self.peak_to_average_ratio = None
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
            utilize_mean_across_time, violation_count_warehouse, max_violation_duration, all_violation_durations, load_imbalance_across_warehouses, imbalance_on_timewindow = self.evaluate_warehouses()

        self.write_result(utilize_mean_across_time, violation_count_warehouse, max_violation_duration,
                          all_violation_durations, load_imbalance_across_warehouses, imbalance_on_timewindow)

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
        trace_vector_history = []
        general_warehouse_capacity_history = []
        # 可视化
        available_items = items.sample(
            n=disk_number, replace=False, random_state=42)
        if self.algorithm_name == "TIDAL":
            available_items.to_csv(os.path.join(
                DirConfig.INTERMEDIATE_DIR, f'available_items_{self.algorithm_name}_{self.trace_vector_interval}h.csv'), index=False, header=True)
        if self.algorithm_name == "RoundRobin":
            available_items = sort_test_items(available_items)
        placed_warehouses = []
        imbalance_across_timewindow = {}
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

            # warehouses_allocation_history.append(
            #     self.warehouses_resource_allocated.copy())
            # cannot_use_by_monitor_history.append(
            #     self.warehouses_cannot_use_by_monitor.copy())
            # if self.algorithm_name == "TIDAL":
            #     trace_vector_history.append(self.warehouse_trace_vector.copy())
            #     general_warehouse_capacity_history.append(
            #         self.general_warehouse_capacity.copy())
            # 监控违规
            self.monitor_warning_violation(
            )
            if items_placed in self.place_items_number_list:
                if items_placed == self.place_items_number_list[0]:
                    with open(os.path.join(self.output_dir, 'violation_count.txt'), 'a') as f:
                        f.write("\n")
                evaluate_warehouses_trace = self.warehouses_trace[DataConfig.DISK_NUMBER:]
                utilize_trace = evaluate_warehouses_trace/self.warehouses_max
                bandwidth_utilize_trace = utilize_trace[:, :, 1]
                violation_count_warehouse = (
                    bandwidth_utilize_trace >= 1).sum(axis=0)
                violation_count = np.mean(violation_count_warehouse)
                with open(os.path.join(self.output_dir, 'violation_count.txt'), 'a') as f:
                    f.write(f"{violation_count},")
            if items_placed in self.place_items_number_list:
                evaluate_warehouses_trace = self.warehouses_trace[DataConfig.DISK_NUMBER:]
                utilize_trace = evaluate_warehouses_trace/self.warehouses_max
                imbalance_across_timewindow[items_placed] = self.evaluate_imbalance_in_one_day(
                    utilize_trace)
        file_dir = os.path.join(
            DirConfig.TEMP_DIR, f'imbalance_across_time_{self.algorithm_name}')
        with open(file_dir, 'wb') as f:
            pickle.dump(imbalance_across_timewindow, f)

        logger.info(f"Placed {items_placed} items."
                    f"Disk number: {DataConfig.DISK_NUMBER}, "
                    f"Warehouse number: {self.warehouse_number}")

        # 以下内容为可视化调试
        # if self.algorithm_name == "TIDAL":
        #     self.plotter.plot_resource_allocation_animation(np.array(trace_vector_history), np.array(
        #         general_warehouse_capacity_history), self.output_dir, "trace_vector_animation")
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
        # # 可视化调试
        return np.array(placed_warehouses), available_items

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
        # self.warehouses_cannot_use_by_monitor = np.sum(
        #     self.warehouses_trace[:DataConfig.DISK_NUMBER, :, 1] > self.warehouses_max[:, 1], axis=0) >= DataConfig.MAX_OVERLOAD_LIFETIME_OCCURRENCE
        for warehouse in range(self.warehouse_number):
            if self.warehouses_cannot_use_by_monitor[warehouse] == 1:
                continue

            if np.any(
                self.warehouses_trace[self.current_time][warehouse] >
                self.warehouses_max[warehouse]
            ):
                self.violation_time_queues[warehouse].append(
                    self.current_time)
                while len(self.violation_time_queues[warehouse]) >= 2 and self.violation_time_queues[warehouse][-1] - self.violation_time_queues[warehouse][0] >= self.violation_window_size:
                    self.violation_time_queues[warehouse].popleft()
                if len(self.violation_time_queues[warehouse]) >= self.max_violation_occurrence:
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
        bandwidth_utilize_trace = utilize_trace[:, :, 1]
        # utilize_trace = np.where(utilize_trace > 1, 1, utilize_trace)
        self.plotter.plot_warehouse_trace(
            bandwidth_utilize_trace, self.output_dir, f"{self.algorithm_name}_warehouse_trace")
        utilize_mean_across_time = np.mean(utilize_trace, axis=0)
        violation_count_warehouse = (
            bandwidth_utilize_trace >= 1).sum(axis=0)
        load_std_across_warehouses = np.mean(
            np.std(utilize_trace, axis=1), axis=0)
        load_mean_across_warehouses = np.mean(
            np.mean(utilize_trace, axis=1), axis=0)
        load_imbalance_across_warehouses = np.where(
            load_mean_across_warehouses == 0, 0, load_std_across_warehouses/load_mean_across_warehouses)
        for i in range(self.warehouse_number):
            series = (bandwidth_utilize_trace >= 1)[:, i]
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
        imbalance_across_timewindow = self.evaluate_imbalance_in_one_day(
            utilize_trace)
        return (utilize_mean_across_time, violation_count_warehouse, max_violation_duration,
                all_violation_durations, load_imbalance_across_warehouses, imbalance_across_timewindow)

    def evaluate_imbalance_in_one_day(self, utilize_trace: np.ndarray) -> dict:
        bandwidth_trace = utilize_trace[:, :, 1]
        peak_bandwidth=np.max(bandwidth_trace, axis=0)
        average_bandwidth=np.mean(bandwidth_trace, axis=0)
        self.peak_to_average_ratio=peak_bandwidth/average_bandwidth
        time_index = pd.date_range(
            start='2025-01-01 00:00:00', periods=len(utilize_trace), freq='5min')
        trace_avg_across_resource = pd.DataFrame(bandwidth_trace, columns=[
            f'warehouse{i}' for i in range(self.warehouse_number)], index=time_index)
        imbalance_across_timewindow = {}
        for windows_length_in_one_day in DataConfig.WINDOWS_LENGTH_IN_ONE_DAY:
            window_means_among_warehouses = trace_avg_across_resource.resample(
                f'{windows_length_in_one_day}').mean()
            daily_cv_among_warehouses = window_means_among_warehouses.resample(
                '1D').std()/window_means_among_warehouses.resample(
                '1D').mean()
            daily_std_among_warehouses = window_means_among_warehouses.resample(
                '1D').std()
            daily_var_among_warehouses = window_means_among_warehouses.resample(
                '1D').var()
            imbalance_across_timewindow[windows_length_in_one_day] = {}
            imbalance_across_timewindow[windows_length_in_one_day]["cv"] = np.array(
                daily_cv_among_warehouses.mean(axis=0))
            imbalance_across_timewindow[windows_length_in_one_day]["std"] = np.array(
                daily_std_among_warehouses.mean(axis=0))
            imbalance_across_timewindow[windows_length_in_one_day]["var"] = np.array(
                daily_var_among_warehouses.mean(axis=0))
        return imbalance_across_timewindow

    def write_result(self, utilize_mean_across_time, violation_count_warehouse, max_violation_duration, all_violation_durations, load_imbalance_across_warehouses, imbalance_across_timewindow):
        """写入结果"""
        result_dir = os.path.join(
            self.output_dir, f"{self.algorithm_name}_result.txt")

        with open(result_dir, "a") as f:
            f.write(
                f"-----------------------warehouse_load------------------------------\n")
            warehouse_load = utilize_mean_across_time*self.warehouses_max
            for warehouse in range(self.warehouse_number):
                f.write(
                    f"warehouse{warehouse}:" + ",".join(map(str, warehouse_load[warehouse])) + "\n")
            f.write(f"average_warehouse_load:" +
                    ','.join(map(str, np.mean(warehouse_load, axis=0))) + "\n")
            f.write(
                f"-----------------------utilize_mean_across_time--------------------------\n")
            for warehouse in range(self.warehouse_number):
                f.write(f"warehouse{warehouse}:" + ",".join(map(str,
                        utilize_mean_across_time[warehouse])) + "\n")
            f.write(
                f"-----------------------violation_count_warehouse--------------------------\n")
            f.write(",".join(map(str, violation_count_warehouse)) + "\n")
            f.write("average_violation_count_warehouse:" +
                    str(np.mean(violation_count_warehouse)) + "\n")
            if self.algorithm_name == "TIDAL" or self.algorithm_name == "TELA":
                self.overload_percentage=np.mean(violation_count_warehouse)/2016
            f.write(
                "--------------------------max_violation_duration-------------------------\n")
            f.write(",".join(map(str, max_violation_duration)) + "\n")
            f.write("average_max_violation_duration:" +
                    str(np.mean(max_violation_duration)) + "\n")
            f.write(
                "--------------------------all_violation_durations-------------------------\n")
            all_warehouse_violation_durations = []
            for warehouse in range(self.warehouse_number):
                all_warehouse_violation_durations.extend(
                    list(all_violation_durations[warehouse][0]))
                f.write(f"warehouse{warehouse}:" + ",".join(map(str,
                        all_violation_durations[warehouse][0])) + "\n")
            if self.algorithm_name == "TIDAL" or self.algorithm_name == "TELA":
                self.overload_duration=all_warehouse_violation_durations
            f.write(f"all_warehouse_violation_duration:" +
                    ",".join(map(str, all_warehouse_violation_durations)) + "\n")
            f.write(
                "--------------------------load_imbalance_across_warehouses-------------------------\n")
            f.write(
                "capacity:"+str(load_imbalance_across_warehouses[0]) + "\nbandwidth:"+str(load_imbalance_across_warehouses[1]) + "\n")
            if self.algorithm_name == "TIDAL" or self.algorithm_name == "TELA":
                self.space_imbalance=load_imbalance_across_warehouses[1]
            f.write(
                "--------------------------peak_to_average_ratio-------------------------\n")
            f.write(f"peak_to_average_ratio:" + str(np.mean(self.peak_to_average_ratio))+",".join(map(str, self.peak_to_average_ratio)) + "\n")
            f.write(
                "--------------------------imbalance_across_timewindow-------------------------\n")
            for windows_length_in_one_day in DataConfig.WINDOWS_LENGTH_IN_ONE_DAY:
                cv_among_warehouses = imbalance_across_timewindow[
                    windows_length_in_one_day]["cv"]
                f.write(f"cv_{windows_length_in_one_day}:" +
                        str(cv_among_warehouses.mean()) + str(cv_among_warehouses.tolist()) + "\n")
            if self.algorithm_name == "TIDAL" or self.algorithm_name == "TELA":
                self.time_imbalance=imbalance_across_timewindow["2h"]["cv"].mean()
            f.write(
                f"-----------------------utilize_mean----------------------------------\n")
            f.write(f"average_utilize_mean:" +
                    ','.join(map(str, np.mean(utilize_mean_across_time, axis=0))) + "\n")
            self.bandwidth_utilization = np.mean(
                utilize_mean_across_time, axis=0)[1]
            self.capacity_utilization = np.mean(
                utilize_mean_across_time, axis=0)[0]
            f.write(
                f"-----------------------config----------------------------------\n")
            f.write(f"warehouse_config: capacity:"+str(WarehouseConfig.CAPACITY_RATIO.tolist()
                                                       )+" bandwidth:"+str(WarehouseConfig.BANDWIDTH_RATIO.tolist())+"\n")
            f.write(f"place_items_number:" +
                    str(self.place_items_number_list) + "\n")
            if self.algorithm_name == "TIDAL" or self.algorithm_name == "TIDAL_threshold":
                f.write(
                    f"trace_vector_interval:{self.trace_vector_interval}\n")
                f.write(f"unknown_threshold:{self.unknown_threshold}\n")
                f.write(f"num_unknown:{self.num_unknown}\n")
                f.write(
                    f"min_bandwidth_warehouse_num:{self.min_bandwidth_warehouse_num}\n")
                f.write(f"noise_ratio:{self.noise_ratio}\n")
                f.write(f"target:{self.target}\n")
                f.write(f"random_state:{self.random_state}\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")

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
