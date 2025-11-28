import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
from config.settings import DataConfig, TestConfig, WarehouseConfig, DirConfig
from collections import Counter
from data.loader import DiskDataLoader
import logging
logger = logging.getLogger(__name__)


def select_disks_for_warehouse_trace():
    loader = DiskDataLoader()
    items = loader.load_items(
        DataConfig.CLUSTER_DIR_LIST)

    # for idx, item in tqdm(items.iterrows(), total=len(items), desc="Pre-loading"):
    #     cluster_index = item["cluster_index"]
    #     disk_id = item["disk_ID"]
    #     raw_trace = disks_trace[cluster_index][disk_id]
    #     first_day_line = plotter.iterate_first_day(raw_trace["timestamp"])
    #     circular_trace = plotter.get_circular_trace(
    #         raw_trace, first_day_line, DataConfig.EVALUATE_TIME_NUMBER
    #     )

    #     if circular_trace is not None and len(circular_trace) == DataConfig.EVALUATE_TIME_NUMBER:
    #         valid_traces.append(circular_trace)
    #         valid_items_indices.append(idx)

    # candidate_matrix = np.array(valid_traces)
    # joblib.dump(candidate_matrix, os.path.join(
    #     TestConfig.TEST_DIR, 'candidate_matrix'))
    valid_items_indices = [i for i in range(len(items))]

    candidate_matrix = joblib.load(os.path.join(
        TestConfig.TEST_DIR, 'candidate_matrix'))
    candidate_count = len(valid_items_indices)

    available_mask = np.ones(candidate_count, dtype=bool)

    warehouse_trace = np.zeros(DataConfig.EVALUATE_TIME_NUMBER)
    selected_items_list = []
    target_disk_num = DataConfig.DISK_NUMBER*4

    for _ in tqdm(range(target_disk_num), desc="Selecting disks (Vectorized)"):
        if not np.any(available_mask):
            break
        current_candidates = candidate_matrix[available_mask]
        combined_traces = current_candidates + warehouse_trace
        means = np.mean(combined_traces, axis=1)
        stds = np.std(combined_traces, axis=1)
        cvs = stds / means
        best_relative_idx = np.argmin(cvs)
        best_absolute_idx = np.where(available_mask)[0][best_relative_idx]
        best_trace = candidate_matrix[best_absolute_idx]
        warehouse_trace += best_trace
        available_mask[best_absolute_idx] = False
        original_df_index = valid_items_indices[best_absolute_idx]
        selected_items_list.append(items.loc[original_df_index])
    selected_items = pd.DataFrame(selected_items_list)
    selected_items.to_csv(os.path.join(
        TestConfig.TEST_DIR, 'selected_items.csv'), index=False)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(warehouse_trace, label='Warehouse Trace')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bandwidth')
    ax.set_title('Warehouse Trace')
    x_ticks = [288*day for day in range(8)]
    x_tick_labels = [f'day{day}' for day in range(8)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    fig.savefig(os.path.join(TestConfig.TEST_DIR, 'warehouse_trace.png'))
    plt.close(fig)
    logger.info(
        f"Combined warehouse trace saved to {os.path.join(TestConfig.TEST_DIR, 'warehouse_trace.png')}")


def iterate_first_day(timestamps: pd.Series) -> int:
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


def get_circular_trace(disk_trace_bandwidth: pd.DataFrame, begin_line: int, trace_len: int) -> np.ndarray:
    """
    return :DataFrame, shape(trace_len, [bandwidth])
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
        return None
    start_index = matches[0]
    base_trace = disk_trace_bandwidth.iloc[begin_line:]["bandwidth"].to_numpy(
    )
    append_trace = disk_trace_bandwidth.iloc[start_index:]["bandwidth"].to_numpy(
    )
    if len(append_trace) == 0:
        return None
    len_base = len(base_trace)

    if len_base >= trace_len:
        circular_bandwidth_trace = base_trace[:trace_len]
    else:
        len_needed = trace_len - len_base
        num_repeats = int(np.ceil(len_needed / len(append_trace)))
        padding = np.tile(append_trace, num_repeats)[:len_needed]
        circular_bandwidth_trace = np.concatenate([base_trace, padding])
    return circular_bandwidth_trace.copy()


class PeakValleyWindowsDistribution:
    def __init__(self):
        self.windows_length_in_one_day = 6
        self.loader = DiskDataLoader()
        self.items = self.loader.load_selected_items()
        self.disks_trace = self.loader.load_all_trace()
        # self.items, self.disks_trace = self.loader.load_items_and_trace(
        #     cluster_index_list=DataConfig.CLUSTER_DIR_LIST)

    def calculate_peak_valley_windows_distribution(self):
        save_dir = os.path.join(DirConfig.TEMP_DIR, 'peak_valley_analyze')
        os.makedirs(save_dir, exist_ok=True)
        total_peak_counter = np.zeros(
            (7, int(24/self.windows_length_in_one_day)))
        total_valley_counter = np.zeros(
            (7, int(24/self.windows_length_in_one_day)))
        for _, item in tqdm(self.items.iterrows(), total=len(self.items), desc="Calculating peak and valley windows distribution"):
            disk_id = item["disk_ID"]
            cluster_index = item["cluster_index"]
            peak_hour_counts, valley_hour_counts = self.get_certain_disk_peak_valley_week(
                disk_id, cluster_index)
            for i in range(7):
                total_peak_counter[i][int(
                    peak_hour_counts[i]/self.windows_length_in_one_day)] += 1
                total_valley_counter[i][int(
                    valley_hour_counts[i]/self.windows_length_in_one_day)] += 1
        total_peak_counter /= np.sum(total_peak_counter, axis=1, keepdims=True)
        total_valley_counter /= np.sum(total_valley_counter,
                                       axis=1, keepdims=True)
        np.savetxt(os.path.join(save_dir,
                   "peak_windows_distribution.txt"), total_peak_counter*100, delimiter=',')
        np.savetxt(os.path.join(save_dir, "valley_windows_distribution.txt"),
                   total_valley_counter*100, delimiter=',')
        logger.info(
            f"Peak and valley windows distribution saved to {os.path.join(save_dir, 'peak_windows_distribution.txt')} and {os.path.join(save_dir, 'valley_windows_distribution.txt')}")
        return total_peak_counter*100, total_valley_counter*100
        return

    def get_certain_disk_peak_valley_week(self, disk_id: str, cluster_index: int):
        disk_trace = self.disks_trace[cluster_index][disk_id]
        first_day_line = iterate_first_day(disk_trace["timestamp"])
        circular_trace = get_circular_trace(
            disk_trace, first_day_line, DataConfig.EVALUATE_TIME_NUMBER)
        time_index = pd.date_range(
            start='2025-01-01 00:00:00', periods=DataConfig.EVALUATE_TIME_NUMBER, freq='5min')
        circular_trace = pd.Series(circular_trace, index=time_index)
        daily_windows_distribution = circular_trace.resample(
            f"{self.windows_length_in_one_day}h").max()
        weekly_peak_distribution = daily_windows_distribution.resample(
            '1D').apply(lambda x: x.idxmax())
        weekly_valley_distribution = daily_windows_distribution.resample(
            '1D').apply(lambda x: x.idxmin())
        weekly_peak_distribution = weekly_peak_distribution.dt.hour.to_list()
        weekly_valley_distribution = weekly_valley_distribution.dt.hour.to_list()
        return weekly_peak_distribution, weekly_valley_distribution
