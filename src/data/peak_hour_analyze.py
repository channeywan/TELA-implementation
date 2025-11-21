import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
import joblib
from config.settings import DirConfig, DataConfig, WarehouseConfig, TestConfig
from data.loader import DiskDataLoader
from visualization.plotter import TelaPlotter
logger = logging.getLogger(__name__)


def iterate_first_day(timestamps) -> int:
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
    return circular_bandwidth_trace


def analyze_peak_hours(df: pd.DataFrame, disks_trace: dict, window_length: str):
    peak_hour_counts = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int)))
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {window_length}"):
        disk_id = row['disk_ID']
        cluster_index = row['cluster_index']
        week_bandwidth = get_circular_trace(disks_trace[cluster_index][disk_id], iterate_first_day(
            disks_trace[cluster_index][disk_id]["timestamp"]), 288*7)
        if week_bandwidth is None:
            continue
        shifted_bandwidth = np.roll(week_bandwidth, -12)
        week_df = pd.DataFrame({
            'datetime': pd.date_range(start='2023-05-09 00:00:00', periods=len(week_bandwidth), freq='5min'),
            'bandwidth': shifted_bandwidth
        }).set_index('datetime')
        daily_peak_timestamps = week_df['bandwidth'].resample(f'{window_length}').mean().resample('D').apply(
            lambda x: x.idxmax() if (not x.empty and x.max() > x.mean()*1.1) else pd.NaT
        )
        # if daily_peak_timestamps.dropna().empty:
        #     daily_peak_timestamps = week_df['bandwidth'].resample(f'{window_length}').apply(lambda x: x.max() if (not x.empty and x.max() > x.mean()*1.1) else np.nan).resample('D').apply(
        #         lambda x: x.idxmax() if (not x.empty and x.max() > 0) else pd.NaT)
        daily_peak_timestamps = daily_peak_timestamps.dropna()
        peak_hours = daily_peak_timestamps.dt.hour
        for hour in peak_hours:
            peak_hour_counts[cluster_index][disk_id][hour] += 1
    return peak_hour_counts


def peak_hour_frequency():
    items, disks_trace = DiskDataLoader().load_items_and_trace(
        cluster_index_list=DataConfig.CLUSTER_DIR_LIST)
    items = pd.read_csv(os.path.join(DirConfig.PLACEMENT_DIR,
                        "motivation", 'selected_items.csv'))
    peak_hour_counts = analyze_peak_hours(
        items, disks_trace, window_length='6h')
    with open(os.path.join(TestConfig.TEST_DIR, 'peak_hour_counts'), 'w') as f:
        for cluster_index, disk_dict in peak_hour_counts.items():
            for disk_id, hour_dict in disk_dict.items():
                f.write(f"{cluster_index},{disk_id}\n")
        logger.info(
            f"peak_hour_counts saved to {os.path.join(TestConfig.TEST_DIR, 'peak_hour_counts')}")
    all_inner_values = []
    for disk_dict in peak_hour_counts.values():
        for hour_dict in disk_dict.values():
            all_inner_values.append(
                np.max(np.array(list(hour_dict.values())))/np.sum(np.array(list(hour_dict.values()))))
    TelaPlotter().plot_cdf(all_inner_values, save_dir=TestConfig.TEST_DIR,
                           title='Peak_Hour_frequency')


def plot_all_disks_peak_hour_distribution(peak_hour_counts: dict, window_length: str):
    output_dir = os.path.join(
        TestConfig.TEST_DIR, 'peak_hour_distribution', window_length)
    os.makedirs(output_dir, exist_ok=True)
    for cluster_index, disk_dict in peak_hour_counts.items():
        os.makedirs(os.path.join(
            output_dir, f'{cluster_index}'), exist_ok=True)
        for disk_id, hour_dict in disk_dict.items():
            counts_list = [hour_dict.get(hour, 0) for hour in range(24)]
            plt.bar(range(24), counts_list)
            plt.xlabel('Hour')
            plt.ylabel('Count')
            plt.title(f'{cluster_index} {disk_id} Peak Hour Distribution')
            plt.savefig(os.path.join(output_dir, f'{cluster_index}', f'{disk_id}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
