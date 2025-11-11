import csv
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from config.settings import DirConfig, WarehouseConfig,  ModelConfig, DataConfig
import pandas as pd
logger = logging.getLogger(__name__)


class DiskDataProcessor:

    def __init__(self):
        self.bandwidth_line = ModelConfig.BANDWIDTH_LINE

    def process_disk_data(self, trace_dir: str) -> Optional[Dict[str, Any]]:
        """处理单个磁盘的跟踪数据"""
        if not os.path.exists(trace_dir):
            return None
        traces = pd.read_csv(trace_dir, sep=',')
        traces.columns = ["timestamp", "read_IOPS",
                          "write_IOPS", "read_BW", "write_BW"]
        traces["timestamp"] = pd.to_datetime(traces["timestamp"])
        if len(traces) == 0:
            logger.warning(f"trace {trace_dir} is empty")
            return None
        if (traces["timestamp"].diff() != "5min").iloc[1:].any():
            logger.warning(f"trace {trace_dir} 采样时间戳错误")
            return None
        first_day_line = self._iterate_first_day(traces["timestamp"])
        disk_bandwidth = traces["read_BW"].values+traces["write_BW"].values
        slice_bandwidth = disk_bandwidth[first_day_line:]
        timestamp_num = len(slice_bandwidth)
        if timestamp_num < DataConfig.MIN_TIMESTAMP_NUM:
            return None
        disk_bandwidth_avg = np.average(slice_bandwidth)
        disk_bandwidth_peak = np.max(slice_bandwidth)
        bandwidth_zero_num = np.sum(slice_bandwidth == 0)
        bandwidth_zero_ratio = bandwidth_zero_num / timestamp_num
        return {
            'avg_bandwidth': disk_bandwidth_avg,
            'peak_bandwidth': disk_bandwidth_peak,
            'timestamp_num': timestamp_num,
            'bandwidth_zero_ratio': bandwidth_zero_ratio
        }

    def calculate_label(self, disk_data: Dict[str, Any]) -> Tuple[int, float]:
        """计算磁盘的标签（稳定或突发）"""
        burst_lable = -1
        bandwidth_mul = (disk_data['peak_bandwidth'] / disk_data['avg_bandwidth']
                         if disk_data['avg_bandwidth'] != 0 else 0)
        if (bandwidth_mul < self.bandwidth_line):
            burst_lable = 0  # 稳定
        else:
            burst_lable = 1  # 突发
        return burst_lable, bandwidth_mul

    def _iterate_first_day(self, timestamps: pd.Series) -> int:
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


class ClusterInfoInitializer:
    """初始化仓库数据的类"""

    def __init__(self):
        self.cluster_info_root = DirConfig.CLUSTER_INFO_ROOT
        self.trace_root = DirConfig.TRACE_ROOT
        self.warehouse_number = WarehouseConfig.WAREHOUSE_NUMBER
        self.disk_processor = DiskDataProcessor()

    def _ensure_cluster_info_dir(self) -> str:
        """确保仓库目录存在"""
        cluster_dir = self.cluster_info_root
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        return cluster_dir

    def _process_cluster(self, cluster_index: int) -> None:
        """处理单个集群的数据"""
        logger.info(f"正在处理集群 {cluster_index}")
        description_dir = f"{self.trace_root}/153_10077{cluster_index}/describe.csv"
        if not os.path.exists(description_dir):
            logger.error(f"description_dir {description_dir} not exists")
            return
        cluster_info_dir = self._ensure_cluster_info_dir()

        with open(description_dir, "r") as descriptions:
            df = pd.read_csv(descriptions, sep=',', usecols=[
                             "disk_uuid", "vm_alias", "vm_cpu", "vm_mem", "vm_type", "is_vip", "ins_type", "project_name", "buss_name", "disk_alias", "disk_size", "disk_type", "volume_type"])
            df.columns = ["disk_ID", "vm_name", "vm_cpu", "vm_memory", "vm_type", "disk_if_VIP", "ins_type",
                          "project_name", "buss_name", "disk_name", "disk_capacity", "disk_type", "volume_type"]
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True, how='any', axis=0)
            if df.empty:
                logger.warning(f"集群 {cluster_index} 过滤后没有有效数据。")
                return
            new_data_df = df.apply(
                self._process_disk_row, axis=1, args=(cluster_index,))
            df = df.join(new_data_df)
            df.dropna(inplace=True, how='any', axis=0)
            cluster_info_dir = self._ensure_cluster_info_dir()
            output_path = f"{cluster_info_dir}/cluster{cluster_index}"
            df.to_csv(output_path, index=False)

    def _process_disk_row(self, description: pd.Series, cluster_index: int,
                          ) -> pd.Series:
        """处理单个磁盘的数据"""
        trace_dir = f"{self.trace_root}/153_10077{cluster_index}/{description['disk_ID']}.csv"
        disk_data = self.disk_processor.process_disk_data(trace_dir)

        if disk_data is None:
            self._log_missing_trace(description['disk_ID'], cluster_index)
            return pd.Series({
                'avg_bandwidth': np.nan, 'peak_bandwidth': np.nan,
                'timestamp_num': np.nan, 'burst_label': np.nan,
                'bandwidth_mul': np.nan, 'bandwidth_zero_ratio': np.nan
            })

        label, bandwidth_mul = self.disk_processor.calculate_label(
            disk_data)
        disk_data['burst_label'] = label
        disk_data['bandwidth_mul'] = bandwidth_mul
        return pd.Series(disk_data)

    def _log_missing_trace(self, disk_id: str, cluster_index: int) -> None:
        """记录缺失的跟踪文件"""
        with open(f"{self.cluster_info_root}/cluster{cluster_index}_not_exist", "a") as f:
            f.write(f"{disk_id}\n")

    def init_cluster_info(self) -> None:
        logger.info("开始初始化磁盘数据")
        for cluster_index in WarehouseConfig.CLUSTER_DIR_LIST:
            self._process_cluster(cluster_index)
