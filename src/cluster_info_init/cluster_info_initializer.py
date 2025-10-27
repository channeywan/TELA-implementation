import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import csv
from datetime import datetime
from config.settings import DirConfig, WarehouseConfig, DataConfig, ModelConfig
import pandas as pd
logger = logging.getLogger(__name__)


class DiskDataProcessor:

    def __init__(self):
        self.read_bandwidth_line = ModelConfig.READ_BANDWIDTH_LINE
        self.write_bandwidth_line = ModelConfig.WRITE_BANDWIDTH_LINE

    def process_disk_data(self, trace_dir: str) -> Optional[Dict[str, Any]]:
        """处理单个磁盘的跟踪数据"""
        if not os.path.exists(trace_dir):
            logger.error(f"trace_dir {trace_dir} not exists")
            return None

        columns = ["timestamp", "read_IOPS",
                   "read_BW", "write_IOPS", "write_BW"]
        traces = pd.read_csv(trace_dir, names=columns, usecols=[
            0, 1, 2, 3, 4], sep=',', header=None)
        first_day_line = self._iterate_first_day(traces["timestamp"])
        disk_avg_RBW = np.average(traces[["read_BW"]][first_day_line:])
        disk_avg_WBW = np.average(traces[["write_BW"]][first_day_line:])
        disk_peak_RBW = np.max(traces[["read_BW"]][first_day_line:])
        disk_peak_WBW = np.max(traces[["write_BW"]][first_day_line:])
        processed_line = len(traces[["read_BW"]][first_day_line:])
        return {
            'avg_RBW': disk_avg_RBW,
            'avg_WBW': disk_avg_WBW,
            'peak_RBW': disk_peak_RBW,
            'peak_WBW': disk_peak_WBW,
            'processed_line': processed_line
        }

    def calculate_label(self, disk_data: Dict[str, Any]) -> Tuple[int, float, float]:
        """计算磁盘的标签（稳定或突发）"""
        burst_lable = -1
        RBW_mul = (disk_data['peak_RBW'] / disk_data['avg_RBW']
                   if disk_data['avg_RBW'] != 0 else 0)
        WBW_mul = (disk_data['peak_WBW'] / disk_data['avg_WBW']
                   if disk_data['avg_WBW'] != 0 else 0)

        if (RBW_mul < self.read_bandwidth_line and
                WBW_mul < self.write_bandwidth_line):
            burst_lable = 0  # 稳定
        else:
            burst_lable = 1  # 突发
        return burst_lable, RBW_mul, WBW_mul

    def _iterate_first_day(self, timestamps: pd.Series) -> int:
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
        description_dir = f"{self.trace_root}/20_136090{cluster_index}/136090{cluster_index}_subscript_info"
        cluster_info_dir = self._ensure_cluster_info_dir()

        with open(description_dir, "r") as descriptions, open(f"{cluster_info_dir}/cluster{cluster_index}", "w") as cluster_info:
            columns = ["appid", "disk_ID", "disk_instance", "vm_uid", "create_time", "finsh_time", "status", "disk_if_local", "disk_attr", "disk_type",
                       "disk_if_VIP", "disk_pay", "pay_type", "vm_name", "vm_cpu", "vm_memory", "app_name", "disk_name", "project_name", "disk_usage", "disk_capacity"]
            df = pd.read_csv(descriptions, names=columns, sep=',', header=None)
            for _, description in df.iterrows():
                if self._is_valid_description(description):
                    self._process_disk(
                        description, cluster_index, cluster_info)

    def _is_valid_description(self, description: pd.Series) -> bool:
        """检查描述是否有效"""
        required_fields = ["disk_ID", "disk_if_local", "disk_attr", "disk_type",
                           "disk_if_VIP", "disk_pay", "vm_cpu", "vm_memory", "disk_capacity"]
        return not any(description[i] == '' or pd.isna(description[i]) for i in required_fields)

    def _process_disk(self, description: pd.Series, cluster_index: int,
                      cluster_info_file) -> None:
        """处理单个磁盘的数据"""
        trace_dir = f"{self.trace_root}/20_136090{cluster_index}/{description['disk_ID']}"
        disk_data = self.disk_processor.process_disk_data(trace_dir)

        if disk_data is None:
            self._log_missing_trace(description['disk_ID'], cluster_index)
            return

        label, RBW_mul, WBW_mul = self.disk_processor.calculate_label(
            disk_data)
        self._write_disk_data(description, disk_data, label,
                              RBW_mul, WBW_mul, cluster_info_file)

    def _log_missing_trace(self, disk_id: str, cluster_index: int) -> None:
        """记录缺失的跟踪文件"""
        with open(f"{self.cluster_info_root}/cluster{cluster_index}_not_exist", "a") as f:
            f.write(f"{disk_id}\n")

    def _write_disk_data(self, description: List[str], disk_data: Dict[str, Any],
                         label: int, RBW_mul: float, WBW_mul: float, cluster_info_file) -> None:
        """写入磁盘数据到仓库文件"""
        cluster_info_file.write(
            # item[0] disk_ID
            # item[1:9] disk_capacity, disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay, vm_cpu, vm_mem
            # item[9:13] average_disk_RBW, average_disk_WBW, peak_disk_RBW, peak_disk_WBW
            # item[13:14] disk_timestamp_num,burst_label
            f"{description['disk_ID']},{description['disk_capacity']},{description['disk_if_local']},{description['disk_attr']},"
            f"{description['disk_type']},{description['disk_if_VIP']},{description['disk_pay']},{description['vm_cpu']},"
            f"{description['vm_memory']},{disk_data['avg_RBW']},{disk_data['avg_WBW']},"
            f"{disk_data['peak_RBW']},{disk_data['peak_WBW']},"
            # label (0 for stable, 1 for burst)
            f"{disk_data['processed_line']},{label},{RBW_mul},{WBW_mul}\n"
        )

    def init_cluster_info(self) -> None:
        logger.info("开始初始化磁盘数据")
        for cluster_index in range(self.warehouse_number):
            self._process_cluster(cluster_index)
