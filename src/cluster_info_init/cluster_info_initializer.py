import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import csv
from datetime import datetime
from config.settings import DirConfig, WarehouseConfig, DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class DiskDataProcessor:

    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()

    def process_disk_data(self, trace_file: str) -> Optional[Dict[str, Any]]:
        """处理单个磁盘的跟踪数据"""
        if not os.path.exists(trace_file):
            return None

        with open(trace_file) as traces:
            first_line = traces.readline().strip().split(",")
            first_day = datetime.fromtimestamp(int(first_line[0])).day
            begin = 0
            processed_line = 0
            disk_avg_RBW = 0.0
            disk_avg_WBW = 0.0
            disk_peak_RBW = -1.0
            disk_peak_WBW = -1.0

            for trace in traces:
                trace = trace.strip().split(",")
                if begin == 0:
                    if (datetime.fromtimestamp(int(trace[0])).day != first_day and
                            datetime.fromtimestamp(int(trace[0])).isoweekday() == 1):
                        begin = 1
                        processed_line = 0
                    else:
                        continue

                if processed_line >= self.data_config.EVALUATE_TIME_NUMBER:
                    break

                rbandwidth = float(trace[2])
                wbandwidth = float(trace[4])
                disk_avg_RBW += rbandwidth
                disk_avg_WBW += wbandwidth
                disk_peak_RBW = max(disk_peak_RBW, rbandwidth)
                disk_peak_WBW = max(disk_peak_WBW, wbandwidth)
                processed_line += 1

            if processed_line == 0:
                return None

            return {
                'avg_RBW': disk_avg_RBW / processed_line,
                'avg_WBW': disk_avg_WBW / processed_line,
                'peak_RBW': disk_peak_RBW,
                'peak_WBW': disk_peak_WBW,
                'processed_line': processed_line
            }

    def calculate_label(self, disk_data: Dict[str, Any]) -> int:
        """计算磁盘的标签（稳定或突发）"""
        RBW_mul = (disk_data['peak_RBW'] / disk_data['avg_RBW']
                   if disk_data['avg_RBW'] != 0 else 0)
        WBW_mul = (disk_data['peak_WBW'] / disk_data['avg_WBW']
                   if disk_data['avg_WBW'] != 0 else 0)

        if (RBW_mul < self.model_config.BANDWIDTH_LINE and
                WBW_mul < self.model_config.BANDWIDTH_LINE):
            return 0  # 稳定
        return 1  # 突发


class ClusterInfoInitializer:
    """初始化仓库数据的类"""

    def __init__(self):
        self.config = DirConfig()
        self.warehouse_config = WarehouseConfig()
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.disk_processor = DiskDataProcessor()

    def _ensure_cluster_info_dir(self) -> str:
        """确保仓库目录存在"""
        cluster_dir = self.config.CLUSTER_INFO_ROOT
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)
        return cluster_dir

    def _process_cluster(self, cluster_index: int) -> None:
        """处理单个集群的数据"""
        logger.info(f"正在处理集群 {cluster_index}")
        description_dir = f"{self.config.TRACE_ROOT}/20_136090{cluster_index}/136090{cluster_index}_subscript_info"
        cluster_info_dir = self._ensure_cluster_info_dir()

        with open(description_dir, "r") as descriptions, open(f"{cluster_info_dir}/cluster{cluster_index}", "w") as cluster_info:
            reader = csv.reader(descriptions)
            for description in reader:
                if self._is_valid_description(description):
                    self._process_disk(
                        description, cluster_index, cluster_info)

    def _is_valid_description(self, description: List[str]) -> bool:
        """检查描述是否有效"""
        required_fields = [1, 19, 7, 8, 9, 10, 11, 14, 15]
        return not any(description[i] == '' for i in required_fields)

    def _process_disk(self, description: List[str], cluster_index: int,
                      cluster_info_file) -> None:
        """处理单个磁盘的数据"""
        trace_dir = f"{self.config.TRACE_ROOT}/20_136090{cluster_index}/{description[1]}"
        disk_data = self.disk_processor.process_disk_data(trace_dir)

        if disk_data is None:
            self._log_missing_trace(description[1], cluster_index)
            return

        label = self.disk_processor.calculate_label(disk_data)
        self._write_disk_data(description, disk_data, label, cluster_info_file)

    def _log_missing_trace(self, disk_id: str, cluster_index: int) -> None:
        """记录缺失的跟踪文件"""
        with open(f"{self.config.CLUSTER_INFO_ROOT}/cluster{cluster_index}_not_exist", "a") as f:
            f.write(f"{disk_id}\n")

    def _write_disk_data(self, description: List[str], disk_data: Dict[str, Any],
                         label: int, cluster_info_file) -> None:
        """写入磁盘数据到仓库文件"""
        cluster_info_file.write(
            # item[0] disk_ID
            # item[1:9] disk_capacity, disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay, vm_cpu, vm_mem
            # item[9:13] average_disk_RBW, average_disk_WBW, peak_disk_RBW, peak_disk_WBW
            # item[13:14] disk_timestamp_num,burst_label
            f"{description[1]},{description[-1]},{description[7]},{description[8]},"
            f"{description[9]},{description[10]},{description[11]},{description[14]},"
            f"{description[15]},{disk_data['avg_RBW']},{disk_data['avg_WBW']},"
            f"{disk_data['peak_RBW']},{disk_data['peak_WBW']},"
            # label (0 for stable, 1 for burst)
            f"{disk_data['processed_line']},{label}\n"
        )

    def init_cluster_info(self) -> None:
        logger.info("开始初始化磁盘数据")
        for cluster_index in range(self.warehouse_config.CLUSTER_NUMBER):
            self._process_cluster(cluster_index)
