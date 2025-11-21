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
        disk_read_bandwidth = traces["read_BW"].values
        disk_write_bandwidth = traces["write_BW"].values
        # slice_bandwidth = disk_bandwidth[first_day_line:]
        slice_bandwidth = disk_bandwidth
        timestamp_num = len(slice_bandwidth)
        if timestamp_num < DataConfig.MIN_TIMESTAMP_NUM:
            return None
        disk_read_bandwidth_avg = np.average(disk_read_bandwidth)
        disk_write_bandwidth_avg = np.average(disk_write_bandwidth)
        disk_bandwidth_avg = np.average(slice_bandwidth)
        disk_bandwidth_peak = np.max(slice_bandwidth)
        bandwidth_zero_num = np.sum(slice_bandwidth == 0)
        bandwidth_zero_ratio = bandwidth_zero_num / timestamp_num

        return {
            'avg_read_bandwidth': disk_read_bandwidth_avg,
            'avg_write_bandwidth': disk_write_bandwidth_avg,
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
        self.column_mapping = {
            "disk_uuid": "disk_ID",
            "vm_alias": "vm_name",
            "vm_cpu": "vm_cpu",
            "vm_mem": "vm_memory",
            "is_vip": "disk_if_VIP",
            "ins_type": "ins_type",
            "project_name": "project_name",
            "buss_name": "buss_name",
            "disk_alias": "disk_name",
            "disk_size": "disk_capacity",
            "add_time": "add_time",
            "create_date_time": "create_date_time",
            "last_op_date_time": "last_op_date_time",
            "modify_time": "modify_time",
            "deadline": "deadline",
            "appid": "appid",
            "depot_id": "depot_id",
            "set_uuid": "set_uuid",
            "set_volume_type": "set_volume_type",
            "vm_os_name": "vm_os_name",
            "vm_type": "vm_type",
            "disk_type": "disk_type",
            "volume_type": "volume_type",
            "pay_mode": "pay_mode",
            "life_state": "life_state"
        }

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
            raw_df = pd.read_csv(descriptions, sep=',',
                                 usecols=self.column_mapping.keys())
            raw_df.rename(columns=self.column_mapping, inplace=True)
            date_columns = ["add_time", "create_date_time",
                            "last_op_date_time", "modify_time", "deadline"]
            for column in date_columns:
                raw_df[column] = pd.to_datetime(
                    raw_df[column], errors='coerce')
            raw_df["finish_time"] = raw_df[date_columns].max(axis=1)
            raw_df["create_time"] = raw_df[date_columns].min(axis=1)
            df = raw_df.drop(date_columns, axis=1)
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True, how='any', axis=0)
            if df.empty:
                logger.warning(f"集群 {cluster_index} 过滤后没有有效数据。")
                return
            new_data_df = df.apply(
                self._process_disk_row, axis=1, args=(cluster_index,))
            df = df.join(new_data_df)
            df.dropna(inplace=True, how='any', axis=0)
            df = df[df['avg_bandwidth'] != 0]
            df.sort_values(by=['appid', 'create_time'],
                           ascending=[True, True], inplace=True)
            df['recent_history_bandwidth_app'] = df.groupby(
                'appid')['avg_bandwidth'].shift(1)
            df['recent_history_bandwidth_cpu'] = df.groupby(['appid', 'vm_cpu'])[
                'avg_bandwidth'].shift(1)
            df['recent_history_bandwidth_memory'] = df.groupby(['appid', 'vm_memory'])[
                'avg_bandwidth'].shift(1)
            df['recent_history_bandwidth_cpu_memory'] = df.groupby(
                ['appid', 'vm_cpu', 'vm_memory'])['avg_bandwidth'].shift(1)
            app_grouped = df.groupby('appid')[
                'avg_bandwidth']
            app_grouped_cpu_memory = df.groupby(['appid', 'vm_cpu', 'vm_memory'])[
                'avg_bandwidth']
            app_grouped_cpu = df.groupby(['appid', 'vm_cpu'])[
                'avg_bandwidth']
            app_grouped_memory = df.groupby(['appid', 'vm_memory'])[
                'avg_bandwidth']
            app_group_sum = app_grouped.transform('sum')
            app_group_count = app_grouped.transform('count')
            app_group_sum_cpu_memory = app_grouped_cpu_memory.transform('sum')
            app_group_count_cpu_memory = app_grouped_cpu_memory.transform(
                'count')
            app_group_sum_cpu = app_grouped_cpu.transform('sum')
            app_group_count_cpu = app_grouped_cpu.transform('count')
            app_group_sum_memory = app_grouped_memory.transform('sum')
            app_group_count_memory = app_grouped_memory.transform('count')
            df["avg_history_bandwidth_app"] = (
                app_group_sum-df["avg_bandwidth"]) / (app_group_count-1)
            df["avg_history_bandwidth_cpu_memory"] = (
                app_group_sum_cpu_memory-df["avg_bandwidth"]) / (app_group_count_cpu_memory-1)
            df["avg_history_bandwidth_cpu"] = (
                app_group_sum_cpu-df["avg_bandwidth"]) / (app_group_count_cpu-1)
            df["avg_history_bandwidth_memory"] = (
                app_group_sum_memory-df["avg_bandwidth"]) / (app_group_count_memory-1)
            cluster_info_dir = self._ensure_cluster_info_dir()
            output_path = f"{cluster_info_dir}/cluster{cluster_index}"
            logger.info(
                f"cluster{cluster_index} info saved to {output_path}, {len(df)} disks processed")
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
        for cluster_index in DataConfig.CLUSTER_DIR_LIST:
            self._process_cluster(cluster_index)
        self.merge_business_type()

    def merge_business_type(self) -> None:
        for cluster_index in DataConfig.CLUSTER_DIR_LIST:
            raw_cluster_info_file = os.path.join(
                DirConfig.CLUSTER_INFO_ROOT, f"cluster{cluster_index}")
            bussiness_info_file = os.path.join(
                DirConfig.CLUSTER_INFO_BUSINESS_TYPE_ROOT, f"cluster{cluster_index}.csv")
            raw_df = pd.read_csv(raw_cluster_info_file)
            bussiness_df = pd.read_csv(bussiness_info_file)
            bussiness_df = bussiness_df[[
                "disk_ID", "business_type", "description"]]
            merged_df = pd.merge(raw_df, bussiness_df,
                                 on="disk_ID", how="left")
            merged_df.to_csv(os.path.join(DirConfig.CLUSTER_INFO_ROOT,
                                          f"cluster{cluster_index}"), index=False)
