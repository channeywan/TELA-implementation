import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
from config.settings import DirConfig, DataConfig, WarehouseConfig, ModelConfig

logger = logging.getLogger(__name__)


class DiskDataLoader:
    """磁盘数据加载器"""

    def __init__(self):
        pass

    def load_items_and_trace(self, cluster_index_list: List[int], type: str = "both"):
        all_items = self.load_items(cluster_index_list, type)
        disks_trace = self._get_disks_trace(cluster_index_list)
        return all_items, disks_trace

    def load_selected_items(self):
        return pd.read_csv(os.path.join(DirConfig.CLUSTER_INFO_ROOT, "selected_disks_all_info.csv"))

    def load_all_trace(self):
        trace_dir = os.path.join(
            DirConfig.CLUSTER_TRACE_DB_ROOT, "all_trace.pkl")
        return joblib.load(trace_dir)

    def load_items(self, cluster_index_list: List[int], type: str = "both"):
        all_cluster_items = []
        for cluster_index in cluster_index_list:
            item_dir = os.path.join(
                DirConfig.CLUSTER_INFO_ROOT,
                f"cluster{cluster_index}"
            )
            # item_dir = os.path.join(
            #     DirConfig.CLUSTER_INFO_BUSINESS_TYPE_ROOT,
            #     f"cluster{cluster_index}.csv"
            # )
            if not os.path.exists(item_dir):
                logger.warning(f"data file not exists: {item_dir}")
                continue
            try:
                items = pd.read_csv(item_dir,  encoding="utf-8")
                if len(items) == 0:
                    logger.warning(f"cluster {cluster_index} has no data")
                    continue
                items["cluster_index"] = cluster_index
                cluster_items = self._parse_disk(
                    items, type)
                if cluster_items is not None and not cluster_items.empty:
                    all_cluster_items.append(cluster_items)
            except (IOError, OSError) as e:
                logger.error(f"read file {item_dir} error: {e}")
                continue
            # logger.info(
            #     f"cluster {cluster_index} loaded {len(cluster_items) if cluster_items is not None else 0} disks")
        # 如果没有加载到任何数据，返回空 DataFrame
        if len(all_cluster_items) == 0:
            return pd.DataFrame()
        all_items = pd.concat(all_cluster_items, ignore_index=True)
        all_items.reset_index(drop=True, inplace=True)
        logger.info(f"total loaded {len(all_items)} disks")
        return all_items

    def _get_disks_trace(self, cluster_index_list: List[int]):
        disks_trace = {}
        for cluster_index in cluster_index_list:
            trace_dir = os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl")
            disks_trace[cluster_index] = joblib.load(trace_dir)
        return disks_trace

    def _parse_disk(self, items: pd.DataFrame, type: str = "both") -> Optional[pd.DataFrame]:
        """
        解析磁盘项目数据
        """
        type_map = {'disk_capacity': int, 'disk_if_VIP': int,
                    'vm_cpu': float, 'vm_memory': float, 'avg_bandwidth': float,
                    'peak_bandwidth': float, 'burst_label': int,
                    'cluster_index': int, 'bandwidth_mul': float, 'bandwidth_zero_ratio': float}

        items = items.astype(type_map)

        mask_cpu_mem = (items['vm_cpu'] > 0) & (items['vm_memory'] > 0)
        mask_avg_bw = (items['avg_bandwidth'] > 0)
        # & (items['avg_bandwidth'] < DataConfig.MAX_BANDWIDTH_THRESHOLD)
        mask_bandwidth_zero_ratio = (items['bandwidth_zero_ratio'] < 0.4)
        validation_mask = mask_cpu_mem & mask_avg_bw & mask_bandwidth_zero_ratio
        if type == "burst":
            type_mask = (items['burst_label'] == 1)
        elif type == "stable":
            type_mask = (items['burst_label'] == 0)
        elif type == "both":
            type_mask = pd.Series(True, index=items.index)
        else:
            raise ValueError(f"Invalid type: {type}")
        items = items[validation_mask & type_mask]
        return items
