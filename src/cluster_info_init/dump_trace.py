from data.loader import DiskDataLoader
from config.settings import WarehouseConfig, DirConfig
import pandas as pd
import os
import joblib
from tqdm import tqdm


class DumpTrace:
    def __init__(self):
        self.disk_data_loader = DiskDataLoader()

    def dump_trace(self):
        for cluster_index in WarehouseConfig.CLUSTER_DIR_LIST:
            items = self.disk_data_loader.load_items(
                cluster_index_list=[cluster_index], type="both")
            if len(items) == 0:
                continue
            trace_db = {}
            for _, item in tqdm(items.iterrows(), total=len(items), desc=f"Dumping cluster {cluster_index} trace"):
                disk_id = item['disk_ID']
                trace_dir = os.path.join(DirConfig.TRACE_ROOT,
                                         f"153_10077{cluster_index}", f"{disk_id}.csv")
                traces = pd.read_csv(trace_dir, sep=',')
                traces.columns = ["timestamp", "read_IOPS",
                                  "write_IOPS", "read_BW", "write_BW"]
                traces["timestamp"] = pd.to_datetime(traces["timestamp"])
                traces["bandwidth"] = traces["read_BW"] + traces["write_BW"]
                trace_db[disk_id] = traces[["timestamp",
                                           "bandwidth"]]
            joblib.dump(trace_db, os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl"))
