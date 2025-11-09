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
        for cluster_index in range(WarehouseConfig.CLUSTER_NUMBER):
            items = self.disk_data_loader.load_items(
                cluster_index_list=[cluster_index], type="both", purpose="train")
            trace_db = {}
            for _, item in tqdm(items.iterrows(), total=len(items), desc=f"Dumping cluster {cluster_index} trace"):
                disk_id = item['disk_ID']
                trace_dir = os.path.join(DirConfig.TRACE_ROOT,
                                         f"20_136090{cluster_index}", f"{disk_id}")
                trace = pd.read_csv(trace_dir, header=None, names=[
                    'timestamp', 'RBW', 'WBW'], delimiter=',', usecols=(0, 2, 4), dtype=float)
                trace["bandwidth"] = trace["RBW"] + trace["WBW"]
                trace_db[disk_id] = trace[["timestamp",
                                           "bandwidth"]]
            joblib.dump(trace_db, os.path.join(
                DirConfig.CLUSTER_TRACE_DB_ROOT, f"cluster_{cluster_index}_trace.pkl"))
