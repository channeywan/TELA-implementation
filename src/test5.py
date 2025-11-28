from visualization.plotter import TelaPlotter
from sklearn.metrics import root_mean_squared_error as rmse, r2_score
import os
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from config.settings import DirConfig, DataConfig, WarehouseConfig, TestConfig
from data.loader import DiskDataLoader
from itertools import permutations
from sklearn.model_selection import train_test_split
from data.request_business_type import run_request_business_type
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    loader = DiskDataLoader()
    df1 = pd.read_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR, 'tencent_disk_business_type.csv'))[
        ["description", "business_type"]]
    df2 = pd.read_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR, 'leave_disk_info_business_type.csv'))[
        ["description", "business_type"]]
    df3 = pd.read_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR, 'remain_disks_valid_business_type.csv'))[
        ["description", "business_type"]]
    df4 = loader.load_items(DataConfig.CLUSTER_DIR_LIST)[
        ["description", "business_type"]]
    df5 = pd.read_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR, 'gemini_generate.csv'))[
        ["description", "business_type"]]
    df = pd.concat([df1, df2, df3, df4, df5])
    df.sort_values(by=["business_type"], inplace=True)
    df.drop_duplicates(subset=["description"], inplace=True)
    # 对每一个business_type只随机保留3000行
    df = df.groupby("business_type", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 3000), random_state=42))

    print(df.value_counts(subset=["business_type"]))
    df.to_csv(os.path.join(DirConfig.BUSINESS_TYPE_DIR,
              'combined_description_business_type.csv'), index=False)
