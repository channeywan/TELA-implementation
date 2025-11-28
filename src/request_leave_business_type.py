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
import re
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def merge_string(row):
    string_list = {s.strip() for s in row.values.astype(str) if s.strip()}
    to_remove = set()
    for s_i in string_list:
        for s_j in string_list:
            if s_i != s_j and s_i in s_j:
                to_remove.add(s_i)
                break
    return '_'.join(sorted([s for s in string_list if s not in to_remove]))


def get_all_disk_info():
    column_mapping = {
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
    all_df = []
    for cluster_index in DataConfig.CLUSTER_DIR_LIST:
        description_dir = f"{DirConfig.TRACE_ROOT}/153_10077{cluster_index}/describe.csv"
        if not os.path.exists(description_dir):
            logger.error(f"description_dir {description_dir} not exists")
            continue
        raw_df = pd.read_csv(description_dir, sep=',',
                             usecols=column_mapping.keys())
        raw_df.rename(columns=column_mapping, inplace=True)
        raw_df["cluster_index"] = cluster_index
        raw_df['description'] = raw_df[["vm_name", "project_name", "buss_name", "disk_name"]].fillna(
            '').replace("未命名", '').apply(merge_string, axis=1)
        description_df = raw_df[['disk_ID',
                                 'cluster_index', 'description', 'appid']]
        descs = description_df['description']
        mask_valid_content = descs.str.contains(r'\S', regex=True, na=False)
        mask_not_unnamed = ~descs.str.strip().str.lower().eq("unnamed")
        mask_not_pattern = ~descs.str.strip().str.match(r"未命名\d*_系统盘", na=False)
        available_df = description_df[mask_valid_content &
                                      mask_not_unnamed & mask_not_pattern].copy().reset_index(drop=True)
        all_df.append(available_df)
    return pd.concat(all_df, ignore_index=True).reset_index(drop=True)


def get_all_tencent_disk_info():
    all_df = []
    for cluster_index in [1360900, 1360901, 1360902, 1360903, 1360904, 1360905, 1360906, 1360907, 1360908, 1360909, 1360919, 1360920, 1360930, 1360931, 1360941, 1360942, 1360943, 1360944, 1360945, 1360946, 1360947, 1360948, 1360953, 1360954, 1360955, 1360956, 1360963, 1360971, 1360972, 1360978, 1360981, 1360982, 1360983, 1360984, 1360985, 1360986, 1360987, 1360988, 1361022, 1361023, 1361130, 1361146, 1361147]:
        description_dir = f"/data/Tencent_CVD/Shanghai/20_{cluster_index}/{cluster_index}_diskinfo"
        if not os.path.exists(description_dir):
            logger.error(f"description_dir {description_dir} not exists")
            continue
        raw_df = pd.read_csv(description_dir, sep=',',
                             usecols=[16, 17, 18], header=None)
        descs = raw_df.fillna(
            '').replace("未命名", '').apply(merge_string, axis=1)
        mask_valid_content = descs.str.contains(r'\S', regex=True, na=False)
        mask_not_unnamed = ~descs.str.strip().str.lower().eq("unnamed")
        mask_not_pattern = ~descs.str.strip().str.match(r"未命名\d*_系统盘", na=False)
        descs = pd.DataFrame({"description": descs})
        available_df = descs[mask_valid_content].copy().reset_index(drop=True)
        all_df.append(available_df)
    return pd.concat(all_df, ignore_index=True).reset_index(drop=True)


def get_leave_disk_info():
    loader = DiskDataLoader()
    current_items = loader.load_items(
        cluster_index_list=DataConfig.CLUSTER_DIR_LIST)
    all_items = get_all_disk_info()
    print("all_items_num:", len(all_items))
    print("current_items_num:", len(current_items))
    index1 = all_items.set_index(['disk_ID', 'cluster_index']).index
    index2 = current_items.set_index(['disk_ID', 'cluster_index']).index
    leave_items = all_items[~index1.isin(index2)]
    print("leave_items_num:", len(leave_items))
    return leave_items


if __name__ == "__main__":
    # df = get_leave_disk_info()
    # df = pd.read_csv(os.path.join(DirConfig.TEMP_DIR,
    #                               "leave_disk_info.csv"))
    # df['fingerprint'] = df['description'].str.replace(
    #     r'\d+', '[NUM]', regex=True)
    # df['fingerprint'] = df['fingerprint'].str.replace(r'[a-z0-9]{8,}', '[ID]', regex=True)
    # df.drop_duplicates(subset=['fingerprint'], keep='first', inplace=True)
    # df.drop(columns=['fingerprint'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # available_leave_items = leave_items.drop_duplicates(
    #     subset=['description'], keep='first')
    # leave_items = available_leave_items.reset_index(drop=True)
    # descs = leave_items['description']
    # mask_not_pattern = ~descs.str.strip().str.match(r"\d*_*PWA_未命名\d*_系统盘", na=False)
    # leave_items = leave_items[mask_not_pattern]
    # df = df[["appid", "cluster_index", "description"]
    #         ].sort_values("description")

    # df.to_csv(os.path.join(DirConfig.TRASH_DIR,
    #                        "raw_leave_disk_info.csv"), index=False)
    # df = pd.read_csv(os.path.join(DirConfig.TRASH_DIR, "remain_disks.csv"))
    # mask_valid_content = ~df['description'].str.match(
    #     r'lh-\d*-lhins-.*_系统盘_lh-cbs',  na=False)
    # df = df[mask_valid_content]
    # df.to_csv(os.path.join(DirConfig.TRASH_DIR,
    #           "remain_disks_valid.csv"), index=False)
    df = pd.read_csv(os.path.join(DirConfig.TRASH_DIR,
                     "all_tencent_disk_description.csv"))
    df.dropna(inplace=True)
    df.sort_values(by=['description'], inplace=True)
    df.to_csv(os.path.join(DirConfig.TRASH_DIR,
              "all_tencent_disk_description.csv"), index=False)
