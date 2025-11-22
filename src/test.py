import logging
import sys
import os
from pathlib import Path
from config.settings import DataConfig, DirConfig
from data.loader import DiskDataLoader
from visualization.plotter import TelaPlotter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# vm_type:['S5' 'S6' nan 'SA3' 'C6' 'SA2' 'TS5' 'MA2' 'IT5' 'MA3' 'GN7' 'GN10X' 'D3' 'C5' 'S4' 'M5' 'M6' 'GN10Xp' 'GN7vw' 'S5se' 'BC1']
# disk_uuid, vm_alias, vm_cpu, vm_mem, vm_type, is_vip, ins_type(cvm,eks),project_name,buss_name,
# disk_alias,disk_size,disk_type(root,data,tfs,tssd),volume_type(cbsssd,cbsBSSD,cbsPremiun,cbsHSSD)

# ,id,region,disk_uuid,zone_id,depot_id,set_uuid,set_name,setType,setSize,set_volume_type,vm_uuid,vm_alias,vm_cpu,vm_mem,vm_os_name,vm_type,vm_deadline,appid,is_vip,is_pdd,ins_type,project_name,buss_name,disk_alias,phy_id,disk_usage,disk_size,disk_type,volume_type,pay_mode,ccbs_status,disk_status,life_state,add_time,create_date_time,last_op_date_time,modify_time,deadline
# depot_id,set_uuid,set_volume_type,vm_os_name,vm_type,disk_type,volume_type,pay_mode,life_state
if __name__ == "__main__":
    loader = DiskDataLoader()
    items = loader.load_selected_items()
    train_items, test_items = train_test_split(
        items, test_size=0.3, random_state=42)
    avaliable_items = test_items.sample(n=6048, random_state=42, replace=False)
    print(np.sum(avaliable_items["avg_bandwidth"]))
    print(np.sum(avaliable_items["disk_capacity"]))
