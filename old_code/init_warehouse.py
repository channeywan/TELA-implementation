from config import warehouse_num, evaluate_time_number, bandwidth_line
import os
import csv
from datetime import datetime
def init_warehouse():
    print("开始初始化磁盘数据")
    for cluster_index in range(warehouse_num):
        print(f"正在处理集群 {cluster_index}")
        description_dir=f"/data/Tencent_CVD/Shanghai/20_136090{cluster_index}/136090{cluster_index}_subscript_info"
        with open(description_dir,"r") as descriptions:
            # appid, disk_uid, disk_instance, vm_uid, create_time, finsh_time, status, is_local, disk_attr, disk_type,  isvip, pay_mode, pay_type, vm_name, cpu, memory, app_name, disk_name, project_name, disk_usage
            warehouse_dir="/home/wcl/warehouse"
            if os.path.exists(warehouse_dir) == False:
                os.mkdir(warehouse_dir)
            with open(f"{warehouse_dir}/warehouse{cluster_index}","w") as warehouse:
                reader=csv.reader(descriptions)
                for description in reader:
                    # description=description.strip().split(",")
                    if '' in [description[1],description[19],description[7],description[8],description[9],description[10],description[11],description[14],description[15]]:
                        continue
                    processed_line=0
                    disk_avg_RBW = 0.0
                    disk_avg_WBW = 0.0
                    disk_peak_RBW = -1.0
                    disk_peak_WBW = -1.0
                    trace_dir=f"/data/Tencent_CVD/Shanghai/20_136090{cluster_index}/{description[1]}"
                    if os.path.exists(trace_dir) == False:
                        with open (f"/home/wcl/warehouse/cluster{cluster_index}_not_exist","a") as trace_dir_not_exist:
                            trace_dir_not_exist.write(f"{description[1]}\n")
                        continue
                    with open(trace_dir) as traces:
                        first_line=traces.readline()
                        first_line=first_line.strip().split(",")
                        first_day=datetime.fromtimestamp(int(first_line[0])).day
                        begin=0
                        for trace in traces:
                            trace=trace.strip().split(",")
                            if begin==0:
                                if datetime.fromtimestamp(int(trace[0])).day != first_day and datetime.fromtimestamp(int(trace[0])).isoweekday() == 1:
                                    begin=1
                                    processed_line=0
                                else:
                                    continue
                            if processed_line>=evaluate_time_number:
                                break
                            rbandwidth=trace[2]
                            wbandwidth=trace[4]
                            disk_avg_RBW += float(rbandwidth)
                            disk_avg_WBW += float(wbandwidth)
                            if float(rbandwidth) > disk_peak_RBW:
                                disk_peak_RBW = float(rbandwidth)
                            if float(wbandwidth) > disk_peak_WBW:
                                disk_peak_WBW = float(wbandwidth)
                            processed_line += 1
                    if processed_line == 0:
                        continue
                    disk_avg_RBW /= processed_line
                    disk_avg_WBW /= processed_line
                    if disk_avg_RBW == 0 and disk_peak_RBW ==0:
                        RBW_mul = 0
                    else:
                        RBW_mul = disk_peak_RBW/disk_avg_RBW 
                    if disk_avg_WBW == 0 and disk_peak_WBW == 0:
                        WBW_mul = 0
                    else:
                        WBW_mul=disk_peak_WBW/disk_avg_WBW
                    label=-1
                    if RBW_mul < bandwidth_line and WBW_mul < bandwidth_line:
                        label = 0
                    else:
                        label = 1
                    # labels (0 for stable, 1 for burst)
                    # item[0] disk_ID
                    # item[1:9] disk_capacity, disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay, vm_cpu, vm_mem
                    # item[9:13] average_disk_RBW, average_disk_WBW, peak_disk_RBW, peak_disk_WBW
                    # item[13:14] disk_timestamp_num,burst_label
                    warehouse.write(f"{description[1]},{description[19]},{description[7]},{description[8]},{description[9]},{description[10]},{description[11]},{description[14]},{description[15]},{disk_avg_RBW},{disk_avg_WBW},{disk_peak_RBW},{disk_peak_WBW},{processed_line},{label}\n")
