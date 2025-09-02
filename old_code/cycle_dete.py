from config import IO_line, bandwidth_line, bandwidth_line_day, evaluate_time_number, min_timestamp_num, warehouse_num, max_capacity, max_IOPS, max_bandwidth
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
# 时间戳、读IOPS、读带宽(KB/s)、写IOPS、写带宽(KB/s)、读延时(us)、写延时(us)、此时的disk_usage(MB)。
# dataset_warehouse(disk_id,disk_capacity, disk_if_local, disk_type, vm_cpu, vm_mem,ave_disk_IOPS, ave_disk_bandwidth, peak_disk_IOPS, peak_disk_bandwidth,disk_timestamp_num)
# item[0] disk_ID
# item[1:9] disk_capacity, disk_if_local, disk_attr, disk_type, disk_if_VIP, disk_pay, vm_cpu, vm_mem
# item[9:13] average_disk_RBW, average_disk_WBW, peak_disk_RBW, peak_disk_WBW
# item[13:15] disk_timestamp_num,burst_label


def load_burst_item():
    print("正在加载磁盘数据")
    item_dir = []
    items = []
    for cluster_index in range(5):
        item_dir = f"/home/wcl/warehouse/warehouse{cluster_index}"
        cluster_item = []
        with open(item_dir, "r") as f:
            line_number = 0
            reader = csv.reader(f)
            for item in reader:
                line_number += 1
                item[1] = int(item[1])
                try:
                    item[7:13] = map(float, item[7:13])
                except ValueError as e:
                    print(
                        f"集群 {cluster_index} 磁盘 {line_number} {str(item)}出错:{e}")
                    continue
                item[13:15] = map(int, item[13:15])
                # 去除时间戳数量小于min_timestamp_num的磁盘
                if item[13] < min_timestamp_num:
                    continue
                # 去除最大值大于max*0.8的磁盘
                if item[11]+item[12] > max_bandwidth[cluster_index]*0.8:
                    continue
                # 去除vm_cpu=0或vm_mem=0的云盘
                if item[7] == 0 or item[8] == 0:
                    continue
                if item[9] < 3 or item[10] < 3:
                    continue
                cluster_item.append(item)
                if item[14] == 0:
                    continue
        items.append(cluster_item)
    return items


def load_stable_item():
    print("正在加载磁盘数据")
    item_dir = []
    items = []
    for cluster_index in range(5):
        item_dir = f"/home/wcl/warehouse/warehouse{cluster_index}"
        cluster_item = []
        with open(item_dir, "r") as f:
            line_number = 0
            reader = csv.reader(f)
            for item in reader:
                line_number += 1
                item[1] = int(item[1])
                try:
                    item[7:13] = map(float, item[7:13])
                except ValueError as e:
                    print(
                        f"集群 {cluster_index} 磁盘 {line_number} {str(item)}出错:{e}")
                    continue
                item[13:15] = map(int, item[13:15])
                # 去除时间戳数量小于min_timestamp_num的磁盘
                if item[13] < min_timestamp_num:
                    continue
                # 去除最大值大于max*0.8的磁盘
                if item[11]+item[12] > max_bandwidth[cluster_index]*0.8:
                    continue
                # 去除vm_cpu=0或vm_mem=0的云盘
                if item[7] == 0 or item[8] == 0:
                    continue
                if item[9] < 3 or item[10] < 3:
                    continue
                cluster_item.append(item)
                if item[14] == 1:
                    continue
        items.append(cluster_item)
    return items
# 已弃用


def filter_burst_items(original_items):
    cluster_dir = []
    burst_items = []
    for i in range(10):
        cluster_dir.append(f"/data/Tencent_CVD/Shanghai/20_136090{i}")
    for cluster_num in range(len(original_items)):
        cluster_item = []
        for item_num_in_cluster in range(len(original_items[cluster_num])):
            item_dir = os.path.join(cluster_dir[cluster_num], str(
                original_items[cluster_num][item_num_in_cluster][0]))
            if not os.path.exists(item_dir):
                print(f"文件不存在: {item_dir}")
                continue
            processed_line = 0
            disk_avg_IO = 0.0
            disk_avg_BW = 0.0
            disk_peak_IO = -1.0
            disk_peak_BW = -1.0
            with open(item_dir, "r") as f:
                for line in f:
                    if processed_line >= evaluate_time_number:
                        break
                    fields = line.split(",")
                    disk_IO = int(fields[1]) + int(fields[3])
                    disk_BW = int(fields[2]) + int(fields[4])
                    disk_avg_IO += disk_IO
                    disk_avg_BW += disk_BW
                    if disk_IO > disk_peak_IO:
                        disk_peak_IO = disk_IO
                    if disk_BW > disk_peak_BW:
                        disk_peak_BW = disk_BW
                    processed_line += 1
                disk_avg_IO /= evaluate_time_number
                disk_avg_BW /= evaluate_time_number
                if disk_avg_IO == 0 and disk_peak_IO == 0:
                    IO_mul = 0
                else:
                    IO_mul = disk_peak_IO/disk_avg_IO
                if disk_avg_BW == 0 and disk_peak_BW == 0:
                    BW_mul = 0
                else:
                    BW_mul = disk_peak_BW/disk_avg_BW
                label = -1
                if IO_mul < IO_line and BW_mul < bandwidth_line:
                    label = 0
                else:
                    label = 1
                # labels (0 for stable, 1 for burst)
                if label == 1:
                    cluster_item.append(list(original_items[cluster_num][item_num_in_cluster][0:9])+[
                                        disk_avg_IO, disk_avg_BW, disk_peak_IO, disk_peak_BW])
        burst_items.append(cluster_item)
    return burst_items


# 计算每个集群每个磁盘在24小时内的IOPS和带宽的峰值出现的小时数
# traces(timestamp,RBW,WBW)
def peak_hour_in_oneday(traces, item_avg_RBW, item_avg_WBW):
    peak_RBW = -1.0
    peak_RBW_hour = []
    peak_WBW = -1.0
    peak_WBW_hour = []
    avg_RBW = 0.0
    avg_WBW = 0.0
    is_a_burst_day = 0
    for i in range(len(traces)):
        avg_RBW += traces[i][1]
        avg_WBW += traces[i][2]
        if traces[i][1] > peak_RBW:
            peak_RBW = traces[i][1]
            peak_RBW_hour = datetime.fromtimestamp(traces[i][0]).hour
        if traces[i][2] > peak_WBW:
            peak_WBW = traces[i][2]
            peak_WBW_hour = datetime.fromtimestamp(traces[i][0]).hour
    if item_avg_RBW == 0:
        if peak_RBW == 0:
            RBW_peak_avg_ratio = 0
        else:
            RBW_peak_avg_ratio = peak_RBW/0.0001
    else:
        RBW_peak_avg_ratio = peak_RBW/item_avg_RBW
    if item_avg_WBW == 0:
        if peak_WBW == 0:
            WBW_peak_avg_ratio = 0
        else:
            WBW_peak_avg_ratio = peak_WBW/0.0001
    else:
        WBW_peak_avg_ratio = peak_WBW/item_avg_WBW
    if RBW_peak_avg_ratio > bandwidth_line*0.7 or WBW_peak_avg_ratio > bandwidth_line*0.7:
        is_a_burst_day = 1
    return is_a_burst_day, peak_RBW_hour, peak_WBW_hour


def peak_hour_in_oneday_scheme1(traces, item_avg_RBW, item_avg_WBW):
    peak_RBW_hour = [0 for _ in range(24)]
    peak_WBW_hour = [0 for _ in range(24)]
    peak_BW_hour = [0 for _ in range(24)]
    avg_RBW = 0.0
    avg_WBW = 0.0
    avg_BW = 0.0
    is_a_burst_day = 0
    for i in range(len(traces)):
        avg_RBW += traces[i][1]
        avg_WBW += traces[i][2]
        avg_BW += traces[i][1]+traces[i][2]
    avg_BW /= len(traces)
    avg_RBW /= len(traces)
    avg_WBW /= len(traces)
    for i in range(len(traces)):
        if traces[i][1] > item_avg_RBW*bandwidth_line_day:
            peak_RBW_hour[datetime.fromtimestamp(traces[i][0]).hour] += 1
        if traces[i][2] > item_avg_WBW*bandwidth_line_day:
            peak_WBW_hour[datetime.fromtimestamp(traces[i][0]).hour] += 1
        if traces[i][1]+traces[i][2] > avg_BW*bandwidth_line_day:
            peak_BW_hour[datetime.fromtimestamp(traces[i][0]).hour] += 1
    return peak_RBW_hour, peak_WBW_hour, peak_BW_hour


def peak_hour_in_oneday_scheme2(traces, item_avg_RBW, item_avg_WBW):
    peak_RBW_hour = [0 for _ in range(24)]
    peak_WBW_hour = [0 for _ in range(24)]
    avg_RBW = 0.0
    avg_WBW = 0.0
    is_a_burst_day = 0
    for i in range(len(traces)):
        avg_RBW += traces[i][1]
        avg_WBW += traces[i][2]
    avg_RBW /= len(traces)
    avg_WBW /= len(traces)
    for i in range(len(traces)):
        if traces[i][1] > item_avg_RBW*bandwidth_line_day:
            peak_RBW_hour[datetime.fromtimestamp(traces[i][0]).hour] = 1
        if traces[i][2] > item_avg_WBW*bandwidth_line_day:
            peak_WBW_hour[datetime.fromtimestamp(traces[i][0]).hour] = 1
    return peak_RBW_hour, peak_WBW_hour

# 已弃用


def select_null_item(burst_items):
    with open('null_items', 'w') as f:
        for cluster_num in range(len(burst_items)):
            for item_num in range(len(burst_items[cluster_num])):
                item = burst_items[cluster_num][item_num]
                disk_id = item[0]
                item_avg_RBW = item[9]
                item_avg_WBW = item[10]
                if item_avg_RBW == 0 or item_avg_WBW == 0:
                    f.write(
                        f"{cluster_num},{disk_id},{item_avg_RBW},{item_avg_WBW}\n")

# 统计每个集群每个磁盘在每个24小时内的IOPS和带宽的峰值出现的小时数的次数
# item[9:13] average_disk_RBW, average_disk_WBW, peak_disk_RBW, peak_disk_WBW


def caculate_peaktime(burst_items):
    print("开始计算峰值时间")
    peak_time = [[[[0 for _ in range(24)] for _ in range(3)] for _ in range(len(
        burst_items[i]))]for i in range(len(burst_items))]  # 第一维:集群，第二维：磁盘，第三维：IOPS和带宽，第四维：24小时
    with open('peak_time_RBW', 'w') as f_readbw, open('peak_time_WBW', 'w') as f_writebw, open('peak_time_BW', 'w') as f_BW:
        for cluster_num in range(len(burst_items)):
            for item_num in range(len(burst_items[cluster_num])):
                if item_num % 1000 == 0:
                    print(f"正在处理集群 {cluster_num} 的磁盘 {item_num}")
                item = burst_items[cluster_num][item_num]
                disk_id = item[0]
                item_avg_RBW = item[9]
                item_avg_WBW = item[10]
                if item_avg_RBW == 0:
                    print(f"集群 {cluster_num} 磁盘 {item_num}的读平均带宽为0")
                if item_avg_WBW == 0:
                    print(f"集群 {cluster_num} 磁盘 {item_num}的写平均带宽为0")
                item_dir = f"/data/Tencent_CVD/Shanghai/20_136090{cluster_num}/{disk_id}"
                if not os.path.exists(item_dir):
                    print(f"文件不存在: {item_dir}")
                    continue
                with open(item_dir, "r") as f:
                    is_first_day = 0  # 是否是第一个00:00:00
                    last_hour = -1
                    traces_oneday = []
                    processed_line = 0
                    for line in f:
                        if processed_line >= evaluate_time_number:
                            break
                        fields = line.split(",")
                        if is_first_day == 0:
                            if datetime.fromtimestamp(int(fields[0])).hour == 0 and last_hour == 23:
                                is_first_day = 1
                                disk_RBW = float(fields[2])
                                disk_WBW = float(fields[4])
                                traces_oneday.append(
                                    [int(fields[0]), disk_RBW, disk_WBW])
                                last_hour = datetime.fromtimestamp(
                                    int(fields[0])).hour
                            last_hour = datetime.fromtimestamp(
                                int(fields[0])).hour
                            continue
                        if datetime.fromtimestamp(int(fields[0])).hour == 0 and last_hour == 23:
                            peak_RBW_hour, peak_WBW_hour, peak_BW_hour = peak_hour_in_oneday_scheme1(
                                traces_oneday, item_avg_RBW, item_avg_WBW)
                            # if is_burst_day == 1:
                            #     if peak_RBW_hour != -1:
                            #         peak_time[cluster_num][item_num][0][peak_RBW_hour] += 1
                            #     if peak_WBW_hour != -1:
                            #         peak_time[cluster_num][item_num][1][peak_WBW_hour] += 1
                            if len(peak_time[cluster_num][item_num][0]) == len(peak_RBW_hour):
                                for i in range(len(peak_time[cluster_num][item_num][0])):
                                    peak_time[cluster_num][item_num][0][i] += peak_RBW_hour[i]
                            else:
                                print("RBW列表长度不同，无法进行元素级相加")
                            if len(peak_time[cluster_num][item_num][1]) == len(peak_WBW_hour):
                                for i in range(len(peak_time[cluster_num][item_num][1])):
                                    peak_time[cluster_num][item_num][1][i] += peak_WBW_hour[i]
                            else:
                                print("RBW列表长度不同，无法进行元素级相加")
                            if len(peak_time[cluster_num][item_num][2]) == len(peak_WBW_hour):
                                for i in range(len(peak_time[cluster_num][item_num][2])):
                                    peak_time[cluster_num][item_num][2][i] += peak_BW_hour[i]
                            else:
                                print("RBW列表长度不同，无法进行元素级相加")
                            traces_oneday = []
                        disk_RBW = float(fields[2])
                        disk_WBW = float(fields[4])
                        traces_oneday.append(
                            [int(fields[0]), disk_RBW, disk_WBW])
                        processed_line += 1
                        last_hour = datetime.fromtimestamp(int(fields[0])).hour
                f_readbw.write(f"{cluster_num},{disk_id}," +
                               ','.join(map(str, peak_time[cluster_num][item_num][0])))
                f_readbw.write('\n')
                f_writebw.write(
                    f"{cluster_num},{disk_id},"+','.join(map(str, peak_time[cluster_num][item_num][1])))
                f_writebw.write('\n')
                f_BW.write(f"{cluster_num},{disk_id}," +
                           ','.join(map(str, peak_time[cluster_num][item_num][2])))
                f_BW.write('\n')
    return peak_time


def calculate_variance(frequencies):
    values = np.arange(len(frequencies))
    mean = np.average(values, weights=frequencies)
    variance = np.average((values - mean)**2, weights=frequencies)
    return variance


def calculate_all_variance_coefficients():
    RBW_var_scores = []
    WBW_var_scores = []
    with open('/home/wcl/cycle_dete/peak_time_RBW', 'r') as fout:
        with open('var_RBW', 'w') as fin:
            for line in fout:
                line = list(line.strip().split(','))
                line = line[2:]
                line = list(map(float, line))
                if np.sum(line) - 0.0 < 1e-6:
                    continue
                var_RBW = calculate_variance(line)
                fin.write(str(var_RBW))
                fin.write('\n')
                RBW_var_scores.append(var_RBW)
    with open('/home/wcl/cycle_dete/peak_time_WBW', 'r') as fout:
        with open('var_WBW', 'w') as fin:
            for line in fout:
                line = line.strip().split(',')
                line = line[2:]
                line = list(map(float, line))
                if np.sum(line) - 0.0 < 1e-6:
                    continue
                var_WBW = calculate_variance(line)
                fin.write(str(var_WBW))
                fin.write('\n')
                WBW_var_scores.append(var_WBW)
    return RBW_var_scores, WBW_var_scores


def plot_variance_density(readbw_scores, writebw_scores, title="Variance of All Disk"):
    fig, axes = plt.subplots(1, 2, figsize=(
        14, 6), sharey=True)  # sharey=True 使密度刻度可比
    fig.suptitle(title, fontsize=16)  # 设置总标题
    if readbw_scores:
        sns.kdeplot(readbw_scores, ax=axes[0],
                    fill=True, color='skyblue', label='RBW')
        axes[0].set_title('readBandwidth peak hourly variance density')
        axes[0].set_xlabel('variance (RBW)')
        axes[0].set_ylabel('PDF')
        # axes[0].set_xlim(0, 1) # 基尼系数范围是 0 到 1
        axes[0].grid(True, linestyle='--', alpha=0.6)  # 添加网格线
        # (可选) 添加平均值和中位数线
        mean_iops = np.mean(readbw_scores)
        median_iops = np.median(readbw_scores)
        axes[0].axvline(mean_iops, color='red', linestyle='dashed',
                        linewidth=1, label=f'mean: {mean_iops:.2f}')
        axes[0].axvline(median_iops, color='green', linestyle='dotted',
                        linewidth=1, label=f'median: {median_iops:.2f}')
        axes[0].legend()  # 显示图例
    else:
        # 如果没有有效的 IOPS 数据
        axes[0].set_title('无有效的 IOPS 基尼系数')
        axes[0].text(0.5, 0.5, '无数据', horizontalalignment='center',
                     verticalalignment='center', transform=axes[0].transAxes)

    # --- 绘制 BW 基尼系数密度图 ---
    if writebw_scores:
        sns.kdeplot(writebw_scores,
                    ax=axes[1], fill=True, color='lightcoral', label='WBW')
        axes[1].set_title('WBW peak hourly variance density')
        axes[1].set_xlabel('variance (WBW)')
        # axes[1].set_ylabel('密度') # Y轴标签已共享
        # axes[1].set_xlim(0, 1) # 基尼系数范围是 0 到 1
        axes[1].grid(True, linestyle='--', alpha=0.6)  # 添加网格线
        # (可选) 添加平均值和中位数线
        mean_bw = np.mean(writebw_scores)
        median_bw = np.median(writebw_scores)
        axes[1].axvline(mean_bw, color='red', linestyle='dashed',
                        linewidth=1, label=f'mean: {mean_bw:.2f}')
        axes[1].axvline(median_bw, color='green', linestyle='dotted',
                        linewidth=1, label=f'median: {median_bw:.2f}')
        axes[1].legend()  # 显示图例
    else:
        # 如果没有有效的 BW 数据
        axes[1].set_title('无有效的带宽基尼系数')
        axes[1].text(0.5, 0.5, '无数据', horizontalalignment='center',
                     verticalalignment='center', transform=axes[1].transAxes)

    # 调整布局以避免重叠，并显示图形
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整矩形区域为总标题留出空间
    fig.savefig('./cdf.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭图形以释放内存


def plot_disk_trace10000(cluster_index, disk_id):
    interval_size = 288
    num_labels = 7
    RBW = []
    WBW = []
    BW = []
    item_dir = f"/data/Tencent_CVD/Shanghai/20_136090{cluster_index}/{disk_id}"
    with open(item_dir, "r") as f:
        first_line = f.readline()
        first_line = first_line.strip().split(",")
        first_day = datetime.fromtimestamp(int(first_line[0])).day
        begin = 0
        for line in f:
            fields = line.split(",")
            if begin == 0:
                if datetime.fromtimestamp(int(fields[0])).day != first_day and datetime.fromtimestamp(int(fields[0])).isoweekday() == 1:
                    begin = 1
                    processed_line = 0
                else:
                    continue
            if processed_line >= 2016:
                break
            disk_RBW = float(fields[2])
            disk_WBW = float(fields[4])
            disk_BW = disk_RBW + disk_WBW
            RBW.append(disk_RBW)
            WBW.append(disk_WBW)
            BW.append(disk_BW)
            processed_line += 1
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(f"Disk {disk_id} trace in a week", fontsize=16)
    fig.supxlabel('TimeStamp (weekday)', fontsize=14)
    tick_locations = [interval_size * k for k in range(num_labels + 1)]
    tick_labels = [str(k) for k in range(num_labels + 1)]
    axes[0].plot(range(len(RBW)), RBW, color='skyblue', label='RBW')
    axes[0].set_ylabel('RBW')
    axes[0].set_xticks(tick_locations, tick_labels)
    axes[1].plot(range(len(WBW)), WBW, color='skyblue', label='WBW')
    axes[1].set_ylabel('WBW')
    axes[1].set_xticks(tick_locations, tick_labels)
    axes[2].plot(range(len(BW)), BW, color='skyblue', label='BW')
    axes[2].set_ylabel('BW')
    axes[2].set_xticks(tick_locations, tick_labels)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_dir = f'./trace/warehouse{cluster_index}/{disk_id}'
    if not os.path.exists(f'./trace/warehouse{cluster_index}'):
        os.makedirs(f'./trace/warehouse{cluster_index}')
    fig.savefig(fig_dir, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


def plot_disk_all_trace(cluster_index, disk_id):
    RBW = []
    WBW = []
    item_dir = f"/data/Tencent_CVD/Shanghai/20_136090{cluster_index}/{disk_id}"
    with open(item_dir, "r") as f:
        processed_line = 0
        for line in f:
            fields = line.split(",")
            disk_RBW = float(fields[2])
            disk_WBW = float(fields[4])
            RBW.append(disk_RBW)
            WBW.append(disk_WBW)
            processed_line += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Disk{disk_id} trace", fontsize=16)
    fig.supxlabel('TimeStamp', fontsize=14)
    axes[0].plot(range(len(RBW)), RBW, color='skyblue', label='RBW')
    axes[0].set_ylabel('RBW')
    axes[1].plot(range(len(WBW)), WBW, color='skyblue', label='WBW')
    axes[1].set_ylabel('WBW')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_dir = f'./all_trace/warehouse{cluster_index}/{disk_id}'
    if not os.path.exists(f'./all_trace/warehouse{cluster_index}'):
        os.makedirs(f'./all_trace/warehouse{cluster_index}')
    fig.savefig(fig_dir, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


def plot_disk_peak_distribution():
    # with open ('peak_time_RBW','r') as frbw, open ('peak_time_WBW','r') as fwbw:
    #     disk_index=0
    #     for rbw,wbw in zip(frbw,fwbw):
    #         if disk_index % 500 != 0 :
    #             disk_index+=1
    #             continue
    #         rbw=list(map(float,rbw.strip().split(',')))
    #         wbw=list(map(float,wbw.strip().split(',')))
    #         fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    #         fig.suptitle(f"Disk{disk_index} Peak Distribution", fontsize=16)
    #         axes[0].bar(range(24),rbw,color='skyblue',  edgecolor='black',alpha=0.8,label='RBW')
    #         axes[0].set_xlabel('peak distribution (RBW)')
    #         axes[1].bar(range(24),wbw,color='skyblue',  edgecolor='black',alpha=0.8,label='WBW')
    #         axes[1].set_xlabel('peak distribution (WBW)')
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         if not os.path.exists(f'./peak_distribution'):
    #             os.makedirs(f'./peak_distribution')
    #         fig.savefig(f'./peak_distribution/{disk_index}.png', dpi=300, bbox_inches='tight')
    #         plt.close(fig)
    #         disk_index+=1
    #         print(f"正在画峰值分布图 {disk_index-1}")
    with open('peak_time_BW', 'r') as fbw:
        disk_index = 0
        for bw in fbw:
            if disk_index % 500 != 0:
                disk_index += 1
                continue
            bw = bw.strip().split(',')
            cluster_index = bw[0]
            disk_id = bw[1]
            bw[2:] = list(map(int, bw[2:]))
            plot_disk_trace10000(cluster_index, disk_id)
            plt.figure(figsize=(14, 6))
            plt.title(f"Disk{disk_id} Peak Distribution", fontsize=16)
            plt.bar(range(24), bw[2:], color='skyblue',
                    edgecolor='black', alpha=0.8, label='BW')
            plt.xlabel('peak distribution (BW)')
            plt.ylabel('count')
            plt.xticks(range(24), rotation=45)
            if not os.path.exists(f'./peak_distribution'):
                os.makedirs(f'./peak_distribution')
            plt.savefig(
                f'./peak_distribution/{disk_index}.png', dpi=300, bbox_inches='tight')
            disk_index += 1
            print(f"正在画峰值分布图 {disk_index-1}")
            plt.close()


if __name__ == "__main__":
    # init_warehouse()
    # burst_items=load_item()
    # caculate_peaktime(burst_items)
    # RBW_variance_scores, WBW_variance_scores = calculate_all_variance_coefficients()

    # plot_variance_density(RBW_variance_scores, WBW_variance_scores)
    stable_items = load_stable_item()
    for cluster_index in range(5):
        for disk_index in range(len(stable_items[cluster_index])):
            if disk_index % 500 != 0:
                continue
            disk_id = stable_items[cluster_index][disk_index][0]
            plot_disk_trace10000(cluster_index, disk_id)

    # print("正在画磁盘trace")
