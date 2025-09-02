import numpy as np
import random

# 轮次
episodes = 20
# dataset_warehouse0.txt   dataset_warehouse2.txt
# 装载货物数量 = len(items) - episode_remain
# 共11058件
#episode_remain = 7058
episode_remain = 9000
# 评价时间数量
evaluate_time_number = 16000
# 时序间隔
time_interval = 1
# 最小时间戳记录
min_timestamp_num = 10000

time_window_length = 1000
max_violation_time_line_window = 20
reservation_rate_for_monitor = 0.95

# 仓库数量
warehouse_number = 9

# 6个仓库
# max_capacity = [140000, 140000, 140000, 140000, 100000, 100000]
# max_IOPS = [400000, 400000, 280000, 280000, 400000, 400000]
# max_bandwidth = [320000, 320000, 4800000, 480000, 480000, 480000]

# 9个仓库
max_capacity = [ 100000,  100000,  100000 , 100000 , 100000 , 100000 , 70000 , 70000 , 70000]
max_IOPS = [450000, 450000, 450000, 260000, 260000, 260000, 280000, 280000, 280000]
max_bandwidth = [300000, 300000, 300000, 600000, 600000, 600000, 320000, 320000, 320000]

# 9个同构仓库
# max_capacity = [100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000]
# max_IOPS = [300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000]
# max_bandwidth = [350000, 350000, 350000, 350000, 350000, 350000, 350000, 350000, 350000]

# 12个仓库
# max_capacity = [140000, 140000, 140000, 140000, 140000, 140000, 140000, 140000, 100000, 100000, 100000, 100000]
# max_IOPS = [400000, 400000, 400000, 400000, 280000, 280000, 280000, 280000, 400000, 400000, 400000, 400000]
# max_bandwidth = [320000, 320000, 320000, 320000, 480000, 480000, 480000, 480000, 480000, 480000, 480000, 480000]

# 15个仓库
# max_capacity = [140000, 140000, 140000, 140000, 140000, 140000, 140000, 140000, 140000, 140000, 100000, 100000, 100000, 100000, 100000]
# max_IOPS = [400000, 400000, 400000, 400000, 400000, 280000, 280000, 280000, 280000, 280000, 400000, 400000, 400000, 400000, 400000]
# max_bandwidth = [320000, 320000, 320000, 320000, 320000, 480000, 480000, 480000, 480000, 480000, 480000, 480000, 480000, 480000, 480000]

# max_capacity = [170000, 170000, 170000, 170000, 170000, 170000, 170000, 170000, 170000]
# max_IOPS = [500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000]
# max_bandwidth = [600000, 600000, 600000, 600000, 600000, 600000, 600000, 600000, 600000]

# for i in range(warehouse_number):
#     max_capacity[i] *= 0.5
#     max_IOPS[i] *= 0.5
#     max_bandwidth[i] *= 0.5

warehouse_max = np.array([max_capacity, max_IOPS, max_bandwidth])

# 预留比例
reservation_rate = 1

IO_line = 100
bandwidth_line = 50

# items_total_num = 10974
items_total_num = 16160

def generate_random_order():
    source_file = '/data2/cloud_disk_code/CDA_new_2/items_seq_16160.txt'
    items_seq = [[] for _ in range(episodes)]
    i = 0
    with open(source_file, "r") as sf:
        for line in sf:
            if i == episodes:
                break
            items_seq_f = line.strip().split(',')
            items_seq_f[0:items_total_num] = map(int, items_seq_f[0:items_total_num])
            items_seq[i] = items_seq_f[0:items_total_num]
            i += 1
    return items_seq

random_items_seq = generate_random_order()
