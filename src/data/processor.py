import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Tuple, Any, Optional

from config.settings import DirConfig, DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self):
        pass

    def _iterate_first_day(self, f_trace) -> int:
        """
        计算第一天的行数
        """
        line_len = 0
        last_hour = -1
        while True:
            line = f_trace.readline()
            if not line:
                break
            fileds = line.split(",")
            timestamp = int(fileds[0])
            current_hour = datetime.fromtimestamp(timestamp).hour
            if current_hour == 0 and last_hour == 23:
                line_len = len(line)
                break
            last_hour = current_hour
        return line_len

    def _iterate_first_week(self, f_trace) -> int:
        """
        计算第一周的行数
        """
        last_hour = -1
        last_day = -1
        line_len = 0
        while True:
            line = f_trace.readline()
            if not line:
                break
            fileds = line.split(",")
            timestamp = int(fileds[0])
            current_hour = datetime.fromtimestamp(timestamp).hour
            current_day = datetime.fromtimestamp(timestamp).weekday()
            if current_hour == 0 and last_hour == 23:
                if current_day == 0 and last_day == 6:
                    line_len = len(line)
                    break
                last_day = current_day
            last_hour = current_hour
        return line_len


class DiskDataProcessor(Processor):
    """磁盘数据处理器"""

    def __init__(self):
        pass

    def calculate_peak_time(self, burst_items: List[Any]) -> List[List[List[List[int]]]]:
        """
        计算每个集群每个磁盘在24小时内的IOPS和带宽的峰值出现的小时数
        Args:
            burst_items: 突发性磁盘数据
        Returns:
            峰值时间统计数据
        """
        logger.info("开始计算峰值时间")

        # 初始化峰值时间数组：[磁盘][类型(RBW/WBW/BW)][24小时]
        peak_time = [[[0 for _ in range(24)] for _ in range(3)]
                     for _ in range(len(burst_items))]

        # 输出文件路径
        rbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_RBW')
        wbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_WBW')
        bw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_BW')

        with open(rbw_file, 'w') as f_readbw, \
                open(wbw_file, 'w') as f_writebw, \
                open(bw_file, 'w') as f_BW:

            for item_num in range(len(burst_items)):
                if item_num % 1000 == 0:
                    logger.info(f"正在处理第{item_num}磁盘 ")

                item = burst_items[item_num]
                disk_id = item[0]
                item_avg_rbw = item[9]
                item_avg_wbw = item[10]
                cluster_index = item[15]
                if item_avg_rbw == 0:
                    logger.warning(
                        f"集群 {cluster_index} 磁盘 {item_num} 的读平均带宽为0")
                if item_avg_wbw == 0:
                    logger.warning(
                        f"集群 {cluster_index} 磁盘 {item_num} 的写平均带宽为0")

                # 处理磁盘trace数据
                peak_hours = self._process_disk_traces(
                    cluster_index, disk_id, item_avg_rbw, item_avg_wbw
                )

                # 累加峰值时间统计
                for i in range(3):  # RBW, WBW, BW
                    for j in range(24):  # 24小时
                        peak_time[item_num][i][j] += peak_hours[i][j]

                # 写入结果文件
                self._write_peak_time_results(
                    f_readbw, f_writebw, f_BW,
                    cluster_index, disk_id, peak_time[item_num]
                )

        return peak_time

    def _process_disk_traces(self, cluster_index: int, disk_id: str,
                             item_avg_rbw: float, item_avg_wbw: float) -> List[List[int]]:
        """
        处理单个磁盘的追踪数据

        Args:
            cluster_index: 集群编号
            disk_id: 磁盘ID
            item_avg_rbw: 平均读带宽
            item_avg_wbw: 平均写带宽

        Returns:
            峰值小时统计 [RBW, WBW, BW][24小时]
        """
        item_dir = os.path.join(
            DirConfig.TRACE_ROOT,
            f"20_136090{cluster_index}",
            str(disk_id)
        )
        peak_hours = [[0 for _ in range(24)] for _ in range(3)]  # [type][hour]
        if not os.path.exists(item_dir):
            logger.warning(f"文件不存在: {item_dir}")
            return peak_hours

        try:
            with open(item_dir, "r") as f:
                last_hour = -1
                traces_oneday = []
                processed_line = -1
                line_len = self._iterate_first_day(f)
                file_current_pos = f.tell()
                f.seek(file_current_pos-line_len)
                for line in f:
                    processed_line += 1
                    if processed_line >= DataConfig.EVALUATE_TIME_NUMBER:
                        break
                    fields = line.split(",")
                    timestamp = int(fields[0])
                    current_hour = datetime.fromtimestamp(timestamp).hour
                    if current_hour == 0 and last_hour == 23:
                        # 处理前一天的数据
                        if traces_oneday:
                            day_peak_hours = self._calculate_daily_peak_hours(
                                traces_oneday, item_avg_rbw, item_avg_wbw
                            )
                            # 累加到总的峰值时间统计
                            for i in range(3):
                                for j in range(24):
                                    peak_hours[i][j] += day_peak_hours[i][j]
                        traces_oneday = []

                    # 收集当天数据
                    disk_rbw = float(fields[2])
                    disk_wbw = float(fields[4])
                    traces_oneday.append([timestamp, disk_rbw, disk_wbw])
                    last_hour = current_hour

        except (IOError, OSError) as e:
            logger.error(f"读取文件 {item_dir} 出错: {e}")

        return peak_hours

    def _calculate_daily_peak_hours(self, traces_oneday: List[List[float]],
                                    item_avg_rbw: float, item_avg_wbw: float) -> List[List[int]]:
        """
        计算单天的峰值小时分布

        Args:
            traces_oneday: 单天的追踪数据
            item_avg_rbw: 平均读带宽
            item_avg_wbw: 平均写带宽

        Returns:
            单天峰值小时统计 [RBW, WBW, BW][24小时]
        """
        peak_rbw_hour = [0 for _ in range(24)]
        peak_wbw_hour = [0 for _ in range(24)]
        peak_bw_hour = [0 for _ in range(24)]

        # 计算当天平均值
        avg_rbw = sum(trace[1] for trace in traces_oneday) / len(traces_oneday)
        avg_wbw = sum(trace[2] for trace in traces_oneday) / len(traces_oneday)
        avg_bw = avg_rbw + avg_wbw

        # 标记峰值小时
        for trace in traces_oneday:
            hour = datetime.fromtimestamp(trace[0]).hour

            if trace[1] > item_avg_rbw * ModelConfig.BANDWIDTH_LINE_DAY:
                peak_rbw_hour[hour] += 1

            if trace[2] > item_avg_wbw * ModelConfig.BANDWIDTH_LINE_DAY:
                peak_wbw_hour[hour] += 1

            if (trace[1] + trace[2]) > (item_avg_rbw + item_avg_wbw) * ModelConfig.BANDWIDTH_LINE_DAY:
                peak_bw_hour[hour] += 1

        return [peak_rbw_hour, peak_wbw_hour, peak_bw_hour]

    def _write_peak_time_results(self, f_readbw, f_writebw, f_bw,
                                 cluster_index: int, disk_id: str,
                                 peak_time_data: List[List[int]]):
        """
        写入峰值时间结果到文件

        Args:
            f_readbw: 读带宽文件句柄
            f_writebw: 写带宽文件句柄
            f_bw: 总带宽文件句柄
            cluster_num: 集群编号
            disk_id: 磁盘ID
            peak_time_data: 峰值时间数据
        """
        f_readbw.write(f"{cluster_index},{disk_id}," +
                       ','.join(map(str, peak_time_data[0])) + '\n')
        f_writebw.write(f"{cluster_index},{disk_id}," +
                        ','.join(map(str, peak_time_data[1])) + '\n')
        f_bw.write(f"{cluster_index},{disk_id}," +
                   ','.join(map(str, peak_time_data[2])) + '\n')

    def calculate_variance(self, frequencies: List[float]) -> float:
        """
        计算方差

        Args:
            frequencies: 频率数组

        Returns:
            float: 方差值
        """
        if not frequencies or sum(frequencies) == 0:
            return 0.0

        values = np.arange(len(frequencies))
        mean = np.average(values, weights=frequencies)
        variance = np.average((values - mean) ** 2, weights=frequencies)
        return variance

    def calculate_all_variance_coefficients(self) -> Tuple[List[float], List[float]]:
        """
        计算所有磁盘的方差系数

        Returns:
            Tuple[List[float], List[float]]: (RBW方差列表, WBW方差列表)
        """
        rbw_var_scores = []
        wbw_var_scores = []

        # 读取RBW峰值时间数据并计算方差
        rbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_RBW')
        var_rbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'var_RBW')

        with open(rbw_file, 'r') as fout, open(var_rbw_file, 'w') as fin:
            for line in fout:
                line_data = line.strip().split(
                    ',')[2:]  # 跳过cluster_num和disk_id
                frequencies = list(map(float, line_data))

                if sum(frequencies) < 1e-6:  # 避免全零数据
                    continue

                var_rbw = self.calculate_variance(frequencies)
                fin.write(f"{var_rbw}\n")
                rbw_var_scores.append(var_rbw)

        # 读取WBW峰值时间数据并计算方差
        wbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_WBW')
        var_wbw_file = os.path.join(DirConfig.OUTPUT_DIR, 'var_WBW')

        with open(wbw_file, 'r') as fout, open(var_wbw_file, 'w') as fin:
            for line in fout:
                line_data = line.strip().split(
                    ',')[2:]  # 跳过cluster_num和disk_id
                frequencies = list(map(float, line_data))

                if sum(frequencies) < 1e-6:  # 避免全零数据
                    continue

                var_wbw = self.calculate_variance(frequencies)
                fin.write(f"{var_wbw}\n")
                wbw_var_scores.append(var_wbw)

        return rbw_var_scores, wbw_var_scores


class CoachProcessor(Processor):
    def __init__(self):
        pass

    def time_window_divide(self, item: List[Any], time_window_length: int) -> List[List[Any]]:
        """
        将数据按时间窗口长度分割
        Args:
            item: 磁盘数据
            time_window_length: 时间窗口长度
            cluster_index: 集群编号
        Returns:
            按照时间窗口分割后,计算出每个窗口的最大利用率,[rbw][wbw]
        """
        time_window_data_rbw = []
        time_window_data_wbw = []
        time_window_boundaries = [time_window_length *
                                  i for i in range(0, 24//time_window_length)]
        item_dir = os.path.join(DirConfig.TRACE_ROOT,
                                f"20_136090{item[15]}", str(item[0]))
        if not os.path.exists(item_dir):
            logger.warning(f"文件不存在: {item_dir}")
            return []
        with open(item_dir, "r") as f:
            processed_line = -1
            line_len = self._iterate_first_week(f)
            file_current_pos = f.tell()
            f.seek(file_current_pos-line_len)
            last_hour = -1
            traces_time_window = []
            traces_oneday_rbw = []
            traces_oneday_wbw = []
            for line in f:
                processed_line += 1
                if processed_line > 2016:
                    break
                fields = line.split(",")
                timestamp = int(fields[0])
                current_hour = datetime.fromtimestamp(timestamp).hour
                if current_hour in time_window_boundaries and last_hour not in time_window_boundaries and traces_time_window:
                    index = time_window_boundaries.index(current_hour)
                    peak_value_rbw, peak_value_wbw = self._get_peak_value_of_window(
                        traces_time_window)
                    traces_oneday_rbw.append(peak_value_rbw)
                    traces_oneday_wbw.append(peak_value_wbw)
                    traces_time_window = []
                    if index == 0 and traces_oneday_rbw and traces_oneday_wbw:
                        time_window_data_rbw.append(traces_oneday_rbw)
                        time_window_data_wbw.append(traces_oneday_wbw)
                        traces_oneday_rbw = []
                        traces_oneday_wbw = []
                disk_rbw = float(fields[2])
                disk_wbw = float(fields[4])
                traces_time_window.append([timestamp, disk_rbw, disk_wbw])
                last_hour = current_hour
        return time_window_data_rbw, time_window_data_wbw

    def _get_peak_value_of_window(self, traces_time_window: List[List[float]]) -> Tuple[float, float]:
        """
        计算时间窗口内的最大利用率
        """
        peak_value_rbw = 0
        peak_value_wbw = 0
        for trace in traces_time_window:
            if trace[1] > peak_value_rbw:
                peak_value_rbw = trace[1]
            if trace[2] > peak_value_wbw:
                peak_value_wbw = trace[2]
        return peak_value_rbw, peak_value_wbw

    def _write_time_window_data(self, time_window_data_rbw: List[List[float]], time_window_data_wbw: List[List[float]]) -> None:
        """
        写入时间窗口数据
        """
        time_window_data_rbw_file = os.path.join(
            DirConfig.OUTPUT_DIR, 'time_window_data_rbw')
        time_window_data_wbw_file = os.path.join(
            DirConfig.OUTPUT_DIR, 'time_window_data_wbw')
        with open(time_window_data_rbw_file, 'w') as f_rbw, open(time_window_data_wbw_file, 'w') as f_wbw:
            for i in range(len(time_window_data_rbw)):
                f_rbw.write(",".join(map(str, time_window_data_rbw[i])) + "\n")
                f_wbw.write(",".join(map(str, time_window_data_wbw[i])) + "\n")

    def _peak_valley_window_of_day(self, time_window_data: List[List[float]]) -> Tuple[List[int], List[int]]:
        """
        计算一周每天的峰值和谷值出现的时间窗口
        """
        peak_window = [-1 for _ in range(len(time_window_data))]
        valley_window = [-1 for _ in range(len(time_window_data))]
        for i in range(len(time_window_data)):
            peak_value = 0
            valley_value = float('inf')
            for j in range(len(time_window_data[i])):
                if time_window_data[i][j] > peak_value:
                    peak_value = time_window_data[i][j]
                    peak_window[i] = j
                if time_window_data[i][j] < valley_value:
                    valley_value = time_window_data[i][j]
                    valley_window[i] = j
        return peak_window, valley_window

    def statistic_peak_valley_window_of_week(self, items: List[Any], time_window_length: int):
        rbw_peak = [[0 for _ in range(24//time_window_length)]
                    for _ in range(7)]
        rbw_valley = [[0 for _ in range(24//time_window_length)]
                      for _ in range(7)]
        wbw_peak = [[0 for _ in range(24//time_window_length)]
                    for _ in range(7)]
        wbw_valley = [[0 for _ in range(24//time_window_length)]
                      for _ in range(7)]
        for item in items:
            time_window_data_rbw, time_window_data_wbw = self.time_window_divide(
                item, time_window_length)
            rbw_peak_window, rbw_valley_window = self._peak_valley_window_of_day(
                time_window_data_rbw)
            wbw_peak_window, wbw_valley_window = self._peak_valley_window_of_day(
                time_window_data_wbw)
            for i in range(7):
                rbw_peak_weekdayi_windows = rbw_peak_window[i]
                rbw_valley_weekdayi_windows = rbw_valley_window[i]
                wbw_peak_weekdayi_windows = wbw_peak_window[i]
                wbw_valley_weekdayi_windows = wbw_valley_window[i]
                rbw_peak[i][rbw_peak_weekdayi_windows] += 1
                rbw_valley[i][rbw_valley_weekdayi_windows] += 1
                wbw_peak[i][wbw_peak_weekdayi_windows] += 1
                wbw_valley[i][wbw_valley_weekdayi_windows] += 1
        return rbw_peak, rbw_valley, wbw_peak, wbw_valley
