from config.settings import DirConfig, DataConfig
from typing import List, Tuple, Optional
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')


logger = logging.getLogger(__name__)


class DiskPlotter:
    """磁盘数据可视化绘图器"""

    def __init__(self):
        self.config = DirConfig()
        self.data_config = DataConfig()

        # 设置中文字体支持
        # 尝试多种中文字体，按优先级排列
        plt.rcParams['font.family'] = [
            'Noto Serif CJK SC'
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    def plot_variance_density(self, rbw_scores: List[float], wbw_scores: List[float],
                              title: str = "所有磁盘的方差分析") -> None:
        """
        绘制方差密度图

        Args:
            rbw_scores: 读带宽方差分数列表
            wbw_scores: 写带宽方差分数列表
            title: 图表标题
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(title, fontsize=16)

        # 绘制RBW方差密度图
        if rbw_scores:
            sns.kdeplot(
                rbw_scores, ax=axes[0], fill=True, color='skyblue', label='RBW')
            axes[0].set_title('读带宽峰值小时方差密度')
            axes[0].set_xlabel('方差 (RBW)')
            axes[0].set_ylabel('概率密度')
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # 添加统计线
            mean_rbw = np.mean(rbw_scores)
            median_rbw = np.median(rbw_scores)
            axes[0].axvline(mean_rbw, color='red', linestyle='dashed',
                            linewidth=1, label=f'均值: {mean_rbw:.2f}')
            axes[0].axvline(median_rbw, color='green', linestyle='dotted',
                            linewidth=1, label=f'中位数: {median_rbw:.2f}')
            axes[0].legend()
        else:
            axes[0].set_title('无有效的RBW方差数据')
            axes[0].text(0.5, 0.5, '无数据', ha='center', va='center',
                         transform=axes[0].transAxes)

        # 绘制WBW方差密度图
        if wbw_scores:
            sns.kdeplot(
                wbw_scores, ax=axes[1], fill=True, color='lightcoral', label='WBW')
            axes[1].set_title('写带宽峰值小时方差密度')
            axes[1].set_xlabel('方差 (WBW)')
            axes[1].grid(True, linestyle='--', alpha=0.6)

            # 添加统计线
            mean_wbw = np.mean(wbw_scores)
            median_wbw = np.median(wbw_scores)
            axes[1].axvline(mean_wbw, color='red', linestyle='dashed',
                            linewidth=1, label=f'均值: {mean_wbw:.2f}')
            axes[1].axvline(median_wbw, color='green', linestyle='dotted',
                            linewidth=1, label=f'中位数: {median_wbw:.2f}')
            axes[1].legend()
        else:
            axes[1].set_title('无有效的WBW方差数据')
            axes[1].text(0.5, 0.5, '无数据', ha='center', va='center',
                         transform=axes[1].transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图片
        output_file = os.path.join(
            DirConfig.OUTPUT_DIR, 'variance_density.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"方差密度图已保存至: {output_file}")

    def plot_disk_trace(self, cluster_index: int, disk_id: str,
                        trace_type: str = "week") -> None:
        """
        绘制磁盘追踪图

        Args:
            cluster_index: 集群索引
            disk_id: 磁盘ID
            trace_type: 追踪类型 ("week" 或 "all")
        """
        if trace_type == "week":
            self._plot_disk_week_trace(cluster_index, disk_id)
        elif trace_type == "all":
            self._plot_disk_all_trace(cluster_index, disk_id)
        else:
            logger.error(f"不支持的追踪类型: {trace_type}")

    def _plot_disk_week_trace(self, cluster_index: int, disk_id: str) -> None:
        """
        绘制磁盘一周的追踪图

        Args:
            cluster_index: 集群索引
            disk_id: 磁盘ID
        """
        item_dir = os.path.join(
            DirConfig.TRACE_ROOT,
            f"20_136090{cluster_index}",
            str(disk_id)
        )

        if not os.path.exists(item_dir):
            logger.warning(f"文件不存在: {item_dir}")
            return

        interval_size = 288  # 一天的数据点数
        num_labels = 7  # 一周7天
        rbw_data = []
        wbw_data = []
        bw_data = []

        try:
            with open(item_dir, "r") as f:
                first_line = f.readline()
                first_fields = first_line.strip().split(",")
                first_day = datetime.fromtimestamp(int(first_fields[0])).day

                begin = False
                processed_line = 0

                for line in f:
                    fields = line.split(",")
                    timestamp = int(fields[0])
                    current_day = datetime.fromtimestamp(timestamp).day
                    current_weekday = datetime.fromtimestamp(
                        timestamp).isoweekday()

                    # 等待到周一开始
                    if not begin:
                        if current_day != first_day and current_weekday == 1:
                            begin = True
                            processed_line = 0
                        else:
                            continue

                    # 收集一周的数据
                    if processed_line >= 2016:  # 7天 * 288数据点/天
                        break

                    disk_rbw = float(fields[2])
                    disk_wbw = float(fields[4])
                    disk_bw = disk_rbw + disk_wbw

                    rbw_data.append(disk_rbw)
                    wbw_data.append(disk_wbw)
                    bw_data.append(disk_bw)
                    processed_line += 1

        except (IOError, OSError) as e:
            logger.error(f"读取文件 {item_dir} 出错: {e}")
            return

        # 绘制图表
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        fig.suptitle(f"磁盘 {disk_id} 一周追踪数据", fontsize=16)
        fig.supxlabel('时间戳 (工作日)', fontsize=14)

        # 设置x轴标签
        tick_locations = [interval_size * k for k in range(num_labels + 1)]
        tick_labels = [str(k) for k in range(num_labels + 1)]

        # 绘制RBW
        axes[0].plot(range(len(rbw_data)), rbw_data,
                     color='skyblue', label='RBW')
        axes[0].set_ylabel('读带宽 (RBW)')
        axes[0].set_xticks(tick_locations, tick_labels)
        axes[0].grid(True, alpha=0.3)

        # 绘制WBW
        axes[1].plot(range(len(wbw_data)), wbw_data,
                     color='lightcoral', label='WBW')
        axes[1].set_ylabel('写带宽 (WBW)')
        axes[1].set_xticks(tick_locations, tick_labels)
        axes[1].grid(True, alpha=0.3)

        # 绘制总带宽
        axes[2].plot(range(len(bw_data)), bw_data, color='green', label='总带宽')
        axes[2].set_ylabel('总带宽 (BW)')
        axes[2].set_xticks(tick_locations, tick_labels)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图片
        trace_dir = os.path.join(
            DirConfig.VISUALIZATION_TRACE_DIR, f'warehouse{cluster_index}')
        os.makedirs(trace_dir, exist_ok=True)
        output_file = os.path.join(trace_dir, f'{disk_id}_week.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"磁盘一周追踪图已保存至: {output_file}")

    def _plot_disk_all_trace(self, cluster_index: int, disk_id: str) -> None:
        """
        绘制磁盘所有追踪数据

        Args:
            cluster_index: 集群索引
            disk_id: 磁盘ID
        """
        item_dir = os.path.join(
            DirConfig.TRACE_ROOT,
            f"20_136090{cluster_index}",
            str(disk_id)
        )

        if not os.path.exists(item_dir):
            logger.warning(f"文件不存在: {item_dir}")
            return

        rbw_data = []
        wbw_data = []

        try:
            with open(item_dir, "r") as f:
                for line in f:
                    fields = line.split(",")
                    disk_rbw = float(fields[2])
                    disk_wbw = float(fields[4])
                    rbw_data.append(disk_rbw)
                    wbw_data.append(disk_wbw)

        except (IOError, OSError) as e:
            logger.error(f"读取文件 {item_dir} 出错: {e}")
            return

        # 绘制图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"磁盘 {disk_id} 所有追踪数据", fontsize=16)
        fig.supxlabel('时间戳', fontsize=14)

        # 绘制RBW
        axes[0].plot(range(len(rbw_data)), rbw_data,
                     color='skyblue', label='RBW')
        axes[0].set_ylabel('读带宽 (RBW)')
        axes[0].grid(True, alpha=0.3)

        # 绘制WBW
        axes[1].plot(range(len(wbw_data)), wbw_data,
                     color='lightcoral', label='WBW')
        axes[1].set_ylabel('写带宽 (WBW)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图片
        trace_dir = os.path.join(
            DirConfig.VISUALIZATION_TRACE_DIR, f'warehouse{cluster_index}')
        os.makedirs(trace_dir, exist_ok=True)
        output_file = os.path.join(trace_dir, f'{disk_id}_all.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"磁盘所有追踪图已保存至: {output_file}")

    def plot_peak_distribution(self, sample_interval: int = 500) -> None:
        """
        绘制磁盘峰值分布图

        Args:
            sample_interval: 采样间隔，每隔多少个磁盘画一次图
        """
        bw_file = os.path.join(DirConfig.OUTPUT_DIR, 'peak_time_BW')

        if not os.path.exists(bw_file):
            logger.warning(f"峰值时间文件不存在: {bw_file}")
            return

        disk_index = 0

        with open(bw_file, 'r') as f:
            for line in f:
                if disk_index % sample_interval != 0:
                    disk_index += 1
                    continue

                data = line.strip().split(',')
                cluster_index = data[0]
                disk_id = data[1]
                bw_distribution = list(map(int, data[2:]))

                # 绘制峰值分布柱状图
                fig, ax = plt.subplots(figsize=(12, 6))
                hours = list(range(24))
                ax.bar(hours, bw_distribution, color='skyblue',
                       edgecolor='black', alpha=0.8)

                ax.set_title(
                    f"磁盘 {disk_id} (集群{cluster_index}) 峰值分布", fontsize=14)
                ax.set_xlabel('小时')
                ax.set_ylabel('峰值频次')
                ax.set_xticks(hours)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # 保存图片
                peak_dist_dir = DirConfig.PEAK_DISTRIBUTION_DIR
                os.makedirs(peak_dist_dir, exist_ok=True)
                output_file = os.path.join(peak_dist_dir, f'{disk_index}.png')
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close(fig)

                disk_index += 1

                if disk_index % 1000 == 0:
                    logger.info(f"已绘制 {disk_index} 个峰值分布图")

        logger.info(f"峰值分布图绘制完成，共处理 {disk_index} 个磁盘")


class CoachPlotter:
    def __init__(self):
        self.config = DirConfig()

    def plot_peak_valley_windows_distribution(self, rbw_peak, rbw_valley, wbw_peak, wbw_valley):
        """
        绘制峰值和谷值出现的时间窗口分布
        """
        rbw_peak = self._normalize_by_column(rbw_peak)
        rbw_valley = self._normalize_by_column(rbw_valley)
        wbw_peak = self._normalize_by_column(wbw_peak)
        wbw_valley = self._normalize_by_column(wbw_valley)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        titles = ["RBW Peak", "RBW Valley", "WBW Peak", "WBW Valley"]
        for ax, data, title in zip(axes.flat, [rbw_peak, rbw_valley, wbw_peak, wbw_valley], titles):
            self.plot_stacked_bar(ax, data, title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle("Peak and Valley Windows Distribution",
                     fontsize=20, fontweight='bold')
        fig.set_facecolor('white')
        output_file = os.path.join(
            DirConfig.OUTPUT_DIR, "peak_valley_windows_distribution.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(
            f"Peak and Valley Windows Distribution saved to {output_file}")

    def _normalize_by_column(self, arr) -> np.ndarray:
        """
        对二维数组的每一列进行归一化（每一列除以该列的总和）
        返回归一化后的 np.ndarray
        """
        arr_np = np.array(arr, dtype=float).T
        col_sum = arr_np.sum(axis=0, keepdims=True)
        # 防止除以0
        col_sum[col_sum == 0] = 1
        normalized = (arr_np / col_sum)*100
        logger.debug(f"归一化后的数组: {normalized}")
        return normalized

    def plot_stacked_bar(self, ax, data_arr, title=""):
        x_pos = np.arange(data_arr.shape[1])
        colors = ['yellow', 'lightgreen', 'cyan',
                  'mediumpurple', 'red', 'saddlebrown']
        hatches = ['//', '', '', '/', 'o', 'x']
        bottom = np.zeros(data_arr.shape[1])
        for i in range(data_arr.shape[0]):
            heights = data_arr[i]
            ax.bar(x_pos, heights, color=colors[i], hatch=hatches[i],
                   bottom=bottom, edgecolor='black', linewidth=1, width=0.8)
            bottom += heights
        ax.axvline(x=4.5, color='red', linestyle='--', linewidth=2.5)
        y_pos_text = 103  # 放在图表内部的顶部
        ax.text(2, y_pos_text, 'Weekdays', ha='center',
                fontsize=16, fontweight='bold', color='black')
        ax.text(5.5, y_pos_text, 'Weekend', ha='center',
                fontsize=16, fontweight='bold', color='black')
        ax.set_ylim(0, 110)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', color='gray', linestyle='-',
                linewidth=1.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=18, fontweight='bold')
