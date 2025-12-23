from config.settings import DirConfig, DataConfig, WarehouseConfig
from typing import List, Tuple, Optional
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import os
import logging
import numpy as np
import matplotlib
import pandas as pd
from data.loader import DiskDataLoader
from data.utils import get_circular_trace, iterate_first_day
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.optimize import brentq
import pickle
import matplotlib.ticker as mticker
matplotlib.use('Agg')


logger = logging.getLogger(__name__)


class BasePlotter:
    def __init__(self):
        self.config = DirConfig()
        self.data_config = DataConfig()
        self.loader = DiskDataLoader()
        plt.rcParams['font.family'] = [
            'serif'
        ]
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_cdf(self, data: List[float], save_dir: str, title: str):
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(data, label='CDF',  color='skyblue', alpha=0.5)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('CDF')
        plt.savefig(os.path.join(save_dir, f'{title}_cdf.png'))
        logger.info(
            f"{title}_cdf.png saved to {os.path.join(save_dir, f'{title}_cdf.png')}")
        plt.close()

    def plot_pdf(self, data: List[float], save_dir: str, title: str):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data, label='PDF', fill=True, color='skyblue', alpha=0.5)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('PDF')
        plt.savefig(os.path.join(save_dir, f'{title}_pdf.png'))
        logger.info(
            f"{title}_pdf.png saved to {os.path.join(save_dir, f'{title}_pdf.png')}")
        plt.close()

    def plot_scatter(self, x: List[float], y: List[float], save_dir: str, title: str, xlabel: str = "X", ylabel: str = "Y"):
        """
        画散点图

        Args:
            x (List[float]): x轴数据
            y (List[float]): y轴数据
            save_dir (str): 保存目录
            title (str): 图表标题
            xlabel (str): x轴标签
            ylabel (str): y轴标签
        """
        plt.figure(figsize=(15, 6))
        plt.scatter(x, y, s=100, alpha=0.9)

        # # 如果有标签，在每个点旁边添加标签
        # labels = ["ODA", "SCDA", "Tela", "TIDAL"]
        # for i, label in enumerate(labels):
        #     plt.annotate(label, (x[i], y[i]),
        #                  xytext=(5, 5), textcoords='offset points',
        #                  fontsize=10, alpha=0.8)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(save_dir, f'{title}_scatter.png'))
        logger.info(
            f"{title}_scatter.png saved to {os.path.join(save_dir, f'{title}_scatter.png')}")
        plt.close()

    def iterate_first_day(self, timestamps: List[int]) -> int:
        return iterate_first_day(timestamps)

    def get_circular_trace(self, trace: pd.DataFrame, first_day_line: int, trace_len: int) -> np.ndarray:
        return get_circular_trace(trace, first_day_line, trace_len)


class TelaPlotter(BasePlotter):
    def __init__(self):
        super().__init__()

    def plot_kMeans_cluster(self, data_burst, data_stable, label_burst, label_stable):
        """绘制KMeans聚类图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 10))
        axes[0].scatter(data_burst,
                        c=label_burst, cmap='viridis')
        axes[0].set_title('突发型磁盘聚类')
        axes[0].set_xlabel('读带宽')
        axes[0].set_ylabel('写带宽')
        axes[0].legend()
        axes[1].scatter(data_stable,
                        c=label_stable, cmap='viridis')
        axes[1].set_title('稳定型磁盘聚类')
        axes[1].set_xlabel('读带宽')
        axes[1].set_ylabel('写带宽')
        axes[1].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(DirConfig.TELA_DIR,
                                 f'KMeans_tela_cluster.png'), dpi=300, bbox_inches='tight')
        logger.info(f"KMeans_tela_cluster.png已保存至{DirConfig.TELA_DIR}")
        plt.close(fig)

    def plot_warehouse_trace(self, warehouse_trace: np.ndarray, algorithm_dir: str, fig_title: str):
        """
        绘制仓库trace图，将所有仓库绘制到同一张图中
        warehouse_trace: 所有仓库的trace数据
        """
        num_rows = (WarehouseConfig.WAREHOUSE_NUMBER + 2) // 3
        num_cols = 3
        picture_dir = os.path.join(algorithm_dir, f"{fig_title}.png")
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 6*num_rows))
        fig.suptitle(fig_title, fontsize=16)
        x_ticks = [288*day for day in range(8)]
        x_tick_labels = [f'day{day}' for day in range(8)]
        for i in range(num_rows):
            for j in range(num_cols):
                warehouse_idx = i*3+j
                axes[i, j].plot(range(len(warehouse_trace[:, warehouse_idx])),
                                warehouse_trace[:, warehouse_idx], color='skyblue')
                axes[i, j].set_title(f'warehouse{warehouse_idx+1}')
                axes[i, j].set_xticks(x_ticks)
                axes[i, j].set_xticklabels(x_tick_labels)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(picture_dir,
                    dpi=300, bbox_inches='tight')
        logger.info(f"warehouse trace已保存至 {picture_dir}")
        plt.close(fig)

    def plot_resource_allocation_animation(self, allocation_history: np.ndarray, algorithm_dir: str, fig_title: str):
        """
        绘制资源分配历史动画

        Args:
            allocation_history: 资源分配历史数据，形状为 (时间步数, 仓库数, 3)
            algorithm_dir: 保存目录
            fig_title: 图表标题
        """
        if len(allocation_history) == 0:
            logger.warning("资源分配历史为空，无法生成动画")
            return

        warehouses_max_ndarray = np.array(WarehouseConfig.WAREHOUSE_MAX).T
        allocation_utilization = allocation_history / warehouses_max_ndarray

        fig, axes = plt.subplots(figsize=(14, 6))
        indices = np.arange(WarehouseConfig.WAREHOUSE_NUMBER)
        bar_width = 0.2
        pos1 = indices - bar_width
        pos2 = indices
        pos3 = indices + bar_width

        # 初始化条形图
        bars1 = axes.bar(
            pos1, allocation_utilization[0, :, 0], bar_width,
            label='capacity', color='dodgerblue')
        bars2 = axes.bar(
            pos2, allocation_utilization[0, :, 1], bar_width,
            label='read_bw', color='sandybrown')
        bars3 = axes.bar(
            pos3, allocation_utilization[0, :, 2], bar_width,
            label='write_bw', color='mediumseagreen')

        axes.set_title(fig_title)
        axes.set_ylim(0, 1.1)
        axes.set_xticks(indices)
        axes.set_xticklabels(
            [f"warehouse{i}" for i in indices], rotation=45)
        axes.set_ylabel('资源利用率')
        axes.axhline(y=1.0, color='red', linestyle='--',
                     linewidth=1, alpha=0.7)
        axes.legend()
        plt.tight_layout()

        def update(frame):
            """更新函数，用于动画的每一帧"""
            current_data = allocation_utilization[frame]
            for i in range(WarehouseConfig.WAREHOUSE_NUMBER):
                bars1[i].set_height(current_data[i, 0])
                bars2[i].set_height(current_data[i, 1])
                bars3[i].set_height(current_data[i, 2])
            axes.set_title(
                f'{fig_title} (时间步: {frame}/{len(allocation_history)})')
            return list(bars1) + list(bars2) + list(bars3)

        ani = animation.FuncAnimation(
            fig, update, frames=len(allocation_history),
            blit=True, interval=10)

        output_path = os.path.join(algorithm_dir, f'{fig_title}.mp4')
        ani.save(output_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)
        logger.info(f"动画已保存至 {output_path}")

    def plot_each_cluster_trace(self, cluster_index_list: List[int], save_dir: str):
        """
        给一个集群列表，绘制每个集群的trace图
        Args:
            cluster_index_list: 集群列表
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        items, disks_trace = self.loader.load_items_and_trace(
            cluster_index_list)
        for cluster_index in cluster_index_list:
            warehouse_trace = np.zeros(DataConfig.EVALUATE_TIME_NUMBER)
            cluster_trace = disks_trace[cluster_index]
            cluster_items = items[items["cluster_index"] == cluster_index]
            for _, item in tqdm(cluster_items.iterrows(), total=len(cluster_items), desc=f"Processing cluster {cluster_index}"):
                disk_id = item["disk_ID"]
                disk_trace = cluster_trace[disk_id]
                first_day_line = self.iterate_first_day(
                    disk_trace["timestamp"].to_list())
                circular_trace = self.get_circular_trace(
                    disk_trace, first_day_line, DataConfig.EVALUATE_TIME_NUMBER)
                warehouse_trace += circular_trace
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(warehouse_trace, label='Warehouse Trace')
            ax.set_xlabel('Time')
            ax.set_ylabel('Bandwidth')
            ax.set_title('Warehouse Trace')
            x_ticks = [288*day for day in range(8)]
            x_tick_labels = [f'day{day}' for day in range(8)]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            fig.savefig(os.path.join(
                save_dir, f'warehouse_trace{cluster_index}.png'))
            plt.close(fig)
            logger.info(
                f"warehouse_trace{cluster_index}.png saved to {os.path.join(save_dir, f'warehouse_trace{cluster_index}.png')}")

    def plot_conbined_items_trace(self, items: pd.DataFrame, save_dir: str):
        """
        给一个items列表，绘制items累计的trace图
        Args:
            items: items列表
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        _, disks_trace = self.loader.load_items_and_trace(
            DataConfig.CLUSTER_DIR_LIST)
        warehouse_trace = np.zeros(DataConfig.EVALUATE_TIME_NUMBER)
        for _, item in tqdm(items.iterrows(), total=len(items), desc="Processing items"):
            cluster_index = item["cluster_index"]
            disk_id = item["disk_ID"]
            disk_trace = disks_trace[cluster_index][disk_id]
            first_day_line = self.iterate_first_day(disk_trace["timestamp"])
            circular_trace = self.get_circular_trace(
                disk_trace, first_day_line, DataConfig.EVALUATE_TIME_NUMBER)
            warehouse_trace += circular_trace
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(warehouse_trace, label='Warehouse Trace')
        ax.set_xlabel('Time')
        ax.set_ylabel('Bandwidth')
        ax.set_title('Warehouse Trace')
        x_ticks = [288*day for day in range(8)]
        x_tick_labels = [f'day{day}' for day in range(8)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        fig.savefig(os.path.join(save_dir, 'warehouse_trace.png'))
        plt.close(fig)
        logger.info(
            f"Combined warehouse trace saved to {os.path.join(save_dir, 'warehouse_trace.png')}")

    def plot_particular_cluster_disk_trace(self, cluster_index_list: List[int], save_dir: str):
        """
        给一个指定的集群列表，绘制该集群下所有磁盘的trace图
        Args:
            cluster_index_list: 集群列表
            save_dir: 保存目录
        """
        items, disks_trace = self.loader.load_items_and_trace(
            cluster_index_list)
        for cluster_index in cluster_index_list:
            cluster_trace = disks_trace[cluster_index]
            cluster_save_dir = os.path.join(
                save_dir, f"cluster{cluster_index}")
            cluster_items = items[items["cluster_index"] == cluster_index]
            os.makedirs(cluster_save_dir, exist_ok=True)
            for _, item in tqdm(cluster_items.iterrows(), total=len(cluster_items), desc=f"Processing cluster {cluster_index}"):
                disk_id = item["disk_ID"]
                disk_trace = cluster_trace[disk_id]
                first_day_line = self.iterate_first_day(
                    disk_trace["timestamp"].to_list())
                circular_trace = self.get_circular_trace(
                    disk_trace, first_day_line, DataConfig.EVALUATE_TIME_NUMBER)
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.plot(circular_trace, label='Disk Trace')
                ax.set_xlabel('Time')
                ax.set_ylabel('Bandwidth')
                ax.set_title('Warehouse Trace')
                x_ticks = [288*day for day in range(8)]
                x_tick_labels = [f'day{day}' for day in range(8)]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_tick_labels)
                fig.savefig(os.path.join(
                    cluster_save_dir, f'{disk_id}.png'))
                plt.close(fig)

    def plot_type_pie(self, cluster_index_list: List[int], save_dir: str):
        items, disks_trace = self.loader.load_items_and_trace(
            cluster_index_list)
        items_type = items['business_type'].value_counts()
        items_type.plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
        plt.savefig(os.path.join(save_dir, 'type_pie.png'))
        logger.info(
            f"type_pie.png saved to {os.path.join(save_dir, 'type_pie.png')}")

    def plot_business_type_vector(self, business_type_vector: pd.DataFrame, save_dir: str):
        trace_vector_interval = DataConfig.TRACE_VECTOR_INTERVAL
        window_number = int(24 / trace_vector_interval)  # 确保是整数
        df_transposed = business_type_vector.T
        df_transposed.index = df_transposed.index.astype(int)
        # new_order = [(start_hour + i*trace_vector_interval) %
        #              24 for i in range(window_number)]
        # df_reordered = df_transposed.loc[new_order]
        # df_reordered = df_reordered.reset_index(drop=True)
        # df_reordered.index = [
        #     i*trace_vector_interval for i in range(window_number)]
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(18, 6))
        df_transposed.plot(ax=plt.gca())
        # plt.legend(loc='upper right')
        plt.xlim(0, 29-trace_vector_interval)
        # plt.ylim(0.02, 0.1)
        plt.xlabel('Hour')
        # x_tick_labels = [f'{h}h' for h in new_order]
        # plt.xticks(df_reordered.index, x_tick_labels)
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'business_type_vector.png'))
        logger.info(
            f"business_type_vector saved to {os.path.join(save_dir, 'business_type_vector.png')}")
        plt.close()


class MotivationPlotter(BasePlotter):
    def __init__(self):
        super().__init__()

    def plot_figure_2a(self,  save_dir: str):
        frequently_peak_hours_ratio = np.loadtxt(os.path.join(
            DirConfig.TEMP_DIR, 'peak_valley_analyze', 'frequently_peak_hours_ratio.txt'), delimiter=',')
        frequently_valley_hours_ratio = np.loadtxt(os.path.join(
            DirConfig.TEMP_DIR, 'peak_valley_analyze', 'frequently_valley_hours_ratio.txt'), delimiter=',')
        plt.figure(figsize=(5, 4.7))
        sns.ecdfplot(frequently_peak_hours_ratio,
                     label='Peak', color='#2a9d8f', linewidth=2.5)
        sns.ecdfplot(frequently_valley_hours_ratio,
                     label='Valley', color='#f4a261', linewidth=2.5)
        plt.legend(fontsize=14)
        plt.xlabel('Stability of Peak/Valley', fontsize=16)
        plt.ylabel('Cumulative Distribution Function', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(os.path.join(save_dir, 'figure_2a.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_2a saved to {os.path.join(save_dir, 'figure_2a.svg')}")

    def plot_figure_2b(self, save_dir: str):
        peak_windows_distribution = np.loadtxt(os.path.join(
            DirConfig.TEMP_DIR, 'peak_valley_analyze', 'peak_windows_distribution.txt'), delimiter=',')
        valley_windows_distribution = np.loadtxt(os.path.join(
            DirConfig.TEMP_DIR, 'peak_valley_analyze', 'valley_windows_distribution.txt'), delimiter=',')
        fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
        plt.subplots_adjust(hspace=0.18)
        x_pos = np.arange(peak_windows_distribution.shape[0])
        peak_bottom = np.zeros(peak_windows_distribution.shape[0])
        valley_bottom = np.zeros(valley_windows_distribution.shape[0])
        colors = ['#2a9d8f', '#f4a261', '#e9c46a', '#e76f51']
        legend_labels = ['1-7hr', '7-13hr', '13-19hr', '19-1hr']
        for i in range(peak_windows_distribution.shape[1]):
            heights = peak_windows_distribution[:, i]
            axes[0].bar(x_pos, heights, color=colors[i], label=legend_labels[i],
                        bottom=peak_bottom, edgecolor='black', linewidth=0.6, width=0.5)
            peak_bottom += heights
        for i in range(valley_windows_distribution.shape[1]):
            heights = valley_windows_distribution[:, i]
            axes[1].bar(x_pos, heights, color=colors[i],
                        bottom=valley_bottom, edgecolor='black', linewidth=0.6, width=0.5)
            valley_bottom += heights
        axes[0].axvline(x=4.5, color='red', linestyle='--', linewidth=1.5)
        axes[1].axvline(x=4.5, color='red', linestyle='--', linewidth=1.5)
        axes[0].set_title('Peak Windows Distribution',
                          fontsize=16)
        axes[1].set_title('Valley Windows Distribution',
                          fontsize=16)
        axes[0].text(2, 104.5, 'Weekdays', ha='center',
                     fontsize=14, color='black')
        axes[0].text(5.5, 104.5, 'Weekend', ha='center',
                     fontsize=14, color='black')
        axes[0].set_ylim(0, 115)
        # axes[0].set_xticks(x_pos)
        axes[0].set_yticks(np.arange(0, 101, 20))
        axes[1].set_yticks(np.arange(0, 101, 20))
        axes[0].tick_params(axis='x', which='major', length=0)
        axes[0].tick_params(axis='y', labelsize=16)
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        axes[1].set_xticks(np.arange(7), labels=[
                           "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        # axes[0].grid(axis='y', color='gray', linestyle='-',
        #              linewidth=1.5, alpha=0.5)
        # axes[1].grid(axis='y', color='gray', linestyle='-',
        #              linewidth=1.5, alpha=0.5)
        axes[0].set_axisbelow(True)
        axes[1].set_axisbelow(True)
        axes[0].legend(loc='lower center', bbox_to_anchor=(
            0.5, 1.1), ncol=4, fontsize=14, frameon=True,  handlelength=0.85, handletextpad=0.6)
        fig.supylabel('Percentage (%)', x=0.02, fontsize=17)
        fig.savefig(os.path.join(save_dir,
                    "figure_2b.svg"), bbox_inches='tight')
        logger.info(
            f"Peak and valley windows distribution saved to {os.path.join(save_dir, 'figure_2b.svg')}")
        plt.close(fig)

    def plot_figure_4a(self, save_dir: str):
        SHADOW_OFFSET_X = 0.01   # 水平偏移
        SHADOW_OFFSET_Y = -0.012  # 垂直偏移 (负数表示向下，即6点钟方向)

        # 阴影颜色和透明度
        SHADOW_COLOR = '#333333'  # 深灰色
        SHADOW_ALPHA = 0.4       # 透明度 (0-1)
        SHADOW_RADIUS = 1
        label_mapping = {
            'game-service': 'Gaming',
            'office-system': 'OA/ERP',
            'gov-public-service': 'Gov/Public',
            'corp-website-portal': 'Corp Web',
            'ecommerce-retail': 'E-comm',
            'local-service-delivery': 'O2O/Local',
            'media-video-streaming': 'Video',
            'media-news-portal': 'News',
            'finance-payment': 'Fin/Pay',
            'data-collection-delivery': 'Data/CDN',
            'ai-machine-learning': 'AI/ML',
            'dev-test-env': 'Dev/Test',
            'education-learning': 'Edu',
            'community-social-forum': 'Social',
            'compute-simulation': 'HPC/Sim',
            'personal-use': 'Personal',
            'iot-saas-platform': 'IoT/SaaS',
            'logistics-mobility': 'Logistics',
            'travel-hospitality': 'Travel',
            'infra-node': 'K8s Node',
            'infra-coordination': 'Coord',
            'infra-database': 'DB',
            'infra-message-queue': 'MQ',
            'infra-cloud-function': 'Serverless',
            'infra-jumpbox': 'Jumpbox',
            'infra-cache': 'Cache',
            'infra-logging-monitoring': 'Monitor',
            'generic-autoscaling': 'ASG',
            'generic-unknown': 'Unknown'
        }
        labels = [
            # --- 左上区 (从小到大) ---
            'News', 'Gov/Public', 'Social', 'Cache', 'Jumpbox',
            'Fin/Pay', 'AI/ML', 'ASG', 'Data/CDN', 'HPC/Sim', 'Gaming', 'MQ',

            # --- 锚定点 ---
            'Unknown',   # 左下 (No.2)
            'K8s Node',  # 底部 (No.1)
            'DB',        # 右下 (No.3)

            # --- 右上区 (从大到小) ---
            'Dev/Test', 'Video', 'Monitor', 'OA/ERP', 'Coord', 'IoT/SaaS',
            'Edu', 'Corp Web', 'E-comm', 'Travel', 'Logistics', 'O2O/Local', 'Serverless'
        ]
        values = [
            # --- 左上区 (从小到大) ---
            40, 76, 104, 164, 233,
            278, 485, 592, 724, 1024, 1341, 1900,

            # --- 锚定点 ---
            2493,  # Unknown
            6526,  # K8s Node
            1978,  # DB

            # --- 右上区 (从大到小) ---
            1757, 1142, 816, 695, 531, 482,
            263, 191, 133, 99, 54, 28, 17
        ]
        colors = [
            # --- 左上区 (蓝紫色系渐变：浅 -> 深) ---
            '#045A90', '#08799E',  '#0B8AA6',
            '#0C93AA', '#2CA7B5', '#199BAE', '#39B0B9', '#4BBBBF', '#4FBFAE', '#54C495', '#57C785', '#91E69C',

            # --- 锚定点颜色 ---
            '#b3ddc5',  # Unknown (青色/Turquoise)
            '#fef1e9',  # K8s Node (米白色/OldLace)
            '#fde3d6',  # DB (淡黄色/LightYellow)

            # --- 右上区 (橙红色系渐变：深 -> 浅) ---
            # 注意：这里是从大到小排，所以颜色是从深到浅，保证靠近中轴线的小类是浅色的
            '#EDDD53', '#E6CB5E', '#E2BE66', '#E0B969', '#CD8E51', '#BA6238',
            '#CE542F', '#E04728', '#C6371B', '#BF3F08', '#8F0000', '#570000', '#004285'
        ]
        data_package = list(zip(values, labels, colors))
        sorted_data = sorted(data_package, key=lambda x: x[0], reverse=True)
        total = sum(values)
        legend_handles = []
        legend_labels = []
        for value, label, color in sorted_data:
            # A. 计算百分比
            percentage = (value / total) * 100
            formatted_label = f'{label} ({percentage:.1f}%)'
            legend_labels.append(formatted_label)
            patch = mpatches.Patch(color=color, label=formatted_label)
            legend_handles.append(patch)

        def make_autopct(pct):
            return ('%1.1f%%' % pct) if pct > 100 else ''

        plt.figure(figsize=(10, 6))
        plt.pie(
            values,
            labels=None,
            colors=[SHADOW_COLOR] * len(values),
            startangle=90,
            counterclock=True,
            radius=SHADOW_RADIUS,
            center=(SHADOW_OFFSET_X, SHADOW_OFFSET_Y),
            wedgeprops={'alpha': SHADOW_ALPHA, 'edgecolor': None}
        )
        plt.pie(values, labels=None, colors=colors, startangle=90,
                counterclock=True,
                autopct=make_autopct,
                wedgeprops={'edgecolor': 'white', 'linewidth': 0.2}
                )
        plt.legend(handles=legend_handles, labels=legend_labels,
                   loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=14, ncol=2, handlelength=0.85, columnspacing=0.3)
        plt.pie(
            x=[1],
            labels=None,
            colors=['white'],
            counterclock=True,
            radius=0.35,
            center=(0, 0)
        )
        plt.axis('equal')
        plt.subplots_adjust(left=0, bottom=0, right=0.6, top=0.95)
        plt.savefig(os.path.join(save_dir, 'figure_4a.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_4a.png saved to {os.path.join(save_dir, 'figure_4a.svg')}")

    def plot_figure_4b(self, save_dir: str):
        label_mapping = {
            'game-service': 'Gaming',
            'office-system': 'OA/ERP',
            'gov-public-service': 'Gov/Public',
            'corp-website-portal': 'Corp Web',
            'ecommerce-retail': 'E-comm',
            'local-service-delivery': 'O2O/Local',
            'media-video-streaming': 'Video',
            'media-news-portal': 'News',
            'finance-payment': 'Fin/Pay',
            'data-collection-delivery': 'Data/CDN',
            'ai-machine-learning': 'AI/ML',
            'dev-test-env': 'Dev/Test',
            'education-learning': 'Edu',
            'community-social-forum': 'Social',
            'compute-simulation': 'HPC/Sim',
            'personal-use': 'Personal',
            'iot-saas-platform': 'IoT/SaaS',
            'logistics-mobility': 'Logistics',
            'travel-hospitality': 'Travel',
            'infra-node': 'K8s Node',
            'infra-coordination': 'Coord',
            'infra-database': 'DB',
            'infra-message-queue': 'MQ',
            'infra-cloud-function': 'Serverless',
            'infra-jumpbox': 'Jumpbox',
            'infra-cache': 'Cache',
            'infra-logging-monitoring': 'Monitor',
            'generic-autoscaling': 'ASG',
            'generic-unknown': 'Unknown'
        }
        vectors = pd.read_csv(os.path.join(
            DirConfig.BUSINESS_TYPE_DIR, 'business_type_vector_all_disk.csv'), delimiter=',', index_col=0)
        vectors.index = vectors.index.map(label_mapping)
        vectors.drop(index=['Personal'], inplace=True)
        style_config = {
            'News': ('#EDDD53', 1.0),
            'Gov/Public': ('#08799E', 0.3),
            'Social': ('#0B8AA6', 0.3),
            'Cache': ('#0C93AA', 0.3),
            'Jumpbox': ('#2a9d8f', 1.0),
            'Fin/Pay': ('#199BAE', 0.3),
            'AI/ML': ('#67b1e2', 1.0),
            'ASG': ('#4BBBBF', 0.3),
            'Data/CDN': ('#4FBFAE', 0.3),
            'HPC/Sim': ('#54C495', 0.3),
            'Gaming': ('#e9c46a', 1.0),
            'MQ': ('#91E69C', 0.3),
            'Unknown': ('#b3ddc5', 0.3),
            'K8s Node': ('#fef1e9', 0.3),
            'DB': ('#67b1e2', 0.3),
            'Dev/Test': ('#EDDD53', 0.3),
            'Video': ('#E6CB5E', 0.3),
            'Monitor': ('#E2BE66', 0.3),
            'OA/ERP': ('#E0B969', 0.3),
            'Coord': ('#CD8E51', 0.3),
            'IoT/SaaS': ('#BA6238', 0.3),
            'Edu': ('#CE542F', 1.0),
            'Corp Web': ('#0074aa', 1.0),
            'E-comm': ('#C6371B', 0.3),
            'Travel': ('#BF3F08', 0.3),
            'Logistics': ('#8F0000', 0.3),
            'O2O/Local': ('#570000', 0.3),
            'Serverless': ('#004285', 0.3)
        }

        def plot_row(label, row_data, color, alpha, is_highlight):
            data = np.array(row_data)
            shifted_data = np.roll(data, -8)
            plt.plot(
                shifted_data,
                label=label,
                color=color if is_highlight else 'gray',
                alpha=alpha,
                linewidth=1.2 if is_highlight else 0.8,
                zorder=10 if is_highlight else 1
            )
        plt.figure(figsize=(10, 5))
        plt.ylabel('Normalized Throughput', fontsize=16)
        plt.xlabel('Time (Hour)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.xticks(range(0, 24, 2), labels=[
                   f'{(8+i)%24}:00' for i in range(0, 24, 2)])
        first_other_label = False
        for label, row in vectors.iterrows():
            color, alpha = style_config[label]
            if alpha == 1.0:
                plot_row(label, row, color, alpha, is_highlight=True)
            else:
                if not first_other_label:
                    plot_row('Others', row, color, alpha, is_highlight=False)
                    first_other_label = True
                else:
                    plot_row(None, row, color, alpha, is_highlight=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        new_handles = []
        new_labels = []
        others_handle = None
        others_label = None

        for h, l in zip(handles, labels):
            if l == 'Others':
                others_handle = h
                others_label = l
            else:
                new_handles.append(h)
                new_labels.append(l)

        if others_handle:
            new_handles.append(others_handle)
            new_labels.append(others_label)
        plt.legend(handles=new_handles, labels=new_labels, fontsize=14)
        plt.savefig(os.path.join(save_dir, f'figure_4b.pdf'),
                    bbox_inches='tight')
        logger.info(
            f"figure_4b.png saved to {os.path.join(save_dir, f'figure_4b.pdf')}")


class EvaluationPlotter(BasePlotter):
    def __init__(self):
        super().__init__()
 
    def softpercentile(self, data: List[float], percentile: float) -> float:
        kde=gaussian_kde(data)
        threshold=brentq(lambda x: kde.integrate_box_1d(-np.inf, x) - percentile, 0, 2016)
        return threshold
    def hardpercentile(self, data: List[float], percentile: float) -> float:
        return np.percentile(data, percentile*100)
    def plot_figure_0(self, save_dir: str):
        plt.figure(figsize=(10, 6))
        data=pd.DataFrame({
            "space_imbalance": [0.1939226908472013,0.20163715308298316,0.20212043471978977,0.04119041969783254,0.10853242793751935,0.7556687704733587],
            "time_imbalance": [0.034176,0.034569,0.038996385,0.025688,0.03096,0.037 ],
            "algorithm_name": ["CBP", "SCDA", "Tela", "Oracle", "TIDAL", "RoundRobin"]
        })
        data.to_csv("figure_0.csv")
        sns.scatterplot(x="space_imbalance", y="time_imbalance", hue="algorithm_name", data=data,s=100)
        for i, algorithm_name in enumerate(data["algorithm_name"]):
            plt.annotate(algorithm_name, (data["space_imbalance"][i], data["time_imbalance"][i]),
                         xytext=(5, 0), textcoords='offset points',
                         fontsize=15, alpha=0.8)
        plt.legend(fontsize=14, ncol=2)
        plt.xlabel('Spatial Imbalance', fontsize=16)
        plt.ylabel('Temporal Imbalance', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(os.path.join(save_dir, f'figure_0.svg'), bbox_inches='tight')
        plt.close()
        logger.info(
            f"figure_0.svg saved to {os.path.join(save_dir, f'figure_0.svg')}")
    def plot_figure_1(self, save_dir: str):
        shape = ['+', '*', 'x', 'o', 's', 'd', '^']
        X = np.array([5200, 5400, 5500,
                      5600, 5700, 5800, 5900, 6000, 6048])/6048*100
        method_list = ["ODA", "SCDA", "Tela", "Oracle", "TIDAL", "RoundRobin"]
        plt.figure(figsize=(6, 6))
        data=pd.DataFrame()
        for method in method_list:
            data_dir = os.path.join(
                DirConfig.PLACEMENT_DIR, method, "violation_count.txt")
            with open(data_dir, "r") as f:
                lines = f.readlines()
                last_line = lines[-1].strip()
                Y_method = np.array([float(x)
                            for x in last_line.split(",") if x.strip() != ""])/2016*100
                data[method]=Y_method
                plt.plot(X, Y_method, label=method, linewidth=2.5,
                         marker=shape[method_list.index(method)])
        data.to_csv("figure_1.csv")
        plt.legend(fontsize=14, ncol=2)
        plt.xlabel('Disk Placed(%)', fontsize=16)
        plt.ylabel('OTF(%)', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(os.path.join(save_dir, f'figure_1.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_2a.svg saved to {os.path.join(save_dir, f'figure_1.svg')}")
        plt.close()

    def plot_figure_2(self, save_dir: str):
        CBP = [2,1,1,3,4,1,3,1,4,3,4,3,4,3,4,2,1,4,3,2,1,1,1,5,1,1,4,2,1,1,4,1,2,1,1,1,2,2,5,2,2,2,3,1,2,3,4,2,3,4,2,9,8,8,1,1,1,1,2,1,4,1,1,2,2,2,8,1,2,1,1,11,1,3,1,1,4,1,2,2,1,1,2,2,1,1,2,1,1,2,6,1,5,7,1,11,1,12,9,41,7,1,1,1,1,2,2,4,1,2,2,10,1,3,1,1,12,1,1,1,2,5,5,1,3,2,5,1,19,14,5,66,10,17,3,2,1,1,2,1,1,2,1,2,6,1,3,2,1,2,2,4,2,6,1,1,1,2,1,2,1,1,1,1,1,1,5,1,2,2,1,3,3,4,1,1,1,2,3,1,1,3,1,1,4,9,4,1,2,3,1,1,1,1,3,1,7,3,5,2,1,4,6,6,4,2,5,1,1,1,1,1,1,1,1,1,4,9,1,1,2,2,1,4,4,5,2,1,1,1,1,1,1,1,2,1,35,7,3,3,1,5,5,1,1,4,1,3,6,1,1,4,2,2,4,1,1,3,1,1,1,1,1,1,2,1,1,8,8,2,1,2,2,1,3,14,3,9,2,1,1,3,2,2,1,2,2,11,7,1,1,1,1,26,8,1,2,7,2,1,2,2,2,2,1,1,1,1,2,4,1,1,3,4,3,7,1,1,1,2,17,3,4,2,3,4,1,1,1,1,2,5,3,1,1,1,1,2,3,1,3,1,3,4,1,1,6,103,7,63,23,4,1,4,2,2,1,1,2,3,2,1,1,1,1,2,2,18,2,55,127,1,7,2,1,5,2,2,4,4,1,7,1,4,31,5,179,3,5,10,5,14,6,11,137,117,18,220,2,1,2,6,1,8,3,1,2,1,11,5,8,1,2,7,23,133,27,18,1,5,1,1,4,1,2,1,3,4,6,2,2,3,4,2,1,8,28,4,3,1,42,121]
        SCDA = [11,3,4,4,13,1,2,4,4,1,5,2,3,1,9,2,15,3,3,107,4,24,2,2,8,5,1,1,1,5,4,2,1,2,1,1,1,2,2,1,1,1,1,1,1,2,1,1,1,1,2,3,5,1,2,9,1,33,4,6,3,2,6,9,1,3,1,1,5,1,5,1,3,2,2,1,4,9,1,1,1,3,1,2,1,1,2,1,4,1,1,3,1,1,1,2,1,2,2,3,15,1,1,7,33,38,28,4,2,18,18,11,1,3,1,1,2,6,2,8,3,1,3,1,1,1,1,6,3,4,1,2,3,1,3,1,3,79,415,1,1,11,3,38,204,12,8,11,1,1,44,178,1,1,2,2,2,2,1,1,1,2,4,1,1,1,2,1,1,1,1,1,1,1,2,6,3,1,5,7,1,1,1,2,6,3,3,4,2,1,1,1,1,4,2,4,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,2,6,5,2,1,2,2,2,1,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,2,1,2,1,1,3,3,1,1,2,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,5,1,1,1,1,1,2,1,1,2,1,1,5,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,5,1,1,3,1,2,1,2,1,1,2,8,1,1,1,1,1,6,14,3,3,1,5,8,5,77,5,1,1,1,1,5,1,2,1,2,1,1,1,1,1,1,1,56,1,1,1,1,1,7,1,1,5,12,24,3,63,1,5,1,1,1,2,1,1,1,1,4,1,1,1,1,5,1,1,5,1,1,1,1,1,8,2,1,13,9,4,3,2,1,1,1,1,4,1,1,1,2,1,5,3,2,2,1,4,1,3,4,1,1,1,5,1,1,1,1,2,4,8,3,1,2,1,1]
        TELA = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
        Oracle = [1,1,1,3,1,1,1,5,3,2,1,3,7,3,1,2,1,1,1,8,4,1,1,1,1,2,1,1,2,3,1,2,1,3,1,4,3,4,3,4,4,4,2,1,4,3]
        TIDAL = [1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        RoundRobin = [2016,2016,1548,467]
        x_grid = np.linspace(1, 2016, 2015)
        cutoff_data = [0, 0.8, 0.95, 0.99, 1, 1.1]
        cutoff_view = [0, 0.2, 0.4, 0.6, 0.9, 1]
        data_dic = {'CBP': CBP, 'SCDA': SCDA, 'TELA': TELA,
                    'Oracle': Oracle, 'TIDAL': TIDAL, 'RoundRobin': RoundRobin}
        data=pd.DataFrame(data_dic)
        data.to_csv("figure_2.csv")
        def forward(x):
            """将数据值映射到视觉坐标"""
            whole_scale = -np.log10(0.0001)
            result = np.where(x <= 0.9999,
                              # 避免 log(0)
                              -np.log10(np.clip(1 - x, 1e-10, None)),
                              (x + 0.0001) * whole_scale)
            return result
            # return np.interp(x, cutoff_data, cutoff_view)

        def inverse(x):
            """将视觉坐标映射回数据值"""
            whole_scale = -np.log10(0.0001)
            result = np.where(x <= whole_scale,
                              1 - 10**(-x),
                              x / whole_scale - 0.0001)
            return result
            # return np.interp(x, cutoff_view, cutoff_data)
        plt.figure(figsize=(9, 6))
        for label, data in data_dic.items():
            kde = gaussian_kde(data)
            kde.set_bandwidth(bw_method='scott')
            pdf = kde(x_grid)
            cdf = np.cumsum(pdf) * (x_grid[1] - x_grid[0])
            cdf = cdf / cdf[-1]
            plt.plot(x_grid, cdf, label=label, linewidth=1)
        plt.xscale("log")
        # plt.yscale("function", functions=(forward, inverse))
        plt.legend(fontsize=14, ncol=2, loc='upper left',
                   bbox_to_anchor=(0.4, 0.5), columnspacing=0.1)
        plt.ylabel('PDF', fontsize=16)
        plt.xlabel('Overload Duration', fontsize=16)
        major_yticks = [0, 0.5, 0.95, 0.99, 1]
        minor_yticks = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8,0.9]
        plt.yticks(major_yticks, [f"{x}" for x in major_yticks])
        plt.gca().set_yticks(minor_yticks, minor=True)
        plt.tick_params(axis='y', which='major', labelsize=12)
        plt.xlim(1, 2016)
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join(save_dir, f'figure_2.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_2b.svg saved to {os.path.join(save_dir, f'figure_2.svg')}")
        plt.close()
    def plot_figure_2_test(self, save_dir: str,tela_overload_duration,violation_count,violation_window_size,max_violation_occurrence):
        CBP = [2,1,1,3,4,1,3,1,4,3,4,3,4,3,4,2,1,4,3,2,1,1,1,5,1,1,4,2,1,1,4,1,2,1,1,1,2,2,5,2,2,2,3,1,2,3,4,2,3,4,2,9,8,8,1,1,1,1,2,1,4,1,1,2,2,2,8,1,2,1,1,11,1,3,1,1,4,1,2,2,1,1,2,2,1,1,2,1,1,2,6,1,5,7,1,11,1,12,9,41,7,1,1,1,1,2,2,4,1,2,2,10,1,3,1,1,12,1,1,1,2,5,5,1,3,2,5,1,19,14,5,66,10,17,3,2,1,1,2,1,1,2,1,2,6,1,3,2,1,2,2,4,2,6,1,1,1,2,1,2,1,1,1,1,1,1,5,1,2,2,1,3,3,4,1,1,1,2,3,1,1,3,1,1,4,9,4,1,2,3,1,1,1,1,3,1,7,3,5,2,1,4,6,6,4,2,5,1,1,1,1,1,1,1,1,1,4,9,1,1,2,2,1,4,4,5,2,1,1,1,1,1,1,1,2,1,35,7,3,3,1,5,5,1,1,4,1,3,6,1,1,4,2,2,4,1,1,3,1,1,1,1,1,1,2,1,1,8,8,2,1,2,2,1,3,14,3,9,2,1,1,3,2,2,1,2,2,11,7,1,1,1,1,26,8,1,2,7,2,1,2,2,2,2,1,1,1,1,2,4,1,1,3,4,3,7,1,1,1,2,17,3,4,2,3,4,1,1,1,1,2,5,3,1,1,1,1,2,3,1,3,1,3,4,1,1,6,103,7,63,23,4,1,4,2,2,1,1,2,3,2,1,1,1,1,2,2,18,2,55,127,1,7,2,1,5,2,2,4,4,1,7,1,4,31,5,179,3,5,10,5,14,6,11,137,117,18,220,2,1,2,6,1,8,3,1,2,1,11,5,8,1,2,7,23,133,27,18,1,5,1,1,4,1,2,1,3,4,6,2,2,3,4,2,1,8,28,4,3,1,42,121]
        SCDA = [11,3,4,4,13,1,2,4,4,1,5,2,3,1,9,2,15,3,3,107,4,24,2,2,8,5,1,1,1,5,4,2,1,2,1,1,1,2,2,1,1,1,1,1,1,2,1,1,1,1,2,3,5,1,2,9,1,33,4,6,3,2,6,9,1,3,1,1,5,1,5,1,3,2,2,1,4,9,1,1,1,3,1,2,1,1,2,1,4,1,1,3,1,1,1,2,1,2,2,3,15,1,1,7,33,38,28,4,2,18,18,11,1,3,1,1,2,6,2,8,3,1,3,1,1,1,1,6,3,4,1,2,3,1,3,1,3,79,415,1,1,11,3,38,204,12,8,11,1,1,44,178,1,1,2,2,2,2,1,1,1,2,4,1,1,1,2,1,1,1,1,1,1,1,2,6,3,1,5,7,1,1,1,2,6,3,3,4,2,1,1,1,1,4,2,4,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,2,6,5,2,1,2,2,2,1,1,2,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,2,1,2,1,1,3,3,1,1,2,2,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,5,1,1,1,1,1,2,1,1,2,1,1,5,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,5,1,1,3,1,2,1,2,1,1,2,8,1,1,1,1,1,6,14,3,3,1,5,8,5,77,5,1,1,1,1,5,1,2,1,2,1,1,1,1,1,1,1,56,1,1,1,1,1,7,1,1,5,12,24,3,63,1,5,1,1,1,2,1,1,1,1,4,1,1,1,1,5,1,1,5,1,1,1,1,1,8,2,1,13,9,4,3,2,1,1,1,1,4,1,1,1,2,1,5,3,2,2,1,4,1,3,4,1,1,1,5,1,1,1,1,2,4,8,3,1,2,1,1]
        # TELA = [1,1,4,1,3,1,3,4,2,1,1,2,1,2,1,1,1,4,2,2,2,2,7,2,2,3,3,2,2,1,1,7,3,2,1,3,2,2,1,3,1,2,3,1,1,1,1,4,2,5,2,4,5,1,2,1,1,3,1,1,3,5,2,1,1,1,4,2,4,2,1,1,10,10,1,2,1,6,17,17,6,3,1,6,87,3,1,3,9,2,3,1,1,10,47,49,1,4,1,1,1,1,3,1,1,1,1,3,9,7,2,28,14,6,4,3,1,1,1,1,3,1,2,1,1,1,1,2,1,1,7,1,1,1,1,2,1,1,3,22,2,2,1,6,1,1,3,2,1,2,1,1,2,1,1,1,1,2,1,1,2,4,5,1,1,19,2,25,2,4,1,2,10,3,11,2,3,2,1,3,2,2,1,2,9,1,1,28,5,6,1,2,1,4,3,1,1,1,3,3,1,5,3,1,2,1,1,2,2,1,1,1,1,1,3,1,1,4,1,2,2,1,1,1,1,1,1,2,1,2,1,1,2,1,1,1,1,1,1,1,4,1,1,2,1,1,1,3,1,1,1,2,1,2,2,2,4,1,1,1,1,1,1,1,1,1,1,1,1,2,3,1,1,1,1,1,1,1,16,8,20,1,12,1,1,1,15,2,17,204,31,1,3,3,1,11,59,4,1,1,1,2,2,3,2,1,1,2,1,5,1,3,4,7,1,1,4,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,3,1,2,2,1,3,2,5,1,8,4,1,1,1,1,1,4,2,1,1,1,1,2,1,2,1,1,2,2,1,1,2,1,1,2,1,1,1,1,2,2,1,2,3,1,1,3,1,2,1,1,1,1,1,1,3,1,1,2,1,2,1,3,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,2,2,6,1,2,1,1,1,1,2,1,1,4,5,2,3,2,1,1,2,2,5,1,5,1,4,1,3,2,1,1,2,2,2,1,3,2,1,2,2,3,2,2,1,1,5,9,4,2,5,2,1,1,1,2,5,2,2,2,1,1,1,1,1,1,1,3,1,1,2,2,2,1,1,1,1,1,1,2,1,1,1,3,1,1,1,1,2,1,1,1,2,1,1,1,1,3,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,2,1,1,2,1,1,1,3,1,1,1,1,1,2,5,1,1,5,3,1,1,2,1,1,3,2,2,1,1,1,1,1,1,4,1,3,1,3,5,1,2,3,7,18,9,1,1,1,1,1,1,1,2,1,3,1,1,1,1,2,1,1,1,7,1,1,1,1,3,1,1,1,1,3,1,1,1,1,1,1,1,1,2,1,1,1,4,3,1,4,1,2,4,1,4,3,4,1,3,1,4,1,1,5,3,1,4,1,2,4,4,3,1]
        TELA = tela_overload_duration
        Oracle = [1,1,1,3,1,1,1,5,3,2,1,3,7,3,1,2,1,1,1,8,4,1,1,1,1,2,1,1,2,3,1,2,1,3,1,4,3,4,3,4,4,4,2,1,4,3]
        TIDAL = [1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        RoundRobin = [2016,2016,1548,467]
        x_grid = np.linspace(1, 2016, 2015)
        data_dic = {'CBP': CBP, 'SCDA': SCDA, 'TELA': TELA,
                    'Oracle': Oracle, 'TIDAL': TIDAL, 'RoundRobin': RoundRobin}

        def forward(x):
            """将数据值映射到视觉坐标"""
            whole_scale = -np.log10(0.0001)
            result = np.where(x <= 0.9999,
                              # 避免 log(0)
                              -np.log10(np.clip(1 - x, 1e-10, None)),
                              (x + 0.0001) * whole_scale)
            return result
            # return np.interp(x, cutoff_data, cutoff_view)

        def inverse(x):
            """将视觉坐标映射回数据值"""
            whole_scale = -np.log10(0.0001)
            result = np.where(x <= whole_scale,
                              1 - 10**(-x),
                              x / whole_scale - 0.0001)
            return result
            # return np.interp(x, cutoff_view, cutoff_data)
        fig,axes = plt.subplots(1,2,figsize=(14, 6))
        fig.suptitle(f'Violation Window Size: {violation_window_size}, Max Violation Occurrence: {max_violation_occurrence}', fontsize=16)
        for label, data in data_dic.items():
            kde = gaussian_kde(data)
            kde.set_bandwidth(bw_method='scott')
            pdf = kde(x_grid)
            cdf = np.cumsum(pdf) * (x_grid[1] - x_grid[0])
            cdf = cdf / cdf[-1]
            axes[0].plot(x_grid, cdf, label=label, linewidth=1)
        axes[0].set_xscale("log")
        axes[0].set_yscale("function", functions=(forward, inverse))
        axes[0].legend(fontsize=14, ncol=2, loc='upper left',
                   bbox_to_anchor=(0.02, 0.97), columnspacing=0.1)
        axes[0].set_ylabel('Mean Bandwidth Utilization', fontsize=16)
        axes[0].set_xlabel('Overload Occurrence', fontsize=16)
        major_yticks = [0, 0.5, 0.95, 0.99, 1]
        minor_yticks = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8,0.9]
        axes[0].set_yticks(major_yticks, [f"{x}" for x in major_yticks])
        axes[0].set_yticks(minor_yticks, minor=True)
        axes[0].tick_params(axis='y', which='major', labelsize=12)
        axes[0].set_xlim(1, 2016)
        axes[0].set_ylim(0, 1.05)


        shape = ['+', '*', 'x', 'o', 's', 'd', '^']
        X = np.array([5200, 5400, 5500,
                      5600, 5700, 5800, 5900, 6000, 6048])/6048*100
        method_list = ["CBP", "SCDA", "Tela", "Oracle", "TIDAL", "RoundRobin"]
        for method in method_list:
            if method == "Tela":
                Y_method = np.array(violation_count)/2016*100
                axes[1].plot(X, Y_method, label=method, linewidth=2.5,
                            marker='o')
            else:
                data_dir = os.path.join(
                    DirConfig.PLACEMENT_DIR, method, "violation_count.txt")
                with open(data_dir, "r") as f:
                    lines = f.readlines()
                    last_line = lines[-1].strip()
                    Y_method = np.array([float(x)
                                for x in last_line.split(",") if x.strip() != ""])/2016*100
                    axes[1].plot(X, Y_method, label=method, linewidth=2.5,
                            marker=shape[method_list.index(method)])
        axes[1].legend(fontsize=14, ncol=2)
        axes[1].set_xlabel('Disk Placed(%)', fontsize=16)
        axes[1].set_ylabel('Overload Percentage(%)', fontsize=16)
        axes[1].tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(os.path.join(save_dir, f'figure_2_test_{violation_window_size}_{max_violation_occurrence}.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_2b.svg saved to {os.path.join(save_dir, f'figure_2_test_{violation_window_size}_{max_violation_occurrence}.svg')}")
        plt.close()
    def plot_figure_3(self,  save_dir: str) -> None:
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_ODA'), 'rb') as f:
            CBP = pickle.load(f)
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_TELA'), 'rb') as f:
            TELA = pickle.load(f)
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_SCDA'), 'rb') as f:
            SCDA = pickle.load(f)
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_Oracle'), 'rb') as f:
            Oracle = pickle.load(f)
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_TIDAL'), 'rb') as f:
            TIDAL = pickle.load(f)
        with open(os.path.join(DirConfig.TEMP_DIR, f'imbalance_across_time_RoundRobin'), 'rb') as f:
            RoundRobin = pickle.load(f)
        data_dic = {'CBP': CBP, 'SCDA': SCDA, 'TELA': TELA,
                    'Oracle': Oracle, 'TIDAL': TIDAL, 'RoundRobin': RoundRobin}

        # data_dic level0:[CBP,SCDA,TELA,Oracle,TIDAL,RoundRobin]
        # data_dic level1:items_placed[5200,5400,5500,5600,5700,5800,5900,6000,6048]
        # data_dic level2:windows_length_in_one_day["5min",  "30min", "1h","2h" ,"3h", "4h", "6h", "8h"]
        # data_dic level3:cv,std
        algorithm_name_list=["CBP","SCDA","TELA","Oracle","TIDAL","RoundRobin"]
        items_placed_list=[5200,5400,5500,5600,5700,5800,5900,6000,6048]
        windows_length_in_one_day_list=["5min",  "30min", "1h","2h" ,"3h", "4h", "6h", "8h"]
        method_list=["cv","std","var"]

        # for items_placed in items_placed_list:
        #     for windows_length_in_one_day in windows_length_in_one_day_list:
        #         cv_violin_data=[]
        #         std_violin_data=[]
        #         var_violin_data=[]
        #         for algorithm_name in algorithm_name_list:
        #             cv_violin_data.append(data_dic[algorithm_name][items_placed][windows_length_in_one_day]["cv"])
        #             std_violin_data.append(data_dic[algorithm_name][items_placed][windows_length_in_one_day]["std"])
        #             var_violin_data.append(data_dic[algorithm_name][items_placed][windows_length_in_one_day]["var"])
        #         fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        #         fig.suptitle(
        #             f'time_window:{windows_length_in_one_day},Disk Number:{items_placed}', fontsize=16, fontweight='bold')
        #         sns.violinplot(cv_violin_data,ax=axes[0])
        #         axes[0].set_title("CV")
        #         axes[0].set_xticks(range(len(algorithm_name_list)), algorithm_name_list)
        #         sns.violinplot(std_violin_data,ax=axes[1])
        #         axes[1].set_title("STD")
        #         axes[1].set_xticks(range(len(algorithm_name_list)), algorithm_name_list)
        #         sns.violinplot(var_violin_data,ax=axes[2])
        #         axes[2].set_title("VAR")
        #         axes[2].set_xticks(range(len(algorithm_name_list)), algorithm_name_list)
        #         plt.savefig(os.path.join(DirConfig.EVALUATION_DIR,"figure3", f'figure3_{items_placed}_{windows_length_in_one_day}.png'), bbox_inches='tight')
        #         plt.close(fig)
        violin_data=[]
        data=pd.DataFrame()
        for algorithm_name in algorithm_name_list:
            data[algorithm_name]=data_dic[algorithm_name][6048]["2h"]["cv"]
            violin_data.append(data_dic[algorithm_name][6048]["2h"]["cv"])
        plt.figure(figsize=(10, 6))
        sns.boxplot(violin_data,palette="vlag",width=0.5)
        data.to_csv("figure_3.csv")
        # sns.despine()
        plt.xticks(range(len(algorithm_name_list)), algorithm_name_list)
        plt.ylabel("Temporal Imbalance",fontsize=16)
        plt.savefig(os.path.join(save_dir, f'figure_3.svg'), bbox_inches='tight')
        plt.close()
        logger.info(
            f"figure_3.svg saved to {os.path.join(save_dir, f'figure_3.svg')}")

    def plot_figure_4(self, save_dir: str) -> None:
        soft_percentile=False
        metric=["OTF", "P95", "P99","Temporal Imbalance","Spatial Imbalance"]
        n_metric=len(metric)
        categories=["CBP","Intensity Only","Full TIDAL"]
        n_method=len(categories)
        plt.figure(figsize=(8, 6))
        overload_count=np.array([474.6666666666667, 202.33333333333334, 64.33333333333333])/2016
        overload_count=overload_count/np.max(overload_count)
        overload_duration1=np.array([2,1,1,3,4,1,3,1,4,3,4,3,4,3,4,2,1,4,3,2,1,1,1,5,1,1,4,2,1,1,4,1,2,1,1,1,2,2,5,2,2,2,3,1,2,3,4,2,3,4,2,9,8,8,1,1,1,1,2,1,4,1,1,2,2,2,8,1,2,1,1,11,1,3,1,1,4,1,2,2,1,1,2,2,1,1,2,1,1,2,6,1,5,7,1,11,1,12,9,41,7,1,1,1,1,2,2,4,1,2,2,10,1,3,1,1,12,1,1,1,2,5,5,1,3,2,5,1,19,14,5,66,10,17,3,2,1,1,2,1,1,2,1,2,6,1,3,2,1,2,2,4,2,6,1,1,1,2,1,2,1,1,1,1,1,1,5,1,2,2,1,3,3,4,1,1,1,2,3,1,1,3,1,1,4,9,4,1,2,3,1,1,1,1,3,1,7,3,5,2,1,4,6,6,4,2,5,1,1,1,1,1,1,1,1,1,4,9,1,1,2,2,1,4,4,5,2,1,1,1,1,1,1,1,2,1,35,7,3,3,1,5,5,1,1,4,1,3,6,1,1,4,2,2,4,1,1,3,1,1,1,1,1,1,2,1,1,8,8,2,1,2,2,1,3,14,3,9,2,1,1,3,2,2,1,2,2,11,7,1,1,1,1,26,8,1,2,7,2,1,2,2,2,2,1,1,1,1,2,4,1,1,3,4,3,7,1,1,1,2,17,3,4,2,3,4,1,1,1,1,2,5,3,1,1,1,1,2,3,1,3,1,3,4,1,1,6,103,7,63,23,4,1,4,2,2,1,1,2,3,2,1,1,1,1,2,2,18,2,55,127,1,7,2,1,5,2,2,4,4,1,7,1,4,31,5,179,3,5,10,5,14,6,11,137,117,18,220,2,1,2,6,1,8,3,1,2,1,11,5,8,1,2,7,23,133,27,18,1,5,1,1,4,1,2,1,3,4,6,2,2,3,4,2,1,8,28,4,3,1,42,121])
        overload_duration2=np.array([1,1,1,5,1,3,3,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,3,1,1,3,1,1,1,1,2,5,4,1,2,2,1,1,1,3,1,1,4,4,1,3,4,1,1,1,1,2,1,1,1,7,1,3,1,1,4,1,2,3,1,4,2,8,11,17,18,1,1,2,2,1,2,1,2,1,1,6,2,3,1,4,5,1,2,2,1,1,1,1,2,1,1,3,2,3,1,1,1,1,1,1,2,1,1,2,6,2,1,3,2,2,10,3,7,3,4,1,1,1,2,2,1,14,1,3,2,1,1,1,6,2,3,3,3,4,4,2,1,1,1,1,1,1,1,3,3,1,2,4,3,1,2,6,1,13,8,6,1,2,86,4,1,7,1,5,1,1,1,10,2,1,1,2,7,12,11,35,2,50,3,61,3,2,1,6,2,1,1,1,2,1,2,1,8,2,1,2,1,8,2,1,3,6,2,3,2,1,2,8,2,6,1,14,2,2,23,26,4,6,2,1,1,2,1,1,4,4,1,2,1,1,1,1,4,1,1,1,1,2,1,1,2,1,1,2,4,2,1,1,6,2,4,1,1,1,1,1,1,1,2,3,1,1,16,4,1,1,1,1,1,10,2,1,1,1,1,1,2,1,2,4,2,1,1,1,1,1,6,6,1,2,3,2,1,51,2,8,3,5,1,8,4,15,1,1,7,17,1,1,20,3,3,1,2,1,3,3,3,2,3,2,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1])
        overload_duration3=np.array([1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1])
        time_imbalance=np.array([0.034176277838632835,0.03679765620390361,0.03095974326429853])
        space_imbalance=np.array([0.1939226908472013,0.13257569241864994,0.10853242793751935])
        if soft_percentile:
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99)])
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95)])
        else:
            p95=np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95)])
            p99=np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99)])
        p95=p95/np.max(p95)
        p99=p99/np.max(p99)
        time_imbalance=time_imbalance/np.max(time_imbalance)
        space_imbalance=space_imbalance/np.max(space_imbalance)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x_base = np.arange(n_metric)
        total_width = 0.8
        bar_width = total_width / n_method
        offsets = [(i - (n_method - 1) / 2) * bar_width for i in range(n_method)]
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#d62828']
        for i in range(n_method):
            values = [overload_count[i], p95[i], p99[i],time_imbalance[i],space_imbalance[i]]
            x = [0 + offsets[i], 1 + offsets[i], 2 + offsets[i],3 + offsets[i],4 + offsets[i]]
            
            ax1.bar(x, values, width=bar_width, 
                    color=colors[i], label=categories[i], edgecolor='black', zorder=10)
        ax1.legend(fontsize=14,bbox_to_anchor=(0.25,0.95),ncol=3)        
        ax1.set_xticks(x_base, metric, fontsize=14)
        plt.savefig(os.path.join(save_dir, f'figure_4.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_4.svg saved to {os.path.join(save_dir, f'figure_4.svg')}")
        plt.close()
    def plot_figure_5(self, save_dir: str) -> None:
        soft_percentile=True
        metric=["OTF", "P95", "P99","Temporal Imbalance","Spatial Imbalance"]
        n_metric=len(metric)
        n_method=3
        mask=np.array([True, True, False, False, True, False])
        categories = ["min_delta_util_var", "min_util_var", "min_load_var", "min_delta_load_var", "min_peak_load_before_placement","min_peak_load_after_placement"]
        categories=np.array(categories)[mask]
        overload_count = np.array([64.33333333333333, 278.5, 278.5, 64.33333333333333, 688.3333333333334,688.3333333333334])[mask]
        overload_count=overload_count/np.max(overload_count)
        overload_duration1=[1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        overload_duration2=[7,3,3,13,5,3,3,3,3,1,3,4,1,12,1,1,2,12,5,49,21,4,15,6,4,4,1,32,1,9,25,2,2,1,7,7,1,3,2,2,2,4,26,2,5,5,1,1,4,1,11,4,3,1,1,1,1,1,1,1,2,1,2,1,1,2,2,3,3,1,5,3,1,1,6,3,1,1,3,1,5,1,4,1,2,1,4,1,1,1,1,1,1,2,1,5,1,1,1,1,2,1,1,1,1,1,3,1,2,2,1,1,4,1,1,1,2,2,1,2,2,1,2,1,1,6,3,1,2,5,5,1,1,5,3,11,38,11,2,5,1,1,4,1,1,1,6,3,1,1,1,1,1,2,1,3,7,3,3,1,10,1,7,22,12,3,1,5,1,9,3,7,3,1,2,3,1,1,1,1,1,1,3,2,1,1,3,1,1,3,1,3,1,1,1,1,2,10,1,4,6,7,1,11,19,12,59,5,1,3,1,1,1,1,5,1,1,4,1,1,24,1,23,143,2,2,1,4,1,1,1,1,1,1,1,1,5,1,1,1,1,1,1,3,1,1,2,3,3,2,4,3,3,3,2,1,1,1,1,2,1,5,1,14,10,2,1,9,3,2,2,2,1,1,1,1,2,1,1,1,1,3,1,1,3,6,3,5,6,4,86,1,1,10,2,1,1,1,1,2,1,1,1,1,1,2,4,1,1,1,6,1,1,2,1,1,1,1,3,1,2,3,10,2,1,1,1,1,2,1,1,2,129,59,1,1,1,1,1,1,3,1,1,1,1,1,4,3,1,1,5,3,1,1,1,1,2,2,2,1,1,1,1,1,4,1,3,1,2,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1]
        overload_duration3=[7,3,3,13,5,3,3,3,3,1,3,4,1,12,1,1,2,12,5,49,21,4,15,6,4,4,1,32,1,9,25,2,2,1,7,7,1,3,2,2,2,4,26,2,5,5,1,1,4,1,11,4,3,1,1,1,1,1,1,1,2,1,2,1,1,2,2,3,3,1,5,3,1,1,6,3,1,1,3,1,5,1,4,1,2,1,4,1,1,1,1,1,1,2,1,5,1,1,1,1,2,1,1,1,1,1,3,1,2,2,1,1,4,1,1,1,2,2,1,2,2,1,2,1,1,6,3,1,2,5,5,1,1,5,3,11,38,11,2,5,1,1,4,1,1,1,6,3,1,1,1,1,1,2,1,3,7,3,3,1,10,1,7,22,12,3,1,5,1,9,3,7,3,1,2,3,1,1,1,1,1,1,3,2,1,1,3,1,1,3,1,3,1,1,1,1,2,10,1,4,6,7,1,11,19,12,59,5,1,3,1,1,1,1,5,1,1,4,1,1,24,1,23,143,2,2,1,4,1,1,1,1,1,1,1,1,5,1,1,1,1,1,1,3,1,1,2,3,3,2,4,3,3,3,2,1,1,1,1,2,1,5,1,14,10,2,1,9,3,2,2,2,1,1,1,1,2,1,1,1,1,3,1,1,3,6,3,5,6,4,86,1,1,10,2,1,1,1,1,2,1,1,1,1,1,2,4,1,1,1,6,1,1,2,1,1,1,1,3,1,2,3,10,2,1,1,1,1,2,1,1,2,129,59,1,1,1,1,1,1,3,1,1,1,1,1,4,3,1,1,5,3,1,1,1,1,2,2,2,1,1,1,1,1,4,1,3,1,2,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1]
        overload_duration4=[1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        overload_duration5=[1,1,7,1,3,4,2,5,2,3,1,2,1,1,3,1,1,1,1,4,1,2,1,7,4,39,19,45,87,22,5,1,241,9,2,9,1,3,34,6,95,23,56,3,1,2,1,1,1,5,4,1,5,3,1,1,1,1,2,4,1,2,3,1,2,1,1,52,16,26,83,3,1,24,2,3,13,8,1,207,2,5,6,1,9,1,7,4,10,4,24,24,129,2,27,21,4,13,1,6,1,17,5,4,36,53,2,9,2,2,1,7,3,70,12,2,1,1,1,3,2,1,5,15,1,6,2,2,1,1,1,2,7,4,10,1,1,6,4,1,1,2,3,6,1,3,16,2,3,6,4,1,9,5,3,1,7,5,3,5,2,2,2,2,2,1,4,5,1,2,2,5,4,4,4,1,1,3,1,1,3,2,3,2,1,1,1,2,1,1,8,2,1,8,3,4,1,3,2,2,19,3,1,13,1,14,13,16,1,2,1,7,23,2,1,1,3,3,2,4,2,3,2,1,3,1,7,1,2,1,1,1,2,1,4,1,1,5,1,1,3,1,4,2,3,10,1,9,8,5,1,1,1,1,5,2,2,1,2,6,3,1,1,2,1,1,1,1,2,30,5,42,19,8,2,10,9,10,2,1,1,1,1,1,4,1,1,2,2,9,25,46,2,3,10,14,19,2,17,1,47,1,12,3,1,3,1,1,15,2,1,1,4,3,10,1,1,1,1,1,1,1,1,3,1,2,1,3,2,2,5,1,2,1,1,2,2,1,1,1,2,2,4,8,1,9,1,1,1,1,1,1,3,1,2,1,2,2,2,1,1,9,1,2,1,2,1,1,1,1,1,1,5,5,4,1,6,10,3,1,2,1,1,1,1,1,4,1,3,4,82,1,1,3,1,1,2,1,2,1,3,1,1,1,1,1,2,1,1,1,2,1,1,1,2,2,1,1,2,1,2,1,1,4,2,2,1,3,3,1,2,1,1,1,1,1,1,2,1,3,6,1,3,1,1,1,1,1,1,1,2,1,1,1,1,1,2,4,2,1,1,1,1,2,1,3,1,1,1,1,2,2,5,2,1,5,1,1,3,1,2,2,2,1,1,1,2,1,1,1,2,4,1,3,1,2,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,2,3,1,2,11,1,2,1,2,1,1,2,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,10,2,5,7,1,1,4,1,1,1,12,9,1,2,4,1,1,4,1,2,1,3,1,1,1,4,1,1,1,2,3,1,1,1,1,1,4,1,1,1,3,1,2,1,1,21,2,2,2,1,2,1,3,8,1,2,1,2,6,1,1,2,3,10,1,10,1,6,4,3,7,5,4,3,1,2,1,1,1,2,1,1,1,1,1,2,1,2,1,4,15,5,7,2,1,3,4,5,27,37,152,62,1,1,1,1,4,1,1,3,1,3,5,1,1,2,3,1,1,1,1,2,1,1,1,1,3,1,1,6,7,1,1,1,2,1,2,1,3,1,9,1,4,1,2,4,1,1,2,1,5,3,1,1,1,2,1,1,1,1,1,1,1,1,2,1,2,2,1,2,1,1,1,1,1,6,4,3,2,1,6,3,6,2,3,2,5,3,1,1,7,1,2,1,3,2,4,2,1,2,2,2,3,2,1,1,4,1,1,1,7,1,2,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,3,1,1,6,1,1,1,1,4,2,1,1,1,1,1,2,1,2,2,1,1,1,1,1,1,1,2,3,2,1,2,1,1,1,1,1,1,1,1,1,1,1,3,8,1,4,1,1,2,1,1,1,2,1,1,1,3,3,6,4,1,1,1,2,1,3,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1]
        overload_duration6=[1,1,7,1,3,4,2,5,2,3,1,2,1,1,3,1,1,1,1,4,1,2,1,7,4,39,19,45,87,22,5,1,241,9,2,9,1,3,34,6,95,23,56,3,1,2,1,1,1,5,4,1,5,3,1,1,1,1,2,4,1,2,3,1,2,1,1,52,16,26,83,3,1,24,2,3,13,8,1,207,2,5,6,1,9,1,7,4,10,4,24,24,129,2,27,21,4,13,1,6,1,17,5,4,36,53,2,9,2,2,1,7,3,70,12,2,1,1,1,3,2,1,5,15,1,6,2,2,1,1,1,2,7,4,10,1,1,6,4,1,1,2,3,6,1,3,16,2,3,6,4,1,9,5,3,1,7,5,3,5,2,2,2,2,2,1,4,5,1,2,2,5,4,4,4,1,1,3,1,1,3,2,3,2,1,1,1,2,1,1,8,2,1,8,3,4,1,3,2,2,19,3,1,13,1,14,13,16,1,2,1,7,23,2,1,1,3,3,2,4,2,3,2,1,3,1,7,1,2,1,1,1,2,1,4,1,1,5,1,1,3,1,4,2,3,10,1,9,8,5,1,1,1,1,5,2,2,1,2,6,3,1,1,2,1,1,1,1,2,30,5,42,19,8,2,10,9,10,2,1,1,1,1,1,4,1,1,2,2,9,25,46,2,3,10,14,19,2,17,1,47,1,12,3,1,3,1,1,15,2,1,1,4,3,10,1,1,1,1,1,1,1,1,3,1,2,1,3,2,2,5,1,2,1,1,2,2,1,1,1,2,2,4,8,1,9,1,1,1,1,1,1,3,1,2,1,2,2,2,1,1,9,1,2,1,2,1,1,1,1,1,1,5,5,4,1,6,10,3,1,2,1,1,1,1,1,4,1,3,4,82,1,1,3,1,1,2,1,2,1,3,1,1,1,1,1,2,1,1,1,2,1,1,1,2,2,1,1,2,1,2,1,1,4,2,2,1,3,3,1,2,1,1,1,1,1,1,2,1,3,6,1,3,1,1,1,1,1,1,1,2,1,1,1,1,1,2,4,2,1,1,1,1,2,1,3,1,1,1,1,2,2,5,2,1,5,1,1,3,1,2,2,2,1,1,1,2,1,1,1,2,4,1,3,1,2,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,2,3,1,2,11,1,2,1,2,1,1,2,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,10,2,5,7,1,1,4,1,1,1,12,9,1,2,4,1,1,4,1,2,1,3,1,1,1,4,1,1,1,2,3,1,1,1,1,1,4,1,1,1,3,1,2,1,1,21,2,2,2,1,2,1,3,8,1,2,1,2,6,1,1,2,3,10,1,10,1,6,4,3,7,5,4,3,1,2,1,1,1,2,1,1,1,1,1,2,1,2,1,4,15,5,7,2,1,3,4,5,27,37,152,62,1,1,1,1,4,1,1,3,1,3,5,1,1,2,3,1,1,1,1,2,1,1,1,1,3,1,1,6,7,1,1,1,2,1,2,1,3,1,9,1,4,1,2,4,1,1,2,1,5,3,1,1,1,2,1,1,1,1,1,1,1,1,2,1,2,2,1,2,1,1,1,1,1,6,4,3,2,1,6,3,6,2,3,2,5,3,1,1,7,1,2,1,3,2,4,2,1,2,2,2,3,2,1,1,4,1,1,1,7,1,2,1,1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,3,1,1,6,1,1,1,1,4,2,1,1,1,1,1,2,1,2,2,1,1,1,1,1,1,1,2,3,2,1,2,1,1,1,1,1,1,1,1,1,1,1,3,8,1,4,1,1,2,1,1,1,2,1,1,1,3,3,6,4,1,1,1,2,1,3,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1]
        time_imbalance=np.array([0.03095974326429853,0.03810242830085577,0.05001068264969687])
        space_imbalance=np.array([0.10853242793751935,0.24023044202901603,0.43785130674827927])
        if soft_percentile:
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95) ,self.softpercentile(overload_duration4, 0.95) ,self.softpercentile(overload_duration5, 0.95) ,self.softpercentile(overload_duration6, 0.95) ])[mask]
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99) ,self.softpercentile(overload_duration4, 0.99) ,self.softpercentile(overload_duration5, 0.99) ,self.softpercentile(overload_duration6, 0.99) ])[mask]
        else:
            p95 = np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95) ,self.hardpercentile(overload_duration4, 0.95) ,self.hardpercentile(overload_duration5, 0.95) ,self.hardpercentile(overload_duration6, 0.95) ])[mask]  
            p99 = np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99) ,self.hardpercentile(overload_duration4, 0.99) ,self.hardpercentile(overload_duration5, 0.99) ,self.hardpercentile(overload_duration6, 0.99) ])[mask]  
        p95=p95/np.max(p95)
        p99=p99/np.max(p99)
        time_imbalance=time_imbalance/np.max(time_imbalance)
        space_imbalance=space_imbalance/np.max(space_imbalance)
        fig, ax1 = plt.subplots(figsize=(9, 6))
        x_base = np.arange(n_metric)
        total_width = 0.8
        bar_width = total_width / n_method
        offsets = [(i - (n_method - 1) / 2) * bar_width for i in range(n_method)]
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#d62828']
        for i in range(n_method):
            values = [overload_count[i], p95[i], p99[i],time_imbalance[i],space_imbalance[i]]
            x = [0 + offsets[i], 1 + offsets[i], 2 + offsets[i],3 + offsets[i],4+offsets[i]]
            
            ax1.bar(x, values, width=bar_width, 
                    color=colors[i], label=categories[i], edgecolor='black', zorder=10)
        ax1.legend(fontsize=14,bbox_to_anchor=(0.25,0.95),ncol=2)        
        ax1.set_xticks(x_base, metric, fontsize=14)
        ax1.set_ylabel("normalized scale",fontsize=16)
        fig.savefig(os.path.join(save_dir, f'figure_5.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_5.svg saved to {os.path.join(save_dir, f'figure_5.svg')}")
        plt.close(fig)
    def plot_figure_6(self, save_dir: str) -> None:
        plt.figure(figsize=(8, 6))
        x = np.array([312.98060405801516, 156.7602685970487, 388.61815700202715, 277.16720626410097, 15.172481301007792, 15.413530452991836,
                     319.6162735300604, 48.457951682969, 81.13847002200782, 121.57102644699626, 1.30035094905179, 2.3383498680777848-0.642])/26100*1000
        logger.info(f"x: {','.join(map(str, x))}")
        y = np.array([0.8366540951583444, 0.9139472920832176, 0.8988617105350565, 0.9101608878849777, 0.7287316997173031,
                     0.7632627551744181, 0.8922469369911449, 0.8702678448600223, 0.8975025818374208, 0.9119772765208676, 0.8041966581057343, 0.3105])
        logger.info(f"y: {','.join(map(str, y))}")
        labels = ["bert-base-chinese", "distilbert-base-multilingual-cased", "Midsummra/CNMBert", "xlm-roberta-base",
                  "uer/chinese_roberta_L-2_H-128", "alibaba-pai/pai-bert-tiny-zh", "google/mobilebert-uncased",
                  "hfl/minirbt-h288", "hfl/rbt3", "hfl/rbtl3", "FastText", "TFIDFLogisticModel"]
        data = pd.DataFrame({"model": labels, "x": x, "y": y})
        selected = ["distilbert-base-multilingual-cased", "Midsummra/CNMBert", "alibaba-pai/pai-bert-tiny-zh",
                    "google/mobilebert-uncased", "hfl/minirbt-h288", "hfl/rbt3", "FastText", "TFIDFLogisticModel"]
        # data=data[data["model"].isin(selected)]
        sns.scatterplot(x="x", y="y", data=data,
                        hue="model")
        plt.xlabel("Latency (ms)", fontsize=16)
        plt.ylabel("F1 Score", fontsize=16)
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(save_dir, f'figure_6.svg'))
        logger.info(
            f"figure_6.svg saved to {os.path.join(save_dir, f'figure_6.svg')}")
        plt.close()
        # for i, label in enumerate(labels):
        #     plt.annotate(label, (x[i], y[i]),
        #                  xytext=(5, 5), textcoords='offset points',
        #                  fontsize=10, alpha=0.8)

    def plot_figure_7(self, save_dir: str) -> None:
        R2 = np.array([0.612042, 0.617795, 0.619709, 0.654557, 0.2788])
        MAE = np.array([116409.835315,
                       123820.041646, 116175.635882, 112339.670056, 201848.1978])/300
        methods = ['CatBoost',
                   'LightGBM', 'XGBoost', 'RandomForest', 'Classification+\nRandomForest']
        x = np.arange(len(methods))
        fig, axe1 = plt.subplots(figsize=(7, 6))
        axe2 = axe1.twinx()
        bar_width = 0.3
        axe1.bar(x, R2, width=bar_width,  label='R2', color='#2a9d8f')
        axe2.bar(x, MAE, width=bar_width,  label='MAE', color='#e9c46a')
        axe1.set_ylabel(r'$R^2$ Score', fontsize=14)
        axe2.set_ylabel('MAE (KB/s)', fontsize=14)
        # axe2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.0e}'))
        axe1.set_ylim(0, max(R2) * 1.8)
        axe2.invert_yaxis()
        axe2.set_ylim(max(MAE) * 1.8, 0)
        axe1.set_xticks(x)
        axe1.set_xticklabels(methods, fontsize=12, fontweight='bold')
        fig.savefig(os.path.join(save_dir, f'figure_7.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_7.svg saved to {os.path.join(save_dir, f'figure_7.svg')}")
        plt.close(fig)

    def plot_figure_8(self, save_dir: str) -> None:
        pass

    def plot_figure_9(self, save_dir: str) -> None:
        soft_percentile=True
        standard_overload_percentage = 0.15269510582010581
        standard_time_imbalance=0.038996385101335566
        standard_space_imbalance=0.20212043471978977
        standard_duration = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
        standard_mean_overload_duration=np.mean(standard_duration)
        overload_duration1=np.array([1,1,1,1,3,1,1,5,3,2,1,1,4,1,1,1,4,1,3,1,4,3,1,1,5,1,1,4,2,2,4,6,1,1,1,1,10,2,4,8,4,1,1,1,3,2,1,2,1,3,1,1,1,1,3,2,1,1,1,1,1,1,1,4,1,1,1,4,1,1,1,1,6,3,1,3,7,1,2,2,3,2,3,1,1,2,3,1,1,1,7,1,2,3,1,2,1,1,1,1,1,1,6,2,1,1,2,3,2,1,3,5,2,7,9,3,1,1,1,3,1,2,3,2,2,1,1,1,3,1,3,3,1,1,1,2,1,1,5,1,3,1,4,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,7,5,1,4,2,5,3,2,1,1,1,1,1,7,1,1,1,2,1,1,1,1,6,1,1,1,1,4,1,1,2,1,1,2,1,10,3,1,2,3,1,1,1,1,1,1,2,1,1,1,1,1,2,3,2,3,1,2,1,2,1,4,3,3,3,2,1,1,1,1,6,1,1,1,1,1,1,2,6,1,1,2,1,1,2,4,1,1,3,1,1,2,1,1,1,2,1,2,11,1,4,1,2,3,1,2,1,2,1,1,6,2,1,1,3,1,1,2,14,1,4,3,1,1,1,4,7,1,2,2,1,1,1,1,1,2,2,1,1,1,2,3,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1])
        overload_duration2=np.array([2,1,2,1,1,3,4,5,3,4,3,2,3,1,1,4,3,1,2,2,1,3,4,3,1,4,2,4,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,3,1,1,2,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,2,5,3,1,3,2,2,20,2,1,1,3,3,10,4,1,2,1,1,1,1,1,1,2,2,1,1,2,2,1,1,1,1,1,1,1,1,1,2,2,1,2,1,1,3,2,2,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,4,2,1,4,6,1,1,1,5,3,2,8,3,1,1,4,1,4,2,1,3,3,1,1,1,1,1,1,2,1,1,2,1,1,1,3,1,3,1,1,1,1,2,1,1,2,2,1,1,4,4,3,2,3,2,3,2,2,3,1,1,1,1,8,1,1,18,1,2,1,3,5,1,1,1,1,1,1,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,4,1,1,6,1,1,1,1,1,2,1,4,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        overload_duration3=np.array([2,1,3,4,3,2,4,5,4,1,3,5,1,4,2,1,3,5,1,1,4,4,1,2,5,1,1,10,2,1,3,5,1,1,1,1,1,2,1,3,3,2,31,23,10,1,2,3,2,4,2,1,5,4,2,1,1,2,17,1,1,3,6,8,5,1,1,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,6,3,12,19,46,2,25,5,1,7,7,23,8,3,2,1,2,2,2,1,2,1,1,2,1,1,1,1,1,3,1,1,1,1,2,3,4,1,1,2,3,1,2,2,1,1,1,1,1,1,2,1,4,2,1,3,3,7,1,1,5,1,2,2,1,1,1,5,1,4,1,4,3,1,2,1,1,2,2,1,1,6,14,3,1,10,26,3,1,3,3,2,5,2,5,4,4,3,11,8,4,1,2,1,1,1,1,1,4,1,3,3,4,3,4,9,33,5,2,9,5,3,1,1,13,35,8,2,1,1,2,1,1,2,1,1,1,2,3,1,1,1,1,4,2,1,2,1,2,1,1,3,3,1,2,3,5,1,4,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,3,1,1,1,1,2,1,1,1,1,1,1,6,1,3,2,2,6,3,2,5,1])
        overload_duration4=np.array([1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1])
        overload_duration5=np.array([4,6,1,1,2,1,11,3,15,3,58,14,14,6,8,1,4,9,6,4,15,1,84,4,18,3,20,1,1,1,1,2,71,3,1,1,1,2,1,2,2,1,2,1,2,1,1,1,1,1,1,1,1,1,2,1,3,1,2,2,1,3,1,1,1,1,1,1,1,1,1,1,1,8,2,1,1,1,1,1,5,7,1,2,1,7,1,1,1,2,3,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,8,2,2,1,1,1,1,4,1,1,1,1,1,2,2,1,1,1,7,2,2,2,1,1,1,1,1,10,6,1,1,5,6,1,1,1,1,1,1,2,1,2,2,1,1,1,1,8,2,1,1,1,1,4,1,6,1,2,1,1,1,1,8,1,1,2,1,1,6,2,6,3,4,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,3,6,2,1,5,1,1,6,1,1,1,1,1,1,1,1,3,19,1,1,1,2,1,1,1,1,1,1,2,5,2,1,1,3,2,1,1,1,1,2,2,2,1,1,4,4,1,1,4,7,6,1,1,1,1,1,3,2,1,1,1,1,1,3,2,4,1,193,4,1,3,9,1,2,1,2,1,2,1,1,3,1,1,1,1,1,1,2,2,2,3,1,1,1,1,1,1,1,1,8,1,4,16,1,1,1,1,1,1,2,1,1,1,1,2,2,4,4,1,4,2,2,1,1,1,3,8,2,10,2,8,1,2,2,1,5,1,1,3,1,1,1,2,1,1,1,2,1,1,1,1,3,6,2,4,4,1,1,1,2,1,1,1,4,1,2,2,7,1,1,1,1,1,1,2,1,2,1,1,1,1,2,1,2,1,2,2,3,1,5,1,5,1,30,1])
        overload_duration6=np.array([3,1,2,1,5,1,1,3,2,31,27,1,11,6,4,6,1,10,1,1,2,1,17,1,1,1,7,13,3,2,1,1,1,1,1,3,1,2,5,1,1,2,2,1,3,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,5,1,1,4,1,1,1,1,1,1,1,2,3,4,3,1,1,1,1,1,2,2,1,1,1,1,1,3,1,3,4,1,3,26,1,4,14,1,4,7,2,3,13,12,2,2,1,3,10,3,20,3,2,5,1,2,1,1,1,3,41,12,159,2,7,1,1,1,2,1,1,2,1,23,2,1,1,3,1,2,2,2,1,3,2,1,5,2,2,1,4,2,1,1,2,1,1,1,3,8,1,1,2,1,2,1,1,1,1,2,1,1,1,1,22,2,4,3,1,1,1,1,1,1,2,2,1,1,2,1,1,3,5,1,2,1,1,1,1,1,1,2,20,1,1,1,1,1,1,1,1,2,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,2,2,1,1,1,1,1,1,1,5,1,1,9,1,1,1,1,2,1,2,1,1,1,1,2,3,2,3,2,1,1,1,2,1,1,1,1,1,1,1,1,1,10,14,1,1,2,4,3,1,1,1,1,3,1,1,1,1,1,2,1,1,1,2,1,1
])      
        overload_percentage=np.array([111.83333333333333, 82.66666666666667, 159.83333333333334,
              64.33333333333333, 221.83333333333334, 166.66666666666666])/2016
        time_imbalance=np.array([0.03622623814081403,0.03621102423495699,0.0349909850250855,0.03095974326429853,0.033733005486304764,0.03788071561319924])
        space_imbalance=np.array([0.1200575987980531,0.10568487420150532,0.1287005151871276,0.10853242793751935,0.15128735030819715,0.13193675052702228])
        if soft_percentile:
            standardp95 = self.softpercentile(standard_duration, 0.95)
            standardp99 = self.softpercentile(standard_duration, 0.99)
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95) ,self.softpercentile(overload_duration4, 0.95) ,self.softpercentile(overload_duration5, 0.95) ,self.softpercentile(overload_duration6, 0.95) ])
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99) ,self.softpercentile(overload_duration4, 0.99) ,self.softpercentile(overload_duration5, 0.99) ,self.softpercentile(overload_duration6, 0.99) ])
        else:
            standardp95 = self.hardpercentile(standard_duration, 0.95)
            standardp99 = self.hardpercentile(standard_duration, 0.99)
            p95 = np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95) ,self.hardpercentile(overload_duration4, 0.95) ,self.hardpercentile(overload_duration5, 0.95) ,self.hardpercentile(overload_duration6, 0.95) ])
            p99 = np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99) ,self.hardpercentile(overload_duration4, 0.99) ,self.hardpercentile(overload_duration5, 0.99) ,self.hardpercentile(overload_duration6, 0.99) ])  
        mean_overload_duration=[np.mean(overload_duration1),np.mean(overload_duration2),np.mean(overload_duration3),np.mean(overload_duration4),np.mean(overload_duration5),np.mean(overload_duration6)]
        # p95=p95/np.max(p95)
        # p99=p99/np.max(p99)
        # overload_count=overload_count/np.max(overload_count)
        # time_imbalance=time_imbalance/np.max(time_imbalance)
        # space_imbalance=space_imbalance/np.max(space_imbalance)
        X = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        degradation_ratio = np.array([62, 230, 485, 767, 1094, 1568])/6048
        fig, axes1 = plt.subplots(figsize=(10, 6))
        # fig.suptitle('Normalized all metrics To Tela', fontsize=16)
        color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
        axes1.set_xlabel(r'$\tau$', fontsize=16)
        axes1.plot(X, overload_percentage/standard_overload_percentage, color=color[0], label='OTF')
        axes1.plot(X, degradation_ratio, color=color[1], label='Generic Unknown (%)')
        axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[2], label='Temporal Imbalance')
        axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[3], label='Spatial Imbalance')
        axes1.plot(X, mean_overload_duration/standard_mean_overload_duration, color=color[4], label='Mean Overload Duration')
        df=pd.DataFrame()
        df["X"]=X
        df["OTF"]=overload_percentage/standard_overload_percentage
        df["Generic Unknown (%)"]=degradation_ratio
        df["Temporal Imbalance"]=time_imbalance/standard_time_imbalance
        df["Spatial Imbalance"]=space_imbalance/standard_space_imbalance
        df["Mean Overload Duration"]=mean_overload_duration/standard_mean_overload_duration
        df.to_csv("fig9.csv")
        # axes1.plot(X, p95/standardp95, color=color[4], label='P95')
        # axes1.plot(X, p99/standardp99, color=color[5], label='P99')
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.legend(loc='upper left', fontsize=14,bbox_to_anchor=(0, 1.1),ncol=3)
        fig.savefig(os.path.join(save_dir, f'figure_9.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_9.svg saved to {os.path.join(save_dir, f'figure_9.svg')}")
        plt.close(fig)

    def plot_figure_10(self, save_dir: str) -> None:
        soft_percentile=True
        percentile1 = 95
        percentile2 = 99
        K = np.array([3,6,12,24])
        X=range(len(K))
        standard_overload_percentage = 0.15269510582010581
        standard_time_imbalance=0.038996385101335566
        standard_space_imbalance=0.20212043471978977
        standard_duration = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
        standard_mean_overload_duration=np.mean(standard_duration)
        overload_percentage=np.array([151.16666666666666,61.333333333333336,64.33333333333333,202.33333333333334])/2016
        overload_duration1 = [2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 2, 4, 2, 2, 2, 1, 1, 3, 4, 1, 1, 3, 1, 2, 1, 1, 1, 3, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 5, 8, 1, 7, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3, 22, 1, 3, 2, 1, 3, 3, 2, 14, 4, 3, 2, 1, 1, 3, 3, 3, 5, 1, 5, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 6, 1, 1, 4, 1, 2, 10, 3, 19, 6, 2, 10, 2, 5, 4, 1, 1, 1, 1, 11, 1, 12, 1, 2, 1, 9, 9,
                      1, 1, 1, 1, 1, 1, 2, 1, 4, 4, 3, 1, 1, 3, 1, 2, 1, 2, 7, 3, 1, 3, 4, 2, 5, 1, 1, 4, 1, 13, 6, 12, 16, 2, 8, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 3, 7, 9, 1, 4, 5, 12, 2, 1, 5, 9, 2, 1, 1, 1, 1, 1, 2, 3, 24, 3, 2, 1, 1, 3, 4, 25, 2, 1, 3, 6, 1, 1, 6, 3, 2, 1, 1, 2, 1, 1, 93, 1, 21, 2, 5, 18, 4, 3, 6, 12, 4, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 3, 4, 1, 4, 1, 4, 4, 4, 4, 4]
        overload_duration2 = [1, 1, 1, 2, 1, 3, 2, 1, 1, 5, 5, 2, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 3, 2, 1, 1, 1, 3, 1, 93, 5, 3, 3, 1, 9, 18, 2, 38, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 4, 1, 2, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 2, 2, 2, 3, 7, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 4, 4, 3, 4, 1, 1, 3, 4, 1, 4, 4, 4, 4, 1, 1, 4, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
        overload_duration3 = [1, 5, 1, 4, 1, 1, 1, 4, 4, 4, 1, 1, 4, 1, 1, 4, 1, 2, 1, 3, 3, 1, 1, 1, 1, 1, 2, 1, 5, 1, 3, 2, 1, 31, 54, 1, 1, 12, 1, 1, 3, 3, 17, 1, 3, 3, 2, 3, 3, 3, 1, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 20, 1, 1, 2, 1, 2,
                      1, 3, 2, 1, 1, 2, 2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 3, 5, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 2, 1, 6, 1, 1, 1, 3, 1, 2, 1, 2, 1, 1, 4, 1, 1, 1, 1, 2, 1, 3, 4, 1, 5, 1, 1, 2, 1, 1, 1, 2, 4, 1, 1]
        overload_duration4 = [1, 1, 1, 5, 1, 3, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 2, 5, 4, 1, 2, 2, 1, 1, 1, 3, 1, 1, 4, 4, 1, 3, 4, 1, 1, 1, 1, 2, 1, 1, 1, 7, 1, 3, 1, 1, 4, 1, 2, 3, 1, 4, 2, 8, 11, 17, 18, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 6, 2, 3, 1, 4, 5, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 6, 2, 1, 3, 2, 2, 10, 3, 7, 3, 4, 1, 1, 1, 2, 2, 1, 14, 1, 3, 2, 1, 1, 1, 6, 2, 3, 3, 3, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 2, 4, 3, 1, 2, 6, 1, 13, 8, 6, 1, 2, 86, 4, 1, 7, 1, 5,
                      1, 1, 1, 10, 2, 1, 1, 2, 7, 12, 11, 35, 2, 50, 3, 61, 3, 2, 1, 6, 2, 1, 1, 1, 2, 1, 2, 1, 8, 2, 1, 2, 1, 8, 2, 1, 3, 6, 2, 3, 2, 1, 2, 8, 2, 6, 1, 14, 2, 2, 23, 26, 4, 6, 2, 1, 1, 2, 1, 1, 4, 4, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 4, 2, 1, 1, 6, 2, 4, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 16, 4, 1, 1, 1, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 2, 4, 2, 1, 1, 1, 1, 1, 6, 6, 1, 2, 3, 2, 1, 51, 2, 8, 3, 5, 1, 8, 4, 15, 1, 1, 7, 17, 1, 1, 20, 3, 3, 1, 2, 1, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        standard_duration = np.array(standard_duration)
        overload_duration1 = np.array(overload_duration1)
        overload_duration2 = np.array(overload_duration2)
        overload_duration3 = np.array(overload_duration3)
        overload_duration4 = np.array(overload_duration4)
        time_imbalance=np.array([0.03503478001752369,0.03465298554224872,0.03095974326429853,0.03679765620390361])
        space_imbalance=np.array([0.14006204484426787,0.09014481896382653,0.10853242793751935,0.13257569241864994])
        if soft_percentile:
            standardp95 = self.softpercentile(standard_duration, 0.95)
            standardp99 = self.softpercentile(standard_duration, 0.99)
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95) ,self.softpercentile(overload_duration4, 0.95)  ])
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99) ,self.softpercentile(overload_duration4, 0.99)  ])
        else:
            standardp95 = self.hardpercentile(standard_duration, 0.95)
            standardp99 = self.hardpercentile(standard_duration, 0.99)
            p95 = np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95) ,self.hardpercentile(overload_duration4, 0.95)])
            p99 = np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99) ,self.hardpercentile(overload_duration4, 0.99)])  
        mean_overload_duration=[np.mean(overload_duration1),np.mean(overload_duration2),np.mean(overload_duration3),np.mean(overload_duration4)]
        fig, axes1 = plt.subplots(figsize=(8, 6))
        fig.suptitle('Normalized all metrics To Tela', fontsize=16)
        color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
        axes1.set_xlabel(r'$K$', fontsize=16)
        axes1.set_xticks(X, labels=K)
        axes1.plot(X, overload_percentage/standard_overload_percentage, color=color[0], label='OTF')
        axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[1], label='Temporal Imbalance')
        axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[2], label='Spatial Imbalance')
        axes1.plot(X, mean_overload_duration/standard_mean_overload_duration, color=color[3], label='Mean Overload Duration')
        df=pd.DataFrame()
        df["X"]=X
        df["OTF"]=overload_percentage/standard_overload_percentage
        df["Temporal Imbalance"]=time_imbalance/standard_time_imbalance
        df["Spatial Imbalance"]=space_imbalance/standard_space_imbalance
        df["Mean Overload Duration"]=mean_overload_duration/standard_mean_overload_duration
        df.to_csv("fig10.csv")
        # axes1.plot(X, p95/standardp95, color=color[3], label='P95')
        # axes1.plot(X, p99/standardp99, color=color[4], label='P99')
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.legend(loc='upper left', fontsize=14,bbox_to_anchor=(0, 1.1),ncol=3)
        fig.savefig(os.path.join(save_dir, f'figure_10.svg'),
                    bbox_inches='tight')
        plt.close(fig)
        logger.info(
            f'figure_10.svg saved to {os.path.join(save_dir, "figure_10.svg")}')

    def plot_figure_11(self,  save_dir: str):
        soft_percentile=True
        percentile1 = 95
        percentile2 = 99
        noise_ratio = [0, 0.1, 0.2, 0.3]
        X=range(len(noise_ratio))
        standard_overload_percentage = 0.15269510582010581
        standard_time_imbalance=0.038996385101335566
        standard_space_imbalance=0.20212043471978977
        standard_duration = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
        standard_mean_overload_duration=np.mean(standard_duration)
        overload_percentage=np.array([64.33333333333333,61.5,99.16666666666667,89.0])/2016
        overload_duration1 = [1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        overload_duration2 = [1,1,1,1,1,4,1,1,2,3,2,1,3,1,2,1,1,2,1,2,1,1,1,1,1,1,1,4,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,2,1,1,2,2,1,2,1,1,2,1,1,2,3,1,1,2,1,1,1,1,2,2,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,3,3,1,1,1,1,1,1,3,1,1,1,3,2,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,2,1,1,1,1,2,3,1,1,1,1,1,1,1,3,3,1,1,1,1,1,2,1,3,1,2,1,1,1,1,1,2,2,3,4,3,2,1,1,1,1,1,1,1,1,3,1,2,1,5,3,2,2,1,8,9,1,1,1,1,2,1,16,1,1,2,4,2,1,1,1,1,4,1,1,2,2,2,1,2,2,3,1,1,1,1,1,1,1,1,3,1,3,2,1,1,7,2,1,1,1,2,5,1,3,1,1,1]
        overload_duration3 = [2,4,4,4,4,4,4,1,1,5,2,1,4,1,2,1,2,1,1,3,5,2,1,3,5,1,4,1,1,4,1,2,3,3,1,5,3,2,1,3,7,2,9,1,1,2,3,1,2,1,1,1,2,2,17,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1,2,2,2,1,2,7,3,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,2,1,1,2,1,2,1,1,3,3,2,1,2,2,1,1,1,1,1,1,4,1,1,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,3,1,3,3,8,2,1,1,2,1,1,3,2,1,1,1,3,2,13,2,17,1,1,2,1,1,1,1,3,2,3,3,1,1,1,1,1,1,3,2,1,3,1,3,1,1,3,1,2,1,1,1,4,1,3,1,1,1,4,1,1,3,3,4,1,2,13,7,1,1,1,1,1,1,3,1,1,1,1,1,2,1,2,2,1,5,1,1,3,1,1,1,3,1,1,1,4,1,2,1,1,2,4,1,5,1,6,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1,1,1,4,2,1,1,1,1,1,1,1,1,1,1,4,1,1,6,1,1,2,1,1,1,1,1,1,1,1,3,1,1,1,1,2,1]
        overload_duration4 = [1,1,4,2,1,1,1,1,2,2,1,1,1,1,1,1,3,2,2,1,4,5,1,5,3,2,3,31,3,1,1,3,1,1,2,1,1,1,1,1,2,1,2,17,3,7,5,3,4,2,1,2,1,3,1,4,1,2,1,3,3,1,1,4,1,1,2,1,1,2,1,1,1,1,1,1,1,1,2,5,11,13,5,2,1,1,7,5,2,4,4,7,1,3,3,3,4,4,1,1,2,2,2,1,4,1,2,1,1,4,4,2,1,2,2,2,2,1,7,1,1,1,1,1,1,1,1,2,1,1,1,2,2,1,1,3,2,3,2,4,2,1,1,1,1,1,1,1,1,1,1,1,1,7,2,2,3,1,4,2,3,5,3,1,2,1,1,1,2,2,3,9,1,1,1,3,1,2,1,1,1,5,1,1,2,1,1,1,1,1,9,1,2,2,3,2,6,6,1,1,1,2,1,1,1,1,3,1,1,1,2,2,1,4,1,1,1,4,1,1,1,1,1,1,1,1,2,4,5,1,2]
        standard_duration = np.array(standard_duration)
        overload_duration1 = np.array(overload_duration1)
        overload_duration2 = np.array(overload_duration2)
        overload_duration3 = np.array(overload_duration3)
        overload_duration4 = np.array(overload_duration4)
        time_imbalance=np.array([0.03095974326429853,0.038070760096871735,0.034569235635083656,0.03446083793766785])
        space_imbalance=np.array([0.10853242793751935,0.11652372784861278,0.12499107662945605,0.10438802869834411])
        if soft_percentile:
            standardp95 = self.softpercentile(standard_duration, 0.95)
            standardp99 = self.softpercentile(standard_duration, 0.99)
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95) ,self.softpercentile(overload_duration4, 0.95)  ])
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99) ,self.softpercentile(overload_duration4, 0.99)  ])
            print("Tela p99:",standardp99) 
            print("Tela p95:",standardp95)  
            print("TITAL p99:",p99[0])
            print("TITAL p95:",p95[0]) 
        else:
            standardp95 = self.hardpercentile(standard_duration, 0.95)
            standardp99 = self.hardpercentile(standard_duration, 0.99)
            p95 = np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95) ,self.hardpercentile(overload_duration4, 0.95)])
            p99 = np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99) ,self.hardpercentile(overload_duration4, 0.99)])  
            print("Tela p99:",standardp99) 
            print("Tela p95:",standardp95)  
            print("TITAL p99:",p99[0])
            print("TITAL p95:",p95[0]) 
        mean_overload_duration=[np.mean(overload_duration1),np.mean(overload_duration2),np.mean(overload_duration3),np.mean(overload_duration4)]
        fig, axes1 = plt.subplots(figsize=(8, 6))
        fig.suptitle('Normalized all metrics To Tela', fontsize=16)
        color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
        axes1.set_xlabel(r'Noise Ratio', fontsize=16)
        axes1.set_xticks(X, labels=noise_ratio)
        axes1.plot(X, overload_percentage/standard_overload_percentage, color=color[0],label='OTF')
        axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[1], label='Temporal Imbalance')
        axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[2], label='Spatial Imbalance')
        axes1.plot(X, mean_overload_duration/standard_mean_overload_duration, color=color[3], label='Mean Overload Duration')
        df=pd.DataFrame()
        df["X"]=X
        df["OTF"]=overload_percentage/standard_overload_percentage
        df["Temporal Imbalance"]=time_imbalance/standard_time_imbalance
        df["Spatial Imbalance"]=space_imbalance/standard_space_imbalance
        df["Mean Overload Duration"]=mean_overload_duration/standard_mean_overload_duration
        df.to_csv("fig11.csv")
        # axes1.plot(X, p95/standardp95, color=color[3], label='P95')
        # axes1.plot(X, p99/standardp99, color=color[4], label='P99')

        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.legend(loc='upper left', fontsize=14,bbox_to_anchor=(0, 1.13),ncol=2)
        plt.savefig(os.path.join(save_dir, f'figure_11.svg'),
                    bbox_inches='tight')
        logger.info(
            f'figure_11.svg saved to {os.path.join(save_dir, "figure_11.svg")}')
        plt.close(fig)
    def plot_figure_11_test(self,  save_dir: str,otf: list, time_imbalance: list, space_imbalance: list, mean_duration: list):
        noise_ratio = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        X=range(len(noise_ratio))
        standard_overload_percentage = 0.15269510582010581
        standard_time_imbalance=0.038996385101335566
        standard_space_imbalance=0.20212043471978977
        standard_duration = np.array([1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3])
        standard_mean_duration=np.mean(standard_duration)


        fig, axes1 = plt.subplots(figsize=(8, 6))
        fig.suptitle('Normalized all metrics To Tela', fontsize=16)
        color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
        axes1.set_xlabel(r'Noise Ratio', fontsize=16)
        axes1.set_xticks(X, labels=noise_ratio)
        axes1.plot(X, otf/standard_overload_percentage, color=color[0],label='Overload Percentage')
        axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[1], label='Time Imbalance')
        axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[2], label='Space Imbalance')
        axes1.plot(X, mean_duration/standard_mean_duration, color=color[3], label='Mean Overload Duration')
        df=pd.DataFrame()
        df["X"]=X
        df["OTF"]=otf/standard_overload_percentage
        df["Temporal Imbalance"]=time_imbalance/standard_time_imbalance
        df["Spatial Imbalance"]=space_imbalance/standard_space_imbalance
        df["Mean Overload Duration"]=mean_duration/standard_mean_duration
        df.to_csv("fig11.csv")
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.legend(loc='upper left', fontsize=14)
        plt.savefig(os.path.join(save_dir, f'figure_11_random_state_mean.svg'),
                    bbox_inches='tight')
        logger.info(
            f'figure_11_random_state_mean.svg saved to {os.path.join(save_dir, f"figure_11_random_state_mean.svg")}')
        plt.close(fig)
    # def plot_figure_11_test(self,  save_dir: str,overload_percentage_list: list, overload_duration_list: list, time_imbalance_list: list, space_imbalance_list: list,random_state: int):
    #     soft_percentile=True
    #     percentile1 = 95
    #     percentile2 = 99
    #     noise_ratio = [0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    #     X=range(len(noise_ratio))
    #     standard_overload_percentage = 0.15269510582010581
    #     standard_time_imbalance=0.038996385101335566
    #     standard_space_imbalance=0.20212043471978977
    #     standard_duration = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
    #     overload_percentage=np.array(overload_percentage_list)
    #     standard_duration = np.array(standard_duration)
    #     time_imbalance=np.array(time_imbalance_list)
    #     space_imbalance=np.array(space_imbalance_list)
    #     if soft_percentile:
    #         standardp95 = self.softpercentile(standard_duration, 0.95)
    #         standardp99 = self.softpercentile(standard_duration, 0.99)
    #         p95=np.array([self.softpercentile(overload_duration, 0.95) for overload_duration in overload_duration_list])
    #         p99=np.array([self.softpercentile(overload_duration, 0.99) for overload_duration in overload_duration_list])
    #     else:
    #         standardp95 = self.hardpercentile(standard_duration, 0.95)
    #         standardp99 = self.hardpercentile(standard_duration, 0.99)
    #         p95 = np.array([self.hardpercentile(overload_duration, 0.95) for overload_duration in overload_duration_list])
    #         p99 = np.array([self.hardpercentile(overload_duration, 0.99) for overload_duration in overload_duration_list])  
        
    #     fig, axes1 = plt.subplots(figsize=(8, 6))
    #     fig.suptitle('Normalized all metrics To Tela', fontsize=16)
    #     color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
    #     axes1.set_xlabel(r'Noise Ratio', fontsize=16)
    #     axes1.set_xticks(X, labels=noise_ratio)
    #     axes1.plot(X, overload_percentage/standard_overload_percentage, color=color[0],label='Overload Percentage')
    #     axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[1], label='Time Imbalance')
    #     axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[2], label='Space Imbalance')
    #     axes1.plot(X, p95/standardp95, color=color[3], label='P95')
    #     axes1.plot(X, p99/standardp99, color=color[4], label='P99')

    #     axes1.tick_params(axis='both', which='major', labelsize=14)
    #     axes1.legend(loc='upper left', fontsize=14)
    #     plt.savefig(os.path.join(save_dir, f'figure_11_random_state_{random_state}.svg'),
    #                 bbox_inches='tight')
    #     logger.info(
    #         f'figure_11_random_state_{random_state}.svg saved to {os.path.join(save_dir, f"figure_11_random_state_{random_state}.svg")}')
        # plt.close(fig)
    def plot_figure_12(self,  save_dir: str):
        soft_percentile=True
        percentile1 = 95
        percentile2 = 99
        M=[2,3,4,5,6]
        X=range(len(M))
        standard_overload_percentage = 0.15269510582010581
        standard_time_imbalance=0.038996385101335566
        standard_space_imbalance=0.20212043471978977
        standard_duration = [1, 1, 1, 1, 3, 1, 1, 10, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 6, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 5, 1, 4, 13, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 2, 1, 193, 1, 1, 1, 4, 3, 5, 1, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 3, 1, 1, 6, 2, 3, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1, 1, 2, 4, 4, 4, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 1, 2, 1, 1, 1, 2, 2, 8, 4, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 15, 2, 19, 2, 2, 1, 5, 1, 1, 12, 31, 190, 7, 1, 1, 5, 2, 1, 1, 1, 2, 1, 4, 1, 68, 2, 2, 1, 1, 1, 1, 6, 6, 11, 5, 1, 6, 1, 3, 2, 1, 7, 2, 2, 1, 4, 1, 1, 4, 4, 19, 9, 3, 21, 35, 21, 13, 29, 3, 7, 1, 15, 9, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 4, 3, 3, 8, 1, 2, 1, 9, 3, 3, 5, 3, 1, 3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 3, 2, 12, 1, 4, 16, 1, 1, 27, 1, 5, 2, 1, 23, 19, 8, 26, 8, 1, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 28, 82, 20, 74, 1, 1, 4, 1, 1, 3, 3, 1, 1, 1, 4, 1, 1, 1, 1, 5, 1, 1, 2, 41, 9, 1, 23, 59, 1, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 3, 1, 3, 4, 1, 4, 3]
        standard_mean_overload_duration=np.mean(standard_duration)
        overload_percentage=np.array([219.66666666666666,57.833333333333336,64.33333333333333,66.0,66.66666666666667])/2016
        overload_duration1 = [1,1,1,2,1,1,1,1,2,1,2,1,1,1,2,2,1,1,1,2,1,2,1,2,3,6,3,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,5,1,1,1,1,1,1,1,1,2,1,2,6,3,3,4,1,7,1,2,1,3,1,2,1,1,1,1,5,1,1,1,1,1,1,1,2,1,1,1,2,2,4,4,4,2,3,1,1,1,1,6,1,1,1,1,1,2,7,2,6,1,2,2,2,1,1,1,1,1,5,1,2,1,3,1,3,2,4,1,4,1,2,3,1,1,1,1,1,1,1,4,10,3,2,1,3,1,1,2,1,1,1,5,1,2,1,1,4,1,1,1,1,1,6,1,6,1,1,1,1,1,1,1,1,6,2,1,2,1,1,1,1,1,2,1,3,1,1,1,7,1,3,10,3,13,6,9,1,8,9,12,49,13,9,17,7,1,3,2,3,2,1,3,1,9,2,1,2,3,212,10,7,7,2,1,4,4,3,2,3,1,1,1,3,31,1,2,1,2,1,1,14,3,1,1,3,4,1,5,2,1,11,9,5,8,5,1,1,5,1,2,4,1,2,1,1,2,1,1,9,3,1,1,1,1,1,1,2,1,1,1,1,1,22,1,2,1,2,10,3,17,1,6,1,1,5,1,1,2,1,2,15,5,5,1,1,1,8,2,6,2,5,3,1,2,1,4,1,5,12,3,2,2,3,1,1,1,3,11,16,1,1,1,1,2,1,2,2,1,2,1,1,2,2,1,1,1,2,1,2,1,3,1,1,1,4,1,1,1,3,1,3,3,3,3,2,1,3,5,1,9,2,1,1,1,1,1,2,4,7,1,1,1,1,1,3,1,4,3,2,4,1,1,1,1,1,1,2,4]
        overload_duration2 = [1,1,1,2,2,1,1,5,1,2,2,1,4,2,1,1,4,4,1,7,3,2,1,1,1,5,1,3,2,1,1,1,2,1,1,3,1,2,2,2,1,1,1,2,4,5,2,1,2,1,3,1,1,11,7,3,5,2,2,4,1,1,3,1,1,1,2,1,1,1,1,3,2,1,1,1,1,1,2,1,1,4,1,1,1,1,1,1,1,5,1,1,4,2,1,1,3,2,1,2,1,1,1,1,1,3,1,1,1,1,1,1,5,7,5,2,2,2,1,1,1,2,5,3,1,1,1,1,4,4,3,5,1,2,1,4,4,1,1,1,1,1,2,2,6,1,1,1,1,1,1,1,1,1,4,4,1,1,1,1,4,2,3,4,1,4,1,2,1,4,1,1]
        overload_duration3 = [1,5,1,4,1,1,1,4,4,4,1,1,4,1,1,4,1,2,1,3,3,1,1,1,1,1,2,1,5,1,3,2,1,31,54,1,1,12,1,1,3,3,17,1,3,3,2,3,3,3,1,3,2,1,1,1,1,1,2,1,1,1,1,1,20,1,1,2,1,2,1,3,2,1,1,2,2,1,3,3,2,1,2,1,2,1,1,2,2,2,1,2,1,1,1,3,5,3,1,1,1,1,1,1,3,1,1,1,1,1,3,2,1,6,1,1,1,3,1,2,1,2,1,1,4,1,1,1,1,2,1,3,4,1,5,1,1,2,1,1,1,2,4,1,1]
        overload_duration4 = [1,1,2,1,1,2,2,1,1,2,3,1,1,1,1,1,1,1,2,1,1,4,1,1,1,1,3,1,1,1,1,1,2,1,5,2,1,2,1,2,5,1,2,4,1,3,1,1,5,1,5,4,1,3,6,1,1,1,1,1,6,6,1,1,1,2,2,1,1,1,1,3,4,2,1,1,148,43,1,1,1,1,1,1,2,1,1,1,1,1,2,1,5,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,4,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        overload_duration5 = [3,1,2,2,1,1,2,3,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,4,1,2,1,4,1,1,7,5,2,1,1,1,1,5,1,1,3,1,1,1,1,1,1,2,3,6,1,6,2,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,3,1,1,1,5,1,3,2,1,1,1,2,1,2,1,1,1,1,1,4,1,1,1,2,3,2,16,1,1,1,5,1,2,8,3,8,1,2,1,1,1,2,1,2,1,5,1,1,5,1,1,1,1,1,1,1,2,1,5,1,1,1,1,5,1,2,1,4,2,4,1,5,1,4,5,1,4,1,4,1,1,7,1,1,1,1,1,1,3,2,1,1,1,1,2,1,1,1,1,1,1,1,1,1,3,1,1,4,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,1,2,1,2,1,1,1,1,1]
        standard_duration = np.array(standard_duration)
        overload_duration1 = np.array(overload_duration1)
        overload_duration2 = np.array(overload_duration2)
        overload_duration3 = np.array(overload_duration3)
        overload_duration4 = np.array(overload_duration4)
        overload_duration5 = np.array(overload_duration5)
        time_imbalance=np.array([0.03899697605121318,0.035993468180723916,0.03095974326429853,0.03532845049399811,0.036746506033317396])
        space_imbalance=np.array([0.14299820920749962,0.09571901936982648,0.10853242793751935,0.1131545366227148,0.14482087102674704])
        if soft_percentile:
            standardp95 = self.softpercentile(standard_duration, 0.95)
            standardp99 = self.softpercentile(standard_duration, 0.99)
            p95=np.array([self.softpercentile(overload_duration1, 0.95) ,self.softpercentile(overload_duration2, 0.95) ,self.softpercentile(overload_duration3, 0.95) ,self.softpercentile(overload_duration4, 0.95),self.softpercentile(overload_duration5, 0.95)  ])
            p99=np.array([self.softpercentile(overload_duration1, 0.99) ,self.softpercentile(overload_duration2, 0.99) ,self.softpercentile(overload_duration3, 0.99) ,self.softpercentile(overload_duration4, 0.99),self.softpercentile(overload_duration5, 0.99)  ])
        else:
            standardp95 = self.hardpercentile(standard_duration, 0.95)
            standardp99 = self.hardpercentile(standard_duration, 0.99)
            p95 = np.array([self.hardpercentile(overload_duration1, 0.95) ,self.hardpercentile(overload_duration2, 0.95) ,self.hardpercentile(overload_duration3, 0.95) ,self.hardpercentile(overload_duration4, 0.95),self.hardpercentile(overload_duration5, 0.95)])
            p99 = np.array([self.hardpercentile(overload_duration1, 0.99) ,self.hardpercentile(overload_duration2, 0.99) ,self.hardpercentile(overload_duration3, 0.99) ,self.hardpercentile(overload_duration4, 0.99),self.hardpercentile(overload_duration5, 0.99)])  
        mean_overload_duration=[np.mean(overload_duration1),np.mean(overload_duration2),np.mean(overload_duration3),np.mean(overload_duration4),np.mean(overload_duration5)]
        fig, axes1 = plt.subplots(figsize=(8, 6))
        fig.suptitle('Normalized all metrics To Tela', fontsize=16)
        color=["#33c5b2","#5f78b6","#ffcd4e","#fc644c","#4fb283","#024e52"]
        axes1.set_xlabel(r'$M$', fontsize=16)
        axes1.set_xticks(X, labels=M)
        axes1.plot(X, overload_percentage/standard_overload_percentage, color=color[0],label='OTF')
        axes1.plot(X, time_imbalance/standard_time_imbalance, color=color[1], label='Temporal Imbalance')
        axes1.plot(X, space_imbalance/standard_space_imbalance, color=color[2], label='Spatial Imbalance')
        # axes1.plot(X, p95/standardp95, color=color[3], label='P95')
        # axes1.plot(X, p99/standardp99, color=color[4], label='P99')
        axes1.plot(X, mean_overload_duration/standard_mean_overload_duration, color=color[3], label='Mean Overload Duration')
        df=pd.DataFrame()
        df["X"]=M
        df["OTF"]=overload_percentage/standard_overload_percentage
        df["Temporal Imbalance"]=time_imbalance/standard_time_imbalance
        df["Spatial Imbalance"]=space_imbalance/standard_space_imbalance
        df["Mean Overload Duration"]=mean_overload_duration/standard_mean_overload_duration
        df.to_csv("fig12.csv")
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.legend(loc='upper left', fontsize=14,bbox_to_anchor=(0, 1.13),ncol=2)
        plt.savefig(os.path.join(save_dir, f'figure_12.svg'),
                    bbox_inches='tight')
        logger.info(
            f"figure_12.svg saved to {os.path.join(save_dir, f'figure_12.svg')}")
        plt.close()


class DiskPlotter(BasePlotter):
    """磁盘数据可视化绘图器"""

    def __init__(self):
        super().__init__()

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
