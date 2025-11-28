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

    def plot_business_type_vector(self, business_type_vector: pd.DataFrame, save_dir: str, start_hour=8):
        trace_vector_interval = DataConfig.TRACE_VECTOR_INTERVAL
        window_number = int(24 / trace_vector_interval)  # 确保是整数
        df_transposed = business_type_vector.T
        df_transposed.index = df_transposed.index.astype(int)
        new_order = [(start_hour + i*trace_vector_interval) %
                     24 for i in range(window_number)]
        df_reordered = df_transposed.reindex(new_order)
        df_reordered = df_reordered.reset_index(drop=True)
        df_reordered.index = [
            i*trace_vector_interval for i in range(window_number)]
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(18, 6))
        df_reordered.plot(ax=plt.gca())
        plt.legend(loc='upper right')
        plt.xlim(0, 29-trace_vector_interval)
        # plt.ylim(0.02, 0.1)
        plt.xlabel('Hour')
        x_tick_labels = [f'{h}h' for h in new_order]
        plt.xticks(df_reordered.index, x_tick_labels)
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
