import matplotlib.pyplot as plt
import seaborn as sns
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
from data.utils import get_circular_trace as get_circular_trace_util
from data.utils import iterate_first_day as iterate_first_day_util
from data.business_type_classify import UnifiedTransformerModel
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# vm_type:['S5' 'S6' nan 'SA3' 'C6' 'SA2' 'TS5' 'MA2' 'IT5' 'MA3' 'GN7' 'GN10X' 'D3' 'C5' 'S4' 'M5' 'M6' 'GN10Xp' 'GN7vw' 'S5se' 'BC1']
# disk_uuid, vm_alias, vm_cpu, vm_mem, vm_type, is_vip, ins_type(cvm,eks),project_name,buss_name,
# disk_alias,disk_size,disk_type(root,data,tfs,tssd),volume_type(cbsssd,cbsBSSD,cbsPremiun,cbsHSSD)

# ,id,region,disk_uuid,zone_id,depot_id,set_uuid,set_name,setType,setSize,set_volume_type,vm_uuid,vm_alias,vm_cpu,vm_mem,vm_os_name,vm_type,vm_deadline,appid,is_vip,is_pdd,ins_type,project_name,buss_name,disk_alias,phy_id,disk_usage,disk_size,disk_type,volume_type,pay_mode,ccbs_status,disk_status,life_state,add_time,create_date_time,last_op_date_time,modify_time,deadline
# depot_id,set_uuid,set_volume_type,vm_os_name,vm_type,disk_type,volume_type,pay_mode,life_state


def generate_prediction_report(trainer, tokenized_dataset, raw_df, id2label):
    output = trainer.predict(tokenized_dataset)
    logits = torch.tensor(output.predictions)
    label_ids = torch.tensor(output.label_ids)
    per_sample_losses = F.cross_entropy(logits, label_ids, reduction='none')
    probs = F.softmax(logits, dim=-1)
    max_probs, pred_ids = torch.max(probs, dim=-1)
    records = []
    descriptions = raw_df['description'].values

    for i in range(len(logits)):
        true_id = label_ids[i].item()
        pred_id = pred_ids[i].item()

        records.append({
            "description": descriptions[i],
            "confidence": max_probs[i].item(),
            "predict_correct": true_id == pred_id,  # 布尔值
            "true_label": id2label.get(true_id, str(true_id)),
            "predict_label": id2label.get(pred_id, str(pred_id)),
            "loss": per_sample_losses[i].item()
        })
    report_df = pd.DataFrame(records)
    cols = ["description", "loss", "confidence",
            "predict_correct", "true_label", "predict_label"]
    report_df = report_df[cols]
    report_df = report_df.sort_values(by='loss', ascending=False)

    return report_df


def plot_error_heatmaps(report_df, model_name, min_error_count=0):
    error_df = report_df[report_df['predict_correct'] == False].copy()
    conf_pivot = error_df.pivot_table(
        index='true_label',
        columns='predict_label',
        values='confidence',
        aggfunc='sum'  # 取平均值
    )

    # 聚合 B: 平均损失 (Mean Loss) -> 寻找 "严重的惩罚"
    loss_pivot = error_df.pivot_table(
        index='true_label',
        columns='predict_label',
        values='loss',
        aggfunc='sum'
    )

    # 聚合 C: 错误计数 (Count) -> 用于过滤低频错误
    count_pivot = error_df.pivot_table(
        index='true_label',
        columns='predict_label',
        values='confidence',
        aggfunc='count'
    )

    # (可选) 过滤掉只有 1-2 个样本的偶然错误，让图更干净
    if min_error_count > 0:
        mask = count_pivot < min_error_count
        conf_pivot[mask] = np.nan
        loss_pivot[mask] = np.nan

    # 3. 绘图设置
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # --- 图 1: 错误置信度热力图 (The "Arrogance" Matrix) ---
    sns.heatmap(
        conf_pivot,
        ax=axes[0],
        cmap='OrRd',       # 红色系，颜色越深越危险         # 保留两位小数
        linewidths=0.5,
        cbar_kws={'label': 'Mean Confidence'}
    )
    axes[0].set_title(
        'Error Confidence Heatmap\n(Darker = Model is more "Confidently Wrong")', fontsize=14)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # --- 图 2: 错误损失热力图 (The "Punishment" Matrix) ---
    sns.heatmap(
        loss_pivot,
        ax=axes[1],
        cmap='Purples',    # 紫色系
        linewidths=0.5,
        cbar_kws={'label': 'Mean Cross-Entropy Loss'}
    )
    axes[1].set_title(
        'Error Loss Heatmap\n(Darker = Model is "Punished" More)', fontsize=14)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(DirConfig.TEMP_DIR,
                model_name, "error_heatmaps.png"))


if __name__ == "__main__":
    model_name = ["bert-base-chinese", "distilbert-base-multilingual-cased", "Midsummra/CNMBert", "xlm-roberta-base",
                  "uer/chinese_roberta_L-2_H-128", "alibaba-pai/pai-bert-tiny-zh", "ymcui/Chinese-MobileBERT"]
    distill_bert = UnifiedTransformerModel(
        model_name="distilbert-base-multilingual-cased")
    items = pd.read_csv(os.path.join(
        DirConfig.BUSINESS_TYPE_DIR, "combined_description_business_type.csv"))
    trainer = distill_bert.run(items)
    items_with_ids = distill_bert.preprocess_test_data(items)
    # trainer = distill_bert.load_trained_model()
    tokenized_dataset = distill_bert.from_df_to_tokenized_dataset(
        items_with_ids)
    report_df = generate_prediction_report(
        trainer, tokenized_dataset, items_with_ids, distill_bert.id2label)
    # distill_bert.plot_confusion_matrix(trainer, tokenized_dataset)
    report_df.to_csv(os.path.join(DirConfig.TEMP_DIR,
                     distill_bert.model_checkpoint, "prediction_report.csv"), index=False)
    report_df = pd.read_csv(os.path.join(
        DirConfig.TEMP_DIR, "distilbert-base-multilingual-cased", "prediction_report.csv"))
    plot_error_heatmaps(
        report_df, model_name="distilbert-base-multilingual-cased")
