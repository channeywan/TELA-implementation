import fasttext
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import jieba
from config.settings import DirConfig, DataConfig
from data.loader import DiskDataLoader
from sklearn.model_selection import train_test_split
import logging
from datasets import Dataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.utils import resample
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import time
logger = logging.getLogger(__name__)


class business_type_classify:
    def __init__(self):
        self.model = None
        self.unknown_threshold = 0.6


class FastTextModel(business_type_classify):
    def __init__(self):
        super().__init__()

    def train(self, items):
        items = items[items['business_type'] != 'generic-unknown'].copy()
        train_items, validation_items = train_test_split(
            items, test_size=0.2, random_state=42, shuffle=True, stratify=items['business_type'])
        X_train = list(train_items['description'])
        y_train = list(train_items['business_type'])
        X_validation = list(validation_items['description'])
        y_validation = list(validation_items['business_type'])
        train_file = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, "fasttext_train.txt")
        validation_file = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, "fasttext_validation.txt")
        self.write_fasttext_file(X_train, y_train, train_file)
        self.write_fasttext_file(X_validation, y_validation, validation_file)
        self.model = fasttext.train_supervised(
            input=train_file,
            autotuneValidationFile=validation_file,
            autotuneDuration=60
        )

    def test(self, test_items):
        begin_time = time.perf_counter()
        X_test = list(test_items['description'])
        y_test = list(test_items['business_type'])
        labels, probs = self.model.predict(X_test, k=1)
        final_preds = []
        confidences = []
        for label_list, prob_list in zip(labels, probs):
            raw_label = label_list[0].replace("__label__", "")
            confidence = prob_list[0]
            confidences.append(confidence)
            if confidence < self.unknown_threshold:
                final_preds.append("generic-unknown")
            else:
                final_preds.append(raw_label)
        end_time = time.perf_counter()
        logger.info(f"FastText predict business type time: {end_time - begin_time} seconds")
        f1_macro = f1_score(y_test, final_preds, average='macro')
        f1_weighted = f1_score(y_test, final_preds, average='weighted')
        acc = accuracy_score(y_test, final_preds)
        print(f"f1_macro: {f1_macro}, f1_weighted: {f1_weighted}, acc: {acc}")
        return

    def write_fasttext_file(self, X, y, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for desc, label in zip(X, y):
                f.write(f"__label__{label} {desc}\n")

    def save_model(self, filename):
        self.model.save_model(filename)
        print(f"模型已保存到: {filename}")

    def load_model(self, filename):
        self.model = fasttext.load_model(filename)
        print(f"模型已加载: {filename}")


class TFIDFLogisticModel(business_type_classify):
    def __init__(self):
        super().__init__()
        self.label_mapping = {
            "game-service": ["game", "gaming", "esports", "游戏", "电竞", "手游", "端游", "副本", "对战", "steam"],
            "office-system": ["office", "oa", "crm", "erp", "hr", "admin", "mail", "办公", "考勤", "人事", "行政", "邮箱", "会议"],
            "gov-public-service": ["gov", "government", "public", "civic", "政务", "政府", "社保", "公积金", "街道", "办事", "税务"],
            "corp-website-portal": ["portal", "website", "homepage", "official", "cms", "官网", "门户", "主页", "展示", "企业站"],
            "ecommerce-retail": ["shop", "store", "mall", "retail", "ecommerce", "order", "buy", "pay", "电商", "商城", "零售", "订单", "支付", "购物"],
            "local-service-delivery": ["delivery", "food", "o2o", "life", "外卖", "团购", "配送", "生活服务", "同城"],
            "media-video-streaming": ["video", "stream", "live", "media", "vod", "视频", "直播", "点播", "流媒体", "影视"],
            "media-news-portal": ["news", "article", "blog", "paper", "info", "新闻", "资讯", "文章", "博客", "阅读"],
            "finance-payment": ["finance", "pay", "bank", "stock", "fund", "wallet", "金融", "支付", "银行", "股票", "基金", "钱包", "理财"],
            "data-collection-delivery": ["data", "collect", "etl", "pipeline", "spider", "crawl", "数据", "采集", "爬虫", "传输", "上报"],
            "ai-machine-learning": ["ai", "ml", "dl", "model", "train", "gpu", "inference", "算法", "模型", "训练", "推理", "人工智能", "机器学习"],
            "dev-test-env": ["dev", "test", "uat", "qa", "ci", "cd", "jenkins", "gitlab", "测试", "开发", "环境", "构建", "部署"],
            "education-learning": ["edu", "learn", "study", "class", "school", "exam", "教育", "学习", "课程", "学校", "考试", "培训"],
            "community-social-forum": ["social", "community", "bbs", "forum", "chat", "im", "sns", "社区", "社交", "论坛", "聊天", "交友"],
            "compute-simulation": ["compute", "sim", "hpc", "render", "calc", "计算", "仿真", "渲染", "模拟", "科学计算"],
            "personal-use": ["personal", "my", "home", "blog", "test", "demo", "个人", "测试", "博客", "私有", "实验"],
            "iot-saas-platform": ["iot", "device", "sensor", "mqtt", "thing", "物联网", "设备", "传感", "连接", "平台"],
            "logistics-mobility": ["logistics", "gps", "map", "car", "driver", "route", "物流", "地图", "出行", "车辆", "司机", "轨迹"],
            "travel-hospitality": ["travel", "hotel", "flight", "trip", "tour", "ticket", "旅游", "酒店", "机票", "旅行", "票务"],
            "infra-node": ["node", "k8s", "pod", "worker", "agent", "节点", "容器", "实例"],
            "infra-coordination": ["zookeeper", "etcd", "consul", "nacos", "conf", "config", "注册中心", "配置", "协调"],
            "infra-database": ["db", "database", "mysql", "redis", "mongo", "oracle", "sql", "数据", "存储"],
            "infra-message-queue": ["mq", "kafka", "rocketmq", "rabbitmq", "topic", "queue", "消息", "队列"],
            "infra-cloud-function": ["function", "serverless", "lambda", "faas", "函数", "计算"],
            "infra-jumpbox": ["jump", "bastion", "ssh", "gateway", "堡垒机", "跳板机", "网关"],
            "infra-cache": ["cache", "redis", "memcached", "缓存", "加速"],
            "infra-logging-monitoring": ["log", "monitor", "prometheus", "grafana", "elk", "alert", "日志", "监控", "告警"],
            "generic-autoscaling": ["scale", "asg", "group", "elastic", "伸缩", "弹性"],
            "generic-unknown": []
        }
        self.flat_vocab = []
        self.reverse_lookup = {}

        for label, keywords in self.label_mapping.items():
            for word in keywords:
                word_lower = word.lower()
                if word_lower not in self.flat_vocab:
                    self.flat_vocab.append(word_lower)
                self.reverse_lookup[word_lower] = label

    def predict_labels_tfidf(self, df):
        begin_time = time.perf_counter()
        corpus = []
        texts = df['description'].fillna("").astype(str).tolist()

        for text in texts:
            text_lower = text.lower()
            words = jieba.cut(text_lower)
            corpus.append(" ".join(words))
        vectorizer = TfidfVectorizer(vocabulary=self.flat_vocab)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        tfidf_array = tfidf_matrix.toarray()
        feature_names = vectorizer.get_feature_names_out()

        predicted_labels = []

        for row_vector in tfidf_array:
            max_score = row_vector.max()

            if max_score == 0:
                predicted_labels.append("generic-unknown")
            else:
                best_word_idx = row_vector.argmax()
                best_word = feature_names[best_word_idx]
                pred_label = self.reverse_lookup.get(
                    best_word, "generic-unknown")
                predicted_labels.append(pred_label)
        end_time = time.perf_counter()
        logger.info(f"TFIDFLogisticModel predict business type time: {end_time - begin_time} seconds")
        result_df = df.copy()
        result_df['predicted_label'] = predicted_labels

        metrics = {}
        y_true = result_df['business_type'].astype(str)
        y_pred = result_df['predicted_label'].astype(str)

        acc = accuracy_score(y_true, y_pred)

        f1 = f1_score(y_true, y_pred, average='macro')

        metrics['accuracy'] = acc
        metrics['f1_macro'] = f1

        print("-" * 50)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("-" * 50)

        return result_df, metrics


class UnifiedTransformerModel(business_type_classify):
    def __init__(self, model_name):
        super().__init__()
        self.model_checkpoint = model_name
        self.model_name = model_name.split("/")[-1]
        self.business_type_list = ["game-service", "office-system", "gov-public-service", "corp-website-portal", "ecommerce-retail", "local-service-delivery", "media-video-streaming", "media-news-portal", "finance-payment", "data-collection-delivery", "ai-machine-learning", "dev-test-env", "education-learning", "community-social-forum",
                                   "compute-simulation", "personal-use", "iot-saas-platform", "logistics-mobility", "travel-hospitality", "infra-node", "infra-coordination", "infra-database", "infra-message-queue", "infra-cloud-function", "infra-jumpbox", "infra-cache", "infra-logging-monitoring", "generic-autoscaling"]
        self.id2label = {i: label for i,
                         label in enumerate(self.business_type_list)}
        self.label2id = {label: i for i,
                         label in enumerate(self.business_type_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint)
        self.time_str = datetime.now().strftime("%d%H%M")

    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.business_type_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def preprocess_data(self, items):
        items['labels'] = items['business_type'].apply(
            lambda x: self.business_type_list.index(
                x) if x in self.business_type_list else -100
        )
        if -1 in items['labels'].values:
            logger.error("有未知的业务类型:")
            unknown_labels = items[items['labels'] == -1]
            logger.error(f"未知业务类型: {unknown_labels['business_type'].unique()}")
            items = items[items['labels'] != -1]
        return items[['description', 'labels']].copy()

    def tokenize_function(self, examples):
        return self.tokenizer(examples['description'], padding="max_length", truncation=True, max_length=128)

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

    def search_best_hyperparameters(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir=DirConfig.MODEL_DISTILL_DIR,
            num_train_epochs=20,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=512,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="f1_macro",
            bf16=True,
            label_smoothing_factor=0.1
        )
        trainer = Trainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )
        best_trial = trainer.hyperparameter_search(
            hp_space=self.optuna_hp_space,
            direction="maximize",
            backend="optuna",
            n_trials=200,
            compute_objective=lambda x: x['eval_f1_macro']
        )

        logger.info(f"最佳超参数组合: {best_trial.hyperparameters}")

        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

        final_trainer = Trainer(
            model_init=self.model_init,
            args=trainer.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )
        final_trainer.train()
        return final_trainer

    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir=DirConfig.MODEL_DISTILL_DIR,
            num_train_epochs=20,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=1024,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="f1_macro",
            bf16=True,
            label_smoothing_factor=0.1
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.business_type_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
        )
        trainer.train()
        return trainer

    def plot_history(self, trainer):
        history = trainer.state.log_history
        train_loss = []
        eval_loss = []
        eval_f1 = []
        steps = []

        for entry in history:
            if 'loss' in entry and 'step' in entry:
                train_loss.append(
                    {'step': entry['step'], 'loss': entry['loss']})
            elif 'eval_loss' in entry:
                eval_loss.append(
                    {'step': entry['step'], 'loss': entry['eval_loss']})
            if 'eval_f1_macro' in entry:
                eval_f1.append(
                    {'step': entry['step'], 'f1': entry['eval_f1_macro']})
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot([x['step'] for x in train_loss], [x['loss']
                 for x in train_loss], label='Train Loss')
        plt.plot([x['step'] for x in eval_loss], [x['loss']
                 for x in eval_loss], label='Eval Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training & Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot([x['step'] for x in eval_f1], [x['f1']
                 for x in eval_f1], label='Eval F1', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.title('Validation F1 Score')
        save_dir = os.path.join(DirConfig.TEMP_DIR, f'{self.model_name}')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(
            save_dir, f'{self.model_name}_loss_curve_{self.time_str}.png'))
        logger.info(
            f"Loss curve saved to {os.path.join(save_dir, f'{self.model_name}_loss_curve_{self.time_str}.png')}")

    def plot_confusion_matrix(self, trainer, test_dataset):
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        cm = confusion_matrix(labels, preds)

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(24, 20))
        class_names = [self.id2label[i] for i in range(len(self.id2label))]

        sns.heatmap(cm_norm, annot=False, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(DirConfig.TEMP_DIR, self.model_name,
                    f'{self.model_name}_confusion_matrix_{self.time_str}.png'))
        logger.info(
            f"混淆矩阵已保存为 {os.path.join(DirConfig.TEMP_DIR,self.model_name,f'{self.model_name}_confusion_matrix_{self.time_str}.png')}")

    def set_logger(self):
        log_path = os.path.join(DirConfig.TEMP_DIR, "optuna_logging.txt")
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter(
            '%(message)s'
        )
        file_handler.setFormatter(formatter)
        optuna_logger.addHandler(file_handler)

    def load_trained_model(self):
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = tokenizer
        inference_args = TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=1024,
            bf16=True,
            report_to="none"
        )
        inference_trainer = Trainer(
            model=model,
            args=inference_args,
            tokenizer=tokenizer
        )
        self.id2label = model.config.id2label
        self.label2id = model.config.label2id
        return inference_trainer

    def from_df_to_tokenized_dataset(self, df):
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True
        )
        return tokenized_dataset

    def write_eval_result(self, eval_result):
        with open(os.path.join(DirConfig.TEMP_DIR, self.model_name, f'{self.model_name}_eval_result_{self.time_str}.txt'), 'w') as f:
            f.write(
                f"将unknown标签从训练集中移除，直接训练，评估结果如下:\n")
            f.write(str(eval_result))
        logger.info(
            f"评估结果已保存到 {os.path.join(DirConfig.TEMP_DIR, self.model_name, f'{self.model_name}_eval_result_{self.time_str}.txt')}")

    def try_best_threshold(self, raw_items):
        items = self.preprocess_data(raw_items)
        tokenized_dataset = self.from_df_to_tokenized_dataset(items)
        trainer = self.load_trained_model()
        output = trainer.predict(tokenized_dataset)
        logits = output.predictions
        labels = output.label_ids
        labels[labels == -100] = len(self.business_type_list)
        logits_tensor = torch.tensor(logits)
        probs = F.softmax(logits_tensor, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)
        UNKNOWN_ID = len(self.business_type_list)
        confidence = max_probs.numpy()
        pred_ids = pred_ids.numpy()
        final_preds = pred_ids.copy()
        best_threshold = 0
        best_f1_macro = 0
        best_acc = 0
        for threshold in np.linspace(0, 1, 50):
            mask_unknown = confidence < threshold
            final_preds[mask_unknown] = UNKNOWN_ID
            f1_macro = f1_score(labels, final_preds, average='macro')
            f1_weighted = f1_score(labels, final_preds, average='weighted')
            acc = accuracy_score(labels, final_preds)
            df = pd.DataFrame({
                'description': items['description'],
                'confidence': confidence,
                'true_label': [self.id2label.get(label, 'generic-unknown') for label in labels],
                'predict_label': [self.id2label.get(pred, 'generic-unknown') for pred in final_preds],
            })
            path = os.path.join(DirConfig.TRASH_DIR, self.model_name)
            os.makedirs(path, exist_ok=True)
            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_acc = acc
                best_threshold = threshold
            # df.to_csv(os.path.join(
            #     path, f'{self.model_name}_threshold{threshold}.csv'), index=False)
            # print(
            #     f"threshold: {threshold}, f1_macro: {f1_macro}, f1_weighted: {f1_weighted}")
        return best_threshold, best_f1_macro, best_acc

    def predict(self, raw_items):
        items = self.preprocess_data(raw_items)
        tokenized_dataset = self.from_df_to_tokenized_dataset(items)
        trainer = self.load_trained_model()
        self.plot_confusion_matrix(trainer, tokenized_dataset)
        output = trainer.predict(tokenized_dataset)
        logits = output.predictions
        labels = output.label_ids
        labels[labels == -100] = len(self.business_type_list)
        logits_tensor = torch.tensor(logits)
        labels_tensor = torch.tensor(labels)
        probs = F.softmax(logits_tensor, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)
        UNKNOWN_ID = len(self.business_type_list)
        confidence = max_probs.numpy()
        pred_ids = pred_ids.numpy()
        final_preds = pred_ids.copy()
        mask_unknown = confidence < self.unknown_threshold
        final_preds[mask_unknown] = UNKNOWN_ID
        f1_macro = f1_score(labels, final_preds, average='macro')
        f1_weighted = f1_score(labels, final_preds, average='weighted')
        acc = accuracy_score(labels, final_preds)
        predict_correct = (final_preds == labels)
        report_df = pd.DataFrame({
            'description': items['description'],
            'confidence': confidence,
            'pred_id': pred_ids,
            'true_label': [self.id2label.get(label, 'generic-unknown') for label in labels],
            'predict_label': [self.id2label.get(pred, 'generic-unknown') for pred in final_preds],
            'predict_correct': predict_correct
        })
        return report_df, f1_macro, f1_weighted, acc

    def run(self, items):
        self.set_logger()
        items = items[items['business_type'] != 'generic-unknown'].copy()
        items = self.preprocess_data(items)
        train_df, test_df = train_test_split(
            items, test_size=0.2, random_state=42, shuffle=True, stratify=items['labels'])
        tokenized_train = self.from_df_to_tokenized_dataset(train_df)
        tokenized_test = self.from_df_to_tokenized_dataset(test_df)
        trainer = self.train(
            tokenized_train, tokenized_test)
        # eval_result = trainer.evaluate(tokenized_test)
        # self.write_eval_result(eval_result)
        self.plot_history(trainer)
        # self.plot_confusion_matrix(trainer, tokenized_test)
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, self.model_name)
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
        return trainer
