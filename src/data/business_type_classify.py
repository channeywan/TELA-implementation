import fasttext
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
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
logger = logging.getLogger(__name__)


class business_type_classify:
    def __init__(self):
        self.model = None

    def split_data(self, items):
        X = items['description']
        y = items['business_type']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
        validation_split_ratio = 1/9
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train, y_train, test_size=validation_split_ratio, random_state=42, shuffle=True, stratify=y_train)
        return X_train, X_test, y_train, y_test, X_validation, y_validation

    def oversample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        max_count = df['labels'].value_counts().max()
        lst_dfs = []
        for class_index, group in df.groupby('labels'):
            current_count = len(group)
            if current_count < max_count:
                resampled_group = resample(
                    group,
                    replace=True,
                    n_samples=max_count,
                    random_state=42
                )
                lst_dfs.append(resampled_group)
            else:
                lst_dfs.append(group)
        df_balanced = pd.concat(lst_dfs)
        df_balanced = df_balanced.sample(
            frac=1, random_state=42).reset_index(drop=True)
        return df_balanced


class FastTextModel(business_type_classify):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train, X_validation, y_validation):
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

    def test(self, X_test, y_test):
        test_file = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, "fasttext_test.txt")
        self.write_fasttext_file(X_test, y_test, test_file)
        results = self.model.test(test_file)
        print("准确率: ", results[1]*100)
        return results

    def write_fasttext_file(self, X, y, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for desc, label in zip(X, y):
                label = label.replace("-", "_")
                f.write(f"__label__{label} {desc}\n")

    def save_model(self, filename):
        fasttext.save_model(filename, self.model)
        print(f"模型已保存到: {filename}")

    def load_model(self, filename):
        self.model = fasttext.load_model(filename)
        print(f"模型已加载: {filename}")


class CountVectorLogisticModel(business_type_classify):
    def __init__(self):
        super().__init__()

    def train(self, X_train, y_train):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                binary=True,
                max_features=30000
            )),
            ('clf', LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=5,
                random_state=42
            ))
        ])
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"整体准确率: {accuracy * 100:.2f}%")
        return accuracy

    def save_model(self, filename):
        joblib.dump(self.model, filename)
        print(f"模型已保存到: {filename}")

    def load_model(self, filename):
        self.model = joblib.load(filename)
        print(f"模型已加载: {filename}")


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        target_device = labels.device
        if self.class_weights.device != target_device:
            self.class_weights = self.class_weights.to(target_device)

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class UnifiedTransformerModel(business_type_classify):
    def __init__(self, model_name):
        super().__init__()
        self.model_checkpoint = model_name
        self.business_type_list = ["game-service", "office-system", "gov-public-service", "corp-website-portal", "ecommerce-retail", "local-service-delivery", "media-video-streaming", "media-news-portal", "finance-payment", "data-collection-delivery", "ai-machine-learning", "dev-test-env", "education-learning", "community-social-forum",
                                   "compute-simulation", "personal-use", "iot-saas-platform", "logistics-mobility", "travel-hospitality", "infra-node", "infra-coordination", "infra-database", "infra-message-queue", "infra-cloud-function", "infra-jumpbox", "infra-cache", "infra-logging-monitoring", "generic-autoscaling", "generic-unknown"]
        self.id2label = {i: label for i,
                         label in enumerate(self.business_type_list)}
        self.label2id = {label: i for i,
                         label in enumerate(self.business_type_list)}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint)
        self.time_str = datetime.now().strftime("%d%H%M")
        self.unknown_threshold = 0.8

    def model_init(self, trial):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.business_type_list),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def optuna_hp_space(self, trial):
        return {
            'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            'per_device_train_batch_size': trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128, 256]),
            'weight_decay': trial.suggest_float("weight_decay", 0.0, 0.5),
            'warmup_ratio': trial.suggest_categorical("warmup_ratio", [0.05, 0.1, 0.15]),
        }

    def preprocess_test_data(self, items):
        def map_eval_label(x):
            if x in self.business_type_list:
                return self.business_type_list.index(x)  # 0-27
            elif x == 'generic-unknown':
                return -100  # <--- 关键！Loss 会忽略它，但 Metrics 能看到它
            else:
                return -1  # 异常数据
        items['labels'] = items['business_type'].apply(map_eval_label)
        if -1 in items['labels'].values:
            logger.error("有未知的业务类型:")
            unknown_labels = items[items['labels'] == -1]
            logger.error(f"未知业务类型: {unknown_labels['business_type'].unique()}")
            items = items[items['labels'] != -1]
        return items[['description', 'labels']].copy()

    def preprocess_data(self, items):
        items['labels'] = items['business_type'].apply(
            lambda x: self.business_type_list.index(
                x) if x in self.business_type_list else -1
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
        logits_tensor = torch.tensor(logits)
        probs = F.softmax(logits_tensor, dim=-1)
        max_probs, pred_ids = torch.max(probs, dim=-1)
        UNKNOWN_ID = len(self.business_type_list)
        final_preds = []
        final_labels = []
        for i in range(len(labels)):
            confidence = max_probs[i].item()
            pred_id = pred_ids[i].item()
            true_label = labels[i]
            if confidence < self.unknown_threshold:
                final_preds.append(UNKNOWN_ID)
            else:
                final_preds.append(pred_id)
            if true_label == -100:
                final_labels.append(UNKNOWN_ID)
            else:
                final_labels.append(true_label)
        f1_macro = f1_score(final_labels, final_preds, average='macro')
        f1_weighted = f1_score(final_labels, final_preds, average='weighted')
        acc = accuracy_score(final_labels, final_preds)
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

    def optuna_logging_callback(self, study: optuna.Study, trial: optuna.Trial):
        """
        Optuna 回调函数：在每次试验结束后被调用，用于记录参数和结果。
        """
        params = trial.params
        value = trial.value
        state = trial.state.name

        log_line = (
            f"[Trial {trial.number}] State: {state}, "
            f"F1-Macro: {value:.4f}, "
            f"Params: {params}\n"
        )

        with open(os.path.join(DirConfig.TEMP_DIR, "optuna_logging.txt"), "a") as f:
            f.write(log_line)
        logger.info("logging write success")
        if value is not None and value == study.best_value:
            best_params_path = os.path.join(
                DirConfig.TEMP_DIR, "best_hpt_params.txt")
            with open(best_params_path, "w") as f:
                f.write(f"Best F1-Macro: {study.best_value:.4f}\n")
                f.write(f"Best Trial: {study.best_trial.number}\n")
                f.write(f"Parameters: {study.best_params}\n")
        logger.info("best params write success")

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
        save_dir = os.path.join(DirConfig.TEMP_DIR, f'{self.model_checkpoint}')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(
            save_dir, f'{self.model_checkpoint}_loss_curve_{self.time_str}.png'))
        logger.info(
            f"Loss curve saved to {os.path.join(save_dir, f'{self.model_checkpoint}_loss_curve_{self.time_str}.png')}")

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
        plt.savefig(os.path.join(DirConfig.TEMP_DIR, self.model_checkpoint,
                    f'{self.model_checkpoint}_confusion_matrix_{self.time_str}.png'))
        logger.info(
            f"混淆矩阵已保存为 {os.path.join(DirConfig.TEMP_DIR,self.model_checkpoint,f'{self.model_checkpoint}_confusion_matrix_{self.time_str}.png')}")

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
            DirConfig.MODEL_DISTILL_DIR, self.model_checkpoint)
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
        with open(os.path.join(DirConfig.TEMP_DIR, self.model_checkpoint, f'{self.model_checkpoint}_eval_result_{self.time_str}.txt'), 'w') as f:
            f.write(
                f"将unknown标签从训练集中移除，然后使用softmax概率阈值{self.unknown_threshold}进行预测，评估结果如下:")
            f.write(str(eval_result))
        logger.info(
            f"评估结果已保存到 {os.path.join(DirConfig.TEMP_DIR, self.model_checkpoint, f'{self.model_checkpoint}_eval_result_{self.time_str}.txt')}")

    def run(self, items):
        self.set_logger()
        train_df, test_df = train_test_split(
            items, test_size=0.2, random_state=42, shuffle=True)
        train_df = train_df[train_df['business_type']
                            != 'generic-unknown'].copy()
        train_df = self.preprocess_data(train_df)
        test_df = self.preprocess_test_data(test_df)
        tokenized_train = self.from_df_to_tokenized_dataset(train_df)
        tokenized_test = self.from_df_to_tokenized_dataset(test_df)
        trainer = self.train(
            tokenized_train, tokenized_test)
        eval_result = trainer.evaluate(tokenized_test)
        self.write_eval_result(eval_result)
        self.plot_history(trainer)
        self.plot_confusion_matrix(trainer, tokenized_test)
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, self.model_checkpoint)
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
        return trainer
