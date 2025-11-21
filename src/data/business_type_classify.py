import fasttext
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
from config.settings import DirConfig, DataConfig
from data.loader import DiskDataLoader
from sklearn.model_selection import train_test_split
import logging
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
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


class DistillBertModel(business_type_classify):
    def __init__(self):
        super().__init__()
        self.model_checkpoint = "distilbert-base-multilingual-cased"
        self.business_type_list = ["game-service", "office-system", "gov-public-service", "corp-website-portal", "ecommerce-retail", "local-service-delivery", "media-video-streaming", "media-news-portal", "finance-payment", "data-collection-delivery", "ai-machine-learning", "dev-test-env", "education-learning", "community-social-forum",
                                   "compute-simulation", "personal-use", "iot-saas-platform", "logistics-mobility", "travel-hospitality", "infra-node", "infra-coordination", "infra-database", "infra-message-queue", "infra-cloud-function", "infra-jumpbox", "infra-cache", "infra-logging-monitoring", "generic-autoscaling", "generic-unknown"]
        self.id2label = {i: label for i,
                         label in enumerate(self.business_type_list)}
        self.label2id = {label: i for i,
                         label in enumerate(self.business_type_list)}
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_checkpoint)

    def preprocess_data(self):
        items = DiskDataLoader().load_items(DataConfig.CLUSTER_DIR_LIST)
        items['labels'] = items['business_type'].apply(
            lambda x: self.business_type_list.index(
                x) if x in self.business_type_list else -1
        )
        if -1 in items['labels'].values:
            logger.error("有未知的业务类型:")
            unknown_labels = items[items['labels'] == -1]
            logger.error(f"未知业务类型: {unknown_labels['business_type'].unique()}")
            items = items[items['labels'] != -1]
        items_for_dataset = items[['description', 'labels']].copy()
        return Dataset.from_pandas(items_for_dataset)

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

    def train(self, train_dataset, eval_dataset):
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.business_type_list), id2label=self.id2label, label2id=self.label2id)
        training_args = TrainingArguments(
            output_dir=DirConfig.MODEL_DISTILL_DIR,
            num_train_epochs=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=1024,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        trainer.train()
        return trainer

    def main(self):
        raw_dataset = self.preprocess_data()
        split_dataset = raw_dataset.train_test_split(
            test_size=0.2,
            seed=42,
            shuffle=True
        )
        tokenized_datasets = split_dataset.map(
            self.tokenize_function,
            batched=True
        )
        trainer = self.train(
            tokenized_datasets['train'], tokenized_datasets['test'])
        eval_result = trainer.evaluate(tokenized_datasets['test'])
        logger.info("评估结果:")
        logger.info(eval_result)
        model_path = os.path.join(
            DirConfig.MODEL_DISTILL_DIR, "distill_bert_model")
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
