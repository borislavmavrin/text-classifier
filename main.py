from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import os
import csv
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing
import fasttext
import nltk
import spacy
import unidecode
import re
from transformers import AutoTokenizer
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from random import shuffle


DATA_FOLDER = 'data'


def download_data(data_url, data_folder):
    if not os.path.exists(DATA_FOLDER):
        with urlopen(data_url) as zip_file:
            with ZipFile(BytesIO(zip_file.read())) as unzip_file:
                unzip_file.extractall(data_folder)


def read_data(data_folder):
    """
    Ingests the data into dict[dict] format.
    NOTE: There are some missing values in the column 'DATE'.
    If 'DATE' is missing - use previous non-missing value.
    This strategy relies on the data specification
    (comments are chronologically order within each file).
    """
    file_lst = os.listdir(data_folder)
    file_lst.sort()
    data_lst = list()
    header = None
    t_default = None
    for file in file_lst:
        file_path = os.path.join(data_folder, file)
        if not os.path.isfile(file_path):
            continue
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    if not header:
                        header = row
                        continue
                    else:
                        continue
                if not row[2].strip():
                    t = t_default
                else:
                    t = datetime.fromisoformat(row[2])
                    t_default = t
                assert isinstance(t, datetime), 'Missing value'
                assert row[4].strip(), 'Missing value'
                data_lst.append({
                    header[0]: row[0],
                    header[1]: row[1],
                    header[2]: t,
                    header[3]: row[3],
                    header[4]: int(row[4])})
    return data_lst


def prep_data(data, lemmatize=False, use_url_feat=False):
    regex = re.compile('[^a-zA-Z. ]')
    nlp = spacy.load("en_core_web_sm")
    for d in data:
        d_prep = unidecode.unidecode(d['CONTENT'])
        d_prep = regex.sub('', d_prep)
        d_prep = d_prep.lower()
        if use_url_feat:
            if 'http' in d['CONTENT']:
                d_prep = ' '.join([d_prep, 'http'])
        if lemmatize:
            d_prep = [x.lemma_ for x in nlp(d_prep)]
        else:
            d_prep = [x.text for x in nlp(d_prep)]
        d['CONTENT_PREP'] = ' '.join(d_prep)


def split_data(data):
    """
    splits dateset by datetime to avoid data pollution
    """
    data = sorted(data, key=lambda x: x['DATE'])
    # train/split ratio taken from:
    # https://www.researchgate.net/profile/Tiago-Almeida-16/publication/300414679_TubeSpam_Comment_Spam_Filtering_on_YouTube/links/588bbff0aca272fa50dddabe/TubeSpam-Comment-Spam-Filtering-on-YouTube.pdf
    train_split_ratio = int(0.7 * len(data))
    return data[: train_split_ratio], data[train_split_ratio:]


class SpamDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class TR:
    def __init__(self):
        self.params = {
            'model': 'bert-base-cased',
            'learning_rate': 5e-5,
            'num_train_epochs': 5,
            'batch_size': 32,
            'weight_decay': 0.01
        }
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.metric = load_metric("f1")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.params['model'], num_labels=2)
        self.model.to(self.device)
        self.training_args = TrainingArguments(output_dir="/tmp/spam/test_trainer",
                                               learning_rate=self.params['learning_rate'],
                                               num_train_epochs=self.params['num_train_epochs'],
                                               per_device_train_batch_size=self.params['batch_size'],
                                               per_device_eval_batch_size=64,
                                               warmup_steps=500,
                                               weight_decay=self.params['weight_decay'],
                                               logging_dir='/tmp/spam/logs',
                                               logging_steps=10000,
                                               evaluation_strategy="epoch")

        self.split_ratio = 0.9
        self.best_model = None

    def __compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def train(self, text, labels):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        idx = list(range(len(labels)))
        shuffle(idx)
        train_idx = idx[:int(self.split_ratio * len(idx))]
        test_idx = idx[int(self.split_ratio * len(idx)):]

        train_text = [text[x] for x in train_idx]
        train_labels = [labels[x] for x in train_idx]
        test_text = [text[x] for x in test_idx]
        test_labels = [labels[x] for x in test_idx]

        train_text_tokenized = tokenizer(train_text, truncation=True, padding=True)
        test_text_tokenized = tokenizer(test_text, truncation=True, padding=True)
        train_dataset = SpamDataset(train_text_tokenized, train_labels)
        test_dataset = SpamDataset(test_text_tokenized, test_labels)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.__compute_metrics,
        )
        trainer.train()
        self.best_model = self.model

    def score(self, text, labels):
        if not self.best_model:
            raise NotImplementedError
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        text_tokenized = tokenizer(text, truncation=True, padding=True)
        test_dataset = SpamDataset(text_tokenized, labels)
        test_dataloader = DataLoader(test_dataset, batch_size=32)
        self.best_model.eval()
        for batch in test_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.best_model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            self.metric.add_batch(predictions=predictions, references=batch["labels"])

        return self.metric.compute()['f1']

    def __str__(self):
        if self.best_model:
            args_dict = self.params
            params_lst = list()
            for param_name in args_dict.keys():
                params_lst.append(f"{param_name}: {args_dict[param_name]}")
            return 'Transformer\n' + '\n'.join(params_lst)
        else:
            return 'Not trained'


class RF:
    def __init__(self):
        nltk.download('punkt')
        pl = Pipeline([
            ('BoW', CountVectorizer()),
            ('TFIDF', TfidfTransformer(norm='l2')),
            ('RF', RandomForestClassifier())])
        param_grid = {
            'BoW__max_features': (100, 500, 1000),
            'BoW__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
            'BoW__stop_words': ('english', None),
            'BoW__min_df': (1, 2, 3, 4, 5),
            'TFIDF__use_idf': (True, False),
            'RF__n_estimators': (30, 50, 80, 100)
        }
        self.search = GridSearchCV(pl, param_grid, scoring='f1', cv=5, refit=True, n_jobs=multiprocessing.cpu_count())
        self.best_model = None

    def train(self, text, labels):
        self.search.fit(text, labels)
        self.best_model = self.search.best_estimator_
        return self.search.best_score_

    def score(self, text, labels):
        if not self.best_model:
            raise NotImplementedError
        return self.best_model.score(text, labels)

    def __str__(self):
        if self.best_model:
            args_dict = self.search.best_params_
            params_lst = list()
            for param_name in args_dict.keys():
                params_lst.append(f"{param_name}: {args_dict[param_name]}")
            return 'RandomForest\n' + '\n'.join(params_lst)
        else:
            return 'Not trained'


class FT:
    def __init__(self):
        self.label_prefix = '__label__'
        self.ft_data_path = os.path.join(DATA_FOLDER, 'ft')
        if not os.path.exists(self.ft_data_path):
            os.makedirs(self.ft_data_path)
        self.best_model = None

    def train(self, text, labels):
        # prepare data for fasttex
        split_ratio = int(0.9 * len(labels))
        train_data = ['\t'.join([self.label_prefix + str(l), t, '\n']) for l, t in zip(labels, text)]
        train_data_path = os.path.join(self.ft_data_path, 'train.tsv')
        with open(train_data_path, 'w') as f_out:
            f_out.writelines(train_data[:split_ratio])
        dev_data_path = os.path.join(self.ft_data_path, 'dev.tsv')
        with open(dev_data_path, 'w') as f_out:
            f_out.writelines(train_data[split_ratio])
        self.best_model = fasttext.train_supervised(input=train_data_path,
                                                    autotuneValidationFile=dev_data_path,
                                                    autotuneModelSize="100M",
                                                    thread=multiprocessing.cpu_count())

    def score(self, text, labels):
        if not self.best_model:
            raise NotImplementedError
        # prepare data for fasttext
        test_data = ['\t'.join([self.label_prefix + str(l), t, '\n']) for l, t in zip(labels, text)]
        test_data_path = os.path.join(self.ft_data_path, 'test.tsv')
        with open(test_data_path, 'w') as f_out:
            f_out.writelines(test_data)
        _, precision, recall = self.best_model.test(test_data_path)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def __str__(self):
        if self.best_model:
            args_obj = self.best_model.f.getArgs()
            params_lst = list()
            for hparam in dir(args_obj):
                if not hparam.startswith('__'):
                    params_lst.append(f"{hparam}: {getattr(args_obj, hparam)}")
            return 'FastText\n' + '\n'.join(params_lst)
        else:
            return 'Not trained'


def main():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
    data_folder = 'data'
    if not os.path.isdir(data_folder):
        download_data(data_url, data_folder)
    data = read_data(data_folder)
    train_split, test_split = split_data(data)

    # check if classes are balanced
    pos_count = len([x['CLASS'] for x in train_split if x['CLASS']])
    neg_count = len([x['CLASS'] for x in train_split if not x['CLASS']])
    print(f'Pos/neg ratio for train_split: {pos_count / (neg_count + 1e-12)}')
    pos_count = len([x['CLASS'] for x in test_split if x['CLASS']])
    neg_count = len([x['CLASS'] for x in test_split if not x['CLASS']])
    print(f'Pos/neg ratio for test_split: {pos_count / (neg_count + 1e-12)}')

    best_model, best_test_score = None, -float('inf')
    model_lst = [TR(), RF(), FT()]
    for lemmatize in [False, True]:
        for use_url_feat in [False, True]:
            prep_data(train_split, lemmatize=lemmatize, use_url_feat=use_url_feat)
            prep_data(test_split)
            for model in model_lst:
                model.train(text=[x['CONTENT_PREP'] for x in train_split], labels=[x['CLASS'] for x in train_split])
                score = model.score([x['CONTENT_PREP'] for x in test_split], [x['CLASS'] for x in test_split])
                if score > best_test_score:
                    best_test_score = score
                    best_data_prep = {
                        'lemmatize': lemmatize,
                        'use_url_feat': use_url_feat
                    }
                    best_model = model
    print('-' * 28)
    print('Best model')
    print(best_test_score)
    print(best_data_prep)
    print(best_model)


if __name__ == '__main__':
    main()
