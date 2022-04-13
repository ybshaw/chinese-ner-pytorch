#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2022/4/2 10:18
# @Author    :ybxiao

"""
数据预处理
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def read_corpus(sents_path, tags_path):
    corpus = []
    with open(sents_path, 'r', encoding='utf-8') as f1, open(tags_path, 'r', encoding='utf-8') as f2:
        for line_sent, line_tag in zip(f1.readlines(), f2.readlines()):
            line_sent = line_sent.strip().split()
            line_tag = line_tag.strip().split()
            assert len(line_sent) == len(line_tag)
            corpus.append((line_sent, line_tag))
    return corpus


def read_tags(file_path):
    tags = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            tags.append(line)
    return tags


def build_vocab2idx(train_corpus):
    vocabs = []
    for sent, _ in train_corpus:
        vocabs.extend(sent)
    vocabs = list(set(vocabs))
    vocab2idx = {vocab: idx + 2 for idx, vocab in enumerate(vocabs)}
    vocab2idx.update({"<pad>": 0, "<unk>": 1})
    return vocab2idx


def build_label2idx(tags_list):
    label2idx = {label: idx for idx, label in enumerate(tags_list)}
    return label2idx


class LSTMTextDataset(Dataset):
    def __init__(self, corpus, vocab2idx=None, label2idx=None, seq_len=50):
        super(LSTMTextDataset, self).__init__()
        self.corpus = corpus
        self.vocab2idx = vocab2idx
        self.label2idx = label2idx
        self.seq_len = seq_len
        self.len = len(self.corpus)

    def __getitem__(self, item):
        sentence, label = self.corpus[item]
        input_ids = [self.vocab2idx.get(word, 1) for word in sentence]
        label_ids = [self.label2idx.get(tag) for tag in label]
        assert len(input_ids) == len(label_ids)

        if len(input_ids) < self.seq_len:
            input_ids.extend([0] * (self.seq_len - len(input_ids)))
            label_ids.extend([0] * (self.seq_len - len(label_ids)))
        else:
            input_ids = input_ids[:self.seq_len]
            label_ids = label_ids[:self.seq_len]
        mask = [1 if ids != 0 else 0 for ids in input_ids]
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long)}

    def __len__(self):
        return self.len


class BERTTextDataset(Dataset):
    def __init__(self, corpus, tokenizer, label2idx, seq_len):
        super(BERTTextDataset, self).__init__()
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.seq_len = seq_len
        self.len = len(corpus)

    def _tokenize_extend_labels(self, sentence, label):
        tokens = []
        labels = []
        for word, tag in zip(sentence, label):
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            tokens.extend(tokenized_word)
            labels.extend([tag] * n_subwords)
        return tokens, labels

    def __getitem__(self, item):
        sentence, label = self.corpus[item]
        tokens, labels = self._tokenize_extend_labels(sentence, label)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels = ['O'] + labels + ['O']

        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
            labels = labels[:self.seq_len]
        else:
            tokens += ['[PAD]' for _ in range(self.seq_len - len(tokens))]
            labels += ['O' for _ in range(self.seq_len - len(labels))]

        attn_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self.label2idx.get(tag) for tag in labels]
        assert len(input_ids) == len(label_ids) == len(attn_mask)

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "label_ids": torch.tensor(label_ids, dtype=torch.long),
                "attn_mask": torch.tensor(attn_mask, dtype=torch.long)}

    def __len__(self):
        return self.len


def build_lstm_dataloader(corpus, vocab2idx, label2idx, seq_len=50, params=None):
    dataset = LSTMTextDataset(corpus, vocab2idx, label2idx, seq_len)
    dataloader = DataLoader(dataset, **params)
    return dataloader


if __name__ == "__main__":
    tags_path = 'dataset/MSRA/tags.txt'
    train_sent_path = 'dataset/MSRA/train/sentences.txt'
    train_tag_path = 'dataset/MSRA/train/tags.txt'

    val_sent_path = 'dataset/MSRA/val/sentences.txt'
    val_tag_path = 'dataset/MSRA/val/tags.txt'

    test_sent_path = 'dataset/MSRA/test/sentences.txt'
    test_tag_path = 'dataset/MSRA/test/tags.txt'

    tags = read_tags(tags_path)     # ['O', 'B-ORG', 'I-PER', 'B-PER', 'I-LOC', 'I-ORG', 'B-LOC']
    train_corpus = read_corpus(train_sent_path, train_tag_path)
    vocab2idx = build_vocab2idx(train_corpus)   # len: 4761
    label2idx = build_label2idx(tags) # {'O': 0, 'B-ORG': 1, 'I-PER': 2, 'B-PER': 3, 'I-LOC': 4, 'I-ORG': 5, 'B-LOC': 6}
    valid_corpus = read_corpus(val_sent_path, val_tag_path)
    test_corpus = read_corpus(test_sent_path, test_tag_path)

    train_params = {"batch_size": 32, "shuffle": True}
    valid_params = {"batch_size": 32, "shuffle": False}
    test_params = {"batch_size": 64, "shuffle": False}

    # train_loader = build_lstm_dataloader(train_corpus, vocab2idx, label2idx, seq_len=50, params=train_params)
    valid_loader = build_lstm_dataloader(valid_corpus, vocab2idx, label2idx, seq_len=50, params=valid_params)
    # test_loader = build_lstm_dataloader(test_corpus, vocab2idx, label2idx, seq_len=50, params=test_params)

    for batch in valid_loader:
        input_ids, label_ids, mask = batch["input_ids"], batch["label_ids"], batch["mask"]
        print(input_ids.size(), label_ids.size(), mask.size())
        break

