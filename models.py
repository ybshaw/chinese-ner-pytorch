#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  :models.py
# @Time      :2022/4/3 15:36
# @Author    :ybxiao


import torch
import torch.nn as nn
from transformers import BertModel
from crf import CRF


class LstmModel(nn.Module):
    def __init__(self, vocab2idx, label2idx, embed_size=None, hidden_size=None, use_crf=True):
        super(LstmModel, self).__init__()
        self.vocab_size = len(vocab2idx)
        self.n_labels = len(label2idx)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_crf = use_crf

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_size, self.n_labels)

    def forwards(self, input_ids, label_ids=None, mask=None):
        embed = self.embedding(input_ids)               # (batch, seq_len, embed_size)
        lstm_out, _ = self.lstm(embed)                  # (batch, seq_len ,hidden_size)
        logits = self.linear(self.dropout(lstm_out))    # (batch, seq_len, n_labels)
        if label_ids is not None:
            if self.use_crf:
                crf = CRF(num_tags=self.n_labels, batch_first=True)
                loss = -1 * crf(emissions=logits, tags=label_ids, mask=mask)
                best_path = crf.decode(emissions=logits, mask=mask, nbest=1)
                return best_path, loss
            else:
                loss_func = nn.CrossEntropyLoss(ignore_index=0)
                preds = logits.view(-1, logits.size(-1))
                targets = label_ids.view(-1)
                loss = loss_func(preds, targets)
                return logits, loss
        else:
            return logits


class BERTModel(nn.Module):
    def __init__(self, bert_path, label2idx):
        super(BERTModel, self).__init__()
        self.n_labels = len(label2idx)
        self.bert = BertModel.from_pretrained(bert_path)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def forwards(self, input_ids=None, label_ids=None, attn_mask=None):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        sequence_out, pool_out = output
        if label_ids is not None:
            bert_out = sequence_out.view(-1, sequence_out.size(-1))
            label_ids = label_ids.view(-1)
            loss = self.loss_func(bert_out, label_ids)
            return loss
        else:
            return sequence_out