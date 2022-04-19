#!/usr/bin/python
# -*- coding:utf-8 -*-
# @FileName  :train_eval.py
# @Time      :2022/4/3 16:58
# @Author    :ybxiao

import random
import numpy as np
from tqdm import trange
import torch
from models import LstmModel, BERTModel
from transformers import BertTokenizer
from seqeval.metrics import accuracy_score, classification_report
from utils import read_corpus, read_tags
from utils import build_loader, build_label2idx, build_vocab2idx
from utils import Parmas


def set_random_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 模型训练
def train_model(model, epochs, train_loader, valid_loader, label2idx, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    for epoch in trange(epochs, desc='Epoch'):
        print("Training epoch: [{}/{}]".format(epoch+1, epochs))
        tr_loss, tr_acc = 0, 0
        n_steps, n_examples = 0, 0
        model.train()

        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)

            output = model.forwards(input_ids, label_ids, mask=mask)
            if model.use_crf:
                best_path, loss = output
                tr_loss += loss.item()
                n_steps += 1
                best_path = best_path.squeeze().contiguous().view(-1)
                active_logits = mask.view(-1) == 1
                preds = torch.masked_select(best_path, active_logits)
                targets = label_ids.view(-1)
                targets = torch.masked_select(targets, active_logits)
                acc = accuracy_score(y_true=targets.cpu().numpy(), y_pred=preds.cpu().numpy())
                tr_acc += acc
            else:
                logits, loss = output
                tr_loss += loss.item()
                n_steps += 1

                logits = logits.view(-1, logits.size(-1))
                preds = torch.argmax(logits, dim=1)
                targets = label_ids.view(-1)
                active_logits = mask.view(-1) == 1
                preds = torch.masked_select(preds, active_logits)
                targets = torch.masked_select(targets, active_logits)
                acc = accuracy_score(y_true=targets.cpu().numpy(), y_pred=preds.cpu().numpy())
                tr_acc += acc
            if idx % 200 == 199:
                avg_loss = tr_loss / n_steps
                print("batch loss is: {:.5f}".format(avg_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / n_steps
        epoch_acc = tr_acc / n_steps
        # 在验证集上评估模型性能
        valid_acc, _ = evaluate(model, valid_loader, device, label2idx)
        print("train_loss: {:.5f} train_acc:{:.4f} valid_acc:{:.4f}".format(epoch_loss, epoch_acc, valid_acc))


# 在测试集上评估
def evaluate(model, data_loader, device, label2idx=None, save_model_path=None):
    idx2label = {v: k for k, v in label2idx.items()}
    y_pred, y_true = [], []
    if save_model_path is not None:
        model.load_state_dict(torch.load(save_model_path))
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            mask = batch['mask'].to(device)

            output = model.forwards(input_ids, label_ids, mask=mask)
            if model.use_crf:
                best_path, loss = output
                best_path = best_path.squeeze().contiguous().view(-1)        # (batch * seq_len, )
                active_logits = mask.view(-1) == 1
                targets = label_ids.view(-1)
                preds = torch.masked_select(best_path, active_logits)
                targets = torch.masked_select(targets, active_logits)
                y_pred.append(list([idx2label[p_idx.item()] for p_idx in preds]))
                y_true.append(list([idx2label[t_idx.item()] for t_idx in targets]))
            else:
                logits, loss = output                                       # (batch, seq_len, n_labels)
                logits = logits.view(-1, logits.size(-1))
                preds = torch.argmax(logits, dim=1)
                targets = label_ids.view(-1)
                active_logits = mask.view(-1) == 1
                preds = torch.masked_select(preds, active_logits)
                targets = torch.masked_select(targets, active_logits)
                y_pred.append(list([idx2label[p_idx.item()] for p_idx in preds]))
                y_true.append(list([idx2label[t_idx.item()] for t_idx in targets]))
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    report = classification_report(y_true=y_true, y_pred=y_pred, digits=4)
    return acc, report


def main():
    tags = read_tags(Parmas.tags_path)
    train_corpus = read_corpus(Parmas.train_sent_path, Parmas.train_tag_path)
    vocab2idx = build_vocab2idx(train_corpus)  # len: 4761
    label2idx = build_label2idx(tags)
    # {'O': 0, 'B-ORG': 1, 'I-PER': 2, 'B-PER': 3, 'I-LOC': 4, 'I-ORG': 5, 'B-LOC': 6}
    tokenizer = BertTokenizer.from_pretrained(Parmas.bert_path, do_lower_case=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(Parmas.train_sent_path, Parmas.train_tag_path, vocab2idx=vocab2idx,
                                label2idx=label2idx, tokenizer=tokenizer, seq_len=80,
                                params=Parmas.train_params, model_type='lstm')
    valid_loader = build_loader(Parmas.val_sent_path, Parmas.val_tag_path, vocab2idx=vocab2idx,
                                label2idx=label2idx, tokenizer=tokenizer, seq_len=80,
                                params=Parmas.valid_params, model_type='lstm')
    test_loader = build_loader(Parmas.test_sent_path, Parmas.test_tag_path, vocab2idx=vocab2idx,
                               label2idx=label2idx, tokenizer=tokenizer, seq_len=80,
                               params=Parmas.test_params, model_type='lstm')

    model = LstmModel(vocab2idx, label2idx, embed_size=100, hidden_size=128, use_crf=True).to(device)
    # model = BERTModel(bert_path=Parmas.bert_path, label2idx=label2idx, use_crf=False).to(device)
    train_model(model, 2, train_loader, valid_loader, label2idx, device)
    acc, report = evaluate(model, test_loader, device, label2idx, save_model_path=Parmas.SAVED_MODEL_PATH)
    print(report)


if __name__ == "__main__":
    set_random_seed(seed=2022)
    main()
