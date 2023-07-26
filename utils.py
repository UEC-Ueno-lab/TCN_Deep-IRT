import json
import numpy as np
from sklearn import metrics
import torch
import json
from collections import defaultdict
import os
from sklearn.metrics import f1_score,cohen_kappa_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def compute_kappa_f1_score(qids, target, pred):
    unique_qids = np.unique(qids)
    f1_scores = np.zeros(len(unique_qids))
    cohen_scores =np.zeros(len(unique_qids))
    weights = np.zeros(len(unique_qids))
    for idx, qid in enumerate(unique_qids):
        qid_mask = qids==qid
        qid_target =  target[qid_mask]
        qid_pred = pred[qid_mask]
        f1_scores[idx] = f1_score(qid_target,qid_pred,average='macro')
        cohen_scores[idx] = cohen_kappa_score(qid_target, qid_pred)
        weights[idx] =  np.sum(qid_mask)/(len(target)+0.)
    cohen_scores[np.isnan(cohen_scores)] = 1.
    avg_f1_score = np.average(f1_scores)
    weighted_f1_score = np.sum(f1_scores*weights)
    avg_cohen_score =  np.average(cohen_scores)
    weighted_cohen_score =  np.sum(cohen_scores*weights)
    return avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score

def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred_new = all_pred.copy()
    all_pred_new[all_pred > 0.5] = 1.0
    all_pred_new[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred_new)

def result_save(file_path, fold, auc, acc, loss):
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as fo:
            info = json.load(fo)
            info['result']['fold'].append(fold)
            info['result']['auc'].append(auc)
            info['result']['acc'].append(acc)
            info['result']['loss'].append(loss)
    else:
        info = defaultdict(dict)
        info['result']['fold']=[fold]
        info['result']['auc']=[auc]
        info['result']['acc']=[acc]
        info['result']['loss']=[loss]
    with open(file_path, 'w', encoding='utf-8') as fo:
        json.dump(info, fo, ensure_ascii=False, indent=4)