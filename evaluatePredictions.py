import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def collect_labels(file_address):
    with open(file_address) as f:
        content = f.readlines()
    labels = [c.split(' ')[0] for c in content]
    return labels

def score_labels(test_labels, predict_labels, label_set):
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predict_labels, average=None, labels=label_set) 
    
    print('precision: {}'.format(precision.mean()))
    print('recall: {}'.format(recall.mean()))
    print('fscore: {}'.format(fscore.mean()))
#     print('support: {}'.format(support))
    
    return precision, recall, fscore, support

def score_txtfiles(test_address, predict_address):
    test_labels = collect_labels(test_address)
    predict_labels = collect_labels(predict_address)
    label_set = sorted(list(set(test_labels)))

    precision, recall, fscore, support = score_labels(test_labels, predict_labels, label_set)
    
    return precision, recall, fscore, support

def score_columns(df, test_column, predict_column):
    test_labels = df[test_column]
    predict_labels = df[predict_column]
    label_set = sorted(list(set(test_labels)))

    precision, recall, fscore, support = score_labels(test_labels, predict_labels, label_set)
    
    return precision, recall, fscore, support    
    