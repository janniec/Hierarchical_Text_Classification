import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def collect_labels(file_address):
    '''
    Collects predicted/actual labels from file at file address.
    Returns list of labels for each text.
    '''
    with open(file_address) as f:
        content = f.readlines()
    labels = [c.split(' ')[0] for c in content]
    return labels

def score_labels(test_labels, predict_labels, label_set):
    '''
    Prints and returns mean scores of lists of labels.
    IN:
    test_labels - list of actual labels 
    predict_labels - list of predicted labels
    label_set - set of labels to be tallied in evaluation
    OUT: scores from evaluation by SkLearn
    '''
    precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predict_labels, average=None, labels=label_set) 
    
    print('precision: {}'.format(precision.mean()))
    print('recall: {}'.format(recall.mean()))
    print('fscore: {}'.format(fscore.mean()))
#     print('support: {}'.format(support))
    
    return precision, recall, fscore, support

def score_txtfiles(test_address, predict_address):
    '''
    IN:
    test_address - test file 
    pred_address - file of predictions
    label_set - set of labels to be tallied in evaluation
    OUT: scores from evaluation by SkLearn
    '''
    test_labels = collect_labels(test_address)
    predict_labels = collect_labels(predict_address)
    label_set = sorted(list(set(test_labels)))

    precision, recall, fscore, support = score_labels(test_labels, predict_labels, label_set)
    
    return precision, recall, fscore, support

def score_columns(df, test_column, predict_column):
    '''
    IN:
    df - dataframe of labels and predictions
    test_column - column name of actual labels
    predict_column - column name of predicted labels
    label_set - set of labels to be tallied in evaluation
    OUT: scores from evaluation by SkLearn    
    '''
    test_labels = df[test_column]
    predict_labels = df[predict_column]
    label_set = sorted(list(set(test_labels)))

    precision, recall, fscore, support = score_labels(test_labels, predict_labels, label_set)
    
    return precision, recall, fscore, support    
    