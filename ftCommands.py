import numpy as np
import pandas as pd
import itertools
import datetime
import os
import subprocess

import evaluatePredictions as ep

def experiment_sweep(ngrams, lrs, dims, wss, epochs, losses):
    starttime = datetime.datetime.now()
    print('Start:\t', starttime)    
    
    num_combos = len(list(itertools.product(ngrams, lrs, \
                                            dims, wss, \
                                            epochs, losses)))
    
    train_address = 'data/train.txt'
    model_address = 'data/model'
    test_address = 'data/test.txt'

    experiment_log = pd.DataFrame(columns=['ngram', 'lr', \
                                           'dim', 'ws', \
                                           'epoch', 'loss', 'score'])

    for index, combo in enumerate(itertools.product(ngrams, lrs, \
                                                    dims, wss, \
                                                    epochs, losses)):
        if index % 10 == 0:
            print(index, '\t', str(float(index)/float(num_combos)), '% done')
        try:
            experiment_log.loc[index, 'ngram'] = combo[0]
            experiment_log.loc[index, 'lr'] = combo[1]
            experiment_log.loc[index, 'dim'] = combo[2]
            experiment_log.loc[index, 'ws'] = combo[3]
            experiment_log.loc[index, 'epoch'] = combo[4]
            experiment_log.loc[index, 'loss'] = combo[5]

            input_output = "../fastText/fasttext supervised -input %s -output %s" %(train_address, model_address)
            paramters = " -label __label__ -wordNgrams %s -lr %s -dim %s -ws %s -epoch %s -loss %s" % combo        
            trainline = input_output + paramters
            os.system(trainline)

            testline = "../fastText/fasttext test %s.bin %s" %(model_address, test_address)
            testing = subprocess.Popen(testline, shell=True, stdout=subprocess.PIPE)
            test_output = testing.communicate()[0].decode("utf-8")

            experiment_log.loc[index, 'score'] = float(test_output.split('\t')[-1])
        except: print(index, combo)
 
    endtime = datetime.datetime.now()
    print('End:\t', endtime)
    print('Duration:\t', endtime-starttime)
    
    return experiment_log

########################## FastText Functions ##########################

def train_model(train_address, ngram, lr, dim, ws, epoch, loss):
    model_address= 'data/%s_model' % train_address.split('/')[-1].split('_')[0]
    input_output = "../fastText/fasttext supervised -input %s -output %s" %(train_address, model_address)
    paramters = " -label __label__ -wordNgrams %s -lr %s -dim %s -ws %s -epoch %s -loss %s" % (ngram, lr, dim, ws, epoch, loss)      
    trainline = input_output + paramters
    os.system(trainline)
    return model_address

def test_model(test_address, model_address):
    testline = "../fastText/fasttext test %s.bin %s" %(model_address, test_address)
    testing = subprocess.Popen(testline, shell=True, stdout=subprocess.PIPE)
    test_output = testing.communicate()[0].decode("utf-8")
    return test_output

def predict_with_model(test_address, model_address):
    random_state = test_address.split('/')[-1].split('_')[-1].split('.')[0]
    predict_address = 'data/%s_prediction_%s.txt' % (test_address.split('/')[-1].split('_')[0], random_state)
    predictline = '../fastText/fasttext predict-prob %s.bin %s 3 > %s' %(model_address, test_address, predict_address)
    os.system(predictline)
    return predict_address

########################## Creating Models 

def train_test_predict(train_address, test_address, ngram, lr, dim, ws, epoch, loss):
    model_address = train_model(train_address, ngram, lr, dim, ws, epoch, loss)
    test_output = test_model(test_address, model_address)
    print(test_output)
    predict_address = predict_with_model(test_address, model_address)
    return model_address, predict_address

def test_predict(test_address, model_address):
    test_output = test_model(test_address, model_address)
    print(test_output)
    predict_address = predict_with_model(test_address, model_address)
    return predict_address