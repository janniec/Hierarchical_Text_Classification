import numpy as np
import pandas as pd
import os
import subprocess

########################## FastText Functions ##########################

def train_model(train_address, ngram, lr, dim, ws, epoch, loss):
    '''
    Trains a FastText model and saves it at model_address.
    IN:
    train_address - where to grab training data from.
    ngram - max length of word ngram.
    lr - step size to convergance.
    dim - size of word vectors.
    ws - size of the context window.
    epoch - number of passes over the training data.
    loss - loss function {ns, hs, softmax}. 
       * ns - negative sampling
       * hs - hierarchical softmax
       * softmax  
    OUT: model_address - where trained model is saved.
    '''
    model_address= 'data/%s_model' % train_address.split('/')[-1].split('_')[0]
    input_output = "../fastText/fasttext supervised -input %s -output %s" %(train_address, model_address)
    parameters = " -label __label__ -wordNgrams %s -lr %s -dim %s -ws %s -epoch %s -loss %s" % (ngram, lr, dim, ws, epoch, loss)      
    trainline = input_output + parameters
    os.system(trainline)
    return model_address

def test_model(test_address, model_address):
    '''
    Tests a trained FastText model.
    IN: 
    test_address - where to grab validation data from.
    model_address - where the trained model is saved.
    OUT: test_output - FastText's evaluation score.
    Please note: FastText has its own evaluation and it is flawed. It is not precision, recall, accuracy, or f1 score. But it is a reliable measure to determine if a model is improving during a bunch of experiments. However, evaluation with SkLearn is recommended. 
    '''
    testline = "../fastText/fasttext test %s.bin %s" %(model_address, test_address)
    testing = subprocess.Popen(testline, shell=True, stdout=subprocess.PIPE)
    test_output = testing.communicate()[0].decode("utf-8")
    return test_output

def predict_with_model(test_address, model_address):
    '''
    Makes predictions with a trained FastText model and saves the predictions to predict_address.
    IN: 
    test_address - where to grab validation data from.
    model_address - where the trained model is saved.
    OUT: predict_address - where predictions have been saved. 
    '''
    random_state = test_address.split('/')[-1].split('_')[-1].split('.')[0]
    predict_address = 'data/%s_prediction_%s.txt' % (test_address.split('/')[-1].split('_')[0], random_state)
    predictline = '../fastText/fasttext predict-prob %s.bin %s 3 > %s' %(model_address, test_address, predict_address)
    os.system(predictline)
    return predict_address

########################## Creating Models #################################

def train_test_predict(train_address, test_address, ngram, lr, dim, ws, epoch, loss):
    '''
    Trains, tests, and predicts with a FastText model.
    IN:
    train_address - where to grab training data from.
    test_address - where to grab validation data from.
    ngram - max length of word ngram.
    lr - step size to convergance.
    dim - size of word vectors.
    ws - size of the context window.
    epoch - number of passes over the training data.
    loss - loss function {ns, hs, softmax}. 
       * ns - negative sampling
       * hs - hierarchical softmax
       * softmax
    OUT:
    model_address - where the trained model is saved.
    predict_address - where predictions have been saved. 
    '''
    model_address = train_model(train_address, ngram, lr, dim, ws, epoch, loss)
    test_output = test_model(test_address, model_address)
    print(test_output)
    predict_address = predict_with_model(test_address, model_address)
    return model_address, predict_address

def test_predict(test_address, model_address):
    '''
    Tests & predicts with a trained FastText model.
    Prints FastText's evaluation.
    Please note: FastText has its own evaluation and it is flawed. It is not precision, recall, accuracy, or f1 score. But it is a reliable measure to determine if a model is improving during a bunch of experiments. However, evaluation with SkLearn is recommended. 
    IN: 
    test_address - where to grab validation data from.
    model_address - where the trained model is saved.
    OUT: predict_address - where predictions have been saved. 
    '''
    test_output = test_model(test_address, model_address)
    print(test_output)
    predict_address = predict_with_model(test_address, model_address)
    return predict_address