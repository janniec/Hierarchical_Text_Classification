import numpy as np
import pandas as pd
import itertools
import datetime
import os
import subprocess
import FTCommands as ftc


def make_addresses(train_address):
    '''
    Outputs file addresses according to formulaic naming convention from train_address.
    '''
    address_path = train_address.split('/')[0]
    file_call = train_address.split('/')[-1].split('_')[0]
    
    test_address = address_path+'/%s_validation_19.txt' % file_call
    model_address = address_path+'/%s_model' % file_call
    return test_address, model_address

def experiment_sweep(train_address, ngrams, lrs, dims, wss, epochs, losses):
    '''
    To find the best parameters to best fit the model to the training data:
    1. Provide ranges of each parameters & datasets.  
    2. Start DateTime timer.  
    3. Create an experiment dataframe to track combination of parameters and scores.  
    4. Print status updates.  
    5. Train FastText models on training set & predict on validation set.     
    6. Evaluate predictions utilizing skLearn.   
    7. Log in dataframe.  
    8. End DateTime timer and prints duration.  
    9. Output dataframe sorted on descending F1 score.
    IN:
    train_address - where to grab training data from.
    ngrams - list of max lengths of word ngram
    lrs - list of step sizes to convergance.
    dims - list of sizees of word vectors
    wss - list of sizees of the context window
    epochs - list of number of passes over the training data.
    losses - list of loss functions {ns, hs, softmax} 
    OUT: Experiment dataframe sorted on descending FastText Scores
    Please note: FastText has its own evaluation and it is flawed. It is not precision, recall, accuracy, or f1 score. But it is a reliable measure to determine if a model is improving during experiments. However, evaluation with SkLearn is recommended.
    '''
    starttime = datetime.datetime.now()
    print('Start:\t', starttime)    
    
    test_address, model_address = make_addresses(train_address)
    
    num_combos = len(list(itertools.product(ngrams, lrs, \
                                            dims, wss, \
                                            epochs, losses)))    

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
            
            _ = ftc.train_model(train_address, combo[0], combo[1], \
                                combo[2], combo[3], combo[4], combo[5])
            
            test_output = ftc.test_model(test_address, model_address)

            experiment_log.loc[index, 'score'] = float(test_output.split('\t')[-1])
        except: print(index, combo)
 
    endtime = datetime.datetime.now()
    print('End:\t', endtime)
    print('Duration:\t', endtime-starttime)
    
    return experiment_log.sort_values(by='score', ascending=False).reset_index(drop=True)