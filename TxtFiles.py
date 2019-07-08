import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#################### Split Data ##############################

def training_validation_holdout_split(doc_ids, random_state=19):
    '''
    Split list of doc ids three ways. 
    Return dictionary with keys "holdout_19", "T1_training_19", "T1_validation_19" if random state is set to 19. Each to values of a mutually exclusive list of doc ids
    '''
    t1_ids = {}
    experiment_ids, holdout_ids = train_test_split(doc_ids,\
                                               random_state=random_state,\
                                               test_size=0.2)
    training_ids, validation_ids = train_test_split(experiment_ids,\
                                               random_state=random_state,\
                                               test_size=0.25)
    t1_ids['holdout_'+str(random_state)] = holdout_ids
    print('holdout\t', len(holdout_ids))
    t1_ids['T1_training_'+str(random_state)] = training_ids
    print('training\t', len(training_ids))
    t1_ids['T1_validation_'+str(random_state)] = validation_ids
    print('validation\t', len(validation_ids))
    return t1_ids

def tier2_training_validation(df, doc_ids, tier2_types, random_state=19):
    '''
    For each of 6 topics, filters doc_ids to one topic at a time and splits it into a training set and validation set. 
    IN: 
    df - dataframe of all messages & labels
    doc_ids - list of doc_ids available for training and validation (remember to leave out the holdout set doc ids)
    tier2_types - dictionary of the 20 newsgroups to the double digit labels
    random_state - how to split the list of doc ids. 
    OUT: 
    Dictionary of 6 topics as keys. each of the 6 topics is a dictionary {"talk_training_19": [], "talk_validation_19":[]} if the topic is "talk" and the random state is 19. Each pairs of values are mutually exclusive list of doc ids.
    '''
    t2_ids = {}
    for label in sorted(tier2_types):
        print('***', label, '***')
        _ids = df[(df['_id'].isin(doc_ids)) &\
                (df['tier1_label']==label)]['_id']
        training_ids, validation_ids = train_test_split(_ids,\
                                                       random_state=random_state,\
                                                        test_size=.25
                                                       )
        t2_ids[label+'_training_'+str(random_state)] = training_ids
        print('training\t', len(training_ids))
        t2_ids[label+'_validation_'+str(random_state)] = validation_ids
        print('validation\t', len(validation_ids))
    return t2_ids

################# Generate FastText TXT #################################

def make_txtfile(filename, df, label_column):
    '''
    Generates FastText txt file
    IN:
    filename - Where to save the txtfile
    df - dataframe filtered by list of doc ids assigned to either training set, validation set, holdout set.
    label_column - designate whether txt file should have tier 1 labels or tier 2 labels. 
    '''
    f = open(filename, 'w')
    for index, row in df.iterrows():
        f.write('__label__%s %s\n'\
                 % (row[label_column], row['clean_text'].replace('\n', '')))
    f.close()    