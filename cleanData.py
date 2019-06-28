import numpy as np
import pandas as pd
import re
import spacy
nlp = spacy.load('en')



def dataframe_the_news(news):
    '''
    news is the 20newsgroup dfset from sklearn
    '''
    df = pd.DataFrame([list(news.target), news.data]).T
    df.columns = ['target', 'text']
    for index, row in df.iterrows():
        df.loc[index, 'label'] = news.target_names[row['target']] 
    return df

################## Clean Texts #############################

def filter_texts(df):
    '''
    remove virtually no text
    remove duplicates
    '''
    df = df[df['text'].str.len()>10].reset_index(drop=True)  
    df.drop_duplicates(keep='first', inplace=True)
    return df

def remove_email_addresses(text):
    emailless_text = re.sub(r"\S*@\S*\s?", '', text)
    return emailless_text

def remove_stopwords_lemmatize(text):
    doc = nlp(text.replace('\n', ' '))
    stopless_lemmas = str(' '.join(\
                                   [str(t.lemma_) \
                                    for t in doc \
                                    if t.is_stop==False]\
                                  ))
    return stopless_lemmas

def remove_special_characters(text):
    unspecial_text = re.sub(r"[^a-zA-Z]+", ' ', text).lower()
    return unspecial_text

def process_texts(df):
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():    
        emailless_text = remove_email_addresses(row['text'])
        stopless_lemmas = remove_stopwords_lemmatize(emailless_text)
        df.loc[index, 'clean_text'] = remove_special_characters(stopless_lemmas)
    return df
        
##################### Unique row Ids ################################    

def make_doc_ids(df):
    df = df.reset_index(drop=False).rename({'index': '_id'}, axis=1)
    return df

#################### Clean Labels & Targets #######################
  
def group_news_topics(df):
    df['label'].replace(to_replace={'talk.religion.misc': 'soc.religion.misc'
                                   , 'alt.atheism': 'soc.religion.atheism'}
                      , inplace=True)
    return df

def tier_labels(df):
    for index, row in df.iterrows():
        df.loc[index, 'tier1_label'] = row['label'].split('.')[0]
        df.loc[index, 'tier2_label'] = '_'.join(row['label'].split('.')[1:])
    return df

def clean_new_labels(df):
    df = group_news_topics(df)
    df = tier_labels(df)
    df = df.drop(['label'], axis=1)
    return df

def labels_to_targets(df):
    tier1_targets = {t:i+1 for i, t in\
                     enumerate(sorted(df['tier1_label'].unique()))\
                    }
    df['tier1_targets'] = df['tier1_label'].map(tier1_targets)
    
    tier2_targets = {}
    for k in tier1_targets.keys():
        t2_labels = sorted(df[df['tier1_label']==k]['tier2_label'].unique())
        for i, l in enumerate(t2_labels):
            tier2_targets[l]= tier1_targets[k]*10+i+1
    df['tier2_targets'] = df['tier2_label'].map(tier2_targets)
    
    df = df.drop(['target'], axis=1)
    return tier1_targets, tier2_targets, df

####################### All Pre-Processing #####################

def cleaned_processed_labeled(news):
    df = dataframe_the_news(news)
    df = process_texts(filter_texts(df))
    print('Text cleaned & processed.')
    df = make_doc_ids(df)
    df = clean_new_labels(df)
    print('Data getting labels & targets.')
    tier1_targets, tier2_targets, df = labels_to_targets(df)
    return (tier1_targets, tier2_targets, df)
    
