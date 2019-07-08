import numpy as np
import pandas as pd
import re
import spacy
nlp = spacy.load('en')



def dataframe_the_news(news):
    '''
    Converts SkLearn dataset into dataframe.
    IN: news - SkLearn object from fetch_20newsgroups()
    OUT: df - Rows of newsgroup messages. Columns of target labels and message text.
    '''
    df = pd.DataFrame([list(news.target), news.data]).T
    df.columns = ['target', 'text']
    for index, row in df.iterrows():
        df.loc[index, 'label'] = news.target_names[row['target']] 
    return df

################## Clean Texts #############################

def filter_texts(df):
    '''
    Removes rows where texts are less than 10 characters. 
    Removes duplicates.
    '''
    df = df[df['text'].str.len()>10].reset_index(drop=True)  
    df.drop_duplicates(keep='first', inplace=True)
    return df

def remove_email_addresses(text):
    '''
    Filters email addresses from text.
    '''
    emailless_text = re.sub(r"\S*@\S*\s?", '', text)
    return emailless_text

def remove_stopwords_lemmatize(text):
    '''
    Tokenizes text with SpaCy. 
    Filters out stopwords. 
    Converts tokens into lemmas.
    '''
    doc = nlp(text.replace('\n', ' '))
    stopless_lemmas = str(' '.join(\
                                   [str(t.lemma_) \
                                    for t in doc \
                                    if t.is_stop==False]\
                                  ))
    return stopless_lemmas

def remove_special_characters(text):
    '''
    Filters out special characters and numbers from text.
    '''
    unspecial_text = re.sub(r"[^a-zA-Z]+", ' ', text).lower()
    return unspecial_text

def process_texts(df):
    '''
    Interates through dataframe and cleans & processes the texts.
    '''
    df = df.reset_index(drop=True)
    for index, row in df.iterrows():    
        emailless_text = remove_email_addresses(row['text'])
        stopless_lemmas = remove_stopwords_lemmatize(emailless_text)
        df.loc[index, 'clean_text'] = remove_special_characters(stopless_lemmas)
    return df
        
##################### Unique row Ids ################################    

def make_doc_ids(df):
    '''
    Converts row index into doc_ids.
    '''
    df = df.reset_index(drop=False).rename({'index': '_id'}, axis=1)
    return df

#################### Clean Labels & Targets #######################
  
def group_news_topics(df):
    '''
    Rename target labels to group talk.religion.misc, alt.atheism, and soc.religion.christion under the "soc" topic.
    '''
    df['label'].replace(to_replace={'talk.religion.misc': 'soc.religion.misc'
                                   , 'alt.atheism': 'soc.religion.atheism'}
                      , inplace=True)
    return df

def tier_labels(df):
    '''
    Split target labels 2 tiered labels. 
    Tier 1 are the 6 topics - comp, rec, sci, talk, soc, misc.
    Tier 2 are the 20 news groups. 
    '''
    for index, row in df.iterrows():
        df.loc[index, 'tier1_label'] = row['label'].split('.')[0]
        df.loc[index, 'tier2_label'] = '_'.join(row['label'].split('.')[1:])
    return df

def clean_new_labels(df):
    '''
    Group the 20 newsgroups in to 6 topics.
    Turn the 6 topics into Tier 1 labels and the 20 newsgroups into Tier 2 labels.
    '''
    df = group_news_topics(df)
    df = tier_labels(df)
    df = df.drop(['label'], axis=1)
    return df

def labels_to_targets(df):
    '''
    Convert the Tier 1 labels into single digit tags - 1, 2, 3, 4, 5, 6
    Convert the Tier 2 lables into double digit tags - 11, 12, 21, 31, etc.
    '''
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
    '''
    Converts SkLearn dataset into dataframe.
    Cleans & processes the texts of newsgroup messages.
    Groups & tiers the labels and converts into number labels. 
    IN: news - SkLearn object from fetch_20newsgroups()
    OUT: 
    tier1_targets - Dictionary of Tier 1 labels to single digit number labels.
    tier2_targets - Dictionary of Tier 2 labels to double digit number labels.
    df - dataframe of cleaned and processed texts & labels.
    '''
    df = dataframe_the_news(news)
    df = process_texts(filter_texts(df))
    print('Text cleaned & processed.')
    df = make_doc_ids(df)
    df = clean_new_labels(df)
    print('Data getting labels & targets.')
    tier1_targets, tier2_targets, df = labels_to_targets(df)
    return (tier1_targets, tier2_targets, df)
    
