# Hierarchical Text Classification
  
  
## FastText
[Facebook's FastText](https://fasttext.cc/) is an open source library for fast and efficient learning of word representation (unsupervised learning) and text classification (supervised learning).  
  
For text classification, FastText essentially measures the probablity of association between label vectors to each text vector. However, it greatly increases speed through hierarchical classification to avoid calculating the probability of every label in the training set for every text. Instead, it uses a binary tree, like the image below, where every leaf node is a label and every node is a probability. So for each text, FastText will only compute the probability of each node along the path to the correct label.  
  
  
<img src="https://github.com/janniec/Hierarchical_Text_Classification/blob/master/images/hierarchical_softmax_example.png" alt="Dimensions" align="middle" height=250px>   
  
  
## Data  
Data for this project came from Scikit Learn's [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) . As described in the [source documentation](http://qwone.com/~jason/20Newsgroups/) of this dataset, there are 20 different news groups associated with 6 different topics.  
  
  
<img src="https://github.com/janniec/Hierarchical_Text_Classification/blob/master/images/2-TableII-1.png" alt="Dimensions" align="middle" height=600px>  
   
   
## Models  
- The classic text classification model uses FastText to train a single model to classify the 20 different news groups.    
See [Classic_Text_Classification.ipynb](https://github.com/janniec/Hierarchical_Text_Classification/blob/master/notebooks/2_Classic_Text_Classification.ipynb).  
- The tiered text classification model uses FastText to train a higher tier model to classify the 6 different topics, and lower tier models, each dedicated to a topic, to classify the different news groups associated with each topic.  
Both models were evaluated on the same holdout set.  
See [Tiered_Text_Classification.ipynb](https://github.com/janniec/Hierarchical_Text_Classification/blob/master/notebooks/3_Tiered_Text_Classification.ipynb).  
  
  
## Tools  
  * FastText  
  * SpaCy  
  * Regex   
  * Scikit Learn  
  * Klepto  
    
  
## Pipeline  
1. Load 20 newsgroup dataset without headers and footers from Scikit Learn.  
2. Clean & tier the labels and targets.  
3. Clean & preprocess the texts.  
      * Remove data points with virtually no text & duplicates.  
      * Remove email addresses and special characters with Regex.  
      * Lemmatize tokens and remove stopwords, utilizing SpaCy.  
4. Save cleaned & processed datasets with Klepto.  
5. Split dataset into holdout set, training sets & validation sets, and generate txt files.  
6. Train FastText models on training sets, predict on validation sets, and adjust parameters .  
7. Test and evaluate FastText models on holdout set.  
  
  
## Next Steps  
Natural Langauge Processing projects generally require large amounts of data, which are almost always in shortage. Next steps will be to gauge FastText's performance at varying amounts of training data against varying amounts of holdout data.  
  
In addition, FastText has option to predict not only labels but also probabilities of labels. Next steps may include cleaning up predictions, especially false positives, by setting thresholds on prediction probabilities. 