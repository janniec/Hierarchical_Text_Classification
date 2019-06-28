# Hierarchical Text Classification
    
## Tools
  * FastText  
  * spaCy  
  * Regular Expressions   
  * SciKit Learn  
  * klepto  
  
## Data  
Data for this project came from SciKit Learn's [20 Newsgroups Text Dataset] (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) . As described in the [source documentation] (http://qwone.com/~jason/20Newsgroups/) of this dataset, there are 20 different news groups associated with 6 different topics.   
<img src="https://github.com/janniec/Hierarchical_Text_Classification/blob/master/images/2-TableII-1.png" alt="Dimensions" align="middle" height=250px>  
  
## Models
- The classic text classification model uses FastText to train a single model to classify the 20 different news groups.    
- The hierarchical text classification model also uses FastTexct to train a higher tier model to classify the 6 different topics, and lower tier models each dedicated to a topic to classify the different news groups associated with each topic.  
Both models were evaluated on the same holdout set. 
  
## Pipeline
