# Hierarchical Text Classification
Hierarchical Text Classifier of News Group Messages using Facebook's FastText.  
   
   
## FastText
[Facebook's FastText](https://fasttext.cc/) is an open source library for fast and efficient learning of word representation (unsupervised learning) and text classification (supervised learning).  
  
For text classification, FastText measures the probablity of association between label vectors to each text vector. However, it greatly increases speed through hierarchical classification to avoid calculating the probability of every label in the training set for every text. Instead, it uses a binary tree, like the image below, where every leaf node is a label and every node is a probability. So for each text, FastText will only compute the probability of each node along the path to the correct label.  
  
  
<img src="https://github.com/janniec/Hierarchical_Text_Classification/blob/master/images/hierarchical_softmax_example.png" alt="Dimensions" align="middle" height=250px>   
  
  
## Data  
Data for this project came from SkLearn's [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) . As described in the [source documentation](http://qwone.com/~jason/20Newsgroups/) of this dataset, there are 20 different news groups associated with 6 different topics.  
  
  
<img src="https://github.com/janniec/Hierarchical_Text_Classification/blob/master/images/2-TableII-1.png" alt="Dimensions" align="middle" height=600px>  
   
  
## Pipeline  

1. Load 20 newsgroup dataset without headers and footers from SkLearn.  
2. Clean & tier the labels and targets.  
3. Clean & preprocess the texts.  
      * Remove data points with virtually no text & duplicates.  
      * Remove email addresses and special characters with Regex.  
      * Lemmatize tokens and remove stopwords, utilizing SpaCy.  
4. Save cleaned & processed datasets with Klepto.  
5. Split dataset into holdout set, training sets & validation sets, and generate txt files.   
6. Train FastText models on training sets, predict on validation sets, and adjust parameters.  
7. Predict 1 of 6 topics with Tier1 model - 'comp', 'misc', 'rec', 'sci', 'soc', or 'talk'.   
8. Pass classied text to 1 of 6 models train to predict newsgroup associated with predicted topic, such that:  
    * the 'comp' model will predict 'graphics', 'os_ms-windows_misc', 'sys_ibm_pc_hardware', 'sys_mac_hardware', or 'windows_x'.  
    * the 'rec' model will predict 'autos', 'motorcycles', 'sport_baseball', or 'sport_hockey'.  
    * the 'sci' model will predict 'crypt', 'electronics', 'med', or 'space'.  
    * the 'soc' model will predict 'religion_atheism', 'religion_christian', or 'religion_misc'.  
    * the 'talk' model will predict 'politics_guns', 'politics_mideast', or 'politics_misc'.  
9. Evaluate predicted Tier2 labels against the actual 20 Newgroup labels using SkLearn.   
  
* In order to find the best parameters to fit the model on the data, please use the ExperimentSweep module.  
    1. Provide ranges of each parameters & datasets.  
    2. Start DateTime timer.  
    3. Create an experiment dataframe to track combination of parameters and scores.  
    4. Print status updates.  
    5. Train FastText models on training set & predict on validation set.     
    6. Evaluate predictions utilizing skLearn.   
    7. Log in dataframe.  
    8. End DateTime timer and prints duration.  
    9. Output dataframe sorted on descending F1 score.  
  
  
## Tools  
  * FastText  
  * SpaCy  
  * Regex   
  * SkLearn  
  * Klepto  
     
  
## Models  
- The classic text classification model uses FastText to train a single model to classify the 20 different news groups. The model was evaluated on a holdout set. This model did really well given that it had to predict 20 labels.    
See [Classic_Text_Classification.ipynb](https://github.com/janniec/Hierarchical_Text_Classification/blob/master/notebooks/2_Classic_Text_Classification.ipynb).   
  
| Labels     	| Recall 	| Precision 	| FScore 	|
|-----------	|--------	|-----------	|---------	|
| 20 Newsgroups	| 80.5   	| 80.8      	| 80.3    	|
  
- The tiered text classification model uses FastText to train a higher tier model to classify the 6 different topics, and lower tier models, each dedicated to a topic, to classify the different news groups associated with each topic.  
By tiering these models, we have the opportuntyto tailor the models to their given dataset. As a result, the tiered models improved in overall performance on the same holdout set.  
See [Tiered_Text_Classification.ipynb](https://github.com/janniec/Hierarchical_Text_Classification/blob/master/notebooks/3_Tiered_Text_Classification.ipynb).  
  
| Labels     	| Recall 	| Precision 	| FScore 	|
|-----------	|--------	|-----------	|---------	|
| 6 Topics  	| 89.7   	| 91.4      	| 90.4    	|
| 20 Newsgroups	| 83.8   	| 84.3      	| 83.8    	|
  
  
## Next Steps  
Natural Langauge Processing projects generally require large amounts of data, which are almost always in shortage. Next steps will be to gauge FastText's performance at varying amounts of training data against varying amounts of holdout data.  
  
In addition, FastText has the ability to output not only predicted labels but also probabilities of associated with predictions. Next steps may include cleaning up predictions, especially false positives, by setting thresholds on prediction probabilities. 