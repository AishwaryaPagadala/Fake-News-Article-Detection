import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#string to test
doc_new = ['obama is running for president in 2016']

#the feature selection has been done in FeatureSelection.py module. here we will create models using those features for prediction

#first we will use bag of words techniques

#building classifier using naive bayes 

#User defined functon for K-Fold cross validatoin
def build_confusion_matrix(classifier):

	k_fold = KFold(2,True,None)
	scores = []
	a      = []
	b      = []
	#rec   = 0
	confusion = np.array([[0,0],[0,0]])

	for train_ind, test_ind in k_fold.split(DataPrep.train_news):
		train_text = DataPrep.train_news.iloc[train_ind]['Statement'] 
		train_y = DataPrep.train_news.iloc[train_ind]['Label']

		test_text = DataPrep.train_news.iloc[test_ind]['Statement']
		test_y = DataPrep.train_news.iloc[test_ind]['Label']

		classifier.fit(train_text,train_y)
		predictions = classifier.predict(test_text)
		confusion += confusion_matrix(test_y,predictions)
		#print('Actual'+'predicte')
		for i in range(len(predictions)):
			print("Actual=%s,   predicted=%s"% (test_y.iloc[i],predictions[i]))
		print('confusion matrix :')
		print(' TN   FP') 
		print(confusion[0])
		print(' FN   TP')
		print(confusion[1])
		#rec+=confusion[1][1]/(confusion[1][0]+confusion[1][1])

		score = f1_score(test_y,predictions)
		sc =precision_score(test_y,predictions)
		sc1 =recall_score(test_y,predictions)
		scores.append(score)
		a.append(sc)
		b.append(sc1)
		

	return (print('Total statements classified:', len(DataPrep.train_news)),
	        print('Confusion matrix:'),
	        print(confusion),
	        print(' f1 Score:',sum(scores)/len(scores)),
	        print(' precision ',sum(a)/len(scores)),
	        print(' Recall ',sum(b)/len(scores)))

nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb_ngram == DataPrep.test_news['Label'])


##Now using n-grams
#logistic regression classifier
logR_pipeline_ngram = Pipeline([
        ('LogR_tfidf',FeatureSelection.tfidf_ngram),
        ('LogR_clf',LogisticRegression(penalty="l2",C=1))
])

logR_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_LogR_ngram = logR_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_LogR_ngram == DataPrep.test_news['Label'])


#K-fold cross validation for all classifiers
build_confusion_matrix(nb_pipeline_ngram)
#build_confusion_matrix(logR_pipeline_ngram)

#=========================================

#print(classification_report(DataPrep.test_news['Label'], predicted_LogR_ngram))
#print(classification_report(DataPrep.test_news['Label'], predicted_nb_ngram))

DataPrep.test_news['Label'].shape

model_file = 'final_model.sav'
pickle.dump(logR_pipeline_ngram,open(model_file,'wb'))

""





