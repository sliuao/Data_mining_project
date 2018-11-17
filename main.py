from sklearn import tree
from sklearn.externals import joblib
import preprocess
import read_data
import pandas as pd
import numpy as np
from preprocess import data
from preprocess import label
from preprocess import data2
from preprocess import label2
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


'''
Here are two functions used to test the score of the classifier
Neural network--single layer with different H
'''
def test_neural(data,label):
	H=[40,50,60,100,200,300]
	for i in H:
		clf = MLPClassifier(hidden_layer_sizes=i)
		score = cross_val_score(clf, data, label, cv=5)

		print "5-fold cross-validation score of H=", i, " is :: ", score.mean()


'''
Here are functions training models using the csv file and predict them
Then find out the accuracy and the running time!
'''
def decisiontree(data,label,data2,label2):
	start = time.time()
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(data, label)
	joblib.dump(clf, 'decisiontree.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of decision tree :: ", accuracy_score(label, train_p)
	print "Test Accuracy of decision tree :: ", accuracy_score(label2, test_p)
	print "Train precision score of decision tree :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of decision tree :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of decision tree :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of decision tree :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start


def randomforest(data,label,data2,label2):
	start = time.time()
	clf = RandomForestClassifier(n_estimators=10)
	clf = clf.fit(data, label)
	joblib.dump(clf, 'randomforest.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of random forest :: ", accuracy_score(label, train_p)
	print "Test Accuracy of random forest :: ", accuracy_score(label2, test_p)
	print "Train precision score of random forest :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of random forest :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of random forest :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of random forest :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start

def neural(data,label,data2,label2):
	start = time.time()
	clf = MLPClassifier(hidden_layer_sizes=50)
	clf = clf.fit(data, label)
	joblib.dump(clf, 'neural_50.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of neural network :: ", accuracy_score(label, train_p)
	print "Test Accuracy of neural network :: ", accuracy_score(label2, test_p)
	print "Train precision score of neural network :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of neural network :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of neural network :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of neural network :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start

def svm(data,label,data2,label2):
	start = time.time()
	clf = svm.SVC(gamma=1)
	clf = clf.fit(data, label)
	joblib.dump(clf, 'svm_1.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of SVM :: ", accuracy_score(label, train_p)
	print "Test Accuracy of SVM :: ", accuracy_score(label2, test_p)
	print "Train precision score of SVM :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of SVM :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of SVM :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of SVM :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start

def xgboost(data,label,data2,label2):
	start = time.time()
	clf = XGBClassifier()
	data_x = np.array(data)
	label_x = np.array(label)
	clf = clf.fit(data_x,label_x)
	joblib.dump(clf, 'xgboost.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of xgboost :: ", accuracy_score(label, train_p)
	print "Test Accuracy of xgboost :: ", accuracy_score(label2, test_p)
	print "Train precision score of xgboost :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of xgboost :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of xgboost :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of xgboost :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start

def adaboost(data,label,data2,label2):
	start = time.time()
	clf = AdaBoostClassifier(n_estimators=100)
	data_x = np.array(data)
	label_x = np.array(label)
	clf = clf.fit(data_x,label_x)
	joblib.dump(clf, 'adaboost.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of adaboost :: ", accuracy_score(label, train_p)
	print "Test Accuracy of adaboost :: ", accuracy_score(label2, test_p)
	print "Train precision score of adaboost :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of adaboost :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of adaboost :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of adaboost :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start

def gradient(data,label,data2,label2):
	start = time.time()
	clf = GradientBoostingClassifier()
	data_x = np.array(data)
	label_x = np.array(label)
	clf = clf.fit(data_x,label_x)
	joblib.dump(clf, 'gradient.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of gradient tree boosting :: ", accuracy_score(label, train_p)
	print "Test Accuracy of gradient tree boosting :: ", accuracy_score(label2, test_p)
	print "Train precision score of gradient tree boosting :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of gradient tree boosting :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of gradient tree boosting :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of gradient tree boosting :: ", recall_score(label2,test_p,average=None).mean()
	end = time.time()
	print "The running time including training and predicting is :: ",end-start



#train models and predict models
def train_model(data,label,data2,label2):
	decisiontree(data,label,data2,label2)
	randomforest(data,label,data2,label2)
	neural(data,label,data2,label2)
	#svm(data,label,data2,label2)
	xgboost(data,label,data2,label2)
	adaboost(data,label,data2,label2)
	gradient(data,label,data2,label2)


#train_model(data,label,data2,label2)


'''
Here the functions are just using the constructed models  ".joblib" stored in the file!!!!
'''
def decisiontree_m(data,label,data2,label2):
	clf = joblib.load('decisiontree.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of decision tree :: ", accuracy_score(label, train_p)
	print "Test Accuracy of decision tree :: ", accuracy_score(label2, test_p)
	print "Train precision score of decision tree :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of decision tree :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of decision tree :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of decision tree :: ", recall_score(label2,test_p,average=None).mean()

def randomdorest_m(data,label,data2,label2):
	clf = joblib.load('randomforest.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of random forest :: ", accuracy_score(label, train_p)
	print "Test Accuracy of random forest :: ", accuracy_score(label2, test_p)
	print "Train precision score of random forest :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of random forest :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of random forest :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of random forest :: ", recall_score(label2,test_p,average=None).mean()

def neural50_m(data,label,data2,label2):
	clf = joblib.load('neural_50.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of neural network with H = 50 :: ", accuracy_score(label, train_p)
	print "Test Accuracy of neural network with H = 50 :: ", accuracy_score(label2, test_p)
	print "Train precision score of neural network with H = 50 :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of neural network with H = 50 :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of neural network with H = 50 :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of neural network with H = 50 :: ", recall_score(label2,test_p,average=None).mean()

def neural_m(data,label,data2,label2):
	clf = joblib.load('neural.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of neural network :: ", accuracy_score(label, train_p)
	print "Test Accuracy of neural network :: ", accuracy_score(label2, test_p)
	print "Train precision score of neural network :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of neural network :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of neural network :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of neural network :: ", recall_score(label2,test_p,average=None).mean()

def svm_m(data,label,data2,label2,file):
	clf = joblib.load(file)
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of SVM :: ", accuracy_score(label, train_p)
	print "Test Accuracy of SVM :: ", accuracy_score(label2, test_p)
	print "Train precision score of SVM :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of SVM :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of SVM :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of SVM :: ", recall_score(label2,test_p,average=None).mean()

def xgboost_m(data,label,data2,label2):
	clf = joblib.load('xgboost.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of xgboost :: ", accuracy_score(label, train_p)
	print "Test Accuracy of xgboost :: ", accuracy_score(label2, test_p)
	print "Train precision score of xgboost :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of xgboost :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of xgboost :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of xgboost :: ", recall_score(label2,test_p,average=None).mean()

def adaboost_m(data,label,data2,label2):
	clf = joblib.load('adaboost.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of adaboost :: ", accuracy_score(label, train_p)
	print "Test Accuracy of adaboost :: ", accuracy_score(label2, test_p)
	print "Train precision score of adaboost :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of adaboost :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of adaboost :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of adaboost :: ", recall_score(label2,test_p,average=None).mean()

def gradient_m(data,label,data2,label2):
	clf = joblib.load('gradient.joblib')
	train_p = clf.predict(data)
	test_p = clf.predict(data2)
	print "Train Accuracy of gradient tree boosting :: ", accuracy_score(label, train_p)
	print "Test Accuracy of gradient tree boosting :: ", accuracy_score(label2, test_p)
	print "Train precision score of gradient tree boosting :: ", precision_score(label,train_p,average=None).mean()
	print "Test precision score of gradient tree boosting :: ", precision_score(label2,test_p,average=None).mean()
	print "Train recall score of gradient tree boosting :: ", recall_score(label,train_p,average=None).mean()
	print "Test recall score of gradient tree boosting :: ", recall_score(label2,test_p,average=None).mean()

def predict_model(data,label,data2,label2):
	decisiontree_m(data,label,data2,label2)
	randomdorest_m(data,label,data2,label2)
	neural50_m(data,label,data2,label2)
	neural_m(data,label,data2,label2)
	#svm_m(data,label,data2,label2)
	xgboost_m(data,label,data2,label2)
	adaboost_m(data,label,data2,label2)
	gradient_m(data,label,data2,label2)

predict_model(data,label,data2,label2)
