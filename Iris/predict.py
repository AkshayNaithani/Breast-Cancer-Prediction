import sys
import csv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

def getAlgo(temp):
	if temp=='rf':
		return "RANDOM FOREST"
	elif temp=='svm':
		return "SUPPORT VECTOR MACHINE"
	elif temp=='nb':
		return "NAIVE NAYES"
	else:
		return "K NEAREST NEIGHBOURS"

def getSpecies(temp):
	if temp=='setosa':
		return "SETOSA"
	elif temp=='versicolor':
		return "VERSICOLOR"
	else:
		return "VIRGINICA"


def performSVM(X_train,X_test,y_train,y_test):
	
	clf=svm.SVC()
	predict=clf.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performKNN(X_train,X_test,y_train,y_test):
	
	neigh=KNeighborsClassifier(n_neighbors=3)
	predict=neigh.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performRandomForest(X_train,X_test,y_train,y_test):
	
	clf = RandomForestClassifier(n_estimators=10)
	predict=clf.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performNB(X_train,X_test,y_train,y_test):

	gnb = GaussianNB()
	predict = gnb.fit(X_train,y_train).predict(X_test)
	clf = RandomForestClassifier(n_estimators=10)	
	return accuracy_score(y_test,predict)*100


def performance():

	feature=[]	
	subfeature=[]
	target=[]
	
	path=os.path.join('input',"Dataset.csv")
	files=glob.glob(path)

	for f1 in files:
		with open(f1,'r') as f:
			reader=csv.reader(f)
			for row in reader:
				subfeature=[float(row[0]),float(row[1]),float(row[2]),float(row[3])]
				feature.append(subfeature)
				target.append(row[4])
				subfeature=[]

	X_train,X_test,y_train,y_test=train_test_split(feature,target, test_size=0.33, random_state=42)
	svmResult=performSVM(X_train,X_test,y_train,y_test)
	knnResult=performKNN(X_train,X_test,y_train,y_test)
	rfResult=performRandomForest(X_train,X_test,y_train,y_test)
	nbResult=performNB(X_train,X_test,y_train,y_test)
	
	return svmResult,knnResult,rfResult,nbResult

def onlySVM(feature,target,test):
		
	return svm.SVC().fit(feature,target).predict(test)

def onlyKNN(feature,target,test):
	
	return KNeighborsClassifier(n_neighbors=3).fit(feature,target).predict(test)
	
def onlyRandomForest(feature,target,test):
	
	return RandomForestClassifier(n_estimators=10).fit(feature,target).predict(test)

def onlyNB(feature,target,test):
	
	return GaussianNB().fit(feature,target).predict(test)

def applyML(algoName,sepalLength,sepalWidth,petalLength,petalWidth):
	
	
	featureA=[]
	subfeatureA=[]
	output=[]
	
	path=os.path.join('input',"Dataset.csv")
	files=glob.glob(path)

	for f1 in files:
		with open(f1,'r') as f:
			reader=csv.reader(f)
			for row in reader:
				subfeatureA=[float(row[0]),float(row[1]),float(row[2]),float(row[3])]
				featureA.append(subfeatureA)
				output.append(row[4])
				subfeatureA=[]

	unicode_featureA = [i.decode('UTF-8') if isinstance(i, basestring) else i for i in featureA]
	test=[float(sepalLength),float(sepalWidth),float(petalLength),float(petalWidth)]
	unicode_test = [i.decode('UTF-8') if isinstance(i, basestring) else i for i in test]

	temp=""
	if algoName=="svm":
		temp=onlySVM(unicode_featureA,output,unicode_test)
	elif algoName=="knn":
		temp=onlyKNN(unicode_featureA,output,unicode_test)
	elif algoName=="rf":
		temp=onlyRandomForest(unicode_featureA,output,unicode_test)
	else:
		temp=onlyNB(unicode_featureA,output,unicode_test)
	return temp
