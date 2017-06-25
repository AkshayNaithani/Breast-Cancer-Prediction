import sys
import csv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def getType(temp):
	if temp=='mean':
		return "MEAN TYPE"
	elif temp=='se':
		return "STANDARD ERROR TYPE"
	else:
		return "WORST TYPE"

def getResult(temp):
	if temp=='B':
		return "BENIGN"
	return "MALIGNANT"

def getAlgo(temp):
	if temp=='rf':
		return "RANDOM FOREST"
	elif temp=='svm':
		return "SUPPORT VECTOR MACHINE"
	else:
		return "K NEAREST NEIGHBOURS"

def performSVM(X_train,X_test,y_train,y_test):
		
	np.predict=[]
	clf=svm.SVC()
	predict=clf.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performKNN(X_train,X_test,y_train,y_test):
		
	np.predict=[]
	neigh=KNeighborsClassifier(n_neighbors=3)
	predict=neigh.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performRandomForest(X_train,X_test,y_train,y_test):
		
	np.predict=[]
	clf = RandomForestClassifier(n_estimators=10)
	predict=clf.fit(X_train,y_train).predict(X_test)
	return accuracy_score(y_test,predict)*100

def performance(status):

	np.feature1=[]
	np.feature2=[]
	np.feature3=[]
	
	np.subfeature1=[]
	np.subfeature2=[]
	np.subfeature3=[]
	np.target=[]
	
	path=os.path.join('input',"Dataset.csv")
	files=glob.glob(path)

	for f1 in files:
		with open(f1,'r') as f:
			reader=csv.reader(f)
			for row in reader:
				#"Mean"
				np.subfeature1=[float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11])]
				np.feature1.append(np.subfeature1)
				#"StandardError"
				np.subfeature2=[float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20]),float(row[21])]
				np.feature2.append(np.subfeature2)
				#"Worst"
				np.subfeature3=[float(row[22]),float(row[23]),float(row[24]),float(row[25]),float(row[26]),float(row[27]),float(row[28]),float(row[29]),float(row[30]),float(row[31])]
				np.feature3.append(np.subfeature3)
				
				np.target.append(row[1])
				
				np.subfeature1=[]
				np.subfeature2=[]
				np.subfeature3=[]

	np.X_train1,np.X_test1,np.y_train1,np.y_test1 = train_test_split(np.feature1,np.target, test_size=0.33, random_state=42)
	np.X_train2,np.X_test2,np.y_train2,np.y_test2 = train_test_split(np.feature2,np.target, test_size=0.33, random_state=42)
	np.X_train3,np.X_test3,np.y_train3,np.y_test3 = train_test_split(np.feature3,np.target, test_size=0.33, random_state=42)
	
	if status=="svm":
		return performSVM(np.X_train1,np.X_test1,np.y_train1,np.y_test1),performSVM(np.X_train2,np.X_test2,np.y_train2,np.y_test2),performSVM(np.X_train3,np.X_test3,np.y_train3,np.y_test3) 
	elif status=="knn":
		return performKNN(np.X_train1,np.X_test1,np.y_train1,np.y_test1),performKNN(np.X_train2,np.X_test2,np.y_train2,np.y_test2),performKNN(np.X_train3,np.X_test3,np.y_train3,np.y_test3)
	else:
		return performRandomForest(np.X_train1,np.X_test1,np.y_train1,np.y_test1),performRandomForest(np.X_train2,np.X_test2,np.y_train2,np.y_test2),performRandomForest(np.X_train3,np.X_test3,np.y_train3,np.y_test3)

def onlySVM(feature,target,test):
		
	return svm.SVC().fit(feature,target).predict(test)

def onlyKNN(feature,target,test):
	
	return KNeighborsClassifier(n_neighbors=3).fit(feature,target).predict(test)
	
def onlyRandomForest(feature,target,test):
	
	return RandomForestClassifier(n_estimators=10).fit(feature,target).predict(test)


def applyML(algoName,typeName,radius,texture,perimeter,area,smoothness,compactness,concavity,concave_points,symmetry,fractal_dimension):

	
	np.featureA=[]
	np.featureB=[]
	np.featureC=[]
	
	np.subfeatureA=[]
	np.subfeatureB=[]
	np.subfeatureC=[]
	np.output=[]
	
	path=os.path.join('input',"Dataset.csv")
	files=glob.glob(path)

	for f1 in files:
		with open(f1,'r') as f:
			reader=csv.reader(f)
			for row in reader:
				#"Mean"
				np.subfeatureA=[float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8]),float(row[9]),float(row[10]),float(row[11])]
				np.featureA.append(np.subfeatureA)
				#"StandardError"
				np.subfeatureB=[float(row[12]),float(row[13]),float(row[14]),float(row[15]),float(row[16]),float(row[17]),float(row[18]),float(row[19]),float(row[20]),float(row[21])]
				np.featureB.append(np.subfeatureB)
				#"Worst"
				np.subfeatureC=[float(row[22]),float(row[23]),float(row[24]),float(row[25]),float(row[26]),float(row[27]),float(row[28]),float(row[29]),float(row[30]),float(row[31])]
				np.featureC.append(np.subfeatureC)
				
				np.output.append(row[1])
				np.subfeatureA=[]
				np.subfeatureB=[]
				np.subfeatureC=[]

	
	np.test=[radius,texture,perimeter,area,smoothness,compactness,concavity,concave_points,symmetry,fractal_dimension]
	
	if algoName=="svm":
		if typeName=="mean":
			return onlySVM(np.featureA,np.output,np.test)
		elif typeName=="se":
			return onlySVM(np.featureB,np.output,np.test) 
		else:
			return onlySVM(np.featureC,np.output,np.test)
	elif algoName=="knn":
		if typeName=="mean":
			return onlyKNN(np.featureA,np.output,np.test)
		elif typeName=="se":
			return onlyKNN(np.featureB,np.output,np.test) 
		else:
			return onlyKNN(np.featureC,np.output,np.test)
	else:
		if typeName=="mean":
			return onlyRandomForest(np.featureA,np.output,np.test)
		elif typeName=="se":
			return onlyRandomForest(np.featureB,np.output,np.test) 
		else:
			return onlyRandomForest(np.featureC,np.output,np.test)
