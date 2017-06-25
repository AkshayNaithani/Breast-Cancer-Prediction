from flask import Flask,render_template,request
from predict import applyML,getAlgo,performance,getSpecies
import win32api

app=Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/performance')
def perform():
	SVM,KNN,RF,NB=performance()
	return render_template("index.html",flagGraph="True",SVM=SVM,KNN=KNN,RF=RF,NB=NB)

@app.route('/Prediction',methods=['POST'])
def Prediction():
	algoName=request.form['algoName']
	sepalLength=request.form['sepalLength']
	sepalWidth=request.form['sepalWidth']
	petalLength=request.form['petalLength']
	petalWidth=request.form['petalWidth']
	
	result=applyML(algoName,sepalLength,sepalWidth,petalLength,petalWidth)
	MESSAGE="\tALGORITHM\t:"+getAlgo(algoName)+"\n\tSEPAL LENGTH\t:"+str(sepalLength)+"\n\tSEPAL WIDTH\t:"+str(sepalWidth)+"\n\tPETAL LENGTH\t:"+str(petalLength)+"\n\tPETAL WIDTH\t:"+str(petalWidth)+"\n\tSPECIES\t\t:"+getSpecies(result)
	win32api.MessageBox(0,MESSAGE,"RESULT", 0x00001000) 
	return render_template("index.html",flagGraph="False",SVM=0,KNN=0,RF=0,NB=0)

if __name__=="__main__":
	app.run(debug=True)