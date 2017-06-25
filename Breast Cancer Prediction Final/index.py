from flask import Flask,render_template,request
from predict import performance,getType,applyML,getResult,getAlgo
from predict import is_float
import win32api

app=Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html",meanSVM=-1,seSVM=-1,worstSVM=-1,meanKNN=-1,seKNN=-1,worstKNN=-1,meanRF=-1,seRF=-1,worstRF=-1)

@app.route('/performance')
def perform():
	meanSVM,seSVM,worstSVM=performance("svm")
	meanKNN,seKNN,worstKNN=performance("knn")
	meanRF,seRF,worstRF=performance("rf")
	return render_template("index.html",meanSVM=meanSVM,seSVM=seSVM,worstSVM=worstSVM,meanKNN=meanKNN,seKNN=seKNN,worstKNN=worstKNN,meanRF=meanRF,seRF=seRF,worstRF=worstRF)

@app.route('/meanPrediction',methods=['POST'])
def meanPrediction():
	algoName=request.form['algoName']
	typeName=request.form['typeName']
	radius=request.form['radius']
	texture=request.form['texture']
	perimeter=request.form['perimeter']
	area=request.form['area']
	smoothness=request.form['smoothness']
	compactness=request.form['compactness']
	concavity=request.form['concavity']
	concave_points=request.form['concave_points']
	symmetry=request.form['symmetry']
	fractal_dimension=request.form['fractal_dimension']
	result=applyML(algoName,typeName,float(radius),float(texture),float(perimeter),float(area),float(smoothness),float(compactness),float(concavity),float(concave_points),float(symmetry),float(fractal_dimension))
	result=getResult(result)
	algoName=getAlgo(algoName)
	typeName=getType(typeName)
	res='\tALGORITHM\t:'+algoName+'\n\n\tTYPE\t\t:'+typeName+'\n\n\tCANCER\t\t:'+result
	win32api.MessageBox(0,res,"RESULT", 0x00001000) 
	return render_template("index.html",meanSVM=-1,seSVM=-1,worstSVM=-1,meanKNN=-1,seKNN=-1,worstKNN=-1,meanRF=-1,seRF=-1,worstRF=-1)

if __name__=="__main__":
	app.run(debug=True)