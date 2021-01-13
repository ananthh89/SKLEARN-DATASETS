
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('XBRegressor.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        LSTAT = float(request.form['LSTAT'])
        RM=float(request.form['RM'])
        TAX=float(request.form['TAX'])
        
        AGE=float(request.form['AGE'])
        CRIM=float(request.form['CRIM'])
        ZN=11.36
        INDUS=11.36
        CHAS=1
        NOX=0.45
        DIS=5
        RAD=4
        PTRATIO=13.40
        B=100
      
        prediction=model.predict([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this house")
        else:
            return render_template('index.html',prediction_text="You Can Sell The House at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)