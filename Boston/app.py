
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
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
        ZN=float(request.form['ZN'])
        INDUS=float(request.form['INDUS'])
        CHAS=float(request.form['CHAS'])
        NOX=float(request.form['NOX'])
        DIS=float(request.form['DIS'])
        RAD=float(request.form['RAD'])
        PTRATIO=float(request.form['PTRATIO'])
        B=float(request.form['B'])
        X=pd.DataFrame([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]],
        columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'],dtype=float)
        prediction=model.predict(X)
        output=round(prediction[0])

        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this house")
        else:
            return render_template('index.html',prediction_text="You Can Sell The House at {} Thousand Dollars".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
