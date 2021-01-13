
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('simpleRegressor.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        age = float(request.form['age'])
        sex=float(request.form['sex'])
        bmi=float(request.form['bmi'])

        bp=float(request.form['bp'])
        s1=float(request.form['s1'])
        s2=float(request.form['s2'])
        s3=float(request.form['s3'])
        s4=float(request.form['s4'])
        s5=float(request.form['s5'])
        s6=float(request.form['s6'])
        X=pd.DataFrame([[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]],
        columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6'],dtype=float)
        prediction=model.predict(X)
        output=round(prediction[0])

        if output<0:
            return render_template('index.html',prediction_texts="Sorry please enter proper values")
        elif output<200:
            return render_template('index.html',prediction_text="Your Diabetes level {}.It is better to show to Doctor ".format(output))

        else:
            return render_template('index.html',prediction_text="Your Diabetes level {}.Maintain same way ".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
