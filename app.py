from flask import Flask,request,render_template

import pandas
import numpy

from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import CustomerData,PredictData
application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    if request.method=='GET':
     return render_template('home.html')
    else:
       data = CustomerData(
        CreditScore=int(request.form.get('credit-score')),
        Age=int(request.form.get('age')),
        Tenure=int(request.form.get('tenure')),
        Balance=float(request.form.get('balance')),
        NumOfProducts=int(request.form.get('num-of-products')),
        HasCrCard=int(request.form.get('has-cr-card')),
        IsActiveMember=int(request.form.get('is-active-member')),  # Corrected the form field name
        EstimatedSalary=float(request.form.get('estimated-salary')),
        Gender=request.form.get('gender'),
        Geography=request.form.get('geography')
            )

       pred_df=data.get_data_as_dataframe()
       print(pred_df)
       predict_pipeline=PredictData()
       results=predict_pipeline.predict(pred_df)
       
       return render_template('home.html',result=results[0])
    

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)