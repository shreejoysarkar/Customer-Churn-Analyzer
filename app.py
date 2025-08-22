from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomeData
applicaiton = Flask(__name__)

app= applicaiton

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
       return render_template('home.html')
    
    else:
        data = CustomeData(
            gender = request.form.get("gender"),
            SeniorCitizen= int(request.form.get("SeniorCitizen")),
            Partner= request.form.get("Partner"),
            Dependents= request.form.get("Dependents"),
            tenure= int(request.form.get("tenure")),
            PhoneService= request.form.get("PhoneService"),
            MultipleLines= request.form.get("MultipleLines"),
            InternetService= request.form.get("InternetService"),
            OnlineSecurity= request.form.get("OnlineSecurity"),
            OnlineBackup= request.form.get("OnlineBackup"),
            DeviceProtection= request.form.get("DeviceProtection"),
            TechSupport= request.form.get("TechSupport"),
            StreamingTV= request.form.get("StreamingTV"),
            StreamingMovies= request.form.get("StreamingMovies"),
            Contract= request.form.get("Contract"),
            PaperlessBilling= request.form.get("PaperlessBilling"),
            PaymentMethod= request.form.get("PaymentMethod"),
            MonthlyCharges= float(request.form.get("MonthlyCharges")),
            TotalCharges= float(request.form.get("TotalCharges")),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictionPipeline()
        results =predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__== "__main__":
    app.run(host = "0.0.0.0",debug=True)