import re
from flask import Flask, render_template, request, Response,make_response
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
async def root():
    return render_template('website.html')

@app.route('/predict', methods=['POST'])
def predict():
    file=request.files['filename']
    file_name= file.filename.split('.')[0]
    data= pd.read_excel(file,engine = 'openpyxl')
    data= data.loc[(data['status']=='Disbursed') | (data['status']=='Rejected')]
    new_dataframe = data[['gender','occupation','loan_amount','monthly_income','cibil_score','product_amount','gst_amount','emi_amount']]
    new_dataframe = new_dataframe.loc[:,~new_dataframe.T.duplicated(keep='first')]
    num_cols = new_dataframe._get_numeric_data().columns
    new_dataframe_array=np.array(new_dataframe[num_cols])

    one_hot = pickle.load(open('one_hot.pkl','rb'))

    
    model = pickle.load(open('predictor.pkl','rb'))

    categorical_values = one_hot.transform(new_dataframe[['occupation','gender']]).toarray()

    final_data = np.column_stack((new_dataframe_array,categorical_values))

    prediction = model.predict(final_data)

    data['Prediction']=prediction


    resp = make_response(data.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename="+str(file_name)+"_export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

if __name__ == "__main__":
    app.run(debug=True, threaded=True,host='0.0.0.0',port=5050)