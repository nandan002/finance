import pickle

import pandas as pd
from flask import Flask, render_template, request, make_response

app = Flask(__name__)


@app.route("/")
def root():
    return render_template('website.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['filename']
    file_name = file.filename.split('.')[0]
    data = pd.read_excel(file, engine='openpyxl')
    data = data.loc[(data['status'] == 'Disbursed') | (data['status'] == 'Rejected')]
    new_dataframe = data[
        ['gender', 'occupation', 'loan_amount', 'monthly_income', 'cibil_score', 'product_amount', 'gst_amount',
         'emi_amount', 'processing_fees_percentage']]
    new_dataframe = new_dataframe.loc[:, ~new_dataframe.T.duplicated(keep='first')]

    new_dataframe = pd.get_dummies(new_dataframe, columns=['gender', 'occupation'])
    model = pickle.load(open('predictor.pkl', 'rb'))

    prediction = model.predict(new_dataframe)

    data['Prediction'] = prediction

    resp = make_response(data.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=" + str(file_name) + "_export.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5050)
