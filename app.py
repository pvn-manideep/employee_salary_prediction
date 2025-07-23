from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model/salary_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'age': int(request.form['age']),
        'workclass': request.form['workclass'],
        'fnlwgt': int(request.form['fnlwgt']),
        'education-num': int(request.form['education_num']),
        'marital-status': request.form['marital_status'],
        'occupation': request.form['occupation'],
        'relationship': request.form['relationship'],
        'race': request.form['race'],
        'gender': request.form['gender'],
        'capital-gain': int(request.form['capital_gain']),
        'capital-loss': int(request.form['capital_loss']),
        'hours-per-week': int(request.form['hours_per_week']),
        'native-country': request.form['native_country']
    }
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)
    model_columns = joblib.load("model/model_columns.pkl")
    for col in model_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_columns]
    prediction = model.predict(df_encoded)[0]
    return render_template('index.html', prediction_text=f'Predicted Income Class: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
