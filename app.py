import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

feature_names = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN',
    'bedroom_ratio', 'household_rooms'
]


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/model')
def model_page():
    return render_template("model.html")


@app.route('/home')
def home():
    return render_template("home.html", prediction=None)


@app.route('/confirmation')
def confirmation():
    name = request.args.get('name')
    email = request.args.get('email')
    props = {
        "name": name,
        "email": email
    }
    return render_template("confirmation.html", data=props)


@app.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        form_data = request.form
        data = {
            'longitude': float(form_data['longitude']),
            'latitude': float(form_data['latitude']),
            'housing_median_age': float(form_data['housing_median_age']),
            'total_rooms': np.log(float(form_data['total_rooms']) + 1),
            'total_bedrooms': np.log(float(form_data['total_bedrooms']) + 1),
            'population': np.log(float(form_data['population']) + 1),
            'households': np.log(float(form_data['households']) + 1),
            'median_income': float(form_data['median_income']),
            'ocean_proximity': form_data['ocean_proximity']
        }

        ocean_proximity_dummies = pd.get_dummies([data['ocean_proximity']]).reindex(
            columns=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], fill_value=0)
        data.update(ocean_proximity_dummies.iloc[0].to_dict())
        del data['ocean_proximity']

        input_df = pd.DataFrame([data])

        input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms']
        input_df['household_rooms'] = input_df['total_rooms'] / input_df['households']

        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        inflation_factor = 2.2391
        prediction = model.predict(input_df)[0]
        prediction = round(prediction * inflation_factor, 2)

    return render_template("home.html", prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
