from django.shortcuts import render
import tensorflow as tf
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os
import numpy as np
from .predict import handle_user_input
from django.http import HttpResponse

# Load the model
model_path = 'myapp/carbon_model.h5'
model = load_model(model_path)

# Define preprocessing function
def preprocess_input(input_df):
    numeric_features = ['product_emissions_MtCO2', 'flaring_emissions_MtCO2', 
                         'venting_emissions_MtCO2', 'total_operational_emissions_MtCO2e']
    categorical_features = ['year', 'parent_entity', 'parent_type', 'commodity']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor.fit_transform(input_df)

# View for displaying the nav
def index(request):
    return render(request, 'index.html')

# View for displaying the home page
def home(request):
    return render(request, 'home.html')


# View for displaying the prediction plot
def display_plot(request):
    plot_path = 'myapp/static/plots/emissions_comparison_plot.png'  # Ensure this path is correct
    if not os.path.exists(plot_path):
        return HttpResponse("Plot not found.", status=404)
    
    with open(plot_path, 'rb') as f:
        return HttpResponse(f.read(), content_type='image/png')
    

# View for displaying historical data on homepage
def historical_data(request):
    return render(request, 'historical.html')

def make_prediction(request):
    if request.method == 'POST':
        year = int(request.POST.get('year'))
        
        # Load your data
        data = pd.read_csv('data.csv')
        
        try:
            prediction, prediction_error, emissions_comparison_plot_url, mse, mae, r2, rmse = handle_user_input(year, data)
            return render(request, 'predict.html', {
                'predicted_emissions': prediction,
                'prediction_error': prediction_error,  # Include this in the context
                'emissions_comparison_plot_url': emissions_comparison_plot_url,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse            })
        except ValueError as e:
            return render(request, 'predict.html', {'error': str(e)})

    return render(request, 'predict.html')