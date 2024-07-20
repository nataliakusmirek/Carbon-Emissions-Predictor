import pandas as pd
import numpy as np
import io
import os
import base64
from flask import Flask, render_template
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor

# Load the dataset and preprocess it
full_data = pd.read_csv('data.csv')

# Define features and target
features = ['year', 'parent_entity', 'parent_type', 'commodity',
            'product_emissions_MtCO2', 'flaring_emissions_MtCO2',
            'venting_emissions_MtCO2', 'total_operational_emissions_MtCO2e']
target = 'total_emissions_MtCO2e'

numeric_features = ['product_emissions_MtCO2', 'flaring_emissions_MtCO2',
                     'venting_emissions_MtCO2', 'total_operational_emissions_MtCO2e']
categorical_features = ['year', 'parent_entity', 'parent_type', 'commodity']

# Log transform skewed features
for feature in numeric_features:
    full_data[feature] = np.log1p(full_data[feature])

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare data for model training
X = full_data[features]
y = full_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess features
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    """
    Evaluate the effectiveness of different Random Forest Regressor parameters to find the most optimal model features for the project.

    Args:
        n_estimators (int): 
        max_depth (int):
        min_samples_split (int):
        min_samples_leaf (int):
        max_features (int):

    Returns:
        mae : The mean absolute error of the selected parameters
    """

    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'max_features': max_features,
        'random_state': 42
    }
    rf = RandomForestRegressor(**params)
    rf.fit(X_train_preprocessed, y_train)
    y_pred = rf.predict(X_test_preprocessed)
    mae = mean_absolute_error(y_test, y_pred)
    return -mae

# Define the parameter bounds
param_bounds = {
    'n_estimators': (50, 300),
    'max_depth': (3, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'max_features': (0.5, 1.0)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=rf_evaluate,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=150)

# Extract the best parameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

# Train the final model
best_model = RandomForestRegressor(**best_params, random_state=42)
best_model.fit(X_train_preprocessed, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test_preprocessed)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Model Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

def preprocess_input(input_data):
    """
    Process input data to prepare it for prediction by the model.

    Args:
        input_data (pd.DataFrame or np.ndarray): The raw input data that needs to be preprocessed.

    Returns:
        input_data_preprocessed (np.ndarray): The preprocessed input data, ready for prediction.

    Raises:
        ValueError: If the preprocessed input data does not match the expected input shape required by the model.
    """
    input_data_preprocessed = preprocessor.transform(input_data)
    expected_input_shape = best_model.n_features_in_
    if input_data_preprocessed.shape[1] != expected_input_shape:
        raise ValueError(f"Expected input shape {expected_input_shape}, but got {input_data_preprocessed.shape[1]}")
    return input_data_preprocessed

def make_prediction(year):
    """
    Predict the total carbon emissions of a specified year using the pre-trained Random Forest Regressor model and historical data.

    Args:
        year (int): The user-inputted year to predict total carbon emissions for.

    Returns:
        float: The model predicted total carbon emissions value.

    Raises:
        ValueError: If the year is greater than 2024.
    """
    if year > 2024:
        raise ValueError("Year must be up to and including 2024")
    
    features = ['year', 'parent_entity', 'parent_type', 'commodity',
                'product_emissions_MtCO2', 'flaring_emissions_MtCO2',
                'venting_emissions_MtCO2', 'total_operational_emissions_MtCO2e']
    
    full_data = pd.read_csv('data.csv')
    
    # Check if the year is within the range of historical data
    if year in full_data['year'].values:
        # Retrieve historical data for the given year
        historical_data = full_data[full_data['year'] == year].copy()
    else:
        # Handle the case where the year is in the future
        most_recent_year = full_data['year'].max()
        historical_data = full_data[full_data['year'] == most_recent_year].copy()
    
    # If there's no data for the most recent year, use default values
    if historical_data.empty:
        historical_data = pd.DataFrame({
            'year': [year],
            'parent_entity': ['unknown'],
            'parent_type': ['unknown'],
            'commodity': ['unknown'],
            'product_emissions_MtCO2': [0.0],
            'flaring_emissions_MtCO2': [0.0],
            'venting_emissions_MtCO2': [0.0],
            'total_operational_emissions_MtCO2e': [0.0]
        })
    
    # Fill missing columns with mean values
    for feature in ['product_emissions_MtCO2', 'flaring_emissions_MtCO2', 'venting_emissions_MtCO2', 'total_operational_emissions_MtCO2e']:
        if historical_data[feature].iloc[0] == 0.0:
            historical_data.loc[:, feature] = full_data[feature].mean()
    
    # Ensure the 'year' column is set correctly
    historical_data.loc[:, 'year'] = year
    historical_data = historical_data[features]
    input_data_preprocessed = preprocess_input(historical_data)
    
    # Make the prediction
    prediction = best_model.predict(input_data_preprocessed)
    
    if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
        return prediction[0][0]  # For a 2D array
    else:
        return prediction[0]

def plot_to_base64(fig):
    """
    Transform plot image to base64 for user output on prediction HTML page.

    Args:
        fig (matplotlib.figure.Figure): The image to be converted to base64.

    Returns:
        str: The base64 string for visualizing the plot.
    """
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

def emissions_comparison_plot(data, year, prediction):
    """
    Plot the model predicted value versus the actual value for webpage output.

    Args:
        data (pd.DataFrame): DataFrame containing historical data.
        year (int): The user-inputted year to predict total carbon emissions for.
        prediction (float): The predicted AI model's value for total carbon emissions of the specified year.

    Returns:
        tuple: A tuple containing:
            - str: The base64 version of the plot for output on the prediction webpage.
            - str: The path to the saved plot.
    """
    plot_dir = 'static/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_path = os.path.join(plot_dir, 'emissions_comparison_plot.png')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter data for the specified year
    year_data = data[data['year'] == year]
    
    actual_emission = year_data['total_emissions_MtCO2e'].values[0] if not year_data.empty else 0
    ax.axhline(y=actual_emission, color='b', linestyle='--', label=f'Actual Emissions for {year}')
    ax.axhline(y=prediction, color='g', linestyle='--', label='Predicted Emissions')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Emissions (MtCO2e)')
    ax.set_title(f'Actual vs. Predicted Total Emissions for {year}')
    ax.set_xlim(year - 1, year + 1)
    
    ax.legend()
    
    fig.savefig(plot_path)
    plt.close(fig)
    
    # Convert the plot to a base64 string
    plot_url = plot_to_base64(fig)
    
    return plot_url, plot_path

def calculate_prediction_accuracy(prediction, actual_value):
    """
    Calculate the accuracy percentage of the AI prediction compared to the actual value.
    
    Args:
        prediction (float): The AI predicted value.
        actual_value (float): The actual value.

    Returns:
        float: The accuracy percentage.
    """
    if actual_value == 0:
        return 0.0  # Avoid division by zero

    accuracy = (1 - abs(prediction - actual_value) / abs(actual_value)) * 100
    return accuracy

def handle_user_input(year, data):
    """
    Take user specified year for prediction and carry out steps for model to predict total carbon emissions for the specified year.
    
    Args:
        year (int): The user-inputted year to predict total carbon emissions for.
        data (pd.DataFrame): DataFrame containing historical data.

    Returns:
        tuple: A tuple containing:
            - float: The predicted value of total carbon emissions by the model for the specified year.
            - str or None: An error message if an exception occurs, otherwise None.
            - str: The base64 version of the plot for output on the prediction webpage.
            - float: The Mean Squared Error of the model prediction.
            - float: The Mean Absolute Error of the model prediction.
            - float: The R-Squared Error of the model prediction.
            - float: The Root Mean Squared Error of the model prediction.
            - float: The accuracy of the model prediction compared to the actual value in the historical data, output as a percentage.

    Raises:
        ValueError: If the year is greater than 2024.
    """
    try:
        if year > 2024:
            raise ValueError("Year must be up to and including 2024")

        prediction = make_prediction(year)

        actual_value = data.loc[data['year'] == year, 'total_emissions_MtCO2e'].values[0]
        accuracy = calculate_prediction_accuracy(prediction, actual_value)

        plot_url, plot_path = emissions_comparison_plot(data, year, prediction)
        X_data_preprocessed = preprocess_input(data[features])
        
        # Evaluate model metrics
        y_pred = best_model.predict(X_data_preprocessed)
        mse = mean_squared_error(data[target], y_pred)
        mae = mean_absolute_error(data[target], y_pred)
        r2 = r2_score(data[target], y_pred)
        rmse = np.sqrt(mse)

        return prediction, None, plot_url, mse, mae, r2, rmse, accuracy

    except ValueError as e:
        return None, str(e), None, None, None, None, None, None



# Initialize the Flask app for webpage
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    return handle_user_input()

if __name__ == '__main__':
    app.run(debug=True)
