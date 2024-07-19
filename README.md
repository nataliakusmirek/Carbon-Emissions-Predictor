# Carbon Emissions Predictor

This project aims to predict carbon emissions using a machine learning model. I used historical data from the Carbon Majors Emissions dataset available on Kaggle. The main goal was to create an accurate predictive model and evaluate its performance.

## Dataset

The dataset used in this project can be found on Kaggle: [Carbon Majors Emissions Data](https://www.kaggle.com/datasets/joebeachcapital/carbon-majors-emissions-data).

## Project Structure

- **data/**: Directory containing the dataset.
- **static/**: Directory containing static files, including plots.
- **templates/**: Directory containing HTML templates.
- **myproject/**: Main project directory containing Django settings and configurations.
  - **myapp/**: Django application containing views, models, and other app-specific files.
    - **migrations/**: Directory containing database migrations.
    - **__init__.py**: Initialization file for the app.
    - **admin.py**: Admin configurations.
    - **apps.py**: App configurations.
    - **models.py**: Database models.
    - **views.py**: Views for handling HTTP requests.
    - **predict.py**: Script containing the model prediction and evaluation logic.
    - **urls.py**: URL routing for the app.

## Techniques Used

1. **Data Preprocessing**:
   - Handling missing values.
   - Feature scaling and normalization.
   - Encoding categorical variables.

2. **Modeling**:
   - Random Forest Regressor.
   - Bayesian Optimization for hyperparameter tuning.

3. **Model Evaluation**:
   - Mean Absolute Error (MAE).
   - Mean Squared Error (MSE).
   - R-squared (R²).
   - Root Mean Squared Error (RMSE).

## What I Learned

- **Data Preprocessing**: Learned how to handle missing values, scale features, and encode categorical variables.
- **Model Training**: Gained experience in training a Random Forest Regressor and optimizing hyperparameters using Bayesian Optimization.
- **Model Evaluation**: Understood various evaluation metrics and their importance in assessing model performance.
- **Django Integration**: Integrated the machine learning model into a Django web application for real-time predictions.

## Model Accuracy

The initial Mean Absolute Error (MAE) of the model was 3.43. After optimization and improvements, the MAE was reduced to 0.57. This significant improvement demonstrates the model's effectiveness in predicting carbon emissions.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning algorithms and model evaluation.
- **matplotlib**: For creating plots and visualizations.
- **bayesian-optimization**: For hyperparameter tuning using Bayesian Optimization.
- **Django**: For building the web application.
- **joblib**: For saving and loading machine learning models.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/carbon-emissions-prediction.git
   cd carbon-emissions-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Django server**:
   ```bash
   python manage.py runserver
   ```

5. **Access the application**:
   Open your web browser and navigate to `http://localhost:8000`.

## Usage

1. **Input Year**: Enter the year for which you want to predict carbon emissions.
2. **View Prediction**: The application will display the predicted emissions and provide a comparison plot of actual vs. predicted values.
3. **Evaluate Model**: The application shows model evaluation metrics including MAE, MSE, R², and RMSE.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
