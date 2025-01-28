from flask import render_template, request, current_app
from . import weather_service, prediction_service
from .weather_service import fetch_weather_data, process_weather_data
from .prediction_service import create_lag_features, train_model, plot_results

@current_app.route('/')
def index():
    return render_template('index.html')

@current_app.route('/predict', methods=['POST'])
def predict():
    data = fetch_weather_data(current_app.config['API_KEY'], current_app.config['CITY'], current_app.config['BASE_URL'])
    df = process_weather_data(data)
    df = create_lag_features(df, lag=3)
    df.dropna(inplace=True)

    target = 'temperature'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test, model, y_pred, mse, plot_url = train_model(X, y, target)

    return render_template('result.html', plot_url=plot_url, mse=mse)
