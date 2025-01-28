# Weather Prediction Project

This project fetches weather data from the OpenWeatherMap API and performs time-series prediction using a linear regression model.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/LTeinturier/weather_app.git
    cd weather_prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add your API key:
    ```
    API_KEY=your_api_key
    ```

4. Run the application:
    open a terminal and run the following :

    ```
    python run.py
    ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Project Structure

- `app/`: Contains the Flask application code.
- `config/`: Configuration settings.
- `static/`: Static files like CSS.
- `templates/`: HTML templates.
- `tests/`: Unit tests.
- `.env`: Environment variables.
- `requirements.txt`: List of dependencies.
- `run.py`: Entry point to run the application.
- `README.md`: Project documentation.

NB: It seems like the API call changed. (free plan changed...)