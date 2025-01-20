import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load dataset and train model
def train_model():
    # Load dataset
    
    data = pd.read_csv('flight_ticket_price_prediction_dataset.csv')

    # Separate features and target variable
    X = data.drop(columns=['TicketPrice', 'FlightID', 'DepartureDate'])
    y = data['TicketPrice']

    # Define categorical and numerical columns
    categorical_features = ['Airline', 'DepartureAirport', 'ArrivalAirport', 'Class', 'DayOfWeek']
    numeric_features = ['Distance', 'FlightDuration', 'NumberOfStops', 'DaysUntilDeparture']

    # Preprocessor and pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train model
    pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline, 'flight_price_model_svm.pkl')

    # Evaluate model performance
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print model performance metrics to console
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Calculate custom accuracy (±10%)
    accuracy = 100 * (1 - (mae / y.mean()))  # assuming y.mean() is the average ticket price
    print(f"Custom Accuracy (±10%): {accuracy:.2f}%\n")

# Predict ticket price
def predict_ticket_price(input_data):
    # Load the trained model
    model = joblib.load('flight_price_model_svm.pkl')

    # Predict ticket price
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('ml.html')

@app.route('/predict', methods=['POST'])
def predict():
    airline = request.form.get("Airline")
    departure_airport = request.form.get("DepartureAirport")
    arrival_airport = request.form.get("ArrivalAirport")
    distance = float(request.form.get("Distance"))
    flight_duration = float(request.form.get("FlightDuration"))
    flight_class = request.form.get("Class")
    num_stops = int(request.form.get("NumberOfStops"))
    days_until_departure = int(request.form.get("DaysUntilDeparture"))
    day_of_week = request.form.get("DayOfWeek")

    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([{
        'Airline': airline,
        'DepartureAirport': departure_airport,
        'ArrivalAirport': arrival_airport,
        'Distance': distance,
        'FlightDuration': flight_duration,
        'Class': flight_class,
        'NumberOfStops': num_stops,
        'DaysUntilDeparture': days_until_departure,
        'DayOfWeek': day_of_week
    }])

    # Make prediction
    predicted_price = predict_ticket_price(input_data)

    return jsonify({'predicted_price': round(predicted_price, 2)})

if __name__ == "__main__":
    train_model()  # Train the model when the script is run
    app.run(debug=True, port=5000)  # Start the Flask app on port 5000

    # Print the URL where the application is running
    print("Server is running on http://127.0.0.1:5000/")
