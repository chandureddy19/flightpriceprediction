import pickle
import xgboost as xgb
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the XGBoost model
with open('xgb_flight_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# List of features used by the model (Make sure this aligns with the model's training features)
feature_order = [
    'Total_Stops', 'Route1', 'Route2', 'Route3', 'Route4', 'Route5',
    'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
    'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
    'Trujet', 'Vistara', 'Vistara Premium economy',
    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad', 
    'Destination_Kolkata', 'Destination_New_Delhi',
    'journey_day', 'journey_month', 'Dep_Time_hour', 'Dep_Time_min',
    'Arrival_Time_hour', 'Arrival_Time_min', 'journey_day_of_week',
    'is_weekend', 'is_holiday_season', 'is_peak_hour_departure',
    'is_peak_hour_arrival', 'dur_hour', 'dur_min',
    'flight_duration_category_Medium', 'flight_duration_category_Short'
]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    try:
        Total_Stops = int(request.form['Total_Stops'])
        Route1 = int(request.form['Route1'])
        Route2 = int(request.form['Route2'])
        Route3 = int(request.form['Route3'])
        Route4 = int(request.form['Route4'])
        Route5 = int(request.form['Route5'])


        
        Airline = request.form['Airline']
        Source = request.form['Source']
        Destination = request.form['Destination']
        journey_day = int(request.form['journey_day'])
        journey_month = int(request.form['journey_month'])
        Dep_Time_hour = int(request.form['Dep_Time_hour'])
        Dep_Time_min = int(request.form['Dep_Time_min'])
        Arrival_Time_hour = int(request.form['Arrival_Time_hour'])
        Arrival_Time_min = int(request.form['Arrival_Time_min'])

        # Handle the journey_day_of_week field gracefully (default value if not found)
        journey_day_of_week = int(request.form.get('journey_day_of_week', 0))  # Default to 0 if not provided
        
        # Checkboxes
        is_weekend = 1 if 'is_weekend' in request.form else 0
        is_holiday_season = 1 if 'is_holiday_season' in request.form else 0
        is_peak_hour_departure = 1 if 'is_peak_hour_departure' in request.form else 0
        is_peak_hour_arrival = 1 if 'is_peak_hour_arrival' in request.form else 0
        
        # Duration
        dur_hour = int(request.form['dur_hour'])
        dur_min = int(request.form['dur_min'])
        
    except KeyError as e:
        # Handle missing keys
        return f"Missing field: {str(e)}", 400

    # Prepare data for model (Ensure correct one-hot encoding)
    sample_data = {
        'Total_Stops': [Total_Stops],
        'Route1': [Route1],
        'Route2': [Route2],
        'Route3': [Route3],
        'Route4': [Route4],
        'Route5': [Route5],
        # One-hot encoding for Airline
        'Air India': [1 if Airline == 'Air India' else 0],
        'GoAir': [1 if Airline == 'GoAir' else 0],
        'IndiGo': [1 if Airline == 'IndiGo' else 0],
        'Jet Airways': [1 if Airline == 'Jet Airways' else 0],
        'Jet Airways Business': [1 if Airline == 'Jet Airways Business' else 0],
        'Multiple carriers': [1 if Airline == 'Multiple carriers' else 0],
        'Multiple carriers Premium economy': [1 if Airline == 'Multiple carriers Premium economy' else 0],
        'SpiceJet': [1 if Airline == 'SpiceJet' else 0],
        'Trujet': [1 if Airline == 'Trujet' else 0],
        'Vistara': [1 if Airline == 'Vistara' else 0],
        'Vistara Premium economy': [1 if Airline == 'Vistara Premium economy' else 0],



        # One-hot encoding for Source city
        'Source_Chennai': [1 if Source == 'Chennai' else 0],
        'Source_Delhi': [1 if Source == 'Delhi' else 0],
        'Source_Kolkata': [1 if Source == 'Kolkata' else 0],
        'Source_Mumbai': [1 if Source == 'Mumbai' else 0],
        # One-hot encoding for Destination city
        'Destination_Cochin': [1 if Destination == 'Cochin' else 0],
        'Destination_Delhi': [1 if Destination == 'Delhi' else 0],
        'Destination_Hyderabad': [1 if Destination == 'Hyderabad' else 0],
        'Destination_Kolkata': [1 if Destination == 'Kolkata' else 0],
        'Destination_New_Delhi': [1 if Destination == 'New Delhi' else 0],

        # Other features
        'journey_day': [journey_day],
        'journey_month': [journey_month],
        'Dep_Time_hour': [Dep_Time_hour],
        'Dep_Time_min': [Dep_Time_min],
        'Arrival_Time_hour': [Arrival_Time_hour],
        'Arrival_Time_min': [Arrival_Time_min],
        'journey_day_of_week': [journey_day_of_week], 
        'is_weekend': [is_weekend],
        'is_holiday_season': [is_holiday_season],
        'is_peak_hour_departure': [is_peak_hour_departure],
        'is_peak_hour_arrival': [is_peak_hour_arrival],
        'dur_hour': [dur_hour],
        'dur_min': [dur_min],
        'flight_duration_category_Medium': [int(request.form.get('flight_duration_category_Medium', 0))],
        'flight_duration_category_Short': [int(request.form.get('flight_duration_category_Short', 0))]
    }
    

    # Create a DataFrame for prediction
    sample_df = pd.DataFrame(sample_data)
    
    # Make the prediction
    prediction = model.predict(sample_df)
    
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
