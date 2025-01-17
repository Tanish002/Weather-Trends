%%writefile app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

st.title("Weather Data Analysis and Prediction")
st.write("Upload your weather dataset (CSV format) to analyze and predict weather metrics for upcoming days.")

# Step 2: File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Step 3: Feature Engineering
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

    # Adding rolling averages and lagged features
    data['Temp_Change'] = data['Max Temperature (C)'] - data['Min Temperature (C)']
    data['Temp_3Day_Avg'] = data['Avg Temperature (C)'].rolling(window=3).mean()
    data['Temp_7Day_Avg'] = data['Avg Temperature (C)'].rolling(window=7).mean()
    data['Temp_30Day_Avg'] = data['Avg Temperature (C)'].rolling(window=30).mean()
    data['Humidity_3Day_Avg'] = data['Avg Humidity'].rolling(window=3).mean()
    data['Humidity_7Day_Avg'] = data['Avg Humidity'].rolling(window=7).mean()
    data['DewPoint_3Day_Min'] = data['Avg Dew Point (C)'].rolling(window=3).min()
    data['WindSpeed_7Day_Max'] = data['Avg Wind Speed'].rolling(window=7).max()
    data['WindSpeed_30Day_Std'] = data['Avg Wind Speed'].rolling(window=30).std()

    # Create next-day targets
    data['Next_Day_Max_Temperature'] = data['Max Temperature (C)'].shift(-1)
    data['Next_Day_Min_Temperature'] = data['Min Temperature (C)'].shift(-1)
    data['Next_Day_Humidity'] = data['Avg Humidity'].shift(-1)
    data['Next_Day_Wind_Speed'] = data['Avg Wind Speed'].shift(-1)
    data.dropna(inplace=True)

    # Defining features and targets
    target_columns = ['Next_Day_Max_Temperature', 'Next_Day_Min_Temperature', 'Next_Day_Humidity', 'Next_Day_Wind_Speed']
    X = data.drop(columns=target_columns + ['Date'])
    y = data[target_columns]

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train Models
    models = {
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'))
    }

    results = {}
    st.write("Training models, please wait...")
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        results[model_name] = {
            'MAE': mae, 'MSE': mse, 'R2': r2
        }

    # Display Results
    # Flatten results to separate each target variable metric
    flattened_results = {}
    for model_name, metrics in results.items():
        for metric_name, metric_values in metrics.items():
            for i, target in enumerate(["Max_Temp", "Min_Temp", "Humidity", "Wind_Speed"]):
                flattened_results[f"{model_name}_{metric_name}_{target}"] = metric_values[i]

    # Convert the flattened dictionary to a DataFrame
    results_df = pd.DataFrame([flattened_results])

    # Display the results in Streamlit
    st.write("\nEvaluation Results:")
    st.write(results_df)


    # Step 6: Prediction
    st.subheader("Predict Weather for Upcoming Days")
    days = st.slider("Select number of days to predict:", 1, 14, 7)

    def prepare_latest_data(latest_data):
        pred_data = latest_data.copy()
        pred_data['Temp_Change'] = pred_data['Max Temperature (C)'] - pred_data['Min Temperature (C)']
        pred_data['Temp_3Day_Avg'] = latest_data['Temp_3Day_Avg'].iloc[-1]
        pred_data['Temp_7Day_Avg'] = latest_data['Temp_7Day_Avg'].iloc[-1]
        pred_data['Temp_30Day_Avg'] = latest_data['Temp_30Day_Avg'].iloc[-1]
        pred_data['Humidity_3Day_Avg'] = latest_data['Humidity_3Day_Avg'].iloc[-1]
        pred_data['Humidity_7Day_Avg'] = latest_data['Humidity_7Day_Avg'].iloc[-1]
        pred_data['DewPoint_3Day_Min'] = latest_data['DewPoint_3Day_Min'].iloc[-1]
        pred_data['WindSpeed_7Day_Max'] = latest_data['WindSpeed_7Day_Max'].iloc[-1]
        pred_data['WindSpeed_30Day_Std'] = latest_data['WindSpeed_30Day_Std'].iloc[-1]
        return pred_data[X.columns]

    def predict_next_days(model, latest_data, days=7):
        predictions = []
        current_data = latest_data.copy()
        for day in range(days):
            pred_input = prepare_latest_data(current_data)
            next_day_pred = model.predict(pred_input)
            pred_date = current_data['Date'].iloc[-1] + pd.Timedelta(days=1)
            pred_dict = {
                'Date': pred_date,
                'Predicted_Max_Temp': next_day_pred[0][0],
                'Predicted_Min_Temp': next_day_pred[0][1],
                'Predicted_Humidity': next_day_pred[0][2],
                'Predicted_Wind_Speed': next_day_pred[0][3]
            }
            predictions.append(pred_dict)
            new_row = current_data.iloc[-1:].copy()
            new_row['Date'] = pred_date
            new_row['Max Temperature (C)'] = next_day_pred[0][0]
            new_row['Min Temperature (C)'] = next_day_pred[0][1]
            new_row['Avg Temperature (C)'] = (next_day_pred[0][0] + next_day_pred[0][1]) / 2
            new_row['Avg Humidity'] = next_day_pred[0][2]
            new_row['Avg Wind Speed'] = next_day_pred[0][3]
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            current_data['Temp_3Day_Avg'] = current_data['Avg Temperature (C)'].rolling(window=3).mean()
            current_data['Temp_7Day_Avg'] = current_data['Avg Temperature (C)'].rolling(window=7).mean()
            current_data['Temp_30Day_Avg'] = current_data['Avg Temperature (C)'].rolling(window=30).mean()
            current_data['Humidity_3Day_Avg'] = current_data['Avg Humidity'].rolling(window=3).mean()
            current_data['Humidity_7Day_Avg'] = current_data['Avg Humidity'].rolling(window=7).mean()
        return pd.DataFrame(predictions)

    best_model = models['Random Forest']
    latest_data = data.tail(30)  # Last 30 days for rolling averages
    predictions_df = predict_next_days(best_model, latest_data, days=days)

    st.subheader(f"Weather Predictions for Next {days} Days")
    st.write(predictions_df.round(2))

    # Optional: Plot predictions
    st.subheader("Temperature Predictions Visualization")
    fig, ax = plt.subplots()
    ax.plot(predictions_df['Date'], predictions_df['Predicted_Max_Temp'], 'r-', label='Max Temperature')
    ax.plot(predictions_df['Date'], predictions_df['Predicted_Min_Temp'], 'b-', label='Min Temperature')
    plt.title('Temperature Predictions for Next Days')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
