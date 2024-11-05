import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

def preprocess_input(data, scaler):
    """
    Preprocess new input data to match the training data format, including encoding,
    one-hot encoding, feature engineering, and scaling.
    """
    # Convert input dictionary to DataFrame
    data_df = pd.DataFrame([data])
    
    # Ensure necessary categorical encodings
    label_encoders = joblib.load(r"C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\label_encoders.joblib")
    try:
        data_df['Building condition'] = label_encoders['Building condition'].transform(data_df['Building condition']) + 1
        data_df['Region'] = label_encoders['Region'].transform(data_df['Region']) + 1
    except KeyError as e:
        raise ValueError(f"Incorrect 'Building condition' or 'Region' format. Details: {e}")

    # One-hot encoding for 'Property type'
    data_df = pd.get_dummies(data_df, columns=['Property type'])

    # Load required columns and ensure consistency
    required_columns = joblib.load(r"C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\required_columns.joblib")
    for col in required_columns:
        if col not in data_df.columns:
            data_df[col] = 0  # Fill missing columns with 0
    data_df = data_df[required_columns]  # Ensure column order matches

    # Feature engineering without 'Price'
    data_df['bedrooms_per_sqm'] = data_df['Number of bedrooms'] / (data_df['Living area m²'] + 1)
    
    # Scale the input data to match training scale
    data_scaled = scaler.transform(data_df)
    
    return data_scaled

def predict_price(new_data):
    """
    Load the trained model and scaler, preprocess the input data, and predict the price.
    
    Parameters:
    - new_data (dict): Dictionary of new property data with keys matching training feature names.
    
    Returns:
    - float: Predicted price for the property.
    """
    # Load trained model and scaler
    model = joblib.load(r"C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\final_xgb_model_advanced.joblib")
    scaler = joblib.load(r"C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\scaler.joblib")
    
    # Validate and preprocess new data
    processed_data = preprocess_input(new_data, scaler)
    
    # Predict and revert log transformation
    log_price_pred = model.predict(processed_data)
    price_pred = np.expm1(log_price_pred)  # Revert log transformation for prediction
    
    return price_pred[0]

def validate_input(data):
    """
    Validate input data format to ensure required fields are present and properly formatted.
    
    Parameters:
    - data (dict): Dictionary of input data.
    
    Returns:
    - bool: True if validation passes; raises ValueError otherwise.
    """
    required_fields = [
        "Number of bedrooms", "Living area m²", "Equipped kitchen", "Furnished", 
        "Swimming pool", "Building condition", "Region", "Property type"
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    return True

if __name__ == "__main__":
    # Example input data for a new property
    new_property = {
        "Number of bedrooms": 3,
        "Living area m²": 120,
        "Equipped kitchen": 1,
        "Furnished": 0,
        "Swimming pool": 0,
        "Building condition": "Good",
        "Region": "Flanders",
        "Property type": "apartment"
    }

    # Validate input data
    try:
        validate_input(new_property)
        predicted_price = predict_price(new_property)
        print(f"Estimated Price: {predicted_price:,.2f}")
    except ValueError as e:
        print(f"Input Error: {e}")
