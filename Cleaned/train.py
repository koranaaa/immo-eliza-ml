# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
import joblib

# Data loading and processing
def load_and_preprocess_data(filepath):
    # Loading data
    data = pd.read_csv(filepath)
    
    # Specify the columns to delete
    columns_to_drop = [
        'Unnamed: 0', 'Property ID', 'Locality data', 
        'Open fire', 'Terrace surface m²', 'Garden area m²', 
        'Price per m²', 'Province'
    ]
    data = data.drop(columns=columns_to_drop)
    
    # Coding of categorical variables
    label_encoders = {}
    for col in ['Building condition', 'Region']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col]) + 1
        label_encoders[col] = le
    data = pd.get_dummies(data, columns=['Property type'], drop_first=True)
    
# Save label encoders for later use
    joblib.dump(label_encoders, r'C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\label_encoders.joblib')

    # Add new features
    data['price_per_sqm'] = (data['Price'] / (data['Living area m²'] + 1)).round(2)
    data['bedrooms_per_sqm'] = (data['Number of bedrooms'] / (data['Living area m²'] + 1)).round(3)
    
    # Definition of the target variable and features
    X = data.drop('Price', axis=1)
    y = data['Price']
    
    return X, y

# Removal of outliers
def remove_outliers(X, y):
    threshold = y.quantile(0.99)
    X_no_outliers = X[y < threshold]
    y_no_outliers = y[y < threshold]
    return X_no_outliers, y_no_outliers

# Data scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# Training the model
def train_model(X_train, y_train, params):
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluation of the model
def evaluate_model(y_test, y_pred, model_name="XGBoost"):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"MAE: {mae:,.3f}")
    print(f"MSE: {mse:,.3f}")
    print(f"RMSE: {rmse:,.3f}")
    print(f"MAPE: {mape * 100:,.3f}%")
    print(f"R² Score: {r2:.2f}")

# Main function
def main():
    # Step 1: Loading and preprocessing the data
    filepath = r'C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Data\cleaned_data_with_region_and_price_per_m2.csv'
    X, y = load_and_preprocess_data(filepath)
    
    # Step 2: Remove outliers
    X_no_outliers, y_no_outliers = remove_outliers(X, y)
    
    # Step 3: Division into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X_no_outliers, y_no_outliers, test_size=0.2, random_state=42)
    
    # An intermediate step for saving the list of required columns for forecasting
    required_columns = X_train.columns.tolist()
    joblib.dump(required_columns, r'C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\required_columns.joblib')

    # Step 4: Logarithmic transformation of the target variable
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # Step 5: Scaling the data
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 6: Setting the model parameters
    best_params_xgb_advanced = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'gamma': 0.3,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5
    }
    
    # Step 7: Training the model
    model = train_model(X_train_scaled, y_train_log, best_params_xgb_advanced)
    
    # Step 8: Prediction and inverse transformation
    y_pred_log = model.predict(X_test_scaled)
    y_pred_final = np.expm1(y_pred_log)  # Reverse conversion to original scale
    
   # Step 9: Model evaluation
    evaluate_model(y_test, y_pred_final)
    
    # Step 10: Save the model
    joblib.dump(model, r'C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\final_xgb_model_advanced.joblib')
    joblib.dump(scaler, r'C:\Users\Anastasiia\Documents\GitHub\immo-eliza-ml\Cleaned\scaler.joblib')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
