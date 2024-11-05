# 🏠 Immo Eliza ML - Real Estate Price Prediction

## 📄 Project Description
Immo Eliza ML is a machine learning project focused on predicting real estate prices in Belgium. Using a dataset of property characteristics, this project involves data preprocessing, feature engineering, training models like linear regression, random forest, and XGBoost, and evaluating their performance for accurate price prediction.

## 📂 Repository Structure

IMMO-ELIZA-ML/
├── .venv/
├── Cleaned/
│   ├── final_xgb_model_advanced.joblib
│   ├── label_encoders.joblib
│   ├── predict.py
│   ├── required_columns.joblib
│   ├── scaler.joblib
│   └── train.py
├── Data/
│   └── cleaned_data_with_region_and_price_per_m2.csv
├── Raw/
│   ├── ANA_with_provinces.ipynb
│   └── ANA_with_Region.ipynb
├── .gitignore
└── README.md

## 📂 Explanation of structure 
1. IMMO-ELIZA-ML/ - The main project folder.
2. .venv/ - Virtual environment directory.
3. Cleaned/ - Folder containing model files and scripts.
- final_xgb_model_advanced.joblib - Saved model file.
- label_encoders.joblib - Label encoder file.
- predict.py - Prediction script.
- required_columns.joblib - Required columns file.
- scaler.joblib - Scaler file.
- train.py - Training script.
4. Data/ - Folder for processed data.
- cleaned_data_with_region_and_price_per_m2.csv - Processed data file.
5. Raw/ - Folder for raw Jupyter notebooks.
- ANA_with_provinces.ipynb - Analysis notebook.
- ANA_with_Region.ipynb - Analysis notebook.
6. .gitignore - Git ignore file.
7. README.md - Project README file.

## 🛠️ Installation
Clone the repository to your local machine:
git clone https://github.com/koranaaa/immo-eliza-ml.git

git clone https://github.com/koranaaa/immo-eliza-ml.git

Navigate to the project directory and install the required dependencies:
pip install -r requirements.txt


## ✔️ Usage
The primary processes—data import, processing, model training, and prediction—can be executed through main.py. To run this file, use the following command:

python main.py

# Example Usage
For predictions, use predict.py. This script loads the prepared model, processes input data, and makes a prediction:

from utils.predict import predict_price
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
predicted_price = predict_price(new_property)
print(f"Estimated Price: {predicted_price:,.2f}")


## 🧠 Models and Metrics
- Linear Regression (R2 on test set: ~0.7)
- Random Forest (R2 on test set: ~0.71)
- XGBoost (R2 on test set: ~0.94, MAPE: ~7%)

Each model is evaluated using MAE, MSE, RMSE, MAPE, and R² Score metrics to compare their accuracy for different property types.


## ⏱️ Project Timeline
The initial setup of this project was completed in 4 days. It involved automating data processing, feature engineering, and model selection.


## 🔄 Planned Updates
- Model Improvement: Experiment with additional models to enhance accuracy.
- Automation: Improve automation of data processing.
- Data Expansion: Incorporate new variables for more detailed predictions.


## 📌 Personal Note
This project was developed as part of my machine learning training at BeCode. It serves as a practical example of applying data preprocessing, feature engineering, and model training and evaluation.