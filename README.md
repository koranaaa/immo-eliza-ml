🏠 Immo Eliza ML - Real Estate Price Prediction
📄 Project Description
Immo Eliza ML is a machine learning project focused on predicting real estate prices in Belgium. Using a dataset of property characteristics, this project involves data preprocessing, feature engineering, training models like linear regression, random forest, and XGBoost, and evaluating their performance for accurate price prediction.

📂 Repository Structure
bash
Copy code
├── data/
│   └── cleaned_data_with_region_and_price_per_m2.csv   # Processed data
├── models/
│   └── final_xgb_model_advanced.joblib                 # Saved model
├── utils/
│   ├── data_import.py
│   ├── model_pipeline.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── README.md
├── requirements.txt
└── main.py
🛠️ Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/koranaaa/immo-eliza-ml.git
Navigate to the project directory and install the required dependencies:

bash
Copy code
pip install -r requirements.txt
✔️ Usage
The primary processes—data import, processing, model training, and prediction—can be executed through main.py. To run this file, use the following command:

bash
Copy code
python main.py
Example Usage
For predictions, use predict.py. This script loads the prepared model, processes input data, and makes a prediction:

python
Copy code
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
🧠 Models and Metrics
Linear Regression (R2 on test set: ~0.7)
Random Forest (R2 on test set: ~0.71)
XGBoost (R2 on test set: ~0.94, MAPE: ~7%)
Each model is evaluated using MAE, MSE, RMSE, MAPE, and R² Score metrics to compare their accuracy for different property types.

⏱️ Project Timeline
The initial setup of this project was completed in 4 days. It involved automating data processing, feature engineering, and model selection.

🔄 Planned Updates
Model Improvement: Experiment with additional models to enhance accuracy.
Automation: Improve automation of data processing.
Data Expansion: Incorporate new variables for more detailed predictions.
📌 Personal Note
This project was developed as part of my machine learning training at BeCode. It serves as a practical example of applying data preprocessing, feature engineering, and model training and evaluation.

This README will give your repository a professional and accessible introduction, making it easier for others to understand and engage with your project.