# Predicting-car-price-across-eras

## Model downloading link 
[https://drive.google.com/file/d/1T9AeHy6OoyFTx4sIIHuakC-QA31x_J0O/view?usp=sharing ]

### 🚗 Car Price Prediction

#### 📌 Overview
This project predicts car prices based on features like brand, model, mileage, engine size, fuel type, and transmission. A Random Forest Regressor model was trained and saved for predictions.

#### 📂 Files
car_price_prediction.ipynb → Jupyter Notebook with full implementation
model.pkl → Saved machine learning model
car_data.csv → Dataset used for training
README.md → Project details
#### 🛠️ Technologies Used
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn

#### Load and Use the Model

import pickle
import pandas as pd
import numpy as np

#### Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

#### Sample input data
new_data = pd.DataFrame({
    'Brand': ['Toyota'],
    'Category': ['SUV'],
    'Mileage': [15000],
    'Engine_Size': [2.0],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Automatic'],
    'Car_Model': ['RAV4'],
    'Car_age': [5],
    'Mileage_per_engine_size': [7500]
})

#### Encode features to match training data  
new_data_encoded = pd.get_dummies(new_data)
new_data_encoded = new_data_encoded.reindex(columns=X_train.columns, fill_value=0)

#### Predict price  
predicted_log_price = model.predict(new_data_encoded)
predicted_price = np.exp(predicted_log_price)

print("Predicted Car Price: $", predicted_price[0])
📈 Model Performance
Best Model: Random Forest Regressor
R² Score: 57%

## 🔗 Future Improvements
Add more features (e.g., owner history, location)
Deploy as a web app using Streamlit
