#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Step 1: Load the dataset
data = pd.read_csv('Housing.csv')

# Step 2: Data Preprocessing

# Step 3: Feature Selection
features = ['area', 'bedrooms', 'bathrooms']

# Step 4: Feature Scaling (Optional for Linear Regression)

# Step 5: Split Data
X = data[features]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Step 8: Make Predictions and Visualization
new_data = np.array([[1500, 3, 2]])  # Example input for prediction
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]}")

# Scatter plot for actual vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs. Predicted House Prices")
plt.show()


# In[ ]:




