
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib

# Step 1: Load the Dataset
data = pd.read_csv("test_full.csv")

# Step 2: Preprocess the Data
# Handle missing values, encode categorical variables, split into features (X) and labels (y)
X = data.drop(columns=["Total Time"])  # Assuming "User" is not a feature
y = data["Total Time"]

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# Step 4: Choose a Model and Train
# model = LinearRegression()
# model = SVR(kernel='rbf')  # You can choose different kernels (e.g., 'linear', 'poly', 'rbf')
model = RandomForestRegressor()


model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 6: Fine-Tune the Model (if necessary)

# Step 7: Make Predictions (on new data if available)

# Step 8: Save the Model
# joblib.dump(model, "your_model.pkl")


# Assuming y_test and y_pred are the true and predicted target values, respectively

# Step 5: Evaluate the Model (Calculate NMSE)
mse = mean_squared_error(y_test, y_pred)
variance_y = np.var(y_test)
nmse = mse / variance_y

print("Normalized Mean Squared Error (NMSE):", nmse)

