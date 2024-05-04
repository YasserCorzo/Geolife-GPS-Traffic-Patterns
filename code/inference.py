
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def prediction(model_index):
    data = pd.read_csv("test_full.csv")

    X = data.drop(columns=["User", "Total Time"])
    y = data["Total Time"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_index == 0:
        model = LinearRegression()
    elif model_index == 1:
        model = SVR(kernel='linear')
    else:
        model = RandomForestRegressor(random_state=80)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    variance_y = np.var(y_test)
    nmse = mse / variance_y

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    print("Normalized Mean Squared Error (NMSE):", nmse)

    return y_test, y_pred


def plot_inference(model_index):
    y_test, y_pred = prediction(model_index)

    y_test_array = y_test.to_numpy()

    sorted_indices = np.argsort(y_test_array)
    y_test_sorted = y_test_array[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.figure()
    plt.plot(y_test_sorted, y_pred_sorted, 'o', label='predicted vs. true')
    plt.plot([min(y_test_sorted), max(y_test_sorted)], [min(y_test_sorted), max(y_test_sorted)], label='y=x')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title('Predicted vs. true regression values')
    plt.legend()
    plt.show()


plot_inference(2)

