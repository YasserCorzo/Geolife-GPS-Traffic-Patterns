
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("test_full.csv")

X = data.drop(columns=["User", "Total Time"])
y = data["Total Time"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

torch.manual_seed(80)

X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = RegressionModel(X_train.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
batch_size = 128

for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        targets = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()

with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()

mse = mean_squared_error(y_test, y_pred)
variance_y = np.var(y_test)
nmse = mse / variance_y

print("Normalized Mean Squared Error (NMSE):", nmse)


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

