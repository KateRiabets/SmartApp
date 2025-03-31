import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

# Завантаження даних з Excel
data = pd.read_excel("final_all_merged.xlsx")
X = data[["X_Air_humidity", "X_Air_temperature"]].values   # Вхідні ознаки
y = data[["Y_Flavour", "Y_Juice", "Y_Weight"]].values       # Цільові значення

# Шляхи до збережених скейлерів
scaler_X_path = "scaler_X.pkl"
scaler_y_path = "scaler_y.pkl"

# Завантаження скейлерів, якщо існують, або створення нових
if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
else:
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(X)
    scaler_y.fit(y)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

# Масштабування даних
X = scaler_X.transform(X)
y = scaler_y.transform(y)

# ✂Розділення на навчальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Перетворення в тензори PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Структура нейронної мережі
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)

# Ініціалізація моделі
model = SimpleNN()

# Визначення функції втрат і оптимізатора
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Навчання моделі
epochs = 200000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Епоха {epoch+1}/{epochs}, Втрата: {loss.item():.4f}")

# Збереження навченої моделі
torch.save(model.state_dict(), "model_checkpoint.pth")
