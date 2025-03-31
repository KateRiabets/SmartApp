import tkinter as tk
from tkinter import messagebox
import requests
import torch
import torch.nn as nn
import pandas as pd
import joblib
import os

# Завантаження скейлерів, що використовувались під час навчання
scaler_X_path = "scaler_X.pkl"
scaler_y_path = "scaler_y.pkl"
if not (os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
    raise FileNotFoundError("Не знайдено scaler_X.pkl и scaler_y.pkl. ")

scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Завантаження нейромережі
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16), # Вхід:
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # Вихід
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN()
model.load_state_dict(torch.load("model_checkpoint.pth"))
model.eval()

# Адреса ESP
ESP_URL = "http://localhost:9080/TEMP"


# Зчитування даних ESP
def zchytaty_dani():
    try:
        response = requests.get(ESP_URL, timeout=5)
        text = response.text.strip()
        print("Отримано:", text)
        parts = text.split(";")
        temp = float(parts[0].split(":")[1])
        hum = float(parts[1].split(":")[1])
        temperature_var.set(f"{temp:.2f}")
        humidity_var.set(f"{hum:.2f}")
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося зчитати дані з ESP:\n{e}")



# Передбачення
def zrobyty_peredbachennya():
    try:
        # Отримання значень
        temp = float(temperature_var.get())
        hum = float(humidity_var.get())
        input_scaled = scaler_X.transform([[hum, temp]])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Отримання результатів з моделі
        with torch.no_grad():
            output = model(input_tensor).numpy()
        # Зворотне масштабування результатів
        output_original = scaler_y.inverse_transform(output)[0]
        # Формуєвання тексту з результатами
        result_text = (
            f"Flavour: {output_original[0]:.2f}\n"
            f"Juice: {output_original[1]:.2f}\n"
            f"Weight: {output_original[2]:.2f} г"
        )
        result_var.set(result_text)
    except Exception as e:
        messagebox.showerror("Помилка", f"Передбачення не виконано:\n{e}")

# GUI
root = tk.Tk()
root.title("Прогноз врожайності")
root.geometry("600x550")
root.configure(bg='#f0f0f0')

STYLE_LABEL = {"font": ("Arial", 13), "bg": "white", "fg": "#ed95ad"}
STYLE_ENTRY = {"font": ("Arial", 13), "bd": 1, "relief": "solid"}
STYLE_BUTTON = {
    "font": ("Arial", 13),
    "bg": "#ed95ad",
    "fg": "white",
    "activebackground": "#ff4081",
    "bd": 0,
    "relief": "flat",
    "padx": 10,
    "pady": 5
}

temperature_var = tk.StringVar()
humidity_var = tk.StringVar()
result_var = tk.StringVar()

container = tk.Frame(root, bg="white", bd=2, relief="groove")
container.place(relx=0.5, rely=0.5, anchor="center", width=500, height=500)

header = tk.Label(container, text="Прогноз врожайності", font=("Arial", 16, "bold"), bg="white", fg="#ed95ad")
header.pack(pady=10)


lbl_temp = tk.Label(container, text="Температура (°C):", **STYLE_LABEL)
lbl_temp.pack(anchor="w", padx=20, pady=(10, 0))
temp_entry = tk.Entry(container, textvariable=temperature_var, **STYLE_ENTRY)
temp_entry.pack(fill="x", padx=20, pady=5)

lbl_hum = tk.Label(container, text="Вологість (%):", **STYLE_LABEL)
lbl_hum.pack(anchor="w", padx=20, pady=(10, 0))
hum_entry = tk.Entry(container, textvariable=humidity_var, **STYLE_ENTRY)
hum_entry.pack(fill="x", padx=20, pady=5)
btn_read = tk.Button(container, text="Зчитати дані", command=zchytaty_dani, **STYLE_BUTTON)
btn_read.pack(fill="x", padx=20, pady=10)

btn_predict = tk.Button(container, text="Зробити передбачення", command=zrobyty_peredbachennya, **STYLE_BUTTON)
btn_predict.pack(fill="x", padx=20, pady=10)

lbl_result = tk.Label(container, textvariable=result_var, font=("Courier", 12), bg="#f9f9f9", justify="left", wraplength=300)
lbl_result.pack(fill="both", expand=True, padx=20, pady=10)

root.mainloop()
