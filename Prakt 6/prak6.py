import tkinter as tk
from tkinter import messagebox
import requests
import matplotlib.pyplot as plt

# Адреса ESP
ESP_URL = "http://localhost:9080/START"


# Перетворення  з АЦП у децибели
def convert_adc_to_db(adc_value):
    return 80 * adc_value / 4095

def on_start():
    try:
        #GET-запит до ESP
        response = requests.get(ESP_URL, timeout=35)
        data = response.json()
        print("Отримано даних:", len(data))

        # Перевірка, що дані у форматі списку
        if not isinstance(data, list):
            messagebox.showerror("Помилка", "Невірний формат даних від ESP32")
            return

        # Список денних та нічних годин
        day_hours = list(range(8, 22))
        night_hours_ordered = [22, 23] + list(range(0, 8))

        # Ініціалізація змінних
        day_labels = []
        day_pot1, day_pot2 = [], []
        night_pot1_dict, night_pot2_dict = {}, {}
        all_db = []

        # Обробка кожного запису
        for entry in data:
            hour = entry.get("hour")
            val1 = entry.get("pot1")
            val2 = entry.get("pot2")

            # Пропуск, якщо немає потрібних значень
            if hour is None or val1 is None or val2 is None:
                continue

            # Перетворення у дБ
            val1_db = convert_adc_to_db(val1)
            val2_db = convert_adc_to_db(val2)

            all_db.extend([val1_db, val2_db])

            # Розділення денних та нічних даних
            if hour in day_hours:
                day_labels.append(hour)
                day_pot1.append(val1_db)
                day_pot2.append(val2_db)
            else:
                night_pot1_dict[hour] = val1_db
                night_pot2_dict[hour] = val2_db

        # Впорядкування нічних годин у правильному порядку
        night_labels = []
        night_pot1 = []
        night_pot2 = []
        for h in night_hours_ordered:
            if h in night_pot1_dict:
                night_labels.append(str(h))
                night_pot1.append(night_pot1_dict[h])
                night_pot2.append(night_pot2_dict[h])

        # Обчислення середнього рівня шуму
        avg_noise = sum(all_db) / len(all_db) if all_db else 0

        # Окремий середній шум для дня і ночі
        avg_day_noise = sum(day_pot1 + day_pot2) / len(day_pot1 + day_pot2) if day_pot1 + day_pot2 else 0
        avg_night_noise = sum(night_pot1 + night_pot2) / len(night_pot1 + night_pot2) if night_pot1 + night_pot2 else 0

        # Пікові значення для дня
        peak_day_noise = max(day_pot1 + day_pot2) if day_pot1 + day_pot2 else 0
        peak_day_hours = [str(day_labels[i // 2]) for i, v in enumerate(day_pot1 + day_pot2) if v == peak_day_noise]

        # Пікові значення для ночі
        peak_night_noise = max(night_pot1 + night_pot2) if night_pot1 + night_pot2 else 0
        night_combined_labels = night_labels + night_labels
        peak_night_hours = [night_combined_labels[i] for i, v in enumerate(night_pot1 + night_pot2) if
                            v == peak_night_noise]

        # Виведення результатів
        avg_label.config(text=(f"Середній рівень шуму: {avg_noise:.2f} дБ\n"
                               f"Середній шум вдень: {avg_day_noise:.2f} дБ\n"
                               f"Середній шум вночі: {avg_night_noise:.2f} дБ\n"
                               f"Піковий шум вдень: {peak_day_noise:.2f} дБ \n(години: {', '.join(peak_day_hours)})\n"
                               f"Піковий шум вночі: {peak_night_noise:.2f} дБ \n(години: {', '.join(peak_night_hours)})"))

        # Побудова графіків
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(day_labels, day_pot1, label="Кімната 1 (день)", marker='o')
        ax1.plot(day_labels, day_pot2, label="Кімната 2 (день)", marker='s')
        ax1.set_title("Денний рівень шуму")
        ax1.set_xlabel("Години")
        ax1.set_ylabel("Шум (дБ)")
        ax1.grid(True)
        ax1.legend()
        ax1.set_xticks(day_labels)

        ax2.plot(range(len(night_labels)), night_pot1, label="Кімната 1 (ніч)", marker='o')
        ax2.plot(range(len(night_labels)), night_pot2, label="Кімната 2 (ніч)", marker='s')
        ax2.set_title("Нічний рівень шуму")
        ax2.set_xlabel("Години")
        ax2.set_ylabel("Шум (дБ)")
        ax2.grid(True)
        ax2.legend()
        ax2.set_xticks(range(len(night_labels)))
        ax2.set_xticklabels(night_labels)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося отримати дані:\n{e}")


root = tk.Tk()
root.title("Визначення рівня шуму")
root.geometry("600x550")
root.configure(bg='#f0f0f0')

STYLE_LABEL = {"font": ("Arial", 13), "bg": "white", "fg": "#ed95ad"}
STYLE_BUTTON = {"font": ("Arial", 13), "bg": "#ed95ad", "fg": "white", "activebackground": "#ff4081", "bd": 0, "relief": "flat", "padx": 10, "pady": 5}

container = tk.Frame(root, bg="white", bd=2, relief="groove")
container.place(relx=0.5, rely=0.5, anchor="center", width=500, height=450)

header = tk.Label(container, text="Визначення рівня шуму", font=("Arial", 16, "bold"), bg="white", fg="#ed95ad")
header.pack(pady=10)

label_info = tk.Label(container, text="Натисніть «Старт» для запуску\nОчікуйте ~12 секунд", **STYLE_LABEL)
label_info.pack(pady=5)

btn_start = tk.Button(container, text="Старт", command=on_start, **STYLE_BUTTON)
btn_start.pack(fill="x", padx=20, pady=10)

avg_label = tk.Label(container, text="", font=("Arial", 12), bg="white", fg="#ed95ad")
avg_label.pack(pady=20)

root.mainloop()
