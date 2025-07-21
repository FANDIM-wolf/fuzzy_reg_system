import pandas as pd
import numpy as np

# Параметры генерации данных
num_samples = 300  # Количество строк данных
temp_min, temp_max = -20, 100      # Температура от 0 до 100
hum_min, hum_max = 0, 100       # Влажность от 0 до 100

# Генерация случайных данных
np.random.seed(42)  # Для воспроизводимости

# T1 - температура в момент времени 1
T1 = np.random.uniform(temp_min, temp_max, num_samples)

# T2 - температура в момент времени 2 (немного коррелирована с T1)
T2 = T1 + np.random.normal(loc=0, scale=5, size=num_samples)
T2 = np.clip(T2, temp_min, temp_max)  # Ограничение диапазона

# H1 - влажность в момент времени 1
H1 = np.random.uniform(hum_min, hum_max, num_samples)

# H2 - влажность в момент времени 2 (немного коррелирована с H1)
H2 = H1 + np.random.normal(loc=0, scale=7, size=num_samples)
H2 = np.clip(H2, hum_min, hum_max)  # Ограничение диапазона

# Расчет comfort_level (уровень комфорта)
# Формула: 0.2*T1 + 0.2*T2 + 0.3*H1 + 0.3*H2
# Округление и ограничение диапазона [0, 10]
comfort_levels = np.round(0.2*T1 + 0.2*T2 + 0.3*H1 + 0.3*H2).astype(int)
comfort_levels = np.clip(comfort_levels, 0, 10)

# Создание DataFrame
data = pd.DataFrame({
    'T1': T1,
    'T2': T2,
    'H1': H1,
    'H2': H2,
    'comfort_level': comfort_levels
})

# Сохранение в CSV-файл
data.to_csv('data.csv', index=False, sep=',')

print("Файл data.csv успешно создан с новыми переменными T1, T2, H1, H2, comfort_level.")
print("\nПервые 5 строк данных:")
print(data.head())