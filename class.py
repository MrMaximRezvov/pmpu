import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_excel('WaveForm1.xlsx', header=None)
X = df.to_numpy()  # 5500 сигналов × 61 точка

# 2. Извлечение признаков (самое важное!)
def extract_features(signals):
    features = []
    
    for signal in signals:
        # Признак 1: Количество пиков
        peaks = 0
        for i in range(1, 60):
            if signal[i] > signal[i-1] + 0.02 and signal[i] > signal[i+1] + 0.02:
                peaks += 1
        
        # Признак 2: Вариация (насколько сильно меняется сигнал)
        variation = np.sum(np.abs(np.diff(signal)))
        
        # Признак 3: Среднее значение
        mean_val = np.mean(signal)
        
        # Признак 4: Стандартное отклонение
        std_val = np.std(signal)
        
        # Признак 5: Максимальное значение
        max_val = np.max(signal)
        
        # Признак 6: Минимальное значение
        min_val = np.min(signal)
        
        # Признак 7: Асимметрия
        center = len(signal) // 2
        left_mean = np.mean(signal[:center])
        right_mean = np.mean(signal[center:])
        asymmetry = np.abs(left_mean - right_mean)
        
        # Признак 8: Количество пересечений среднего уровня
        crossings = 0
        for i in range(1, len(signal)):
            if (signal[i-1] - mean_val) * (signal[i] - mean_val) < 0:
                crossings += 1
        
        features.append([peaks, variation, mean_val, std_val, max_val, min_val, asymmetry, crossings])
    
    return np.array(features)

# 3. Извлекаем признаки
X_features = extract_features(X)

# 4. Нормализация признаков (важно для многих моделей)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# 5. Создаём и обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Но нам нужны метки для обучения... Используем полу-автоматический подход:

# Размечаем часть данных эвристикой (то, что мы точно знаем)
y_partial = np.zeros(len(X))

for i, signal in enumerate(X):
    peaks = 0
    for j in range(1, 60):
        if signal[j] > signal[j-1] + 0.02 and signal[j] > signal[j+1] + 0.02:
            peaks += 1
    
    # Если ≥2 пиков - точно синусоида (режим 1)
    if peaks >= 2:
        y_partial[i] = 1
    # Если 1 пик в центре - вероятно импульс (режим 2)
    elif peaks == 1:
        peak_idx = np.argmax(signal[1:60]) + 1
        if 10 <= peak_idx <= 50:  # t ∈ [1,5]
            y_partial[i] = 2
        else:
            y_partial[i] = 1
    else:
        y_partial[i] = 0  # Неизвестно

# Берём только размеченные данные для обучения
known_indices = y_partial != 0
X_train = X_scaled[known_indices]
y_train = y_partial[known_indices]

# Обучаем модель
model.fit(X_train, y_train)

# 6. Предсказываем для всех данных
predictions = model.predict(X_scaled)

# 7. Считаем ответ
mode2_count = np.sum(predictions == 2)

print(f"Количество экспериментов в Режиме №2: {int(mode2_count)}")
