import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def generate_synthetic_data(n_samples=1000):
    """Генерируем обучающие данные по формулам из PDF"""
    X = []
    y = []
    
    for _ in range(n_samples):
        if np.random.random() < 0.5:  # Режим №1: синусоида
            A = np.random.uniform(0.3, 0.6)
            omega = np.random.uniform(0.5, 1.5)
            phi = np.random.uniform(0, np.pi)
            
            t = np.linspace(0, 6, 61)  # 61 точка от 0 до 6
            signal = A * np.sin(omega * t + phi) + A
            
            label = 1
        else:  # Режим №2: импульс
            omega0 = np.random.uniform(1, 5)
            gamma = np.random.uniform(0.01, 0.1)
            
            t = np.linspace(0, 6, 61)
            signal = t / np.sqrt((t**2 - omega0**2)**2 + gamma * t**2 + 1e-10)  # +1e-10 для стабильности
            
            label = 2
        
        # Применяем НОРМАЛИЗАЦИЮ из условия: y = x/max(x) + h
        h = np.random.uniform(0, 0.1)
        signal = signal / np.max(np.abs(signal)) + h
        
        X.append(signal)
        y.append(label)
    
    return np.array(X), np.array(y)

def extract_features(signals):
    """Извлекаем признаки, устойчивые к нормализации"""
    features = []
    
    for signal in signals:
        # 1. Количество ЗНАЧИМЫХ пиков (с порогом для шума)
        peaks = 0
        for i in range(1, 60):
            if signal[i] > signal[i-1] + 0.02 and signal[i] > signal[i+1] + 0.02:
                peaks += 1
        
        # 2. Вариация (синусоида меняется чаще)
        variation = np.sum(np.abs(np.diff(signal)))
        
        # 3. Частота пересечений среднего уровня
        mean_val = np.mean(signal)
        crossings = 0
        for i in range(1, 61):
            if (signal[i-1] - mean_val) * (signal[i] - mean_val) < 0:
                crossings += 1
        
        # 4. Ширина основного пика (для импульса - узкий)
        max_idx = np.argmax(signal)
        left_width = 0
        right_width = 0
        
        # Ищем, где сигнал падает до 0.5 от максимума
        max_val = signal[max_idx]
        threshold = 0.5 * max_val
        
        for i in range(max_idx, 0, -1):
            if signal[i] < threshold:
                left_width = max_idx - i
                break
        
        for i in range(max_idx, 61):
            if signal[i] < threshold:
                right_width = i - max_idx
                break
        
        peak_width = left_width + right_width
        
        features.append([peaks, variation, crossings, peak_width])
    
    return np.array(features)

def solve_olympiad_task():
    """Основное решение для олимпиады"""
    # 1. Генерируем обучающие данные
    X_train, y_train = generate_synthetic_data(2000)
    
    # 2. Извлекаем признаки
    X_train_features = extract_features(X_train)
    
    # 3. Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_features, y_train)
    
    # 4. Загружаем реальные данные
    real_data = pd.read_excel("classification/WaveForm1.xlsx", header=None).values
    
    # 5. Извлекаем признаки для реальных данных
    real_features = extract_features(real_data)
    
    # 6. Предсказываем
    predictions = model.predict(real_features)
    
    # 7. Считаем ответ
    mode2_count = np.sum(predictions == 2)
    
    return int(mode2_count)

if __name__ == '__main__':
    answer = solve_olympiad_task()
    print(f"Ответ: {answer}")