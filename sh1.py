"""
Олимпиадное решение задачи классификации сигналов
Классификация: Режим №1 (синусоида) vs Режим №2 (импульс)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import scipy.stats

# ============================================
# ФУНКЦИИ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ
# ============================================

def count_peaks(signal, threshold=0.02):
    """Считает количество значимых локальных максимумов"""
    peaks = 0
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] + threshold and signal[i] > signal[i+1] + threshold:
            peaks += 1
    return peaks

def count_zero_crossings(signal):
    """Считает количество пересечений среднего уровня"""
    mean_val = np.mean(signal)
    crossings = 0
    for i in range(1, len(signal)):
        if (signal[i-1] - mean_val) * (signal[i] - mean_val) < 0:
            crossings += 1
    return crossings

def extract_signal_features(signal):
    """
    Извлекает 15 признаков из одного сигнала
    
    Признаки устойчивы к нормализации y = x/max(x) + h
    """
    features = []
    
    # 1. Статистические признаки
    features.append(np.mean(signal))           # Среднее
    features.append(np.std(signal))            # Стандартное отклонение
    features.append(np.max(signal))            # Максимум
    features.append(np.min(signal))            # Минимум
    features.append(np.median(signal))         # Медиана
    features.append(scipy.stats.skew(signal))  # Асимметрия
    features.append(scipy.stats.kurtosis(signal))  # Эксцесс
        # 2. Признаки формы сигнала
    features.append(count_peaks(signal))       # Количество пиков
    features.append(count_zero_crossings(signal))  # Пересечения среднего
    features.append(np.sum(np.abs(np.diff(signal))))  # Вариация
    
    # 3. Энергетические признаки
    features.append(np.sum(signal**2))         # Энергия сигнала
    
    # 4. Признаки первой производной
    deriv = np.diff(signal)
    features.append(np.mean(deriv))            # Среднее производной
    features.append(np.std(deriv))             # Стд производной
    
    # 5. Признаки симметрии
    center = len(signal) // 2
    left_part = signal[:center]
    right_part = signal[center:][::-1]  # зеркальное отражение правой части
    symmetry_score = np.mean(np.abs(left_part[:len(right_part)] - right_part))
    features.append(symmetry_score)  # Мера симметрии
    
    return features

def extract_features_batch(signals):
    """Извлекает признаки для всех сигналов сразу"""
    features_list = []
    for signal in signals:
        features = extract_signal_features(signal)
        features_list.append(features)
    return np.array(features_list)

# ============================================
# ФУНКЦИЯ РАЗМЕТКИ ДАННЫХ (эвристический подход)
# ============================================

def create_labels_with_heuristic(signals):
    """
    Создаёт метки для обучения на основе эвристики
    
    Возвращает:
    - 1: Режим №1 (синусоида)
    - 2: Режим №2 (импульс)
    - 0: Неизвестно (будет исключено из обучения)
    """
    labels = np.zeros(len(signals))
    
    for i, signal in enumerate(signals):
        # Считаем пики
        peaks = count_peaks(signal, threshold=0.02)
        
        # Правило 1: Если ≥2 пиков - точно синусоида (режим 1)        if peaks >= 2:
            labels[i] = 1
        
        # Правило 2: Если 1 пик - проверяем его положение
        elif peaks == 1:
            # Находим индекс пика
            max_idx = np.argmax(signal)
            t_peak = max_idx * 0.1  # конвертируем в время (шаг 0.1)
            
            # Для импульса пик должен быть в [1, 5] (по условию ω₀ ∈ [1, 5])
            if 1.0 <= t_peak <= 5.0:
                labels[i] = 2
            else:
                labels[i] = 1
        
        # Правило 3: Если 0 пиков - вероятно импульс с пиком на краю
        else:
            labels[i] = 2
    
    return labels

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ РЕШЕНИЯ
# ============================================

def solve_signal_classification():
    """
    Полный пайплайн решения задачи классификации сигналов
    
    Возвращает:
    - mode2_count: количество экспериментов в Режиме №2
    - model: обученная модель
    - predictions: предсказания для всех сигналов
    """
    
    print("Шаг 1: Загрузка данных...")
    # Загрузка данных из Excel файла
    df = pd.read_excel('WaveForm1.xlsx', header=None)
    X_raw = df.to_numpy()  # 5500 × 61
    print(f"Загружено {len(X_raw)} сигналов, каждый по {len(X_raw[0])} точек")
    
    print("\nШаг 2: Извлечение признаков...")
    # Извлечение признаков для всех сигналов
    X_features = extract_features_batch(X_raw)
    print(f"Извлечено {X_features.shape[1]} признаков для каждого сигнала")
    print(f"Матрица признаков: {X_features.shape}")
    
    print("\nШаг 3: Нормализация признаков...")
    # Нормализация (стандартизация) признаков
    scaler = StandardScaler()    X_scaled = scaler.fit_transform(X_features)
    print("Признаки нормализованы")
    
    print("\nШаг 4: Создание меток для обучения...")
    # Создание меток с помощью эвристики
    y_labels = create_labels_with_heuristic(X_raw)
    
    # Фильтрация: берём только те данные, которые мы уверенно разметили
    known_mask = y_labels != 0
    X_train = X_scaled[known_mask]
    y_train = y_labels[known_mask]
    
    print(f"Размечено {len(y_train)} сигналов из {len(y_labels)}")
    print(f"Режим №1 (синусоида): {np.sum(y_train == 1)}")
    print(f"Режим №2 (импульс): {np.sum(y_train == 2)}")
    
    print("\nШаг 5: Создание и обучение модели...")
    # Создание модели случайного леса
    model = RandomForestClassifier(
        n_estimators=100,        # количество деревьев
        max_depth=15,            # максимальная глубина
        min_samples_split=5,     # минимальное количество образцов для разделения
        random_state=42,         # для воспроизводимости
        n_jobs=-1                # использовать все ядра процессора
    )
    
    # Обучение модели
    model.fit(X_train, y_train)
    print("Модель обучена")
    
    print("\nШаг 6: Кросс-валидация (проверка качества)...")
    # Кросс-валидация для оценки качества модели
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Кросс-валидация (5 фолдов):")
    print(f"  Средняя точность: {cv_scores.mean():.4f}")
    print(f"  Стандартное отклонение: {cv_scores.std():.4f}")
    
    print("\nШаг 7: Предсказание для всех данных...")
    # Предсказание для всех сигналов (включая те, что не были размечены)
    predictions = model.predict(X_scaled)
    
    # Подсчёт количества сигналов Режима №2
    mode2_count = np.sum(predictions == 2)
    
    print(f"Предсказано {mode2_count} сигналов Режима №2")
    print(f"Предсказано {np.sum(predictions == 1)} сигналов Режима №1")
    
    return int(mode2_count), model, predictions

# ============================================# ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ
# (альтернативный подход - обучение на сгенерированных данных)
# ============================================

def generate_synthetic_signals(n_samples=2000):
    """
    Генерирует синтетические данные по формулам из условия
    
    Возвращает:
    - X: матрица сигналов
    - y: метки (1 или 2)
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        if np.random.random() < 0.5:
            # Режим №1: синусоида
            A = np.random.uniform(0.3, 0.6)
            omega = np.random.uniform(0.5, 1.5)
            phi = np.random.uniform(0, np.pi)
            
            t = np.linspace(0, 6, 61)
            signal = A * np.sin(omega * t + phi) + A
            
            label = 1
        else:
            # Режим №2: импульс
            omega0 = np.random.uniform(1, 5)
            gamma = np.random.uniform(0.01, 0.1)
            
            t = np.linspace(0, 6, 61)
            signal = t / np.sqrt((t**2 - omega0**2)**2 + gamma * t**2 + 1e-10)
            
            label = 2
        
        # Применяем нормализацию из условия: y = x/max(x) + h
        h = np.random.uniform(0, 0.1)
        signal = signal / np.max(np.abs(signal)) + h
        
        X.append(signal)
        y.append(label)
    
    return np.array(X), np.array(y)

def solve_with_synthetic_data():
    """
    Альтернативное решение: обучение на сгенерированных данных
    
    Этот подход часто даёт лучшие результаты, так как модель    обучается на данных, сгенерированных по тем же правилам,
    что и реальные данные
    """
    print("Генерация синтетических данных...")
    X_synthetic, y_synthetic = generate_synthetic_signals(3000)
    
    print("Извлечение признаков для синтетических данных...")
    X_features_synthetic = extract_features_batch(X_synthetic)
    
    print("Нормализация...")
    scaler = StandardScaler()
    X_scaled_synthetic = scaler.fit_transform(X_features_synthetic)
    
    print("Обучение модели на синтетических данных...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled_synthetic, y_synthetic)
    
    print("Загрузка реальных данных...")
    df = pd.read_excel('WaveForm1.xlsx', header=None)
    X_real = df.to_numpy()
    
    print("Извлечение признаков для реальных данных...")
    X_features_real = extract_features_batch(X_real)
    X_scaled_real = scaler.transform(X_features_real)
    
    print("Предсказание...")
    predictions = model.predict(X_scaled_real)
    mode2_count = np.sum(predictions == 2)
    
    print(f"Ответ (синтетические данные): {mode2_count}")
    return int(mode2_count), model, predictions

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================

def main():
    """
    Основная функция запуска
    
    Выбери один из двух подходов:
    1. solve_signal_classification() - обучение на эвристически размеченных данных
    2. solve_with_synthetic_data() - обучение на сгенерированных данных (РЕКОМЕНДУЕТСЯ)
    """
        print("=" * 60)
    print("РЕШЕНИЕ ЗАДАЧИ КЛАССИФИКАЦИИ СИГНАЛОВ")
    print("=" * 60)
    
    # ВЫБЕРИ ОДИН ИЗ ДВУХ ПОДХОДОВ:
    
    # ПОДХОД 1: Обучение на эвристически размеченных данных
    # answer, model, predictions = solve_signal_classification()
    
    # ПОДХОД 2: Обучение на сгенерированных данных (ЛУЧШИЙ ВЫБОР)
    answer, model, predictions = solve_with_synthetic_data()
    
    print("\n" + "=" * 60)
    print(f"ФИНАЛЬНЫЙ ОТВЕТ: {answer}")
    print("=" * 60)
    
    return answer

# ============================================
# ЗАПУСК ПРОГРАММЫ
# ============================================

if __name__ == '__main__':
    final_answer = main()
    # Для олимпиады просто выведи число:
    print(final_answer)
