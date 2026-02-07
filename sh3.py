"""
=====================================================
РАБОТА С ПРОПУЩЕННЫМИ ЗНАЧЕНИЯМИ (NaN)
=====================================================
"""

import numpy as np
import pandas as pd

# =====================================================
# СОЗДАНИЕ ДАННЫХ С ПРОПУЩЕННЫМИ ЗНАЧЕНИЯМИ
# =====================================================

# Создание DataFrame с NaN
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, 4, 5],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan]
})

print("Исходный DataFrame:")
print(df)
print()

# =====================================================
# ОБНАРУЖЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# =====================================================

# Проверка на наличие NaN
print("df.isnull():")
print(df.isnull())
print()

# Проверка на отсутствие NaN
print("df.notnull():")
print(df.notnull())
print()

# Подсчёт пропущенных значений
print("Количество пропущенных по столбцам:")
print(df.isnull().sum())
print()

print("Общее количество пропущенных:")
print(df.isnull().sum().sum())
print()

print("Процент пропущенных по столбцам:")
print(df.isnull().mean() * 100)print()

# Проверка, есть ли хоть один NaN в столбце/строке
print("Столбцы с пропущенными значениями:")
print(df.columns[df.isnull().any()])
print()

print("Строки с пропущенными значениями:")
print(df[df.isnull().any(axis=1)])
print()

# =====================================================
# УДАЛЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# =====================================================

# Удаление строк с хотя бы одним NaN
df_dropped_rows = df.dropna()
print("Удалены строки с NaN:")
print(df_dropped_rows)
print()

# Удаление столбцов с хотя бы одним NaN
df_dropped_cols = df.dropna(axis=1)
print("Удалены столбцы с NaN:")
print(df_dropped_cols)
print()

# Удаление только если все значения в строке NaN
df_dropped_all = df.dropna(how='all')
print("Удалены только полностью пустые строки:")
print(df_dropped_all)
print()

# Удаление только если все значения в столбце NaN
df_dropped_all_cols = df.dropna(axis=1, how='all')
print("Удалены только полностью пустые столбцы:")
print(df_dropped_all_cols)
print()

# Удаление строк, где меньше 3 не-NaN значений
df_dropped_thresh = df.dropna(thresh=3)
print("Удалены строки с < 3 не-NaN значениями:")
print(df_dropped_thresh)
print()

# Удаление строк с пропусками только в определённых столбцах
df_dropped_subset = df.dropna(subset=['A', 'B'])
print("Удалены строки с пропусками в A или B:")
print(df_dropped_subset)
print()
# =====================================================
# ЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# =====================================================

# Заполнение конкретным значением
df_filled_const = df.fillna(0)
print("Заполнено нулями:")
print(df_filled_const)
print()

# Заполнение разными значениями для разных столбцов
df_filled_dict = df.fillna({'A': 0, 'B': -1, 'C': 999})
print("Заполнено разными значениями:")
print(df_filled_dict)
print()

# Заполнение средним значением
df_filled_mean = df.fillna(df.mean())
print("Заполнено средним:")
print(df_filled_mean)
print()

# Заполнение медианой
df_filled_median = df.fillna(df.median())
print("Заполнено медианой:")
print(df_filled_median)
print()

# Заполнение модой (наиболее частым значением)
df_filled_mode = df.fillna(df.mode().iloc[0])
print("Заполнено модой:")
print(df_filled_mode)
print()

# Заполнение минимальным/максимальным значением
df_filled_min = df.fillna(df.min())
df_filled_max = df.fillna(df.max())

# Заполнение прямой (вперёд) - последним известным значением
df_filled_ffill = df.fillna(method='ffill')
print("Заполнено прямой (ffill):")
print(df_filled_ffill)
print()

# Заполнение обратной (назад) - следующим известным значением
df_filled_bfill = df.fillna(method='bfill')
print("Заполнено обратной (bfill):")
print(df_filled_bfill)
print()
# Заполнение интерполяцией
df_filled_interp = df.interpolate(method='linear')
print("Заполнено линейной интерполяцией:")
print(df_filled_interp)
print()

# Заполнение полиномиальной интерполяцией
df_filled_poly = df.interpolate(method='polynomial', order=2)
print("Заполнено полиномиальной интерполяцией:")
print(df_filled_poly)
print()

# =====================================================
# ПРОДВИНУТЫЕ МЕТОДЫ ЗАПОЛНЕНИЯ
# =====================================================

# Заполнение с ограничением количества заполненных значений
df_filled_limit = df.fillna(method='ffill', limit=1)
print("Заполнено с ограничением (максимум 1):")
print(df_filled_limit)
print()

# Заполнение с использованием группировки
df_grouped = pd.DataFrame({
    'group': ['A', 'A', 'A', 'B', 'B', 'B'],
    'value': [1, np.nan, 3, 4, np.nan, 6]
})

df_grouped_filled = df_grouped.groupby('group').transform(lambda x: x.fillna(x.mean()))
print("Заполнено средним по группе:")
print(df_grouped_filled)
print()

# Заполнение с использованием скользящего среднего
df_rolling = pd.DataFrame({'value': [1, 2, np.nan, 4, 5, np.nan, 7]})
df_rolling['filled'] = df_rolling['value'].fillna(
    df_rolling['value'].rolling(window=3, min_periods=1).mean()
)
print("Заполнено скользящим средним:")
print(df_rolling)
print()

# =====================================================
# ЗАМЕНА КОНКРЕТНЫХ ЗНАЧЕНИЙ
# =====================================================

# Замена одного значения на другое
df_replace_single = df.replace(np.nan, -999)
print("Замена NaN на -999:")print(df_replace_single)
print()

# Замена нескольких значений
df_replace_multi = df.replace({np.nan: -999, 1: 100})
print("Замена NaN на -999 и 1 на 100:")
print(df_replace_multi)
print()

# Замена с использованием регулярных выражений (для строк)
df_strings = pd.DataFrame({'text': ['hello', 'world', 'missing', 'test']})
df_strings_replaced = df_strings.replace('missing', np.nan)
print("Замена строки на NaN:")
print(df_strings_replaced)
print()

# =====================================================
# РАБОТА С БЕСКОНЕЧНОСТЯМИ (inf, -inf)
# =====================================================

# Создание данных с бесконечностями
df_inf = pd.DataFrame({
    'A': [1, 2, np.inf, 4, -np.inf],
    'B': [1, 2, 3, 4, 5]
})

print("Исходные данные с бесконечностями:")
print(df_inf)
print()

# Замена бесконечностей на NaN
df_inf_replaced = df_inf.replace([np.inf, -np.inf], np.nan)
print("Бесконечности заменены на NaN:")
print(df_inf_replaced)
print()

# Замена бесконечностей на конкретное значение
df_inf_filled = df_inf.replace([np.inf, -np.inf], 999)
print("Бесконечности заменены на 999:")
print(df_inf_filled)
print()

# =====================================================
# ПОЛНЫЙ ПАЙПЛАЙН ОБРАБОТКИ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# =====================================================

def clean_missing_values(df, strategy='auto'):
    """
    Полный пайплайн обработки пропущенных значений
        Параметры:
    - df: DataFrame для очистки
    - strategy: стратегия заполнения ('mean', 'median', 'mode', 'ffill', 'bfill', 'interpolate', 'auto')
    
    Возвращает:
    - Очищенный DataFrame
    - Словарь с информацией о пропущенных значениях
    """
    # Сохраняем информацию о пропусках
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Копируем DataFrame, чтобы не изменять оригинал
    df_clean = df.copy()
    
    if strategy == 'auto':
        # Автоматический выбор стратегии для каждого столбца
        for column in df_clean.columns:
            missing_pct = df_clean[column].isnull().mean()
            
            if missing_pct > 0.5:
                # Если пропущено больше 50% - удаляем столбец
                df_clean = df_clean.drop(columns=[column])
                missing_info[f'{column}_action'] = 'dropped (too many missing)'
            elif missing_pct > 0.1:
                # Если пропущено 10-50% - заполняем медианой/модой
                if df_clean[column].dtype in ['float64', 'int64']:
                    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                    missing_info[f'{column}_action'] = 'filled with median'
                else:
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
                    missing_info[f'{column}_action'] = 'filled with mode'
            else:
                # Если пропущено < 10% - заполняем средним
                if df_clean[column].dtype in ['float64', 'int64']:
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                    missing_info[f'{column}_action'] = 'filled with mean'
                else:
                    df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
                    missing_info[f'{column}_action'] = 'filled with mode'
    
    elif strategy == 'mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    
    elif strategy == 'median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif strategy == 'mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    elif strategy == 'ffill':
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'bfill':
        df_clean = df_clean.fillna(method='bfill').fillna(method='ffill')
    
    elif strategy == 'interpolate':
        df_clean = df_clean.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean, missing_info

# Пример использования
df_test = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5, np.nan],
    'B': [np.nan, 2, 3, 4, np.nan, 6],
    'C': [1, 2, 3, 4, 5, 6]
})

print("Исходные данные:")
print(df_test)
print()

cleaned_df, info = clean_missing_values(df_test, strategy='auto')
print("Очищенные данные:")
print(cleaned_df)
print()

print("Информация о пропусках:")
for key, value in info.items():
    print(f"{key}: {value}")
print()

# =====================================================
# ВИЗУАЛИЗАЦИЯ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# =====================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Тепловая карта пропущенных значений
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Тепловая карта пропущенных значений')
plt.show()
# Гистограмма количества пропущенных по столбцам
plt.figure(figsize=(10, 6))
missing_counts = df.isnull().sum()
missing_counts.plot(kind='bar')
plt.title('Количество пропущенных значений по столбцам')
plt.ylabel('Количество пропущенных')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Гистограмма процентов пропущенных
plt.figure(figsize=(10, 6))
missing_pct = df.isnull().mean() * 100
missing_pct.plot(kind='bar')
plt.title('Процент пропущенных значений по столбцам')
plt.ylabel('Процент (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================================
# ПРОДВИНУТЫЕ МЕТОДЫ ИМПУТАЦИИ
# =====================================================

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# SimpleImputer - простая импутация
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
imputer_constant = SimpleImputer(strategy='constant', fill_value=-999)

# Пример использования
df_numeric = df[['A', 'B', 'C']].copy()
df_imputed = pd.DataFrame(
    imputer_mean.fit_transform(df_numeric),
    columns=df_numeric.columns
)
print("Imputed with mean (SimpleImputer):")
print(df_imputed)
print()

# KNNImputer - импутация с использованием K-ближайших соседей
knn_imputer = KNNImputer(n_neighbors=3)
df_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)
print("Imputed with KNN:")print(df_knn_imputed)
print()

# IterativeImputer - итеративная импутация (экспериментальная)
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# 
# iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
# df_iterative_imputed = pd.DataFrame(
#     iterative_imputer.fit_transform(df_numeric),
#     columns=df_numeric.columns
# )

# =====================================================
# ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ В РЕАЛЬНОМ ПРОЕКТЕ
# =====================================================

def comprehensive_data_cleaning(df):
    """
    Комплексная очистка данных
    
    Шаги:
    1. Обнаружение и анализ пропущенных значений
    2. Обработка бесконечностей
    3. Удаление дубликатов
    4. Заполнение пропусков
    5. Проверка результатов
    """
    print("=" * 60)
    print("НАЧАЛО ОЧИСТКИ ДАННЫХ")
    print("=" * 60)
    
    # Шаг 1: Анализ пропущенных значений
    print("\n1. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
    print(f"Общее количество пропущенных: {df.isnull().sum().sum()}")
    print(f"Процент пропущенных: {(df.isnull().sum().sum() / df.size * 100):.2f}%")
    
    missing_by_col = df.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    
    if len(cols_with_missing) > 0:
        print(f"\nСтолбцы с пропущенными значениями:")
        for col, count in cols_with_missing.items():
            pct = count / len(df) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
    
    # Шаг 2: Обработка бесконечностей
    print("\n2. ОБРАБОТКА БЕСКОНЕЧНОСТЕЙ")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:        print(f"Найдено {inf_count} бесконечных значений")
        df = df.replace([np.inf, -np.inf], np.nan)
        print("Бесконечности заменены на NaN")
    else:
        print("Бесконечностей не найдено")
    
    # Шаг 3: Удаление дубликатов
    print("\n3. УДАЛЕНИЕ ДУБЛИКАТОВ")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Найдено {duplicates} дубликатов")
        df = df.drop_duplicates()
        print("Дубликаты удалены")
    else:
        print("Дубликатов не найдено")
    
    # Шаг 4: Заполнение пропусков
    print("\n4. ЗАПОЛНЕНИЕ ПРОПУСКОВ")
    
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            dtype = df[column].dtype
            
            if missing_count / len(df) > 0.5:
                print(f"  {column}: удалён (слишком много пропусков)")
                df = df.drop(columns=[column])
            elif dtype in ['float64', 'int64']:
                median_val = df[column].median()
                df[column] = df[column].fillna(median_val)
                print(f"  {column}: заполнен медианой ({median_val:.2f})")
            elif dtype == 'object':
                mode_val = df[column].mode()[0] if len(df[column].mode()) > 0 else 'UNKNOWN'
                df[column] = df[column].fillna(mode_val)
                print(f"  {column}: заполнен модой ('{mode_val}')")
            else:
                df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
                print(f"  {column}: заполнен прямой/обратной")
    
    # Шаг 5: Финальная проверка
    print("\n5. ФИНАЛЬНАЯ ПРОВЕРКА")
    final_missing = df.isnull().sum().sum()
    if final_missing == 0:
        print("✓ Все пропущенные значения обработаны!")
    else:
        print(f"⚠ Осталось {final_missing} пропущенных значений")
    
    print(f"\nИтоговый размер данных: {df.shape}")
    print("=" * 60)
        return df

# Пример использования
df_real = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40, np.inf],
    'income': [50000, 60000, 55000, np.nan, 70000, 65000],
    'city': ['NYC', 'LA', np.nan, 'NYC', 'LA', 'NYC'],
    'score': [85, 90, 88, 92, np.nan, 87]
})

print("Исходные данные:")
print(df_real)
print()

df_cleaned = comprehensive_data_cleaning(df_real)

print("\nОчищенные данные:")
print(df_cleaned)
