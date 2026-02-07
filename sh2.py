"""
=====================================================
ПОЛНАЯ ШПАРГАЛКА ПО PYTHON ДЛЯ ОЛИМПИАД ПО ДАННЫМ
=====================================================
NumPy, Pandas, Matplotlib, SciPy, Scikit-learn
=====================================================
"""

# =====================================================
# NUMPY - ОСНОВЫ РАБОТЫ С МАССИВАМИ
# =====================================================

import numpy as np

# ===== СОЗДАНИЕ МАССИВОВ =====
# Создание массива из списка
arr = np.array([1, 2, 3, 4, 5])

# Создание массива с нулями
zeros = np.zeros((3, 4))  # матрица 3×4 из нулей

# Создание массива с единицами
ones = np.ones((2, 3))  # матрица 2×3 из единиц

# Создание массива со случайными числами
random_arr = np.random.rand(3, 3)  # случайные числа [0, 1]
random_int = np.random.randint(0, 10, size=(2, 5))  # случайные целые [0, 10)

# Создание массива с последовательностью
seq = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 равноотстоящих точек от 0 до 1

# Создание единичной матрицы
identity = np.eye(3)  # единичная матрица 3×3

# ===== ИНДЕКСИРОВАНИЕ И СРЕЗЫ =====
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr[0]      # первая строка: [1, 2, 3]
arr[:, 1]   # второй столбец: [2, 5, 8]
arr[1:3, 0:2]  # срез: [[4, 5], [7, 8]]

# Булево индексирование
arr[arr > 5]  # все элементы > 5: [6, 7, 8, 9]

# ===== ОПЕРАЦИИ С МАССИВАМИ =====
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b  # [5, 7, 9]a * b  # [4, 10, 18]
a ** 2  # [1, 4, 9]

# Поэлементные операции
np.sin(a)  # синус каждого элемента
np.exp(a)  # экспонента каждого элемента
np.log(a)  # натуральный логарифм

# ===== СТАТИСТИЧЕСКИЕ ОПЕРАЦИИ =====
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

np.mean(arr)      # среднее: 5.5
np.median(arr)    # медиана: 5.5
np.std(arr)       # стандартное отклонение
np.var(arr)       # дисперсия
np.min(arr)       # минимум: 1
np.max(arr)       # максимум: 10
np.sum(arr)       # сумма: 55
np.prod(arr)      # произведение

# ===== ЛИНЕЙНАЯ АЛГЕБРА =====
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B           # матричное умножение
np.dot(A, B)    # тоже матричное умножение
np.transpose(A) # транспонирование
np.linalg.inv(A) # обратная матрица
np.linalg.det(A) # определитель
np.linalg.eig(A) # собственные значения и векторы

# ===== ПОЛЕЗНЫЕ ФУНКЦИИ =====
np.reshape(arr, (2, 5))  # изменение формы
np.flatten(arr)          # преобразование в 1D
np.concatenate([a, b])   # объединение массивов
np.vstack([a, b])        # вертикальное объединение
np.hstack([a, b])        # горизонтальное объединение
np.unique(arr)           # уникальные значения
np.sort(arr)             # сортировка
np.argsort(arr)          # индексы для сортировки

# =====================================================
# PANDAS - РАБОТА С ТАБЛИЧНЫМИ ДАННЫМИ
# =====================================================

import pandas as pd

# ===== СОЗДАНИЕ DataFrame =====
# Из словаря
df = pd.DataFrame({    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Из numpy массива
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# ===== ЗАГРУЗКА ДАННЫХ =====
# Из CSV
df = pd.read_csv('data.csv')

# Из Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1', header=0)

# Из JSON
df = pd.read_json('data.json')

# Из SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table", conn)

# ===== ПРОСМОТР ДАННЫХ =====
df.head()        # первые 5 строк
df.tail()        # последние 5 строк
df.info()        # информация о данных
df.describe()    # статистическое описание
df.shape         # размер (строки, столбцы)
df.columns       # названия столбцов
df.dtypes        # типы данных

# ===== ИНДЕКСИРОВАНИЕ =====
df['A']          # столбец A
df[['A', 'B']]   # столбцы A и B
df.loc[0]        # строка с индексом 0
df.iloc[0]       # первая строка
df.loc[0, 'A']   # элемент на пересечении строки 0 и столбца A
df.iloc[0, 0]    # первый элемент

# ===== ФИЛЬТРАЦИЯ =====
df[df['A'] > 2]              # строки где A > 2
df[(df['A'] > 2) & (df['B'] < 5)]  # несколько условий
df[df['A'].isin([1, 3])]     # A равно 1 или 3

# ===== ДОБАВЛЕНИЕ/УДАЛЕНИЕ СТОЛБЦОВ =====
df['D'] = df['A'] + df['B']  # новый столбец
df['E'] = [10, 20, 30]       # новый столбец со значениями
df.drop('C', axis=1)         # удалить столбец Cdf.drop(0, axis=0)           # удалить строку с индексом 0

# ===== ГРУППИРОВКА И АГРЕГАЦИЯ =====
df.groupby('A').mean()       # среднее по группам A
df.groupby('A').agg({'B': 'sum', 'C': 'mean'})
df.pivot_table(values='B', index='A', aggfunc='mean')

# ===== СОРТИРОВКА =====
df.sort_values('A')          # сортировка по столбцу A
df.sort_values(['A', 'B'], ascending=[True, False])

# ===== РАБОТА С ПРОПУЩЕННЫМИ ЗНАЧЕНИЯМИ =====
df.isnull()                  # маска пропущенных значений
df.dropna()                  # удалить строки с пропусками
df.fillna(0)                 # заполнить нулями
df.fillna(df.mean())         # заполнить средним

# ===== ОБЪЕДИНЕНИЕ ТАБЛИЦ =====
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [1, 2], 'C': [5, 6]})

pd.merge(df1, df2, on='A')   # объединение по столбцу A
pd.concat([df1, df2])        # вертикальное объединение

# =====================================================
# MATPLOTLIB - ВИЗУАЛИЗАЦИЯ ДАННЫХ
# =====================================================

import matplotlib.pyplot as plt

# ===== НАСТРОЙКА =====
plt.figure(figsize=(10, 6))  # размер фигуры
plt.rcParams['font.size'] = 12  # размер шрифта

# ===== ЛИНЕЙНЫЙ ГРАФИК =====
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График синуса')
plt.legend()
plt.grid(True)
plt.show()

# ===== ТОЧЕЧНАЯ ДИАГРАММА =====
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y, c='red', s=50, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Точечная диаграмма')
plt.show()

# ===== ГИСТОГРАММА =====
data = np.random.randn(1000)

plt.hist(data, bins=30, color='green', alpha=0.7)
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма')
plt.show()

# ===== ЯЩИК С УСАМИ (BOXPLOT) =====
data = [np.random.randn(100) for _ in range(3)]

plt.boxplot(data, labels=['A', 'B', 'C'])
plt.ylabel('Значение')
plt.title('Ящик с усами')
plt.show()

# ===== МАТРИЦА КОРРЕЛЯЦИЙ =====
corr_matrix = df.corr()
plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Матрица корреляций')
plt.show()

# ===== НЕСКОЛЬКО ГРАФИКОВ =====
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')

axes[1, 0].scatter(x, np.random.rand(100))
axes[1, 0].set_title('Scatter')

axes[1, 1].hist(np.random.randn(1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
# =====================================================
# SCIPY - НАУЧНЫЕ ВЫЧИСЛЕНИЯ
# =====================================================

import scipy.stats as stats
import scipy.signal as signal
from scipy.optimize import minimize, curve_fit
from scipy.integrate import quad
from scipy.fft import fft, fftfreq

# ===== СТАТИСТИКА =====
data = np.random.randn(100)

# Описательная статистика
stats.describe(data)

# Тесты на нормальность
stats.shapiro(data)  # тест Шапиро-Уилка
stats.normaltest(data)  # тест на нормальность

# Проверка гипотез
stats.ttest_ind(data[:50], data[50:])  # t-тест
stats.f_oneway(data[:33], data[33:66], data[66:])  # ANOVA

# Распределения
stats.norm.pdf(x, loc=0, scale=1)  # PDF нормального распределения
stats.norm.cdf(x, loc=0, scale=1)  # CDF нормального распределения
stats.norm.rvs(size=100)  # случайные числа из нормального распределения

# ===== ОБРАБОТКА СИГНАЛОВ =====
# Фильтрация
butter_filter = signal.butter(3, 0.1, 'low', output='sos')
filtered = signal.sosfilt(butter_filter, data)

# Поиск пиков
peaks, _ = signal.find_peaks(data, height=0.5, distance=5)

# Сглаживание
smoothed = signal.savgol_filter(data, window_length=11, polyorder=2)

# Свертка
kernel = np.ones(5) / 5
convolved = signal.convolve(data, kernel, mode='same')

# Быстрое преобразование Фурье
freq = fftfreq(len(data), d=0.1)
fft_vals = fft(data)

# ===== ОПТИМИЗАЦИЯ =====
# Минимизация функцииdef func(x):
    return x[0]**2 + x[1]**2

result = minimize(func, x0=[1, 1])
print(result.x)  # оптимальные параметры

# Подгонка кривой
def model(x, a, b, c):
    return a * np.exp(-b * x) + c

x_data = np.linspace(0, 4, 50)
y_data = model(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

params, _ = curve_fit(model, x_data, y_data)
print(params)  # [a, b, c]

# ===== ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ =====
def integrand(x):
    return x**2

result, error = quad(integrand, 0, 1)
print(result)  # интеграл от 0 до 1

# =====================================================
# SCIKIT-LEARN - МАШИННОЕ ОБУЧЕНИЕ
# =====================================================

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ===== РЕГРЕССИЯ =====
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(f"R²: {lr.score(X_test, y_test):.3f}")

# Регуляризация
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Деревья и ансамбли
dt = DecisionTreeRegressor(max_depth=5)
rf = RandomForestRegressor(n_estimators=100)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

# ===== КЛАССИФИКАЦИЯ =====
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Логистическая регрессия
log_reg = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
proba = log_reg.predict_proba(X_test)  # вероятности классов

# Деревья решений
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
dt.fit(X_train, y_train)

# Случайный лес (РЕКОМЕНДУЕТСЯ для олимпиад)
rf = RandomForestClassifier(
    n_estimators=100,      # количество деревьев
    max_depth=15,          # максимальная глубина
    min_samples_split=5,   # минимальное количество для разделения
    random_state=42,
    n_jobs=-1              # использовать все ядра
)
rf.fit(X_train, y_train)

# Градиентный бустинг
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)

# K-ближайших соседей
knn = KNeighborsClassifier(n_neighbors=5)knn.fit(X_train, y_train)

# Наивный байес
nb = GaussianNB()
nb.fit(X_train, y_train)

# ===== ПОДГОТОВКА ДАННЫХ =====
# Разделение на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Масштабирование в [0, 1]
minmax = MinMaxScaler()
X_scaled = minmax.fit_transform(X)

# Кодирование категориальных переменных
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_categorical)

# Отбор признаков
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Рекурсивное исключение признаков
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# PCA (снижение размерности)
pca = PCA(n_components=0.95)  # сохранить 95% дисперсии
X_pca = pca.fit_transform(X_scaled)

# ===== ОЦЕНКА КАЧЕСТВА =====
# Классификация
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Подробный отчет
print(classification_report(y_test, y_pred))

# ROC-кривая (для бинарной классификации)
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Регрессия
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# ===== КРОСС-ВАЛИДАЦИЯ =====
# Простая кросс-валидация
scores = cross_val_score(model, X, y, cv=5)
print(f"CV scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Стратифицированная кросс-валидация
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# ===== ПОДБОР ГИПЕРПАРАМЕТРОВ =====
# Grid Searchparam_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
print(f"Лучшие параметры: {grid.best_params_}")
print(f"Лучший скор: {grid.best_score_:.3f}")

best_model = grid.best_estimator_

# Randomized Search (быстрее)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# ===== ПАЙПЛАЙНЫ =====
# Создание пайплайна
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)predictions = pipeline.predict(X_test)

# Пайплайн с поиском параметров
param_grid = {
    'pca__n_components': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200]
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)

# ===== КЛАСТЕРИЗАЦИЯ =====
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Оценка качества кластеризации
silhouette = silhouette_score(X, clusters)
print(f"Silhouette Score: {silhouette:.3f}")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Иерархическая кластеризация
agg = AgglomerativeClustering(n_clusters=3)
clusters = agg.fit_predict(X)

# =====================================================
# ПОЛЕЗНЫЕ УТИЛИТЫ И СОВЕТЫ
# =====================================================

# ===== СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛЕЙ =====
import joblib
import pickle

# Сохранение модели
joblib.dump(model, 'model.pkl')
# или
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Загрузка модели
model = joblib.load('model.pkl')
# или
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# ===== РАБОТА С ВРЕМЕННЫМИ РЯДАМИ =====
import pandas as pd

# Создание временного ряда
dates = pd.date_range('2020-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Скользящее среднее
rolling_mean = ts.rolling(window=7).mean()

# Сезонная декомпозиция
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts, model='additive', period=7)

# Автокорреляция
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts)
plot_pacf(ts)

# ===== ОБРАБОТКА ТЕКСТА =====
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Count Vectorizer
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(texts)

# ===== ВИЗУАЛИЗАЦИЯ МОДЕЛЕЙ =====
# Важность признаков (для деревьев)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Важность признаков")
plt.bar(range(10), importances[indices[:10]])
plt.xticks(range(10), feature_names[indices[:10]], rotation=45)
plt.tight_layout()
plt.show()

# Кривая обучения
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Train')
plt.plot(train_sizes, test_mean, label='Test')
plt.xlabel('Размер обучающей выборки')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# =====================================================
# ЧЕК-ЛИСТ ДЛЯ ОЛИМПИАДНЫХ ЗАДАЧ
# =====================================================

"""
✅ ЗАГРУЗКА ДАННЫХ
- [ ] Проверить формат файла (CSV, Excel, JSON)
- [ ] Проверить наличие заголовков
- [ ] Проверить типы данных

✅ РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA)
- [ ] Посмотреть первые/последние строки
- [ ] Проверить пропущенные значения
- [ ] Посмотреть статистику
- [ ] Построить визуализации

✅ ПОДГОТОВКА ДАННЫХ
- [ ] Обработать пропущенные значения
- [ ] Закодировать категориальные переменные
- [ ] Нормализовать/масштабировать признаки
- [ ] Отобрать важные признаки

✅ ВЫБОР МОДЕЛИ
- [ ] Для классификации: начать с Random Forest
- [ ] Для регрессии: начать с Linear Regression
- [ ] Для кластеризации: начать с K-Means

✅ ОБУЧЕНИЕ И ВАЛИДАЦИЯ
- [ ] Разделить данные на обучение/тест
- [ ] Использовать кросс-валидацию
- [ ] Подобрать гиперпараметры

✅ ОЦЕНКА КАЧЕСТВА
- [ ] Вычислить метрики (accuracy, F1, MSE и т.д.)
- [ ] Построить матрицу ошибок
- [ ] Проанализировать важность признаков

✅ СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
- [ ] Сохранить модель
- [ ] Сохранить предсказания- [ ] Подготовить финальный ответ
"""

# =====================================================
# БЫСТРЫЙ СТАРТ ДЛЯ ОЛИМПИАДЫ
# =====================================================

"""
# 1. ЗАГРУЗКА ДАННЫХ
import pandas as pd
df = pd.read_excel('data.xlsx', header=None)
X = df.to_numpy()

# 2. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
def extract_features(signals):
    features = []
    for signal in signals:
        features.append([
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            np.su
