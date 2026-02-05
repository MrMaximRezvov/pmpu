# # import numpy as np
# # import pandas as pd
# # from scipy.signal import find_peaks

# # df = pd.read_excel('classification/WaveForm1.xlsx', header=None)
# # X_main = df.to_numpy()

# # mode2_count = 0

# # for row in X_main:
# #     # Поиск пиков с фильтрацией шума (минимальная высота 0.05)
# #     max_peak, _ = find_peaks(row, distance=1, height=0.01)
# #     num_peaks = len(max_peak)
    
# #     # Вычисление первой производной
# #     deriv1 = np.diff(row)
    
# #     # Нулевые пересечения первой производной
# #     # Это количество раз, когда первая производная меняет знак
# #     zero_crossings = np.sum(np.diff(np.sign(deriv1)) != 0)
    
# #     # Классификация по ключевым признакам:
# #     # - Импульс: 1 пик + 2 нулевых пересечения (рост и падение)
# #     # - Синусоида: ≥2 пика + ≥4 нулевых пересечения
# #     if num_peaks == 1 and zero_crossings == 2:
# #         mode2_count += 1

# # print(f"Количество экспериментов в Режиме №2: {mode2_count}")

# import numpy as np
# import pandas as pd
# from scipy.signal import find_peaks

# # Исправлено: убран лишний пробел перед именем файла
# df = pd.read_excel('classification/WaveForm1.xlsx', header=None)
# X_main = df.to_numpy()

# max_peaks = []
# zero_crossings = []

# for row in X_main:
#     # Поиск пиков с минимальной высотой для фильтрации шума
#     max_peak, _ = find_peaks(row, distance=5, height=0.05)
#     max_peaks.append(len(max_peak))
    
#     # Вычисление первой производной
#     deriv1 = np.diff(row)
    
#     # Нулевые пересечения первой производной
#     crossings = np.sum(np.diff(np.sign(deriv1)) != 0)
#     zero_crossings.append(crossings)

# # Классификация: 
# # Режим №2 (импульс): 1 пик + 2 нулевых пересечения
# # Режим №1 (синусоида): ≥2 пика + ≥4 нулевых пересечения
# mode2_indices = []
# for i in range(len(X_main)):
#     if max_peaks[i] == 1 and zero_crossings[i] == 2:
#         mode2_indices.append(i)

# # Вывод результата
# print(len(mode2_indices))

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

df = pd.read_excel('classification/WaveForm1.xlsx', header = None)

X_main = df.to_numpy()
max_peaks = []
for row in X_main:
    max_peak, _ = find_peaks(row, distance=5)
    max_peaks.append(len(max_peak))

dataDeriv = np.diff(X_main)
dataDeriv2 = np.diff(dataDeriv)

zero_crossings = [len(np.where(np.diff(np.sign(dataRow)))[0]) for dataRow in dataDeriv]
num_of_neg = [len(np.where(np.diff(dataRow) == -1)[0]) for dataRow in dataDeriv2]

print(len(np.intersect1d(np.intersect1d(np.where(np.array(max_peaks)==1)[0],
np.where(np.array(zero_crossings)==1)[0]),
np.where(np.array(num_of_neg) < 10)[0])))