import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn as skl
from sklearn.linear_model import LinearRegression 

datamodel = pd.read_excel("DataModel1.xls", header = None)

print(datamodel)

# xarr = datamodel.columns[0]
# yarr = datamodel.columns[1]

xarr = datamodel.iloc[:, 0] 
yarr = datamodel.iloc[:, 1]

plt.figure(figsize=(8, 6)) # задаем размер окна
plt.scatter(xarr, yarr, s=1, c='blue') # s - размер точки, c - цвет
plt.title("Орбита небесного тела")
plt.xlabel("Координата X")
plt.ylabel("Координата Y")
plt.axis('equal') # Важно! Чтобы круги не превращались в овалы из-за масштаба осей
plt.show()

print(xarr, type(xarr))
print(yarr, type(yarr))

x = xarr
y = yarr

Sx4 = (x**4).sum()
Sx2y2 = (x**2 * y**2).sum()
Sx2 = (x**2).sum()
Sy4 = (y**4).sum()
Sy2 = (y**2).sum()

A = [[Sx4, Sx2y2],
     [Sx2y2, Sy4]]
B = [[Sx2],
     [Sy2]]

u, v = np.linalg.solve(A, B)
print(u, v)

a = np.sqrt(1/u)
print(a[0])

# Решение с помощью модели и обучения

X_feat = np.column_stack([x**2, y**2])

target = np.ones(len(x))

model = LinearRegression(fit_intercept=False)
model.fit(X_feat, target)

print(model.coef_)

a = np.sqrt(1/model.coef_[0])

print(a)