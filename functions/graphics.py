import numpy as np
import matplotlib.pyplot as plt

# 1. Создаем 100 точек от -10 до 10
x = np.linspace(-10, 10, 100)

# 2. Вычисляем функцию (например, квадратичную или тригонометрическую)
y = np.sin(x)   

# 3. Строим линейный график
plt.plot(x, y, label='sin(x)', color='green', linestyle='-')
plt.grid(True) # добавляем сетку для удобства
plt.legend()   # показываем легенду
plt.axis('equal') # Закрепляем масштаб по осям
plt.show()

x = np.linspace(-2*np.pi, 2*np.pi, 200)

plt.plot(x, np.sin(x), label='Синус')
plt.plot(x, np.cos(x), label='Косинус', linestyle='--') # прерывистая линия

plt.title("Тригонометрические функции")
plt.grid(True)
plt.legend() # Чтобы понимать, где какая линия
plt.axis('equal') # Закрепляем масштаб по осям
plt.show()