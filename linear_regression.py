import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Tworzymy sztuczne dane
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10]) 

#Tworzymy i trenujemy model regresji liniowej
model = LinearRegression()
model.fit(X, y)

#Przewidywanie wartości dla X
y_pred = model.predict(X)

#Wizualizacja
plt.scatter(X, y, label="Dane rzeczywiste", color="blue")
plt.plot(X, y_pred, label="Regresja liniowa", color="red")
plt.xlabel("Cecha")
plt.ylabel("Wartość")
plt.legend()
plt.show()