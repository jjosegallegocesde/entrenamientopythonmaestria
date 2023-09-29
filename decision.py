import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Crea un DataFrame de ejemplo con datos de tecnologías y sus características
datos = {
    'Costo': [2, 3, 1, 5, 4, 7, 8, 6],
    'Capacidad': [3, 2, 4, 1, 2, 1, 1, 3],
    'Adecuada': ['Si', 'No', 'Si', 'No', 'Si', 'No', 'No', 'Si']
}
df = pd.DataFrame(datos)

# Divide los datos en características (X) y etiquetas (Y)
X = df[['Costo', 'Capacidad']]
Y = df['Adecuada']

# Divide los datos en conjuntos de entrenamiento y prueba
randomState = random.randint(1, 1000)
X_entrenado, X_prueba, Y_entrenado, Y_prueba = train_test_split(X, Y, test_size=0.3, random_state=45)

# Crea un modelo SVM
svm_model = SVC(kernel='linear')

# Entrena el modelo en el conjunto de entrenamiento
svm_model.fit(X_entrenado, Y_entrenado)

# Realiza predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_prueba)

# Calcula la precisión del modelo
accuracy = accuracy_score(Y_prueba, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Clasifica una nueva tecnología ficticia
nueva_tecnología = [[8, 4]]
clasificación = svm_model.predict(nueva_tecnología)
print(f"Clasificación de la nueva tecnología: {clasificación[0]}")

# Visualizar la clasificación
plt.figure(figsize=(10, 6))

# Usar una paleta de colores para visualizar la clasificación
colores = {'Si': 'blue', 'No': 'red'}
colores_entrenamiento = [colores[label] for label in Y_entrenado]
plt.scatter(X_entrenado['Costo'], X_entrenado['Capacidad'], c=colores_entrenamiento, marker='o', edgecolors='k', label='Datos de Entrenamiento')

# Graficar la clasificación de la nueva tecnología ficticia
plt.scatter(nueva_tecnología[0][0], nueva_tecnología[0][1], c='purple', marker='x', s=100, label='Nueva Tecnología')

plt.xlabel('Costo')
plt.ylabel('Capacidad')
plt.title('Clasificación SVM con kernel lineal')
plt.legend()
plt.show()
