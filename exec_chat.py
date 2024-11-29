from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from kfoldchat import *
from chat_algorithm import *

# Cargar y preprocesar los datos
data = pd.read_csv('processed.cleveland.data', header=None)
data.replace("?", pd.NA, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.median(), inplace=True)
data[13] = data[13].apply(lambda x: 0 if x == 0 else 1)
scaler = MinMaxScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

X = data.iloc[:, :-1].values  # Solo características
y = data[13].values  # Clases binarias (sano/enfermo)

# Validación cruzada con 10 pliegues
kf_validation = KFoldCHATValidation(n_splits=10, random_state=42)
resultados = kf_validation.evaluate(X, y, CHATAlgorithm)
print("\nResultados de la validación cruzada con 10 pliegues:")
print("Precisión promedio:", resultados["accuracy"])
print("Precisión por clase promedio:", resultados["precision"])
print("Recall promedio:", resultados["recall"])
print("F1-score promedio:", resultados["f1_score"])
