import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Cargar y preprocesar los datos
def load_and_preprocess_data(filename):
    # Cargar datos, reemplazar "?" con NaN y luego imputar con la media de la columna
    data = pd.read_csv(filename, header=None, na_values="?")
    data.fillna(data.mean(), inplace=True)
    
    # Extraer características (X) y clases (y)
    X = data.iloc[:, :-1].values  # Todas las columnas menos la última
    y = data.iloc[:, -1].values   # Última columna es la clase
    
    # Normalización de los datos (traslación de ejes)
    X_normalized = X - np.mean(X, axis=0)
    
    return X_normalized, y

# 2. Codificación One-Hot para las clases
def encode_classes(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    return y_encoded, encoder

# 3. Fase de Aprendizaje: generar matriz de pesos (similar a Linear Associator)
def train_CHAT(X_train, y_train_encoded):
    # Calcular la matriz de pesos: M = sum(y * x.T)
    num_features = X_train.shape[1]
    num_classes = y_train_encoded.shape[1]
    
    # Inicializar matriz de pesos
    M = np.zeros((num_classes, num_features))
    
    # Acumular las asociaciones entre patrones de entrada y clases
    for i in range(X_train.shape[0]):
        M += np.outer(y_train_encoded[i], X_train[i])
    
    return M

# 4. Fase de Clasificación
def classify_CHAT(X_test, M):
    # Multiplicar los patrones de prueba por la matriz de pesos
    y_pred_scores = np.dot(X_test, M.T)
    
    # Seleccionar la clase con la puntuación más alta
    y_pred = np.argmax(y_pred_scores, axis=1)
    
    return y_pred

# 5. Evaluación del modelo
def evaluate_model(y_test_encoded, y_pred, encoder):
    # Decodificar las etiquetas reales (si y_test_encoded está en formato one-hot)
    y_test_decoded = np.argmax(y_test_encoded, axis=1)  # Decodificar y_test
    # Calcular precisión
    accuracy = accuracy_score(y_test_decoded, y_pred)
    return accuracy


# 6. Ejecución del algoritmo
def main(filename):
    # Cargar y preprocesar los datos
    X, y = load_and_preprocess_data(filename)
    
    # Codificar las clases en One-Hot
    y_encoded, encoder = encode_classes(y)
    
    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    # Fase de Aprendizaje (entrenar CHAT)
    M = train_CHAT(X_train, y_train)
    
    # Fase de Clasificación (clasificar patrones de prueba)
    y_pred = classify_CHAT(X_test, M)
    
    # Evaluación del modelo
    accuracy = evaluate_model(y_test, y_pred, encoder)
    print(f"Precisión del modelo CHAT: {accuracy * 100:.2f}%")

# Llamada al main (aquí se proporciona el nombre del archivo CSV)
# filename = "processed.cleveland.data"
# filename = "processed.switzerland.data"
filename = "reprocessed.hungarian.data"
main(filename)
