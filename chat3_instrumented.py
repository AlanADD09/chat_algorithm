import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Función para cargar el archivo y eliminar filas con "?"
def load_data_to_dataframe(file_path):
    df = pd.read_csv(file_path, header=None)
    df.replace(["?", "-9"], np.nan, inplace=True)
    # Convertir columnas a numéricas y llenar NaN con la mediana
    df = df.apply(pd.to_numeric, errors='coerce')
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
    # Normalización y eliminación de NaN
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df.dropna(inplace=True)
    return df

# Normalización de los patrones
def normalize_patterns(X):
    epsilon = 1e-8
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon)

# Aplicar la traducción a los datos
def apply_translation(X):
    print('Translation: Mean values:', X.mean(axis=0))
    print('Translated data:', X - X.mean(axis=0))

# Fase de aprendizaje del Linear Associator con vectorización de clases
def linear_associator_learning(X, Y):
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    print('Learning phase: Initial M matrix:', M)
    for i in range(len(X)):
        print(f'Updating M with outer product of Y[{i}] and X[{i}]:', np.outer(Y[i], X[i]))
        print('M matrix after update:', M)
    return M

# Fase de recuperación del CHAT usando similitud máxima
def recover_class_vectorized(M, X):
    print('Recovery phase: Similarity matrix:', similarities)
    print('Recovered class indices:', recovered_classes)
    return recovered_classes

def main_chat():
    # Ejemplo de uso (sustituye file_path por la ruta del archivo que desees cargar)
    # file_path = "processed.cleveland.data"
    # file_path = "processed.switzerland.data"
    # file_path = "reprocessed.hungarian.data"
    file_path = "example.data"

    df = load_data_to_dataframe(file_path)
    X = df.iloc[:, :-1].values  # Características
    Y = df.iloc[:, -1].values   # Clases

    # Preparación de los datos
    X_translated = apply_translation(X)
    X_normalized = normalize_patterns(X_translated)

    # Vectorización de la salida
    Y_classes = np.unique(Y)
    Y_vectorized = np.array([np.eye(len(Y_classes))[int(y)] for y in Y])

    # Fase de aprendizaje
    M = linear_associator_learning(X_normalized, Y_vectorized)

    # Fase de recuperación
    recovered_classes = recover_class_vectorized(M, X_normalized)
    recovered_classes_labels = [Y_classes[i] for i in recovered_classes]

    # Calcular precisión
    accuracy = np.mean(recovered_classes_labels == Y)
    print(f"Precisión: {accuracy * 100:.2f}%")

def chat_algorithm_M(X_train, Y_train, X_test):
    print('Running CHAT algorithm...')
    """
    Función que implementa el algoritmo CHAT basado en la teoría.
    Realiza la fase de aprendizaje en los datos de entrenamiento (X_train, Y_train)
    y la fase de recuperación en los datos de prueba (X_test).
    
    Parámetros:
    - X_train: np.array, características de entrenamiento.
    - Y_train: np.array, etiquetas de clase para el entrenamiento.
    - X_test: np.array, características de prueba para clasificación.
    
    Retorna:
    - recovered_classes_labels: List[int], clases predichas para X_test.
    """
    
    # Paso 1: Normalizar los datos de entrada
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Paso 2: Aplicar la traslación (centrado de datos)
    X_train = X_train - X_train.mean(axis=0)
    X_test = X_test - X_test.mean(axis=0)
    
    # Paso 3: Convertir las etiquetas Y en formato vectorizado (one-hot encoding)
    Y_classes = np.unique(Y_train)
    Y_train_vectorized = np.array([np.eye(len(Y_classes))[int(y)] for y in Y_train])
    
    # Paso 4: Fase de aprendizaje con el Linear Associator
    M = np.zeros((Y_train_vectorized.shape[1], X_train.shape[1]))
    for i in range(len(X_train)):
        M += np.outer(Y_train_vectorized[i], X_train[i])
    
    # Paso 5: Fase de recuperación en los datos de prueba (X_test)
    similarities = M.dot(X_test.T)
    print('Recovered class indices:', recovered_classes)
    
    # Convertimos las clases de vuelta a etiquetas originales
    recovered_classes_labels = [Y_classes[i] for i in recovered_classes]
    
    return recovered_classes_labels

# Función para cargar el archivo y eliminar filas con "?"
def load_data_to_dataframe_delete(file_path):
    # Cargar el archivo sin encabezados y reemplazar "?" con NaN
    df = pd.read_csv(file_path, header=None)
    df.replace(["?", "-9"], np.nan, inplace=True)  # Reemplazar "?" con NaN

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Solo columnas numéricas
            df[col].fillna(df[col].median(), inplace=True)  # Imputación con la mediana

    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns  # Selecciona solo columnas numéricas
    df[num_cols] = scaler.fit_transform(df[num_cols])  # Normalización
    
    # Eliminar filas con valores NaN (originalmente "?" en el archivo)
    df.dropna(inplace=True)
    
    # Convertir todas las columnas a tipo numérico
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df

main_chat()