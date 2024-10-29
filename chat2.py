import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Función para cargar el archivo .data y reemplazar los "?" con el promedio de la columna
def load_data_to_dataframe(file_path):
    df = pd.read_csv(file_path, header=None)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    for column in df.columns:
        mean_value = df[column].mean(skipna=True)
        df[column] = df[column].fillna(mean_value)
    return df

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

# Normalización de los patrones
def normalize_patterns(X):
    # Evitar la división por cero al agregar un valor pequeño (epsilon) al denominador
    epsilon = 1e-8
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon)


# Función para la fase de aprendizaje del Linear Associator
def linear_associator_learning(X, Y):
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]
    M = np.zeros((Y.shape[1], X.shape[1]))
    for i in range(len(X)):
        M += np.outer(Y[i], X[i])
    return M

# Función optimizada para la fase de aprendizaje del Linear Associator
def linear_associator_learning_optimized(X, Y):
    # Asegura que Y es una columna y tiene el mismo número de filas que X
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]  # Convertir a una matriz columna si es necesario

    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Mismatch: X tiene {X.shape[0]} filas pero Y tiene {Y.shape[0]} filas.")
        
    M = Y.T @ X  # Multiplicación matricial
    return M

# Función para la fase de recuperación del Linear Associator
# def linear_associator_recovery(M, x_single, threshold=0):
#     y_recovered = np.dot(M, x_single)
#     return 1 if y_recovered > threshold else 0

def linear_associator_recovery(M, x):
    """
    Fase de recuperación en el clasificador CHAT usando la matriz de aprendizaje M.
    Dado un patrón de entrada, identifica la clase de salida basada en la activación máxima.
    
    Parameters:
    - x: numpy array de un único patrón de entrada.
    - M: Matriz de aprendizaje construida durante la fase de entrenamiento.
    
    Returns:
    - Clase predicha para el patrón de entrada.
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]  # Convierte a 2D si es necesario (columna)
    activations = M @ x  # Multiplicación de la matriz de aprendizaje con el patrón de entrada
    predicted_class = activations.argmax()  # Selecciona la clase con la activación más alta
    return predicted_class


# Función modificada para la fase de aprendizaje de la Lernmatrix sin epsilon (versión "vanilla")
def lernmatrix_learning_vanilla(X, Y):
    num_classes = len(np.unique(Y))  # Número de clases únicas en Y
    M = np.zeros((num_classes, X.shape[1]))  # Matriz M de tamaño (clases, atributos)
    
    for i in range(len(X)):
        class_index = int(Y[i])  # Índice de la clase correspondiente
        for j in range(len(X[i])):
            # Solo añadimos 1 cuando el valor de X[i][j] es mayor a 0.5
            if X[i][j] > 0.5:
                M[class_index][j] += 1  # Refuerzo sin epsilon
    
    return M

# Función para la fase de aprendizaje de la Lernmatrix (eliminamos las penalizaciones negativas)
def lernmatrix_learning(X, Y, epsilon=3):
    num_classes = len(np.unique(Y)) + 1 # Número de clases únicas en Y
    M = np.zeros((num_classes, X.shape[1]))  # Matriz M de tamaño (clases, atributos)
    
    for i in range(len(X)):
        class_index = int(Y[i])  # Índice de la clase correspondiente
        for j in range(len(X[i])):
            if X[i][j] > 0.5:  # Usamos 0.5 como umbral para patrones normalizados
                M[class_index][j] += epsilon  # Refuerzo positivo fuerte
            else:
                # Evitamos penalizaciones negativas, simplemente no sumamos nada para evitar restas excesivas
                M[class_index][j] += 0.1  # Refuerzo leve
    
    return M

# Función para la fase de recuperación de la Lernmatrix (sin umbral)
def lernmatrix_recovery(M, x_single):
    # Calcular las puntuaciones para cada clase
    scores = np.dot(M, x_single)
    
    # Imprimir las puntuaciones para depuración
    # print("Puntuaciones de la Lernmatrix:", scores)
    
    # Seleccionar la clase con la mayor puntuación
    return np.argmax(scores)

# Función para la traslación de los patrones
def translate_patterns(X, mean_vector):
    return X - mean_vector

# Función para calcular el vector medio (prototipo)
def calculate_mean_vector(X):
    return np.mean(X, axis=0)

# Clasificación de patrones desconocidos
# def classify_patterns(X_new, M_linear, M_lernmatrix):
#     # Trasladar los nuevos patrones a los ejes traslados
#     mean_vector = calculate_mean_vector(X_new)
#     X_translated = translate_patterns(X_new, mean_vector)
    
#     # Normalizamos los patrones trasladados
#     X_translated = normalize_patterns(X_translated)
    
#     # Aplicamos la recuperación sobre cada patrón individualmente
#     y_linear_results = []
#     y_lernmatrix_results = []
    
#     for i in range(len(X_translated)):
#         # Recuperar para un solo patrón (fila)
#         y_linear = linear_associator_recovery(M_linear, X_translated[i])
#         y_lernmatrix = lernmatrix_recovery(M_lernmatrix, X_translated[i])  # Sin umbral para ahora
        
#         y_linear_results.append(y_linear)
#         y_lernmatrix_results.append(y_lernmatrix)
    
#     return y_linear_results, y_lernmatrix_results

def classify_patterns(X_new, M_linear, M_lernmatrix):
    """
    Clasifica patrones desconocidos usando las matrices de aprendizaje.
    
    Parameters:
    - X_new: Matriz de patrones de entrada (normalizados y trasladados).
    - M_linear: Matriz de aprendizaje del Linear Associator.
    - M_lernmatrix: Matriz de aprendizaje de la Lernmatrix.
    
    Returns:
    - Resultados de clasificación del Linear Associator y la Lernmatrix.
    """
    mean_vector = calculate_mean_vector(X_new)
    X_translated = translate_patterns(X_new, mean_vector)
    X_translated = normalize_patterns(X_translated)

    y_linear_results = []
    y_lernmatrix_results = []

    for i in range(len(X_translated)):
        x = X_translated[i]  # Seleccionar un solo patrón
        y_linear = linear_associator_recovery(M_linear, x)  # Usar el patrón como una matriz 2D
        y_lernmatrix = lernmatrix_recovery(M_lernmatrix, x)  # Mantener x en 2D para lernmatrix
        
        y_linear_results.append(y_linear)
        y_lernmatrix_results.append(y_lernmatrix)

    return y_linear_results, y_lernmatrix_results


# Implementación del algoritmo CHAT
def chat_algorithm(file_path, epsilon=3):
    # df = load_data_to_dataframe(file_path)
    df = load_data_to_dataframe_delete(file_path)
    X = df.iloc[:, :-1].values  # patrones de entrada (todas las columnas menos la última)
    Y = df.iloc[:, -1].values   # clases (última columna)
    
    # Normalizar los patrones de entrada
    X = normalize_patterns(X)
    
    # Calcular el vector medio
    mean_vector = calculate_mean_vector(X)
    
    # Traslación de los patrones
    X_translated = translate_patterns(X, mean_vector)
    
    # Fase de aprendizaje (Linear Associator y Lernmatrix)
    M_linear = linear_associator_learning(X_translated, Y)
    # M_linear = linear_associator_learning_optimized(X_translated, Y)
    M_lernmatrix = lernmatrix_learning(X_translated, Y, epsilon=epsilon)
    # M_lernmatrix = lernmatrix_learning_vanilla(X_translated, Y)
    
    # Clasificar patrones desconocidos
    results = classify_patterns(X_translated, M_linear, M_lernmatrix)
    
    return results

def chat_algorithm_M(X_test, y_test, epsilon=2):
    # Normalizar los patrones de entrada
    X = normalize_patterns(X_test)
    
    # Calcular el vector medio
    mean_vector = calculate_mean_vector(X)
    
    # Traslación de los patrones
    X_translated = translate_patterns(X, mean_vector)
    
    # Fase de aprendizaje (Linear Associator y Lernmatrix)
    # NOTA: Si estás entrenando un modelo ya aprendido, necesitarás matrices preentrenadas
    M_linear = linear_associator_learning(X_translated, y_test)
    # M_linear = linear_associator_learning_optimized(X_translated, y_test)
    M_lernmatrix = lernmatrix_learning(X_translated, y_test, epsilon=epsilon)
    # M_lernmatrix = lernmatrix_learning_vanilla(X_translated, y_test)
    
    # Aplicar la predicción sobre cada muestra en el conjunto de prueba
    y_pred = []
    for x in X_translated:
        y_linear, y_lernmatrix = classify_patterns([x], M_linear, M_lernmatrix)
        # Puedes decidir si quieres usar uno de los dos modelos o una combinación
        y_pred.append(y_lernmatrix[0])  # Usando la predicción de la Lernmatrix como ejemplo
    
    return y_pred  # Devolvemos las predicciones para cada muestra



# Datos del ejemplo del PDF (2 clases, 2 dimensiones)
# data = pd.DataFrame({
#     0: [2.1, 6.3, 2.5, 5.8],
#     1: [3.8, 3.8, 3.0, 2.9],
#     2: [0, 1, 0, 1]
# })

# Guardar el archivo como CSV para usarlo en el algoritmo
# data.to_csv('chat_example.data', header=False, index=False)

# Ejecutar el algoritmo CHAT con el ejemplo del PDF
# file_path = 'example.data'
# results = chat_algorithm(file_path, epsilon=3)  # Aumentamos epsilon para refuerzo
# print("Linear Associator results:", results[0])
# print("Lernmatrix results:", results[1])