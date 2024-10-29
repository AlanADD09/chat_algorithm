import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paso 1: Inicialización de la Memoria
def initialize_memory(dim_input, dim_output):
    memory_matrix = np.zeros((dim_output, dim_input))
    print("Memoria inicializada en ceros:\n", memory_matrix)
    return memory_matrix

# Paso 2: Aprendizaje del Linear Associator
def linear_associator_learning(memory_matrix, X_fundamental, y_fundamental):
    for i, (x, y) in enumerate(zip(X_fundamental, y_fundamental)):
        y_vector = np.zeros(memory_matrix.shape[0])
        y_vector[int(y)] = 1
        memory_matrix += np.outer(y_vector, x)
        print(f"Producto externo para el patrón {i} (Clase {y}):\n", np.outer(y_vector, x))
        print("Matriz de memoria tras actualizar con el patrón:\n", memory_matrix)

# Paso 3: Traslación de Ejes
def translate_axes(memory_matrix, X_train, X_test):
    mean_vector = np.mean(X_train, axis=0)
    print("Vector de media calculado para la traslación de ejes:\n", mean_vector)
    memory_matrix_translated = memory_matrix - mean_vector
    X_test_translated = X_test - mean_vector
    print("Matriz de memoria después de la traslación:\n", memory_matrix_translated)
    print("Conjunto de prueba después de la traslación:\n", X_test_translated)
    return memory_matrix_translated, X_test_translated

# Paso 4: Recuperación de la Lernmatrix
def lernmatrix_recovery(memory_matrix, input_pattern):
    output_vector = np.dot(memory_matrix, input_pattern)
    class_index = np.argmax(output_vector)
    print("Patrón de entrada:\n", input_pattern)
    print("Vector de salida tras multiplicación:\n", output_vector)
    print("Clase recuperada:", class_index)
    return class_index, output_vector

# Paso 5: Clasificación de Patrones Desconocidos
def classify_unknown_patterns(memory_matrix_translated, X_test_translated):
    predicciones = []
    for i, test_pattern in enumerate(X_test_translated):
        print(f"\n--- Clasificación del patrón de prueba {i} ---")
        class_index, _ = lernmatrix_recovery(memory_matrix_translated, test_pattern)
        predicciones.append(class_index)
    return predicciones

# Función para evaluación de precisión
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Evaluación del modelo:")
    print("Precisión (accuracy):", accuracy)
    print("Precisión por clase (precision):", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Matriz de Confusión:\n", conf_matrix)

# Función Principal: Ejecuta todo el Algoritmo
def run_chat_algorithm(file_path, learning_rate=0.1, max_iterations=100):
    # Cargar y preprocesar datos
    data = pd.read_csv(file_path, header=None)
    data.replace("?", pd.NA, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.fillna(data.median(), inplace=True)

    # Normalización de datos
    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    print("Datos después de la normalización:\n", data.head())

    # División de datos en entrenamiento y prueba
    X = data.iloc[:, :-1]
    y = data[13]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dimensiones de entrada y salida
    dim_input = X_train.shape[1]
    dim_output = len(np.unique(y_train))

    # Inicialización de la memoria
    memory_matrix = initialize_memory(dim_input, dim_output)

    # Aprendizaje del Linear Associator
    linear_associator_learning(memory_matrix, X_train.values, y_train.values)
    print("Matriz de memoria después del aprendizaje:\n", memory_matrix)

    # Traslación de Ejes
    memory_matrix_translated, X_test_translated = translate_axes(memory_matrix, X_train.values, X_test.values)
    print("Matriz de memoria después de la traslación:\n", memory_matrix_translated)

    # Clasificación de Patrones Desconocidos
    predicciones = classify_unknown_patterns(memory_matrix_translated, X_test_translated)
    print("Clasificaciones para los primeros patrones desconocidos:", predicciones[:10])

    # Evaluación del Modelo
    evaluate_model(y_test.values, np.array(predicciones))

def test_chat():
    # Conjunto de datos reducido para pruebas
    X_example = np.array([
        [2.1, 3.8],
        [6.3, 3.8],
        # [2.1, 4.0],
        # [6.3, 4.0],
        # [2.0, 3.5],
        # [6.5, 3.5]
    ])

    y_example = np.array([0, 1, 0, 1, 0, 1])

    # Separar los datos en entrenamiento y prueba (usaremos el mismo conjunto en este caso)
    X_train_example = X_example
    y_train_example = y_example
    X_test_example = np.array([
        [2.2, 3.9],  # Esperado: Clase 0
        [6.2, 3.9]   # Esperado: Clase 1
    ])

    # Dimensiones de entrada y salida
    dim_input_example = X_train_example.shape[1]
    dim_output_example = len(np.unique(y_train_example))

    # Inicialización de la memoria
    memory_matrix_example = initialize_memory(dim_input_example, dim_output_example)

    # Aprendizaje del Linear Associator
    linear_associator_learning(memory_matrix_example, X_train_example, y_train_example)
    print("Matriz de memoria después del aprendizaje:\n", memory_matrix_example)

    # Traslación de Ejes
    memory_matrix_translated_example, X_test_translated_example = translate_axes(memory_matrix_example, X_train_example, X_test_example)
    print("Matriz de memoria después de la traslación:\n", memory_matrix_translated_example)

    # Clasificación de Patrones Desconocidos
    predicciones_example = classify_unknown_patterns(memory_matrix_translated_example, X_test_translated_example)
    print("Clasificaciones para los patrones desconocidos:", predicciones_example)

# Ejecutar el algoritmo
# run_chat_algorithm('processed.cleveland.data')
test_chat()