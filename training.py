from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from chat2 import *

# Entrenar el algoritmo CHAT en los datos de entrenamiento
def train_chat(X_train, y_train):
    # Aquí usaríamos la función chat_algorithm adaptada para tomar X_train y y_train
    results = chat_algorithm_M(X_train, y_train)
    return results

# Evaluación del modelo
def evaluate_chat(X_test, y_test, y_pred):
    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    
    # Imprimir matriz de confusión
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    # Reporte detallado de clasificación
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Precisión total
    print(f"Precisión: {accuracy}")

# Cargar los datos de Cleveland
file_path = 'processed.cleveland.data'
df = load_data_to_dataframe(file_path)

# Dividir los datos en características (X) y la variable objetivo (y)
X = df.iloc[:, :-1].values  # Todos los atributos menos el último
y = df.iloc[:, -1].values   # La última columna es la clase

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el algoritmo en los datos de entrenamiento y almacenar las predicciones
results = train_chat(X_train, y_train)

# Generar predicciones para el conjunto de prueba
y_pred = chat_algorithm_M(X_test, y_test)

# Evaluar el rendimiento en el conjunto de prueba
evaluate_chat(X_test, y_test, y_pred)
