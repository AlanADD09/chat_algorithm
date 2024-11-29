from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KFoldCHATValidation:
    def __init__(self, n_splits=30, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate(self, X, y, chat_algorithm):
        """
        Realiza validación cruzada con k-fold utilizando el algoritmo CHAT.
        
        Parámetros:
        - X: Características (DataFrame o array).
        - y: Clases (Serie o array).
        - chat_algorithm: Función o clase que implementa el algoritmo CHAT.
        
        Retorna:
        - resultados: Diccionario con las métricas promedio de la validación.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }
        
        # Ejecutar k-fold validation
        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"\n--- Pliegue {fold + 1} ---")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Entrenar y evaluar el algoritmo CHAT
            chat = chat_algorithm(X_train, y_train, X_test, y_test)
            y_pred = chat.run()  # Ejecuta el algoritmo y obtiene las predicciones

            # Calcular métricas
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics["recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics["f1_score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Calcular promedios
        resultados = {key: sum(values) / len(values) for key, values in metrics.items()}
        return resultados
