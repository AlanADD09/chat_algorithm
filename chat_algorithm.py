from cha import *

class CHATAlgorithm:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        # Traslación Inicial
        X_train_translated, mean_vector = initial_translation(self.X_train)

        # Inicialización de la memoria
        dim_input = self.X_train.shape[1]
        dim_output = len(np.unique(self.y_train))
        memory_matrix = initialize_memory(dim_input, dim_output)

        # Aprendizaje
        linear_associator_learning(memory_matrix, X_train_translated, self.y_train)

        # Traslación para Clasificación
        memory_matrix_translated, X_test_translated = translate_axes(memory_matrix, self.X_train, self.X_test)

        # Clasificación
        predicciones = classify_unknown_patterns(memory_matrix_translated, X_test_translated)
        return np.array(predicciones)
