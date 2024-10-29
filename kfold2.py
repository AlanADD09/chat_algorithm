from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from chat3 import chat_algorithm_M, load_data_to_dataframe_delete

# --- Función de evaluación inicial ---
def evaluate_initial_population(X, y, population):
    kfold = KFold(n_splits=10)
    scores = []

    for individual in population:
        selected_features = list(range(X.shape[1]))  # Usar todas las características inicialmente
        X_selected = X[:, selected_features]

        # Evaluar usando validación cruzada y chat_algorithm_M
        fold_scores = []
        for train_index, test_index in kfold.split(X_selected):
            X_train, X_test = X_selected[train_index], X_selected[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Entrenar usando los datos de entrenamiento y evaluar en datos de prueba
            y_pred = chat_algorithm_M(X_train, y_train, X_test)  # Aquí se usa el conjunto de prueba X_test, y_test

            # Calcular precisión comparando predicciones con etiquetas verdaderas
            accuracy = np.mean(np.array(y_pred) == y_test[:len(y_pred)])
            fold_scores.append(accuracy)

        # Promedio de precisión en los folds de este individuo
        scores.append(np.mean(fold_scores))

    return scores

# --- Función de selección genética optimizada con reglas de Deb ---
def genetic_feature_selection_with_deb(X, y, n_generations=20, pop_size=50):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_features, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

    def deb_tournament(population, k=2):
        offspring = []
        for _ in range(len(population) // 2):  # Crear parejas de padres
            parent1 = tools.selTournament(population, 1, tournsize=3)[0]
            parent2 = tools.selTournament(population, 1, tournsize=3)[0]
            
            # Clonar los padres para crear los hijos
            child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)

            # Realizar cruzamiento y mutación
            if np.random.rand() < 0.5:
                toolbox.mate(child1, child2)
                toolbox.mutate(child1)
                toolbox.mutate(child2)

            # Evaluar la aptitud de los hijos
            toolbox.evaluate(child1)
            toolbox.evaluate(child2)

            # Aplicar reglas de Deb para seleccionar el mejor entre padres e hijos
            offspring.append(child1 if child1.fitness >= parent1.fitness else parent1)
            offspring.append(child2 if child2.fitness >= parent2.fitness else parent2)

        return offspring

    # Registrar la función de selección en el toolbox sin pasar toolbox explícitamente como parámetro
    toolbox.register("select", deb_tournament)

    population = toolbox.population(n=pop_size)

    # Evaluación inicial con todas las características
    initial_scores = evaluate_initial_population(X, y, population)
    for i, ind in enumerate(population):
        ind.fitness.values = (initial_scores[i],)

    # Ejecución del algoritmo genético
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=True)

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual, best_individual.fitness.values[0]

# Uso: Llamar a genetic_feature_selection_with_deb para ejecutar la selección genética

# Función para normalizar y trasladar los patrones
def preprocess_patterns(X):
    epsilon = 1e-8
    min_X, max_X = X.min(axis=0), X.max(axis=0)
    mean_vector = X.mean(axis=0)
    return (X - mean_vector - min_X) / (max_X - min_X + epsilon)

# Función de evaluación para la selección genética de características
def evaluate_features(individual, X, y):
    selected_features = [index for index, bit in enumerate(individual) if bit == 1]
    if not selected_features:  # Si no se seleccionan características
        return 0,
    X_selected = X[:, selected_features]
    classifier = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(classifier, X_selected, y, cv=5)
    return np.mean(scores),

# Función de selección de características con algoritmos genéticos
def genetic_feature_selection(X, y, n_generations=20, pop_size=50):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_features, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, verbose=False)

    # Selección del mejor conjunto de características
    best_individual = tools.selBest(population, 1)[0]
    selected_features = [index for index, bit in enumerate(best_individual) if bit == 1]
    return selected_features

# Define la ruta al archivo de datos
file_path = "processed.cleveland.data"
# file_path = "processed.switzerland.data"
# file_path = "reprocessed.hungarian.data"

# Cargar y preprocesar los datos
df = load_data_to_dataframe_delete(file_path)
# df = load_data_to_dataframe(file_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X = preprocess_patterns(X)

# Ejecutar la validación cruzada con 10 folds en cada iteración
# selected_features, _ = genetic_feature_selection_with_deb(X, y, n_generations=50, pop_size=100)
# X = X[:, selected_features]

# Convertir el vector de etiquetas y a valores discretos para clasificación
y = np.round(y).astype(int)

# Validación cruzada de 10 pliegues y registro de precisión
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_list = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Asegurar que y_train y y_test sean enteros
    y_train = np.round(y_train).astype(int)
    y_test = np.round(y_test).astype(int)

    # Realizar la predicción con el algoritmo CHAT
    y_pred = chat_algorithm_M(X_train, y_train, X_test)

    # Convertir predicciones a etiquetas discretas
    y_pred = np.round(y_pred).astype(int)

    # Calcular precisión solo si y_pred y y_test tienen el mismo tamaño
    if len(y_pred) == len(y_test):
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
    else:
        print("Error: y_pred y y_test no coinciden en tamaño")

# Calcular y mostrar precisión promedio y desviación estándar
valid_accuracies = [acc for acc in accuracy_list if not np.isnan(acc)]
if valid_accuracies:
    mean_accuracy = np.mean(valid_accuracies)
    std_accuracy = np.std(valid_accuracies)
    print(f"\nPrecisión promedio en 10-fold cross-validation: {mean_accuracy:.2f}")
    print(f"Desviación estándar de la precisión: {std_accuracy:.2f}")
else:
    print("No se calcularon precisiones válidas.")