import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier  # Suponiendo que el clasificador CHAT es similar a un árbol de decisión

# Función para cargar el archivo .data y convertirlo en un DataFrame
def load_data_to_dataframe(file_path):
    # Cargar el archivo .data usando pandas
    df = pd.read_csv(file_path, header=None)
    
    # Reemplazar los valores faltantes "?" por NaN
    df.replace('?', 0, inplace=True)
    
    # Convertir las columnas a tipos de datos apropiados
    for column in df.columns:
        # Intentar convertir a float si es posible
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass
    
    return df

file_path = 'processed.cleveland.data'

df = load_data_to_dataframe(file_path)

# Paso 1: Definir los parámetros
num_iterations = 100
population_size = 50
crossover_probability = 0.8
mutation_probability = 0.01
num_folds = 10

# Supongamos que el DataFrame `df` está cargado con las características y etiquetas
X = df.iloc[:, :-1]  # Características
y = df.iloc[:, -1]   # Etiquetas

# Paso 2: Generar la población inicial
def initialize_population(size, num_features):
    return np.random.randint(2, size=(size, num_features))

# Paso 3: Evaluar la población usando validación cruzada
def evaluate_population(population, X, y):
    scores = []
    for individual in population:
        selected_features = X.loc[:, individual == 1]
        if selected_features.shape[1] == 0:
            scores.append(0)  # No tiene sentido evaluar un conjunto vacío de características
            continue
        classifier = DecisionTreeClassifier()  # Reemplazar con el clasificador CHAT
        score = np.mean(cross_val_score(classifier, selected_features, y, cv=num_folds))
        scores.append(score)
    return np.array(scores)

# Paso 4: Función de selección (por ejemplo, ruleta)
def select_parents(population, scores):
    # Selección por ruleta
    total_score = np.sum(scores)
    selection_probabilities = scores / total_score
    parents_indices = np.random.choice(len(population), size=len(population), p=selection_probabilities)
    return population[parents_indices]

# Funciones de cruce y mutación
def crossover(parents, crossover_probability):
    offspring = []
    for i in range(0, len(parents), 2):
        if np.random.rand() < crossover_probability:
            crossover_point = np.random.randint(1, parents.shape[1]-1)
            offspring1 = np.concatenate([parents[i][:crossover_point], parents[i+1][crossover_point:]])
            offspring2 = np.concatenate([parents[i+1][:crossover_point], parents[i][crossover_point:]])
            offspring.append(offspring1)
            offspring.append(offspring2)
        else:
            offspring.extend([parents[i], parents[i+1]])
    return np.array(offspring)

def mutate(offspring, mutation_probability):
    for individual in offspring:
        if np.random.rand() < mutation_probability:
            mutation_point = np.random.randint(0, offspring.shape[1])
            individual[mutation_point] = 1 - individual[mutation_point]  # Flip bit
    return offspring

# Paso 4: Evolución de la población
def genetic_algorithm(X, y, num_iterations, population_size, crossover_probability, mutation_probability, df):
    num_features = X.shape[1]
    population = initialize_population(population_size, num_features)
    best_score = 0
    best_individual = None
    
    for iteration in range(num_iterations):
        scores = evaluate_population(population, X, y)
        
        # Actualizar la mejor población
        max_score_index = np.argmax(scores)
        if scores[max_score_index] > best_score:
            best_score = scores[max_score_index]
            best_individual = population[max_score_index]
        
        # Selección de padres
        parents = select_parents(population, scores)
        
        # Cruce
        offspring = crossover(parents, crossover_probability)
        
        # Mutación
        population = mutate(offspring, mutation_probability)
        
        print(f"Iteration {iteration+1}/{num_iterations}, Best Score: {best_score}")
    
    return best_individual, best_score

# Ejecutar el algoritmo genético
best_individual, best_score = genetic_algorithm(X, y, num_iterations, population_size, crossover_probability, mutation_probability, df)

# Mostrar el mejor conjunto de características encontrado
print("Best individual:", best_individual)
print("Best score:", best_score)