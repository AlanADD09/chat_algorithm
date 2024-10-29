import pandas as pd

# Función para cargar el archivo .data y convertirlo en un DataFrame
def load_data_to_dataframe(file_path):
    # Cargar el archivo .data usando pandas
    df = pd.read_csv(file_path, header=None)
    
    # Reemplazar los valores faltantes "?" por NaN
    # df.replace('?', pd.NA, inplace=True)
    
    # Convertir las columnas a tipos de datos apropiados
    for column in df.columns:
        # Intentar convertir a float si es posible
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass
    
    return df

# Ruta al archivo .data
file_path = 'processed.cleveland.data'

# Cargar el archivo y almacenarlo en un DataFrame
df = load_data_to_dataframe(file_path)

# Mostrar las primeras filas del DataFrame
print(df)
