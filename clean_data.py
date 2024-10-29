import numpy as np
import pandas as pd
from chat2 import load_data_to_dataframe_delete

# Define la ruta al archivo de datos
file_path_cleveland = "processed.cleveland.data"
file_path_switzerland = "processed.switzerland.data"
file_path_hungarian = "reprocessed.hungarian.data"

# Cargar y preprocesar los datos
cleveland_data = load_data_to_dataframe_delete(file_path_cleveland)
switzerland_data = load_data_to_dataframe_delete(file_path_switzerland)
hungarian_data = load_data_to_dataframe_delete(file_path_hungarian)

# Reemplazar los valores "?" y "-9" con NaN
cleveland_data.replace(["?", "-9"], np.nan, inplace=True)
switzerland_data.replace(["?", "-9"], np.nan, inplace=True)
hungarian_data.replace(["?", "-9"], np.nan, inplace=True)

# Imputación de valores faltantes (media o mediana, como ejemplo)
for df in [cleveland_data, switzerland_data, hungarian_data]:
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:  # Solo columnas numéricas
            df[col].fillna(df[col].median(), inplace=True)  # Imputación con la mediana

# Normalización
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for df in [cleveland_data, switzerland_data, hungarian_data]:
    num_cols = df.select_dtypes(include=[np.number]).columns  # Selecciona solo columnas numéricas
    df[num_cols] = scaler.fit_transform(df[num_cols])  # Normalización

# Mostrar datos procesados
print(cleveland_data.head())
print(switzerland_data.head())
print(hungarian_data.head())