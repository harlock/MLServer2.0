import pandas as pd
import io
import os
from json import loads, dumps
from pathlib import Path
from sklearn.model_selection import train_test_split
from decouple import config


def loadfile(url: str):
    df = pd.read_csv(url)
    buffer = io.StringIO()

    description_dataset = df.describe(include='all').to_json(orient="split")

    # Captura la información del DataFrame
    df.info(buf=buffer, verbose=True)
    lines = buffer.getvalue().splitlines()
    dataInfo = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
                .drop('Count', axis=1)
                .rename(columns={'Non-Null': 'Non-Null Count'})).to_json(orient="split")

    # Convertir DataFrame a JSON
    result = df.to_json(orient="split")
    parsed = loads(result)
    dataParsed = dumps(parsed)

    # Obtener filas y columnas
    rowss = df.shape[0]
    colss = df.shape[1]

    # Valores únicos
    unique_valuess = df[list(df.columns.values)].apply(lambda x: x.unique()).to_json()

    # Valores duplicados
    duplicate_valuess = df.duplicated(subset=None, keep=False).to_json()
    duplicate_datas = df.groupby(df.columns.tolist(), as_index=False).size()
    duplicate_datas = duplicate_datas[duplicate_datas['size'] > 1]
    duplicate_datas = duplicate_datas.to_json()

    return [rowss, colss, dataParsed, dataInfo, description_dataset, unique_valuess, duplicate_valuess, duplicate_datas]

def changeValue(path_file,column_title, back_value,new_value):
    print(type(back_value))
    df = pd.read_csv(path_file)
    if (back_value is None):
        df[column_title] = df[column_title].fillna(new_value)

    else:
        df[column_title] = df[column_title].replace([back_value], new_value)
    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return "Valores de los datos cambiados"

def deleteduplicates(path_file):
    df = pd.read_csv(path_file)
    df = df.drop_duplicates()

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return "Datos duplicados borrados"

def splitData(pathFile,  laboratoryId, testPercentage):
    testPercentage = int(testPercentage)
    if testPercentage == 0:
        testPercentage = 1
    if testPercentage == 100:
        testPercentage = 97
    testPercentage = testPercentage/100
    print(testPercentage)
    df = pd.read_csv(pathFile)
    print('Dataset original:')
    print(df)

    features = df.columns # Seleccionar la información de las columnas, no el nombre de estass

    # Colocar las características y la etiqueta (En este caso van todas para que el dataset al momento de dividir seleccione todas)
    X = df[features]
    y = df[features]    #Normalmente el target es solo una columna pero solo buscamos dividir el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPercentage, random_state=42) #Aqui se divide el datasett

    dataset_path = config('dataset_path')  # Importar path para guardar desde el .env

    #Guardamos en disco en la ubicación donde se guardan los datasets de laravel para poder consultar despues
    X_test.to_csv(dataset_path+'test(unTercio)--'+laboratoryId+'.csv', index=False)  ##1/3
    X_train.to_csv(dataset_path+'training(unTercio)--'+laboratoryId+'.csv', index=False)  ##2/3

    # Guarda el dataset pero en ML no en laravel (dice path_laravel pero es porque es la que se envia a laravel para consultar
    path_test_laravel = 'datasets/test(unTercio)--' + laboratoryId + '.csv'
    path_training_laravel = 'datasets/training(unTercio)--' + laboratoryId + '.csv'

    pathtest = Path(path_test_laravel)
    pathtest.parent.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(pathtest, index=False)

    pathtraining = Path(path_training_laravel)
    pathtraining.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(pathtraining, index=False)

    # Obtener el número total de muestras y calcular el porcentaje y mostrar los resultados
    total_samples = len(df)
    train_percentage = (X_train.shape[0] / total_samples) * 100
    test_percentage = (X_test.shape[0] / total_samples) * 100
    print(f"\nPorcentaje de entrenamiento: {train_percentage:.2f}%")
    print(f"Porcentaje de prueba: {test_percentage:.2f}%")

    # Imprimir los conjuntos de datos de entrenamiento y prueba
    print("\nConjunto de entrenamiento (X_train):")
    print(X_train)
    print("\nConjunto de prueba (X_test):")
    print(X_test)

    return [path_test_laravel, path_training_laravel, laboratoryId]

def dropcolumn(path_file, column):
    df = pd.read_csv(path_file)
    df = df.drop(columns=[column])

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return "Columna borrada correctamente"