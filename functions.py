import pandas as pd
import io
from json import loads, dumps
from pathlib import Path
from sklearn.model_selection import train_test_split


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

    # Seleccionar la información de las columnas, no el nombre de estass
    features = df.columns

    # Colocar las características y la etiqueta (En este caso van todas para que el dataset al momento de dividir seleccione todas)
    X = df[features]
    y = df[features]    #Normalmente el target es solo una columna pero solo buscamos dividir el dataset

    #Aqui se divide el datasett
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPercentage, random_state=42)

    #Guardamos en disco en la ubicación donde se guardan los datasets de laravel para poder consultar despues
    X_test.to_csv('C:/xampp/htdocs/Memla/MEMLA/storage/app/public/datasets/test'+laboratoryId+'.csv', index=False)  ##1/3
    X_train.to_csv('C:/xampp/htdocs/Memla/MEMLA/storage/app/public/datasets/training'+laboratoryId+'.csv', index=False)  ##2/3
    path_test = 'datasets/test'+laboratoryId+'.csv'
    path_training = 'datasets/training'+laboratoryId+'.csv'

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
    #print("\nEtiquetas de entrenamiento (y_train):")
    #print(y_train)
    #print("\nEtiquetas de prueba (y_test):")
    #print(y_test)

    pathtest = Path(path_test)
    pathtest.parent.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(pathtest, index=False)

    pathtraining = Path(path_training)
    pathtraining.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(pathtraining, index=False)

    return [path_test, path_training, laboratoryId]

def dropcolumn(path_file, column):
    df = pd.read_csv(path_file)
    df = df.drop(columns=[column])

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return "Columna borrada correctamente"