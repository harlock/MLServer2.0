import pandas as pd
import io
import os
from json import loads, dumps
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from decouple import config
from scipy.stats import norm
from math import ceil


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

    encode_valuess = {}
    for columna in df.columns:
        num_valores_unicos = df[columna].nunique()  # Contar los valores únicos de la columna
        encode_valuess[columna] = num_valores_unicos  # Guardar en el diccionario

    # Convertir el diccionario a formato JSON
    encode_valuess = dumps(encode_valuess)

    return [rowss, colss, dataParsed, dataInfo, description_dataset, unique_valuess, duplicate_valuess, duplicate_datas, encode_valuess]

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
    print(df.shape)

    features = df.columns # Seleccionar la información de las columnas, no el nombre de estass
    X = df[features]
    y = df[features]    #Normalmente el target es solo una columna pero solo buscamos dividir el dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testPercentage, random_state=42) #Aqui se divide el datasett

    dataset_path = config('dataset_path')  # Importar path para guardar desde el .env

    #Guardamos en disco en la ubicación donde se guardan los datasets de laravel para poder consultar despues
    X_test.to_csv(dataset_path+'test(unTercio)--'+laboratoryId+'.csv', index=False)  ##1/3
    X_train.to_csv(dataset_path+'training(unTercio)--'+laboratoryId+'.csv', index=False)  ##2/3

    # Guarda el dataset pero en ML no en laravel (dice path_laravel pero es porque es la que se envia a laravel para consultar desde el propio Laravel
    path_test_laravel = 'datasets/test(unTercio)--'+laboratoryId+'.csv'
    X_test.to_csv(path_test_laravel, index=False)
    path_training_laravel = 'datasets/training(unTercio)--'+laboratoryId+'.csv'
    X_train.to_csv(path_training_laravel, index=False)

    # Obtener el número total de muestras y calcular el porcentaje y mostrar los resultados
    total_samples = len(df)
    train_percentage = (X_train.shape[0] / total_samples) * 100
    test_percentage = (X_test.shape[0] / total_samples) * 100
    print(f"\nPorcentaje de entrenamiento: {train_percentage:.2f}%")
    print(f"Porcentaje de prueba: {test_percentage:.2f}%")

    # Imprimir los conjuntos de datos de entrenamiento y prueba
    print("\nConjunto de entrenamiento (X_train):")
    print(X_train.shape)
    print("\nConjunto de prueba (X_test):")
    print(X_test.shape)

    return [path_test_laravel, path_training_laravel, laboratoryId]

def splitRepresentative(pathFile, laboratoryId, confidenceLevel, marginError):

    df = pd.read_csv(pathFile)
    confidence_level = float(float(confidenceLevel)/100)
    margin_of_error = float(float(marginError)/100)
    print(f"Confidence Level : {confidence_level}")
    print(f"Margin of error: {margin_of_error}")
    p = 0.5  # Proporción esperada, por lo general es 0.5, entonces así lo dejé
    N = len(df)  # Tamaño de la población
    print(f"Tamaño de la población: {N}")
    print(f"Dataset original: {df.shape}")

    # Z-score para el nivel de confianza
    Z = norm.ppf(1 - (1 - confidence_level) / 2)
    print(f"Z: {Z}")

    sample_size = ceil((Z ** 2 * p * (1 - p)) / (margin_of_error ** 2))  # Tamaño de la muestra
    sample_size = ceil(sample_size / (1 + (sample_size - 1) / N))        # Si la población es finita, ajustar el tamaño de la muestra

    training = df.sample(n=sample_size, random_state=42)                 # Tomar la muestra
    test = df.drop(training.index)                                       # Extraer el restante del dataset (será el de test)

    print(f"Tamaño de la muestra: {sample_size}")
    print(f"Dimensiones de la muestra (training): {training.shape}")
    print(f"Dimensiones del restante (test): {test.shape}")

    dataset_path = config('dataset_path')  # Importar path para guardar desde el .env

    # Guardamos en disco en la ubicación donde se guardan los datasets de laravel para poder consultar despues
    test.to_csv(dataset_path+'test(RS)--'+laboratoryId+'.csv', index=False)  ##1/3
    training.to_csv(dataset_path+'training(RS)--'+laboratoryId+'.csv', index=False)  ##2/3

    # Guarda el dataset pero en ML no en laravel (dice path_laravel pero es porque es la que se envia a laravel para consultar desde el propio Laravel
    path_test_laravel = 'datasets/test(RS)--'+laboratoryId+'.csv'
    test.to_csv(path_test_laravel, index=False)
    path_training_laravel = 'datasets/training(RS)--'+laboratoryId+'.csv'
    training.to_csv(path_training_laravel, index=False)

    return [path_test_laravel, path_training_laravel, laboratoryId]

def splitKFold(pathFile, laboratoryId, crossValidation):

    df = pd.read_csv(pathFile)
    crossValidation = int(crossValidation)
    kf = KFold(n_splits=crossValidation, shuffle=True, random_state=1)  # Usamos 10 pliegues

    # Inicializamos las variables acumulativas como DataFrames vacíos
    training = pd.DataFrame()
    test = pd.DataFrame()
    fold_number = 1

    for data_index, val_index in kf.split(df):
        # Obtener los conjuntos de entrenamiento y validación para el pliegue actual
        data_train, data_val = df.iloc[data_index], df.iloc[val_index]

        # Concatenar los pliegues a las variables acumulativas
        training = pd.concat([data_train])
        test = pd.concat([data_val])

        # Puedes guardar o usar estos conjuntos para entrenar tu modelo
        print(f"Pliegue {fold_number}:")
        print(f"  - Tamaño de data_train: {data_train.shape}")
        print(f"  - Tamaño de data_val: {data_val.shape}")
        fold_number += 1

    print(training)
    print(test)

    dataset_path = config('dataset_path')  # Importar path para guardar desde el .env

    # Guardamos en disco en la ubicación donde se guardan los datasets de laravel para poder consultar despues
    test.to_csv(dataset_path + 'test(K-fold)--' + laboratoryId + '.csv', index=False)  ##1/3
    training.to_csv(dataset_path + 'training(K-fold)--' + laboratoryId + '.csv', index=False)  ##2/3

    # Guarda el dataset pero en ML no en laravel (dice path_laravel pero es porque es la que se envia a laravel para consultar desde el propio Laravel
    path_test_laravel = 'datasets/test(K-fold)--' + laboratoryId + '.csv'
    test.to_csv(path_test_laravel, index=False)
    path_training_laravel = 'datasets/training(K-fold)--' + laboratoryId + '.csv'
    training.to_csv(path_training_laravel, index=False)

    return [path_test_laravel, path_training_laravel, laboratoryId]


def dropcolumn(path_file, column):
    df = pd.read_csv(path_file)
    df = df.drop(columns=[column])

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return "Columna borrada correctamente"

def encodecolumno(path_file, column, values):
    df = pd.read_csv(path_file)
    values = values.split(',')

    order = {column: idx + 1 for idx, column in enumerate(values)}
    print(order)
    print(type(order))
    # Aplicar la codificación ordinal
    df[column] = df[column].map(order)

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return "Columna codificada ordinal correctamente"

def encodecolumnn(path_file, column):
    df = pd.read_csv(path_file)
    df[column] = df[column].astype('category')
    df = pd.get_dummies(df, columns=[column])

    filepath = Path(path_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return "Columna codificada Nominal correctamente"