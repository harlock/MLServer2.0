import pandas as pd
import io
from json import loads, dumps
from pathlib import Path


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