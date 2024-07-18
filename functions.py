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
