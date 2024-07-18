# This is a sample Python script.
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from django.http import  JsonResponse
import os.path
import pandas as pd
from pathlib import Path

from functions import loadfile, deleteduplicates, changeValue

from django.views.decorators.http import require_http_methods

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

@require_http_methods(["GET"])
def load_file(request):
    print("Funcion de loadFile llamada")
    pathfile = request.GET["path_file"]
    namefile = request.GET["name_file"]
    check_file = os.path.isfile(namefile)
    pathfile = pathfile if not check_file else namefile

    if not check_file:
        df = pd.read_csv(pathfile)
        filepath = Path(namefile)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    # Llamar a la función loadfile con el pathfile
    result = loadfile(pathfile)

    response_data = {
        "rowss": result[0],
        "colss": result[1],
        "dataParsed": result[2],
        "dataInfo": result[3],
        "description_dataset": result[4],
        "unique_valuess": result[5],
        "duplicate_valuess": result[6],
        "duplicate_datas": result[7],
    }
    ##return JsonResponse({'pathfile':pathfile, 'namefile': namefile})
    return JsonResponse(response_data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def changeValueRequest(request):
    print("Funcion de changeValueRequest llamada")
    pathfile = request.GET["name_file"]
    back_value = request.GET["back_value"]
    new_value = request.GET["new_value"]
    column_title = request.GET["columnTitle"]
    column_type = request.GET["columnType"]

    if (column_type == 'int64'):
        if (back_value == 'null'):
            back_value = None
        else:
            back_value = int(back_value)
    if (column_type == 'float64'):
        if (back_value == 'null'):
            back_value = None
        else:
            back_value = float(back_value)
    if (back_value == 'null'):
        back_value = None

    result = changeValue(pathfile, column_title, back_value, new_value)

    return JsonResponse({'Estado': result})

def delete_duplicates(request):
    print("Funcion de delete_duplicates llamada")
    pathfile = request.GET["path_file"]
    result = deleteduplicates(pathfile)
    return JsonResponse({'Estado': result})