# This is a sample Python script.
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from django.http import  JsonResponse
import os.path
import pandas as pd
from pathlib import Path

from functions import loadfile

from django.views.decorators.http import require_http_methods

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

@require_http_methods(["GET"])
def load_file(request):
    print("llego a loadfile")
    pathfile = request.GET["path_file"]
    namefile = request.GET["name_file"]
    check_file = os.path.isfile(namefile)
    pathfile = pathfile if not check_file else namefile

    if not check_file:
        print("El archivo no existe aun")
        df = pd.read_csv(pathfile)
        print("Lee el archivo con el path")
        filepath = Path(namefile)
        print("asigna al filepath el path del nombre del archivo creo")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        print("crea el directorio si es que no existe")
        df.to_csv(filepath, index=False)
        print("Ultima linea xd")

    # Llamar a la función loadfile con el pathfile
    print("Esta a punto de llamar a load file functions")
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

def delete_duplicates(request):
    nombre = request.GET["name_file"]
    print(request.GET["name_file"])
    return JsonResponse({'message': 'Hola desde delete_duplicates.', 'nombre de archivo' : nombre})