import pandas as pd
import requests
from django.http import JsonResponse

def probar():
    pathfile = "http://localhost:8000/storage/datasets/6AZioTUu64N8yAusUIpM8f3pWCBGl6H2a58nmNOI.csv"
    namefile = "datasets/6AZioTUu64N8yAusUIpM8f3pWCBGl6H2a58nmNOI.csv"

    print("antes de download")
    download_file(pathfile, namefile)
    print("Archivo guardado correctamente")

    return JsonResponse({'message': 'realizando conexión',
                         'nueva variable': 'soy el contenido'})

def download_file(url, save_path):
    try:
        # Send GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the content of the response to a local file
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully: {save_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

probar()