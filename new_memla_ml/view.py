from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd

def index(request):
    print("hola, estoy llegando")
    return JsonResponse({'message' : 'realizando conexión',
                         'nueva variable' : 'soy el contenido'})