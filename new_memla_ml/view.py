from django.http import HttpResponse
from django.http import JsonResponse

def index(request):
    print("hola, estoy llegando")
    return JsonResponse({'message' : 'realizando conexión',
                         'nueva variable' : 'soy el contenido'})