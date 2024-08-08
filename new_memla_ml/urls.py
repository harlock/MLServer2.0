"""
URL configuration for new_memla_ml project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import main
from . import view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('probarconexion/', view.index),
    path('load_file/', main.load_file, name='load_file'),
    path('delete_duplicate/', main.delete_duplicates, name='delete_duplicate'),
    path('change_value/', main.changeValueRequest, name='change_value'),
    path('split_data/', main.split_data, name='split_data'),
    path('drop_column/', main.drop_column, name='drop_column'),
]
