from django.urls import path
from . import views

urlpatterns = [
    # Ruta vacía ('') significa la página de inicio
    path('', views.home, name='home'),
]