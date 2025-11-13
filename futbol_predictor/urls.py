from django.contrib import admin
from django.urls import path, include # <--- Asegúrate de importar 'include'

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Esto conecta la raíz del sitio (http://127.0.0.1:8000/) con tu app 'core'
    path('', include('core.urls')), 
]