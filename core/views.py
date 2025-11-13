from django.shortcuts import render
from .services import predictor

from django.shortcuts import render
from .services import predictor

from django.shortcuts import render
from django.http import JsonResponse # <--- IMPORTANTE
from .services import predictor
import json

from django.shortcuts import render
from django.http import JsonResponse
from .services import predictor

def home(request):
    context = {}
    equipos = []

    # Cargar equipos (Igual que siempre)
    if predictor.params and 'encoders' in predictor.params:
        encoders = predictor.params['encoders']
        if 'Equipo' in encoders:
            equipos = list(encoders['Equipo'].classes_)
        elif 'group_id' in encoders:
            equipos = list(encoders['group_id'].classes_)
        equipos.sort()
    context['equipos'] = equipos

    if request.method == 'POST':
        # Detectar si es una llamada AJAX (JavaScript)
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'

        local = request.POST.get('local')
        visitante = request.POST.get('visitante')
        fecha = request.POST.get('fecha')
        hora = request.POST.get('hora')
        jornada = request.POST.get('jornada')

        if local and visitante and fecha and hora and jornada:
            if local == visitante:
                error_msg = "¡Un equipo no puede jugar contra sí mismo!"
                if is_ajax: return JsonResponse({'success': False, 'error': error_msg})
                context['error'] = error_msg
            else:
                try:
                    resultado = predictor.predict(local, visitante, fecha, hora, int(jornada))
                    
                    # --- AQUÍ ESTÁ EL TRUCO ---
                    if is_ajax:
                        # Si es JS, devolvemos JSON para los gráficos dinámicos
                        return JsonResponse({'success': True, 'data': resultado})
                    else:
                        # Si es envío normal, devolvemos la página HTML completa
                        context['resultado'] = resultado
                        
                except Exception as e:
                    print(f"[ERROR VISTA]: {e}")
                    if is_ajax: return JsonResponse({'success': False, 'error': str(e)})
                    context['error'] = str(e)
        else:
            error_msg = "Faltan datos."
            if is_ajax: return JsonResponse({'success': False, 'error': error_msg})
            context['error'] = error_msg

    return render(request, 'core/home.html', context)