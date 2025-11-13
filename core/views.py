from django.shortcuts import render
from .services import predictor

def home(request):
    context = {}
    equipos = []

    # --- Lógica Robusta para obtener lista de equipos ---
    if predictor.params and 'encoders' in predictor.params:
        encoders = predictor.params['encoders']
        
        # Buscamos la clave correcta (Equipo o group_id)
        if 'Equipo' in encoders:
            equipos = list(encoders['Equipo'].classes_)
        elif 'group_id' in encoders:
            equipos = list(encoders['group_id'].classes_)
        else:
            context['error'] = "No se encontraron equipos. Claves: " + str(list(encoders.keys()))
        
        equipos.sort()
    else:
        context['error'] = "El modelo no se ha cargado. Revisa la consola."

    context['equipos'] = equipos

    if request.method == 'POST':
        local = request.POST.get('local')
        visitante = request.POST.get('visitante')
        fecha = request.POST.get('fecha')
        hora = request.POST.get('hora')
        
        if local and visitante and fecha:
            if local == visitante:
                context['error'] = "¡Un equipo no puede jugar contra sí mismo!"
            else:
                try:
                    resultado = predictor.predict(local, visitante, fecha, hora, jornada_num=1)
                    context['resultado'] = resultado
                except Exception as e:
                    # Imprimimos el error en la consola para depurar
                    print(f"[ERROR VISTA]: {e}")
                    context['error'] = str(e)
        else:
            context['error'] = "Por favor, completa todos los campos."

    return render(request, 'core/home.html', context)