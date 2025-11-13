import pickle
import os
import sys

# Configurar entorno para poder cargar cosas de Django si fuera necesario (aunque aquí usamos pickle puro)
base_path = os.path.dirname(os.path.abspath(__file__))
# LÍNEA CORRECTA (Si 'ml_models' está junto a 'manage.py')
pkl_path = os.path.join(base_path, 'ml_models', 'tft_params.pkl')

print(f"--- Inspeccionando: {pkl_path} ---")

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\n✅ Archivo cargado correctamente.")
    print(f"Claves principales: {list(data.keys())}")
    
    if 'model_params' in data:
        params = data['model_params']
        print("\n--- Dentro de 'model_params' ---")
        print(f"Claves: {list(params.keys())}")
        
        if 'encoders' in params:
            encoders = params['encoders']
            print("\n--- Dentro de 'encoders' (ESTO ES LO IMPORTANTE) ---")
            print(f"Claves disponibles: {list(encoders.keys())}")
            
            # Verificamos qué tipo de objeto es
            first_key = list(encoders.keys())[0]
            print(f"Tipo del primer encoder ({first_key}): {type(encoders[first_key])}")
            
            # Intentamos ver las clases del primer encoder
            try:
                print(f"Clases de ejemplo para {first_key}: {list(encoders[first_key].classes_)[:5]}")
            except:
                print(f"No se pudieron leer las clases de {first_key}")
        else:
            print("❌ NO se encontró la clave 'encoders' dentro de model_params")
    else:
        print("❌ NO se encontró la clave 'model_params' en la raíz")

except FileNotFoundError:
    print("❌ ERROR: No se encuentra el archivo. Verifica la ruta.")
except Exception as e:
    print(f"❌ ERROR al leer el archivo: {e}")