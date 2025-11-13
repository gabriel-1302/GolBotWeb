import torch
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from django.conf import settings
import torch.nn.functional as F

# Importamos la arquitectura del modelo
try:
    from futbol_predictor.tft_arch import TemporalFusionTransformer
except ImportError:
    from core.tft_arch import TemporalFusionTransformer

class PredictorFutbol:
    def __init__(self):
        self.model = None
        self.params = None
        self.config = None
        self.df_history = None
        self.device = torch.device('cpu') # CPU para evitar problemas de concurrencia en web
        
        # Cargar recursos al iniciar la clase
        self._load_resources()

    def _load_resources(self):
        print("--- [ML] Cargando recursos de Inteligencia Artificial... ---")
        base_path = os.path.join(settings.BASE_DIR, 'ml_models')
        
        # 1. Cargar Parámetros y Datos Procesados
        try:
            with open(os.path.join(base_path, 'tft_params.pkl'), 'rb') as f:
                info = pickle.load(f)
                self.params = info['model_params']
                self.config = info['model_config']
                
            # Usamos el DataFrame procesado que guardamos en el entrenamiento
            if 'processed_df_long' in self.params:
                self.df_history = self.params['processed_df_long']
                print(f"--- [ML] Historial cargado desde Pickle: {len(self.df_history)} registros ---")
            else:
                print("ERROR CRÍTICO: El archivo .pkl no contiene 'processed_df_long'.")
                return

        except FileNotFoundError:
            print("ERROR CRÍTICO: No se encontró tft_params.pkl en ml_models/")
            return

        # 2. Cargar Modelo
        try:
            self.model = TemporalFusionTransformer(**self.config).to(self.device)
            model_path = os.path.join(base_path, 'tft_modelo_futbol_entrenado.pth')
            
            # map_location es vital para cargar en CPU un modelo entrenado en GPU
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("--- [ML] Modelo TFT Cargado y Listo ---")
        except Exception as e:
            print(f"ERROR cargando el modelo: {e}")

    def _get_scaled_history_tensor(self, equipo_nombre, encoder_len):
        """Construye el tensor histórico para un equipo."""
        scalers = self.params['scalers']
        past_num_cols = [col for col in scalers.keys() if col.startswith('avg_')]
        
        raw_cols_rolling = [
            'Goles_Favor', 'Goles_Contra', 'Puntos', 'posesion', 
            'remates_favor', 'remates_contra', 'remates_puerta_favor', 'remates_puerta_contra',
            'corners_favor', 'corners_contra', 'amarillas_favor', 'amarillas_contra',
            'faltas_favor', 'faltas_contra', 'rojas_favor', 'rojas_contra'
        ]

        # Filtrar historial del equipo
        df_team = self.df_history[self.df_history['Equipo'] == equipo_nombre].sort_values(by='datetime').copy()
        
        # Recalcular Rolling (para asegurar frescura de datos)
        for col in raw_cols_rolling:
            new_col_name = f"avg_{col}_5"
            if new_col_name in past_num_cols:
                df_team[new_col_name] = df_team[col].shift(1).rolling(5, min_periods=1).mean()
        
        df_team = df_team.fillna(0)
        df_last_n = df_team.tail(encoder_len)
        
        # Padding si faltan datos (equipos nuevos)
        if len(df_last_n) < encoder_len:
            padding_len = encoder_len - len(df_last_n)
            padding = np.zeros((padding_len, len(past_num_cols)))
            if len(df_last_n) > 0:
                hist_list = [scalers[col].transform(df_last_n[[col]]) for col in past_num_cols]
                data_part = np.hstack(hist_list)
                final_hist = np.vstack([padding, data_part])
            else:
                final_hist = padding
        else:
            hist_list = [scalers[col].transform(df_last_n[[col]]) for col in past_num_cols]
            final_hist = np.hstack(hist_list)
            
        return final_hist

    def predict(self, equipo_local, equipo_visitante, fecha_str, hora_str, jornada_num=1):
        if self.model is None:
            raise Exception("El modelo no está cargado. Revisa la consola del servidor.")

        encoders = self.params['encoders']
        scalers = self.params['scalers']
        target_scaler = self.params['target_scaler']
        encoder_len = self.params['encoder_length']

        # --- 1. Obtener Encoder de Equipos (Robustez) ---
        if 'Equipo' in encoders:
            team_encoder = encoders['Equipo']
        elif 'group_id' in encoders:
            team_encoder = encoders['group_id']
        else:
            raise Exception(f"No se encontró codificador de equipos. Claves: {list(encoders.keys())}")

        # --- 2. Construir Tensores ---
        
        # Histórico (X_hist)
        hist_local = self._get_scaled_history_tensor(equipo_local, encoder_len)
        hist_visit = self._get_scaled_history_tensor(equipo_visitante, encoder_len)
        batch_hist = torch.FloatTensor(np.array([hist_local, hist_visit])).to(self.device)

        # Estático (X_static)
        try:
            local_id = team_encoder.transform([equipo_local])[0]
            visit_id = team_encoder.transform([equipo_visitante])[0]
        except ValueError:
            raise Exception(f"Equipo desconocido: '{equipo_local}' o '{equipo_visitante}'")
            
        batch_static = torch.LongTensor([[local_id], [visit_id]]).to(self.device)

        # Futuro (X_fut)
        dt = datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M")
        
        jornada_scaled = scalers['Jornada_Num_Raw'].transform(np.array([[jornada_num]]))[0, 0]
        
        # Encoder Oponente (Robustez)
        if 'Oponente' in encoders:
            op_encoder = encoders['Oponente']
        elif 'Oponente_id' in encoders:
            op_encoder = encoders['Oponente_id']
        else:
            op_encoder = team_encoder 

        op_local_id = op_encoder.transform([equipo_visitante])[0]
        op_visit_id = op_encoder.transform([equipo_local])[0]
        
        mes_id = encoders['Mes'].transform([dt.month])[0]
        dia_id = encoders['Dia_semana'].transform([dt.weekday()])[0]

        fut_data_local = [jornada_scaled, 1, op_local_id, mes_id, dia_id] 
        fut_data_visit = [jornada_scaled, 0, op_visit_id, mes_id, dia_id] 
        
        batch_fut = torch.FloatTensor([fut_data_local, fut_data_visit]).unsqueeze(1).to(self.device)

        # --- 3. Inferencia ---
        with torch.no_grad():
            # Tupla: (pred_goles, pred_resultado)
            predictions_tuple, _ = self.model(batch_static, batch_hist, batch_fut)
            pred_goles_scaled, pred_res_logits = predictions_tuple

        # --- 4. Post-Procesamiento (LIMPIEZA DE DATOS) ---
        
        # A. Goles (Desnormalizar P50)
        pred_goles_np = pred_goles_scaled.cpu().numpy()
        
        # Obtener valores crudos
        raw_local = target_scaler.inverse_transform(pred_goles_np[0, 0, 1].reshape(1, 1))[0, 0]
        raw_visit = target_scaler.inverse_transform(pred_goles_np[1, 0, 1].reshape(1, 1))[0, 0]
        
        # B. Resultado (Probabilidades)
        probs = F.softmax(pred_res_logits, dim=-1).cpu().numpy()
        raw_probs = probs[0, 0, :] # [Derrota, Empate, Victoria]
        
        # --- C. CONVERSIÓN A PYTHON NATIVO Y REDONDEO ---
        # Usamos float(f"{...}") para cortar decimales y asegurar que sea serializable por JSON
        
        clean_goles_local = float(f"{raw_local:.2f}")
        clean_goles_visit = float(f"{raw_visit:.2f}")
        
        clean_prob_der = float(f"{raw_probs[0] * 100:.2f}") # Derrota
        clean_prob_emp = float(f"{raw_probs[1] * 100:.2f}") # Empate
        clean_prob_vic = float(f"{raw_probs[2] * 100:.2f}") # Victoria

        # --- 5. Formatear Respuesta Final ---
        return {
            'local': equipo_local,
            'visitante': equipo_visitante,
            
            'goles_local': clean_goles_local,
            'goles_visitante': clean_goles_visit,
            
            'prob_victoria': clean_prob_vic,
            'prob_empate': clean_prob_emp,
            'prob_derrota': clean_prob_der,
            
            'marcador_predicho': f"{int(round(clean_goles_local))} - {int(round(clean_goles_visit))}"
        }

# Instancia global
predictor = PredictorFutbol()