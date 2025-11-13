import torch
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from django.conf import settings
import torch.nn.functional as F

# Importamos la arquitectura
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
        self.device = torch.device('cpu') 
        self._load_resources()

    def _load_resources(self):
        print("--- [ML] Cargando recursos de Inteligencia Artificial... ---")
        base_path = os.path.join(settings.BASE_DIR, 'ml_models')
        
        # 1. Cargar Parámetros y DATOS PROCESADOS
        try:
            with open(os.path.join(base_path, 'tft_params.pkl'), 'rb') as f:
                info = pickle.load(f)
                self.params = info['model_params']
                self.config = info['model_config']
                
            # --- CORRECCIÓN CLAVE AQUÍ ---
            # En lugar de leer el CSV crudo, usamos el DataFrame que guardamos en el entrenamiento.
            # Este ya tiene la columna 'Equipo' y las fechas listas.
            if 'processed_df_long' in self.params:
                self.df_history = self.params['processed_df_long']
                print(f"--- [ML] Historial cargado desde Pickle: {len(self.df_history)} registros ---")
            else:
                print("ERROR CRÍTICO: El archivo .pkl no contiene 'processed_df_long'. Re-entrena el modelo.")
                return
            # -----------------------------

        except FileNotFoundError:
            print("ERROR CRÍTICO: No se encontró tft_params.pkl en ml_models/")
            return

        # 2. Cargar Modelo
        try:
            self.model = TemporalFusionTransformer(**self.config).to(self.device)
            model_path = os.path.join(base_path, 'tft_modelo_futbol_entrenado.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("--- [ML] Modelo TFT Cargado y Listo ---")
        except Exception as e:
            print(f"ERROR cargando el modelo: {e}")

    def _get_scaled_history_tensor(self, equipo_nombre, encoder_len):
        scalers = self.params['scalers']
        past_num_cols = [col for col in scalers.keys() if col.startswith('avg_')]
        
        raw_cols_rolling = [
            'Goles_Favor', 'Goles_Contra', 'Puntos', 'posesion', 
            'remates_favor', 'remates_contra', 'remates_puerta_favor', 'remates_puerta_contra',
            'corners_favor', 'corners_contra', 'amarillas_favor', 'amarillas_contra',
            'faltas_favor', 'faltas_contra', 'rojas_favor', 'rojas_contra'
        ]

        # Ahora esto funcionará porque self.df_history tiene la columna 'Equipo'
        df_team = self.df_history[self.df_history['Equipo'] == equipo_nombre].sort_values(by='datetime').copy()
        
        # Recalcular Rolling (por si acaso)
        for col in raw_cols_rolling:
            new_col_name = f"avg_{col}_5"
            if new_col_name in past_num_cols:
                df_team[new_col_name] = df_team[col].shift(1).rolling(5, min_periods=1).mean()
        
        df_team = df_team.fillna(0)
        df_last_n = df_team.tail(encoder_len)
        
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

        # --- 1. Obtener Encoder de Equipos ---
        # Tu inspección confirmó que la clave es 'Equipo'
        if 'Equipo' in encoders:
            team_encoder = encoders['Equipo']
        elif 'group_id' in encoders:
            team_encoder = encoders['group_id']
        else:
            raise Exception(f"Claves disponibles en encoders: {list(encoders.keys())}")

        # --- 2. Construir Tensores ---
        hist_local = self._get_scaled_history_tensor(equipo_local, encoder_len)
        hist_visit = self._get_scaled_history_tensor(equipo_visitante, encoder_len)
        batch_hist = torch.FloatTensor(np.array([hist_local, hist_visit])).to(self.device)

        try:
            local_id = team_encoder.transform([equipo_local])[0]
            visit_id = team_encoder.transform([equipo_visitante])[0]
        except ValueError:
            raise Exception(f"Equipo desconocido: '{equipo_local}' o '{equipo_visitante}'.")
            
        batch_static = torch.LongTensor([[local_id], [visit_id]]).to(self.device)

        dt = datetime.strptime(f"{fecha_str} {hora_str}", "%Y-%m-%d %H:%M")
        jornada_scaled = scalers['Jornada_Num_Raw'].transform(np.array([[jornada_num]]))[0, 0]
        
        # Encoder de Oponente
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
            predictions_tuple, _ = self.model(batch_static, batch_hist, batch_fut)
            pred_goles_scaled, pred_res_logits = predictions_tuple

        # --- 4. Resultados ---
        pred_goles_np = pred_goles_scaled.cpu().numpy()
        local_p50 = target_scaler.inverse_transform(pred_goles_np[0, 0, 1].reshape(1, 1))[0, 0]
        visit_p50 = target_scaler.inverse_transform(pred_goles_np[1, 0, 1].reshape(1, 1))[0, 0]
        
        probs = F.softmax(pred_res_logits, dim=-1).cpu().numpy()
        probs_local = probs[0, 0, :] 
        
        return {
            'local': equipo_local,
            'visitante': equipo_visitante,
            'goles_local': float(f"{local_p50:.2f}"),
            'goles_visitante': float(f"{visit_p50:.2f}"),
            'prob_victoria': round(probs_local[2] * 100, 1),
            'prob_empate': round(probs_local[1] * 100, 1),
            'prob_derrota': round(probs_local[0] * 100, 1),
            'marcador_predicho': f"{int(round(local_p50))} - {int(round(visit_p50))}"
        }

predictor = PredictorFutbol()