import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Gated Linear Unit (GLU) ---
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, output_size=None, dropout=0.1):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        values = self.linear1(x)
        gates = torch.sigmoid(self.linear2(x))
        output = values * gates
        return self.dropout(output)

# --- 2. Gated Residual Network (GRN) ---
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=None, output_size=None,
                dropout=0.1, context_size=None):
        super().__init__()
        if hidden_size is None: hidden_size = input_size
        if output_size is None: output_size = input_size
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.linear2 = nn.Linear(hidden_size + context_size, hidden_size)
        else:
            self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        if input_size != output_size:
            self.skip_projection = nn.Linear(input_size, output_size)
        else:
            self.skip_projection = None

    def forward(self, x, context=None):
        residual = x
        x = F.elu(self.linear1(x))
        if context is not None and self.context_size is not None:
            x = torch.cat([x, context], dim=-1)
        x = F.elu(self.linear2(x))
        x = self.glu(x)
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        return self.layer_norm(x + residual)

# --- 3. Variable Selection Network (VSN) ---
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_variables, hidden_size, dropout=0.1):
        super().__init__()
        self.num_variables = num_variables
        self.input_size = input_size
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_variables)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_size * num_variables, hidden_size, num_variables, dropout
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, variables):
        processed_vars = []
        for i, var in enumerate(variables):
            processed = self.variable_grns[i](var)
            processed_vars.append(processed)
        all_vars = torch.cat(variables, dim=-1)
        selection_weights = self.selection_grn(all_vars)
        selection_weights = self.softmax(selection_weights)
        weighted_vars = []
        for i, processed_var in enumerate(processed_vars):
            weight = selection_weights[..., i:i+1]
            weighted_var = processed_var * weight
            weighted_vars.append(weighted_var)
        selected_output = torch.cat(weighted_vars, dim=-1)
        return selected_output, selection_weights

# --- 4. Multi-Head Attention ---
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_query = query.size(0), query.size(1)
        seq_len_key_val = key.size(1)
        residual = query
        Q = self.w_q(query).view(batch_size, seq_len_query, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_key_val, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_key_val, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        attention_output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.d_model)
        output = self.w_o(attention_output)
        return self.layer_norm(output + residual), attention_weights

# --- 6. Arquitectura Principal TFT (CORREGIDA: SIN GLOBAL DEVICE) ---
class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_static_cat, num_hist_numeric, num_fut_numeric, num_fut_categorical,
                 vocab_sizes, categorical_indices, sequence_length, prediction_length,
                 hidden_size=64, num_heads=4, dropout=0.1, num_quantiles=3, num_classes=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.prediction_length = prediction_length
        self.num_quantiles = num_quantiles
        self.num_classes = num_classes
        self.num_static_cat = num_static_cat
        self.num_hist_numeric = num_hist_numeric
        self.num_fut_numeric = num_fut_numeric
        self.num_fut_categorical = num_fut_categorical
        self.cat_indices = categorical_indices
        self.vocab_sizes = vocab_sizes

        self.static_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_sizes['group_id'], hidden_size)
        ])
        self.future_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_sizes[col], hidden_size) 
            for col in ['Localia', 'Oponente_id', 'Mes', 'Dia_semana']
        ])
        self.hist_numeric_projection = nn.Linear(1, hidden_size)
        self.fut_numeric_projection = nn.Linear(1, hidden_size)

        self.static_vsn = VariableSelectionNetwork(hidden_size, self.num_static_cat, hidden_size, dropout)
        self.historical_vsn = VariableSelectionNetwork(hidden_size, self.num_hist_numeric, hidden_size, dropout)
        self.future_vsn = VariableSelectionNetwork(hidden_size, self.num_fut_numeric + self.num_fut_categorical, hidden_size, dropout)

        self.lstm_encoder = nn.LSTM(hidden_size * self.num_hist_numeric, hidden_size, batch_first=True, dropout=dropout)
        self.lstm_decoder = nn.LSTM(hidden_size * (self.num_fut_numeric + self.num_fut_categorical), hidden_size, batch_first=True, dropout=dropout)

        self.static_context_grn_c = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.static_context_grn_h = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.static_context_grn_enrich = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
        self.attention_gating = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.final_gating = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.final_projection_goles = nn.Linear(hidden_size, num_quantiles)
        self.final_projection_resultado = nn.Linear(hidden_size, num_classes)

    def process_static_inputs(self, x_static_cat):
        embedded_static = [self.static_embeddings[i](x_static_cat[..., i]) for i in range(self.num_static_cat)]
        selected_static, _ = self.static_vsn(embedded_static)
        return selected_static.squeeze(1)

    def process_temporal_inputs(self, x_num, x_cat, projections, embeddings, cat_indices, vsn):
        all_vars = []
        for i in range(x_num.shape[-1]):
            all_vars.append(projections[i](x_num[..., i:i+1]))
        for i in range(x_cat.shape[-1]):
            all_vars.append(embeddings[i](x_cat[..., i]))
        selected_vars, _ = vsn(all_vars)
        return selected_vars

    def forward(self, x_static, x_hist, x_fut):
        x_static_cat = x_static.long()
        x_hist_num = x_hist
        
        # --- CORRECCIÓN CRÍTICA: Usar el dispositivo del tensor de entrada ---
        current_device = x_hist.device # Detectamos dónde está x_hist (CPU o GPU)
        x_hist_cat = torch.empty(x_hist.shape[0], x_hist.shape[1], 0, device=current_device) 
        # -------------------------------------------------------------------
        
        x_fut_num = x_fut[..., 0:1].float()
        x_fut_cat = x_fut[..., 1:].long()

        static_context = self.process_static_inputs(x_static_cat)
        historical_features = self.process_temporal_inputs(
            x_hist_num, x_hist_cat, [self.hist_numeric_projection] * self.num_hist_numeric,
            [], self.cat_indices['historical'], self.historical_vsn
        )
        future_features = self.process_temporal_inputs(
            x_fut_num, x_fut_cat, [self.fut_numeric_projection], 
            self.future_embeddings, self.cat_indices['future'], self.future_vsn
        )

        static_c = self.static_context_grn_c(static_context).unsqueeze(0)
        static_h = self.static_context_grn_h(static_context).unsqueeze(0)
        
        encoder_out, (hidden, cell) = self.lstm_encoder(historical_features, (static_h, static_c))
        decoder_out, _ = self.lstm_decoder(future_features, (hidden, cell))

        static_enrichment = self.static_context_grn_enrich(static_context).unsqueeze(1)
        lstm_all_out = torch.cat([encoder_out, decoder_out], dim=1) + static_enrichment

        encoder_len = x_hist.shape[1]
        attention_out, _ = self.attention(
            query=lstm_all_out[:, encoder_len:, :],
            key=lstm_all_out, value=lstm_all_out
        )
        
        attention_out = self.attention_gating(attention_out)
        attention_out = self.attention_norm(attention_out + lstm_all_out[:, encoder_len:, :])
        
        final_out = self.final_gating(attention_out)
        pred_goles = self.final_projection_goles(final_out)
        pred_resultado = self.final_projection_resultado(final_out)
        
        return (pred_goles, pred_resultado), None