import math
import torch
import torch.nn as nn


class CausalTemporalTransformer(nn.Module):
    """
    Part 3: Transition Dynamics Engine (The World Model)
    Bespoke Temporal Transformer with causal masking and K-step rollout
    P(z_{t+1} | z_1..z_t), Eq (1) PE, Eq (2) Masked Attention, Eq (3) MSE
    """

    def __init__(self, latent_dim, num_heads=4, num_layers=2, hidden_dim=64, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, latent_dim))
        nn.init.normal_(self.pos_encoder, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.state_predictor = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def positional_encoding_sin(self, seq_len, d_model, device):
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe.unsqueeze(0)

    def forward(self, src):
        seq_len = src.size(1)
        src = src + self.pos_encoder[:, :seq_len, :]
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(src, mask=mask)
        output = self.norm(output)
        next_state_pred = self.state_predictor(output)
        return next_state_pred

    def rollout(self, current_sequence, k_steps):
        self.eval()
        simulated_trajectory = []
        seq = current_sequence.clone()
        with torch.no_grad():
            for _ in range(k_steps):
                predictions = self.forward(seq)
                next_state = predictions[:, -1:, :]
                simulated_trajectory.append(next_state)
                seq = torch.cat([seq[:, 1:, :], next_state], dim=1)
        return torch.cat(simulated_trajectory, dim=1)

    def dynamics_loss(self, src):
        pred = self.forward(src[:, :-1, :])
        target = src[:, 1:, :]
        loss = nn.functional.mse_loss(pred, target)
        return loss