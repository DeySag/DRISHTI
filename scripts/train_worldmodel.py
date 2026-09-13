import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle
from telemetry.cic_adapter import load_all_cic
from telemetry.graph_builder import build_windowed_graphs

print("=== Loading trained encoder ===")
encoder = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
encoder.classifier = torch.nn.Linear(8, 1)
encoder.load_state_dict(torch.load("checkpoints/encoder.pt", map_location="cpu"))
encoder.eval()
print(f"Encoder loaded from checkpoints/encoder.pt")

print("\n=== Loading CIC data ===")
flows = load_all_cic(window_duration=60.0, nrows_per_file=10000, max_files=2, global_reindex=True)
print(f"Total flows {len(flows)}, windows {flows['window_idx'].nunique()}")

graphs = build_windowed_graphs(flows.drop(columns=["label", "timestamp", "source_file"], errors="ignore"))
print(f"Graphs built {len(graphs)}")

# Generate latent sequence Z from trained encoder
print("\n=== Generating latent sequence Z ===")
Z_seq = []
labels_seq = []
for wid in sorted(graphs.keys()):
    X, Ei, Ea = graphs[wid]
    if X.size(0) < 2:
        continue
    batch = torch.zeros(X.size(0), dtype=torch.long)
    with torch.no_grad():
        zt = encoder(X, Ei, Ea, batch)
    Z_seq.append(zt.squeeze(0))
    win_labels = flows[flows["window_idx"] == wid]["label"]
    is_attack = (win_labels != "Benign").mean() > 0.1
    labels_seq.append(1 if is_attack else 0)

Z = torch.stack(Z_seq).unsqueeze(0)  # [1, T, d]
print(f"Z shape: {Z.shape}")
print(f"Z mean: {Z.mean(dim=1).detach().numpy().round(3)}")
print(f"Z std:  {Z.std(dim=1).detach().numpy().round(3)}")
print(f"Attack ratio: {sum(labels_seq)/len(labels_seq):.2f}")

# Train world model
print("\n=== Training World Model (Causal Transformer) ===")
wm = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)

wm.train()
best_loss = float("inf")
for epoch in range(100):
    loss = wm.dynamics_loss(Z)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(wm.state_dict(), "checkpoints/world_model_best.pt")

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: dynamics MSE loss = {loss.item():.4f}")

print(f"  Final loss: {loss.item():.4f}")
print(f"  Best loss:  {best_loss:.4f}")

# Load best and evaluate
wm.load_state_dict(torch.load("checkpoints/world_model_best.pt", map_location="cpu"))
wm.eval()
with torch.no_grad():
    fore = wm.rollout(Z, k_steps=5)
    loss_final = wm.dynamics_loss(Z)

print(f"\nFinal dynamics MSE: {loss_final.item():.4f}")
print(f"Forecast shape: {fore.shape}")

# Save final world model
torch.save(wm.state_dict(), "checkpoints/world_model.pt")
print("Saved world model to checkpoints/world_model.pt")

# Oracle on real data
print("\n=== Training Oracle on real labels ===")
all_Z = Z.squeeze(0).detach().numpy()
y = np.array(labels_seq)
oracle = ExplainableOracle(latent_dim=8)
oracle.fit(all_Z, y)
risk, mitre, attr = oracle.decode_trajectory(fore.squeeze(0).detach().numpy())
print(f"Oracle risk: {risk.round(3)}")
print(f"MITRE stages: {mitre}")

print("\nWorld model retraining complete.")
