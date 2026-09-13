import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import pandas as pd
from telemetry.cic_adapter import load_all_cic
from telemetry.graph_builder import build_windowed_graphs, build_graph_tensor
from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle

print("Loading CIC flows...")
flows = load_all_cic(window_duration=60.0, nrows_per_file=15000, max_files=2)
print(f"Total flows {len(flows)}, windows {flows['window_idx'].nunique()}")

# Build graphs per window
graphs = build_windowed_graphs(flows.drop(columns=["label", "timestamp", "source_file"], errors="ignore"))
print(f"Graphs built {len(graphs)}")

encoder = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
Z_seq = []
labels_seq = []
for wid in sorted(graphs.keys()):
    X, Ei, Ea = graphs[wid]
    batch = torch.zeros(X.size(0), dtype=torch.long)
    with torch.no_grad():
        zt = encoder(X, Ei, Ea, batch)
    Z_seq.append(zt.squeeze(0))
    # window label = majority attack
    win_labels = flows[flows["window_idx"] == wid]["label"]
    is_attack = (win_labels != "Benign").mean() > 0.1
    labels_seq.append(1 if is_attack else 0)

Z = torch.stack(Z_seq).unsqueeze(0)  # [1, T, d]
print(f"Z shape {Z.shape}, attack ratio {sum(labels_seq)/len(labels_seq):.2f}")

# Train world model
wm = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
wm.train()
for epoch in range(30):
    loss = wm.dynamics_loss(Z)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss {loss.item():.4f}")

# Oracle
wm.eval()
with torch.no_grad():
    fore = wm.rollout(Z, k_steps=5)
print(f"Forecast {fore.shape}")

all_Z = Z.squeeze(0).detach().numpy()
y = np.array(labels_seq)
oracle = ExplainableOracle(latent_dim=8)
oracle.fit(all_Z, y)
risk, mitre, attr = oracle.decode_trajectory(fore.squeeze(0).detach().numpy())
print(f"Oracle risk {risk}")
print(f"Mitre {mitre}")
print("Demo done, saving models to checkpoints/")
Path("checkpoints").mkdir(exist_ok=True)
torch.save(encoder.state_dict(), "checkpoints/encoder.pt")
torch.save(wm.state_dict(), "checkpoints/world_model.pt")
print("Saved")
