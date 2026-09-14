import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from latent_encoder.model import LatentGraphEncoder
from latent_encoder.utils import capped_attention_regularizer
from telemetry.cic_adapter import load_all_cic
from telemetry.graph_builder import build_windowed_graphs

print("=== PHASE 1: Self-Supervised Encoder Pretraining (DGI per window) ===")
flows = load_all_cic(window_duration=5.0, nrows_per_file=10000, max_files=2, global_reindex=True)
print(f"Total flows {len(flows)}, windows {flows['window_idx'].nunique()}")

graphs = build_windowed_graphs(flows.drop(columns=["label", "timestamp", "source_file"], errors="ignore"))
print(f"Graphs built {len(graphs)}")

encoder = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
encoder.classifier = torch.nn.Linear(8, 1)
print(f"Encoder params: {sum(p.numel() for p in encoder.parameters())}")

# Build labels
labels_seq = []
for wid in sorted(graphs.keys()):
    win_labels = flows[flows["window_idx"] == wid]["label"]
    is_attack = (win_labels != "Benign").mean() > 0.1
    labels_seq.append(1 if is_attack else 0)
print(f"Labels: {sum(labels_seq)}/{len(labels_seq)} attack windows ({sum(labels_seq)/len(labels_seq)*100:.1f}%)")

# Collect all windows as individual graphs
window_data = []
for wid in sorted(graphs.keys()):
    X, Ei, Ea = graphs[wid]
    if X.size(0) < 2:
        continue
    window_data.append((X, Ei, Ea, labels_seq[sorted(graphs.keys()).index(wid)]))

print(f"Usable windows (nodes>=2): {len(window_data)}")

# Phase 1: DGI per-window (mini-batched for speed)
BATCH_SIZE = 32
optimizer = torch.optim.Adam(encoder.parameters(), lr=5e-4)

print("\nTraining DGI (self-supervised)...")
encoder.train()
for epoch in range(100):
    total_loss = 0.0
    count = 0
    np.random.shuffle(window_data)
    for i in range(0, len(window_data), BATCH_SIZE):
        batch_data = window_data[i:i+BATCH_SIZE]
        batch_loss = 0.0
        for X, Ei, Ea, label in batch_data:
            batch = torch.zeros(X.size(0), dtype=torch.long)
            perm = torch.randperm(X.size(0))
            X_corrupt = X[perm]
            z_real = encoder(X, Ei, Ea, batch)
            z_corrupt = encoder(X_corrupt, Ei, Ea, batch)
            pos_sim = F.cosine_similarity(z_real, z_real, dim=1)
            neg_sim = F.cosine_similarity(z_real, z_corrupt, dim=1)
            dgi_loss = -torch.mean(torch.log(torch.sigmoid(pos_sim) + 1e-8) +
                                   torch.log(1 - torch.sigmoid(neg_sim) + 1e-8))
            att_reg = capped_attention_regularizer(None)
            loss = dgi_loss + 0.01 * att_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            count += 1
        total_loss += batch_loss

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: DGI loss = {total_loss/count:.4f}")

print(f"  Final DGI loss: {total_loss/count:.4f}")

# Phase 2: Supervised fine-tuning (mini-batched)
print("\n=== PHASE 2: Supervised Fine-Tuning ===")
optimizer_sup = torch.optim.Adam(encoder.parameters(), lr=1e-4)

for epoch in range(50):
    total_loss = 0.0
    correct = 0
    total = 0
    np.random.shuffle(window_data)
    for i in range(0, len(window_data), BATCH_SIZE):
        batch_data = window_data[i:i+BATCH_SIZE]
        for X, Ei, Ea, label in batch_data:
            batch = torch.zeros(X.size(0), dtype=torch.long)
            z = encoder(X, Ei, Ea, batch)
            target = torch.tensor(label, dtype=torch.float).unsqueeze(0)
            pred = encoder.classifier(z).squeeze(0)
            loss = F.binary_cross_entropy_with_logits(pred, target)

            optimizer_sup.zero_grad()
            loss.backward()
            optimizer_sup.step()
            total_loss += loss.item()

            pred_label = (pred > 0).float().item()
            correct += int(pred_label == label)
            total += 1

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: sup loss = {total_loss/total:.4f}, acc = {correct}/{total} ({correct/total*100:.1f}%)")

print(f"  Final: sup loss = {total_loss/total:.4f}, acc = {correct}/{total} ({correct/total*100:.1f}%)")

# Phase 3: Evaluation
print("\n=== PHASE 3: Evaluation ===")
encoder.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for X, Ei, Ea, label in window_data:
        batch = torch.zeros(X.size(0), dtype=torch.long)
        z = encoder(X, Ei, Ea, batch)
        pred = (encoder.classifier(z).squeeze(0) > 0).float().item()
        correct += int(pred == label)
        total += 1
        all_preds.append(pred)
        all_labels.append(label)

accuracy = correct / total
print(f"Attack detection accuracy: {correct}/{total} = {accuracy*100:.1f}%")

# Save
Path("checkpoints").mkdir(exist_ok=True)
torch.save(encoder.state_dict(), "checkpoints/encoder_trained.pt")
torch.save(encoder.state_dict(), "checkpoints/encoder.pt")
print(f"\nSaved trained encoder to checkpoints/encoder.pt")

# Phase 4: Latent space
print("\n=== PHASE 4: Latent Space ===")
Z_all = []
with torch.no_grad():
    for X, Ei, Ea, label in window_data:
        batch = torch.zeros(X.size(0), dtype=torch.long)
        z = encoder(X, Ei, Ea, batch)
        Z_all.append(z.squeeze(0).numpy())

Z_all = np.array(Z_all)
Z_benign = Z_all[np.array(all_labels) == 0]
Z_attack = Z_all[np.array(all_labels) == 1]
print(f"Z shape: {Z_all.shape}")
if len(Z_benign) > 0 and len(Z_attack) > 0:
    print(f"Benign Z mean: {Z_benign.mean(axis=0).round(3)}")
    print(f"Attack Z mean: {Z_attack.mean(axis=0).round(3)}")
    print(f"Separation: {np.abs(Z_benign.mean(axis=0) - Z_attack.mean(axis=0)).mean():.3f}")

print("\nEncoder training complete.")
