import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
import csv
from telemetry.cic_adapter import CACHE_ROOT, load_all_cic
from telemetry.graph_builder import build_windowed_graphs
from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle
import numpy as np

DATA_DIR = CACHE_ROOT
CHECKPOINT = Path("checkpoints")
WINDOW = 60.0
K_STEPS = 5
LOG_PATH = Path("data/pull_logs.csv")

_pull_count = 0

def run_once(offset=None):
    global _pull_count
    import random
    if offset is None:
        random.seed(_pull_count * 999)
        offset = random.randint(0, 80000)
        max_files = random.choice([1, 2, 3])
    else:
        max_files = 2
    flows = load_all_cic(window_duration=WINDOW, nrows_per_file=5000, max_files=max_files, offset=offset, global_reindex=True)
    graphs = build_windowed_graphs(flows.drop(columns=["label","timestamp","source_file"], errors="ignore"))
    enc = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
    enc.classifier = torch.nn.Linear(8, 1)
    if (CHECKPOINT/"encoder.pt").exists():
        enc.load_state_dict(torch.load(CHECKPOINT/"encoder.pt"))
    enc.eval()
    with torch.no_grad():
        Z_seq = [enc(*graphs[w], torch.zeros(graphs[w][0].size(0), dtype=torch.long)).squeeze(0) for w in sorted(graphs.keys())]
    Z = torch.stack(Z_seq).unsqueeze(0)
    wm = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
    if (CHECKPOINT/"world_model.pt").exists():
        wm.load_state_dict(torch.load(CHECKPOINT/"world_model.pt"))
    wm.eval()
    with torch.no_grad():
        noise = torch.randn_like(Z) * 0.08 * (1 + _pull_count % 4 * 0.25)
        Z_noisy = Z + noise
        fore = wm.rollout(Z_noisy, k_steps=K_STEPS)
    oracle = ExplainableOracle(latent_dim=8)
    y = np.array([1 if (flows[flows.window_idx==w]["label"]!="Benign").mean()>0.1 else 0 for w in sorted(graphs.keys())])
    oracle.fit(Z.squeeze(0).detach().numpy(), y)
    risk, mitre, _ = oracle.decode_trajectory(fore.squeeze(0).detach().numpy())
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] pull#{_pull_count} offset {offset} windows {len(graphs)} -> risk {risk.round(2).tolist()} mitre {list(mitre)[:3] if hasattr(mitre,'__len__') else mitre} attack_ratio {y.mean():.2f}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["pull","timestamp","windows","risk_mean","risks","attack_ratio"])
        w.writerow([_pull_count, ts, len(graphs), float(risk.mean()), ";".join(f"{x:.2f}" for x in risk), float(y.mean())])
    _pull_count += 1
    return risk

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="continuous pull")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--offset", type=int, default=None)
    args=p.parse_args()
    if args.loop:
        print(f"Starting pull loop every {args.interval}s (Ctrl+C to stop) - sliding offset 5000/pull")
        try:
            while True:
                run_once(offset=None if args.offset is None else args.offset)
                if args.offset is not None:
                    args.offset += 5000
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\nStopped. Log at {LOG_PATH}")
    else:
        run_once(offset=args.offset)