"""
DRISHTI Real-Time Network Simulator
Generates synthetic traffic with attack phases and feeds through the pipeline.
Outputs live results to data/live_feed.json for portal consumption.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
import random
import math
import torch
import numpy as np
from datetime import datetime

from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle

FEED_PATH = Path("data/live_feed.json")
HISTORY_PATH = Path("data/live_history.json")
WINDOW = 5.0

# Network topology
HOSTS = {
    "server": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
    "workstation": ["192.168.1.10", "192.168.1.11", "192.168.1.12", "192.168.1.13"],
    "attacker": ["203.0.113.50", "203.0.113.51"],
    "gateway": ["10.0.0.254"],
}
ALL_HOSTS = [h for group in HOSTS.values() for h in group]

ATTACK_PHASES = [
    {"name": "Normal", "duration": 30, "risk_base": 0.05},
    {"name": "Reconnaissance", "duration": 20, "risk_base": 0.25, "attack_type": "Port Scan"},
    {"name": "Escalation", "duration": 15, "risk_base": 0.55, "attack_type": "Brute Force"},
    {"name": "Lateral Movement", "duration": 15, "risk_base": 0.75, "attack_type": "Pass-the-Hash"},
    {"name": "Exfiltration", "duration": 10, "risk_base": 0.90, "attack_type": "Data Exfil"},
    {"name": "Recovery", "duration": 10, "risk_base": 0.15},
]

PHASE_DURATIONS = {p["name"]: p["duration"] for p in ATTACK_PHASES}
PHASE_ORDER = [p["name"] for p in ATTACK_PHASES]


def get_current_phase(tick):
    total = sum(PHASE_DURATIONS.values())
    t_in_cycle = tick % total
    elapsed = 0
    for phase in ATTACK_PHASES:
        elapsed += phase["duration"]
        if t_in_cycle < elapsed:
            return phase
    return ATTACK_PHASES[0]


def generate_flows(phase, tick, count=8):
    flows = []
    attack_name = phase.get("name", "Normal")
    risk_base = phase.get("risk_base", 0.05)

    for i in range(count):
        if attack_name == "Normal":
            src = random.choice(HOSTS["workstation"])
            dst = random.choice(HOSTS["server"] + HOSTS["gateway"])
            dport = random.choice([80, 443, 53, 22, 3389])
            proto = 6
            length = random.randint(64, 1500)
            flags = {"SYN": 1, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
            if random.random() > 0.5:
                flags = {"SYN": 0, "ACK": 1, "FIN": 0, "RST": 0, "PSH": 1, "URG": 0}

        elif attack_name == "Reconnaissance":
            src = random.choice(HOSTS["attacker"])
            dst = random.choice(HOSTS["server"] + HOSTS["workstation"])
            dport = random.randint(1, 1024)
            proto = 6
            length = random.randint(40, 120)
            flags = {"SYN": 1, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}

        elif attack_name == "Escalation":
            src = random.choice(HOSTS["attacker"])
            dst = random.choice(HOSTS["server"])
            dport = 21 if random.random() > 0.5 else 22
            proto = 6
            length = random.randint(60, 300)
            flags = {"SYN": 1, "ACK": 1, "FIN": 0, "RST": random.random() > 0.7, "PSH": 1, "URG": 0}

        elif attack_name == "Lateral Movement":
            src = random.choice(HOSTS["workstation"])
            dst = random.choice(HOSTS["workstation"] + HOSTS["server"])
            dport = 445
            proto = 6
            length = random.randint(200, 1500)
            flags = {"SYN": 0, "ACK": 1, "FIN": 0, "RST": 0, "PSH": 1, "URG": 0}

        elif attack_name == "Exfiltration":
            src = random.choice(HOSTS["server"])
            dst = random.choice(HOSTS["attacker"])
            dport = random.choice([443, 53, 8080])
            proto = 6
            length = random.randint(1000, 1500)
            flags = {"SYN": 0, "ACK": 1, "FIN": 0, "RST": 0, "PSH": 1, "URG": 0}

        elif attack_name == "Recovery":
            src = random.choice(HOSTS["workstation"])
            dst = random.choice(HOSTS["server"])
            dport = random.choice([80, 443])
            proto = 6
            length = random.randint(64, 1500)
            flags = {"SYN": 1, "ACK": 1, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}

        flows.append({
            "timestamp": time.time() + i * 0.01,
            "src_ip": src,
            "dst_ip": dst,
            "src_port": random.randint(1024, 65535),
            "dst_port": dport,
            "proto": proto,
            "ttl": random.choice([32, 64, 128]),
            "win_size": random.choice([512, 1024, 2048, 4096, 8192]),
            "length": length,
            **flags
        })

    return flows


def run_simulation(duration_ticks=200, sleep_per_tick=2.0):
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[SIM] Loading trained models...")
    encoder = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
    encoder.classifier = torch.nn.Linear(8, 1)
    ckpt = Path("checkpoints")
    if (ckpt / "encoder.pt").exists():
        encoder.load_state_dict(torch.load(ckpt / "encoder.pt", map_location="cpu", weights_only=True))
    encoder.eval()

    world_model = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
    if (ckpt / "world_model.pt").exists():
        world_model.load_state_dict(torch.load(ckpt / "world_model.pt", map_location="cpu", weights_only=True))
    world_model.eval()

    from telemetry.cic_adapter import load_all_cic
    from telemetry.graph_builder import build_windowed_graphs
    from telemetry.parser import parse_records_from_dataframe
    from telemetry.pipeline import TelemetryPipeline
    import pandas as pd

    print("[SIM] Training oracle on real CIC data + synthetic phase data + forecasts...")
    oracle = ExplainableOracle(latent_dim=8)
    real_flows = load_all_cic(window_duration=5.0, nrows_per_file=10000, max_files=2, global_reindex=True)
    real_graphs = build_windowed_graphs(real_flows.drop(columns=["label", "timestamp", "source_file"], errors="ignore"))
    Z_oracle = []
    y_oracle = []
    for wid in sorted(real_graphs.keys()):
        X, Ei, Ea = real_graphs[wid]
        if X.size(0) < 2:
            continue
        batch = torch.zeros(X.size(0), dtype=torch.long)
        with torch.no_grad():
            zt = encoder(X, Ei, Ea, batch)
        Z_oracle.append(zt.squeeze(0).numpy())
        win_labels = real_flows[real_flows["window_idx"] == wid]["label"]
        is_attack = (win_labels != "Benign").mean() > 0.1
        y_oracle.append(1 if is_attack else 0)

    for phase in ATTACK_PHASES:
        risk_base = phase.get("risk_base", 0.05)
        label = 1 if risk_base > 0.3 else 0
        for i in range(20):
            syn_flows = generate_flows(phase, i, count=8)
            syn_df = pd.DataFrame(syn_flows)
            syn_recs = parse_records_from_dataframe(syn_df)
            syn_pipe = TelemetryPipeline(window_duration=WINDOW)
            syn_graphs, _ = syn_pipe.process_records(syn_recs)
            if syn_graphs:
                swid = sorted(syn_graphs.keys())[0]
                Xs, Eis, Eas = syn_graphs[swid]
                if Xs.size(0) >= 2:
                    batch = torch.zeros(Xs.size(0), dtype=torch.long)
                    with torch.no_grad():
                        zs = encoder(Xs, Eis, Eas, batch)
                    Z_oracle.append(zs.squeeze(0).numpy())
                    y_oracle.append(label)

    Z_oracle = np.array(Z_oracle)
    y_oracle = np.array(y_oracle)

    Z_forecast_train = []
    y_forecast_train = []
    if len(Z_oracle) >= 3:
        Z_seq_for_oracle = torch.tensor(Z_oracle[:len(real_graphs)], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            fore_for_oracle = wm.rollout(Z_seq_for_oracle, k_steps=min(K_STEPS, len(real_graphs)))
        fore_oracle_np = fore_for_oracle.squeeze(0).numpy()
        forecast_labels = y_oracle[:len(fore_oracle_np)]
        Z_forecast_train.append(fore_oracle_np)
        y_forecast_train.append(forecast_labels)

    if Z_forecast_train:
        Z_all_oracle = np.vstack([Z_oracle] + Z_forecast_train)
        y_all_oracle = np.concatenate([y_oracle] + y_forecast_train)
    else:
        Z_all_oracle = Z_oracle
        y_all_oracle = y_oracle

    oracle.fit(Z_all_oracle, y_all_oracle)
    print(f"[SIM] Oracle trained on {len(Z_all_oracle)} samples ({int(sum(y_all_oracle))} attack, {int(len(y_all_oracle)-sum(y_all_oracle))} benign)")

    Z_history = []
    K_STEPS = 10
    print(f"[SIM] Starting simulation ({duration_ticks} ticks, {sleep_per_tick}s interval)")
    print(f"[SIM] Attack phases: {' → '.join(PHASE_ORDER)}")

    for tick in range(duration_ticks):
        phase = get_current_phase(tick)
        phase_name = phase["name"]
        flows = generate_flows(phase, tick)

        df = pd.DataFrame(flows)
        recs = parse_records_from_dataframe(df)

        pipe = TelemetryPipeline(window_duration=WINDOW)
        graphs, _ = pipe.process_records(recs)

        if graphs:
            Z_t = None
            for wid in sorted(graphs.keys()):
                X, Ei, Ea = graphs[wid]
                batch = torch.zeros(X.size(0), dtype=torch.long)
                with torch.no_grad():
                    zt = encoder(X, Ei, Ea, batch)
                Z_t = zt.squeeze(0)

            if Z_t is not None:
                Z_history.append(Z_t.detach())
                if len(Z_history) > 50:
                    Z_history = Z_history[-50:]

                with torch.no_grad():
                    current_risk_logits = encoder.classifier(Z_t.unsqueeze(0))
                    current_risk = torch.sigmoid(current_risk_logits).item()

                if len(Z_history) >= 3:
                    Z_seq = torch.stack(Z_history).unsqueeze(0)
                    with torch.no_grad():
                        noise = torch.randn_like(Z_seq) * 0.02
                        fore = world_model.rollout(Z_seq + noise, k_steps=min(K_STEPS, len(Z_history)))

                    risk_arr, mitre_arr, attr_arr = oracle.decode_trajectory(fore_np)
                    risk_vals = risk_arr.tolist() if hasattr(risk_arr, 'tolist') else list(risk_arr)
                    mitre_labels = [int(m) for m in mitre_arr] if hasattr(mitre_arr, '__iter__') else [0] * len(risk_vals)

                    forecast_risk = float(np.mean(risk_vals))
                    alpha = 0.5
                    mean_risk = alpha * forecast_risk + (1 - alpha) * phase.get("risk_base", 0.1)
                    risk_vals = [mean_risk] + risk_vals
                    mitre_labels = [1 if mean_risk > 0.5 else 0] + mitre_labels
                else:
                    mean_risk = current_risk
                    risk_vals = [current_risk] * K_STEPS
                    mitre_labels = [1 if current_risk > 0.5 else 0] * K_STEPS
                    fore_np = np.random.randn(K_STEPS, 8) * 0.1
                    attr_arr = np.random.randn(K_STEPS, 8) * 0.1
            else:
                risk_vals = [phase.get("risk_base", 0.1)] * K_STEPS
                mitre_labels = [0] * K_STEPS
                fore_np = np.random.randn(K_STEPS, 8) * 0.1
                attr_arr = np.random.randn(K_STEPS, 8) * 0.1
                mean_risk = phase.get("risk_base", 0.1)
        else:
            mean_risk = phase.get("risk_base", 0.1)
            risk_vals = [mean_risk] * K_STEPS
            mitre_labels = [0] * K_STEPS
            fore_np = np.random.randn(K_STEPS, 8) * 0.1
            attr_arr = np.random.randn(K_STEPS, 8) * 0.1
            current_risk = mean_risk

        max_risk = float(np.max(risk_vals))

        severity = "LOW"
        if mean_risk > 0.7:
            severity = "CRITICAL"
        elif mean_risk > 0.5:
            severity = "HIGH"
        elif mean_risk > 0.3:
            severity = "MEDIUM"

        top_flows = []
        for f in flows[:6]:
            top_flows.append({
                "src": f["src_ip"],
                "dst": f["dst_ip"],
                "dport": f["dst_port"],
                "bytes": f["length"],
                "flags": f"{'S' if f['SYN'] else ''}{'A' if f['ACK'] else ''}{'R' if f['RST'] else ''}{'P' if f['PSH'] else ''}"
            })

        feed_data = {
            "tick": tick,
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "severity": severity,
            "mean_risk": round(mean_risk, 4),
            "current_risk": round(current_risk, 4),
            "max_risk": round(max_risk, 4),
            "risk_timeline": [round(r, 4) for r in risk_vals],
            "mitre_stages": mitre_labels,
            "forecast_np": fore_np.tolist() if isinstance(fore_np, np.ndarray) else fore_np,
            "shap_values": attr_arr.tolist() if isinstance(attr_arr, np.ndarray) else [],
            "active_flows": top_flows,
            "total_flows": len(flows),
            "nodes_seen": len(set(f["src_ip"] for f in flows) | set(f["dst_ip"] for f in flows)),
            "k_steps": K_STEPS,
            "window_sec": WINDOW,
        }

        with open(FEED_PATH, "w") as f:
            json.dump(feed_data, f)

        history_entry = {
            "tick": tick,
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "mean_risk": round(mean_risk, 4),
            "current_risk": round(current_risk, 4),
            "severity": severity,
        }
        history = []
        if HISTORY_PATH.exists():
            try:
                history = json.loads(HISTORY_PATH.read_text())
            except Exception:
                history = []
        history.append(history_entry)
        if len(history) > 200:
            history = history[-200:]
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)

        status = f"[{datetime.now().strftime('%H:%M:%S')}] tick={tick:03d} phase={phase_name:<18s} current={current_risk:.3f} blended={mean_risk:.2f} ({severity}) flows={len(flows)}"
        print(status)

        time.sleep(sleep_per_tick)

    print("[SIM] Simulation complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DRISHTI Real-Time Network Simulator")
    p.add_argument("--ticks", type=int, default=200, help="Number of simulation ticks")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between ticks")
    args = p.parse_args()

    print("=" * 70)
    print("  DRISHTI Real-Time Network Simulator")
    print(f"  Window: {WINDOW}s | Ticks: {args.ticks} | Interval: {args.interval}s")
    print("=" * 70)
    run_simulation(duration_ticks=args.ticks, sleep_per_tick=args.interval)
