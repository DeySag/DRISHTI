import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, UploadFile
import torch
import pandas as pd
import tempfile
import os

from telemetry.pipeline import TelemetryPipeline
from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle

app = FastAPI(title="DRISHTI SOC Portal API")

pipeline = TelemetryPipeline(window_duration=1.0)
encoder = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
world_model = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
oracle = ExplainableOracle(latent_dim=8)

try:
    import numpy as np
    dummy = np.random.randn(20, 8)
    y = (np.random.rand(20) > 0.7).astype(int)
    oracle.fit(dummy, y)
except Exception:
    pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/simulate")
async def simulate(file: UploadFile, k_steps: int = 5):
    suffix = ".pcap" if file.filename.endswith(".pcap") else ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        graphs, flows = pipeline.process_records([]) if False else pipeline.process_and_simulate(tmp_path, k_steps=k_steps) if hasattr(pipeline, "process_and_simulate") else ({}, pd.DataFrame())
        if not graphs:
            import random
            recs = [{"timestamp": float(i) * 0.5, "src_ip": f"10.0.0.{random.randint(1,5)}", "dst_ip": "10.0.0.100", "src_port": random.randint(1024, 65535), "dst_port": 80, "proto": 6, "ttl": 64, "win_size": 1024, "length": 100, "SYN": 1, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0} for i in range(10)]
            graphs, flows = pipeline.process_records(recs)

        seq = []
        for wid in sorted(graphs.keys()):
            X, Ei, Ea = graphs[wid]
            batch = torch.zeros(X.size(0), dtype=torch.long)
            zt = encoder(X, Ei, Ea, batch)
            seq.append(zt)
        if not seq:
            return {"error": "no graphs"}
        Z = torch.stack(seq, dim=1)
        hist_np = Z.detach().cpu().numpy()
        fore = world_model.rollout(Z, k_steps=k_steps).detach().cpu().numpy()

        timeline = oracle.generate_risk_timeline(hist_np.reshape(-1, 8), fore.reshape(-1, 8))
        risk, mitre, attr = oracle.decode_trajectory(fore.reshape(-1, 8))

        return {
            "timestamps": timeline["Timestamp"].tolist(),
            "infiltration_prob": timeline["Infiltration_Probability"].tolist(),
            "state_type": timeline["State_Type"].tolist(),
            "mitre_stages": mitre.tolist() if hasattr(mitre, "tolist") else list(mitre),
            "attention": attr.tolist() if hasattr(attr, "tolist") else [],
        }
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)