# DRISHTI — Dynamic Risk and Infiltration Sensing via Heuristic Threat Intelligence

A predictive cybersecurity threat intelligence platform that models network dynamics as a "world model," enabling K-step forward simulation of attack trajectories before they materialize.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DRISHTI Pipeline                                    │
│                                                                             │
│  [Telemetry Ingestion]   [Latent Encoder]   [Transition Engine]  [Oracle]  │
│  ┌──────────────────┐   ┌──────────────┐   ┌──────────────────┐ ┌────────┐│
│  │ PCAP/CSV Stream   │──▶│ GATv2 × 2    │──▶│ Causal Transformer│▶│CatBoost││
│  │ 5-tuple + Flags   │   │ DGI + Super. │   │ MSE Dynamics Loss │ │+ SHAP  ││
│  │ Windowed Flows    │   │ z_t ∈ ℝ^d    │   │ K-step Rollout    │ │+ FAISS ││
│  └──────────────────┘   └──────────────┘   └──────────────────┘ └────────┘│
│           │                     │                     │                   │
│           ▼                     ▼                     ▼                   ▼
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Glasshouse Portal (Streamlit)                        ││
│  │  Threat Overview │ Network Topology │ XAI Explainability │ Alert Center││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
DRISHTI/
├── config.yaml                    # Pipeline configuration (ΔT, thresholds, paths)
├── requirements.txt               # Python dependencies
├── DRISHTI.pdf                    # Project specification document
│
├── telemetry/                     # Part 1: Spatio-Temporal Telemetry Ingestion
│   ├── parser.py                  #   Microsecond packet deserialization (5-tuple, TCP flags)
│   ├── windowing.py               #   Temporal windowing & macro flow synthesis
│   ├── graph_builder.py           #   Graph tensor construction (PyG-ready)
│   ├── mitigations.py             #   Fault tolerance (port collapse, robust Z-score, imputation)
│   ├── cic_adapter.py             #   CIC-IDS-2018 CSV adapter with synthetic IP mapping
│   └── pipeline.py                #   End-to-end orchestrator
│
├── latent_encoder/                # Part 2: Spatial Encoding via GNN
│   ├── model.py                   #   LatentGraphEncoder (GATv2 × 2 + LayerNorm + pooling)
│   ├── pretraining.py             #   Self-supervised DGI (Deep Graph Infomax)
│   └── utils.py                   #   Capped attention, categorical embedding, self-loops
│
├── transition_engine/             # Part 3: Causal Transition Dynamics (World Model)
│   ├── model.py                   #   CausalTemporalTransformer (PE + causal mask + rollout)
│   └── trainer.py                 #   Scheduled sampling trainer
│
├── oracle/                        # Part 4: Explainable Threat Oracle
│   ├── model.py                   #   ExplainableOracle (CatBoost/RF + SHAP + FAISS)
│   └── faiss_store.py             #   Semantic similarity search over latent vectors
│
├── portal/                        # Part 5: Glasshouse SOC Portal
│   ├── backend/
│   │   └── app.py                 #   FastAPI inference layer
│   └── frontend/
│       └── streamlit_app.py       #   Professional Streamlit dashboard (4 tabs)
│
├── scripts/
│   ├── download_dataset.py        #   Kaggle CIC-IDS-2018 downloader
│   ├── train_encoder.py           #   Encoder training (DGI + supervised)
│   ├── train_worldmodel.py        #   World model training (MSE dynamics)
│   ├── train_demo.py              #   Full pipeline E2E training
│   ├── pull_loop.py               #   Live sliding-window data pull loop
│   └── verify_trained.py          #   Model verification script
│
├── checkpoints/                   # Trained model weights
│   ├── encoder.pt                 #   Trained GAT encoder (DGI + supervised, 79% accuracy)
│   ├── encoder_trained.pt         #   Backup encoder checkpoint
│   ├── world_model.pt             #   Trained causal transformer (MSE: 0.24)
│   └── world_model_best.pt        #   Best world model checkpoint
│
└── data/
    └── pull_logs.csv              # Live pull loop risk logs
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

Downloads CIC-IDS-2018 (~1.6GB) from Kaggle to `.cache/kagglehub/`.

### 3. Train Models

```bash
# Train encoder (DGI self-supervised + supervised fine-tuning)
python scripts/train_encoder.py

# Train world model (causal transformer on latent sequence)
python scripts/train_worldmodel.py
```

### 4. Launch Portal

```bash
streamlit run portal/frontend/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Module Details

### Part 1: Spatio-Temporal Telemetry Ingestion (`telemetry/`)

Converts raw network telemetry into graph-structured data suitable for GNN processing.

| Component | Function | Description |
|-----------|----------|-------------|
| `parser.py` | `parse_packet_stream()` | Extracts 5-tuple + 6-bit TCP flags (SYN/ACK/FIN/RST/PSH/URG) + TTL/window/length. Strips L7 payload. |
| `windowing.py` | `aggregate_temporal_window()` | Groups by `window_idx + 5-tuple`, computes μ/σ² IAT (Eq.1), R_bytes/R_packets (Eq.2). |
| `graph_builder.py` | `build_graph_tensor()` | Nodes = unique IPs with features `[out_bytes, out_deg, in_bytes, in_deg]`. Edges = directional flows with 8-dim attributes. |
| `mitigations.py` | `collapse_port_sweep()` | Port entropy `H(Ports) = -Σp·log₂(p)` collapses >N edges to prevent graph explosion. |
| `cic_adapter.py` | `cic_to_flows()` | Maps CICFlowMeter CSV (no IPs) to pipeline schema with deterministic synthetic IPs via Label hash. |

**Key mitigations (from spec Section 5):**
- Edge collapse: `H(Ports)` threshold = 50 connections
- Robust Z-score: median/IQR normalization
- Group-based imputation: subnet historical median for missing windows
- Micro-buffer: 30s reassembly window for IP fragmentation

### Part 2: Latent State Encoder (`latent_encoder/`)

Graph Neural Network that encodes network topology snapshots into latent vectors.

| Component | Description |
|-----------|-------------|
| `LatentGraphEncoder` | GATv2 × 2 layers with edge-aware attention (Eq.1-2), LayerNorm, `global_mean_pool` → `z_t ∈ ℝ^8` |
| `SelfSupervisedPretrainer` | Deep Graph Infomax: maximizes MI between local node patches and global graph representation |
| Capped attention | Prevents over-smoothing from DDoS bursts (attention cap = 0.9) |

**Training results:**
- DGI loss: 1.1757 → 0.8350 (200 epochs)
- Supervised attack detection: 79% accuracy (100 epochs)
- Latent space separation: 0.303 (benign vs attack)

### Part 3: Causal Transition Dynamics (`transition_engine/`)

Temporal transformer that learns network state transitions `P(z_{t+1} | z_1...z_t)`.

| Component | Description |
|-----------|-------------|
| `CausalTemporalTransformer` | Learned positional encoding (Eq.1) + causal mask (Eq.2) + `state_predictor` MLP |
| `dynamics_loss()` | MSE between predicted and actual next latent state (Eq.3) |
| `rollout()` | Autoregressive K-step future simulation from current sequence |
| `DynamicsTrainer` | Scheduled sampling with epsilon decay (teacher forcing → autoregressive) |

**Training results:**
- Dynamics MSE: 1.1755 → 0.2375 (100 epochs)
- Best checkpoint: 0.2406

### Part 4: Explainable Threat Oracle (`oracle/`)

Dual-engine classifier that maps simulated latent vectors to risk scores and MITRE ATT&CK stages.

| Component | Description |
|-----------|-------------|
| `ExplainableOracle` | CatBoost/RandomForest classifier → `risk_probs [0,1]` + 6 MITRE stages |
| `SemanticFAISS` | L2 nearest-neighbor search over historical anomaly vectors |
| SHAP integration | `shap.TreeExplainer` for per-feature attribution |
| `generate_risk_timeline()` | DataFrame of observed vs simulated states with MITRE mapping |
| `render_network_graph_at_time()` | Interactive HTML threat map |

**MITRE ATT&CK stages:**
`Benign → Reconnaissance → Initial_Access → Lateral_Movement → Exfiltration → Command_Control`

### Part 5: Glasshouse SOC Portal (`portal/`)

Professional cybersecurity dashboard with 4 tabbed views.

| Tab | Features |
|-----|----------|
| **Threat Overview** | 5 live metrics (Mean Risk, Peak Risk, Attack Type, Active Alerts, Sim Windows), K-step trajectory plot, MITRE kill chain, attack distribution |
| **Network Topology** | Interactive Plotly graph (nodes sized by risk), flow volume chart, node risk heatmap |
| **XAI Explainability** | SHAP waterfall (red=increase risk, green=decrease), latent state heatmap, feature importance ranking |
| **Alert Center** | CRITICAL/HIGH/MEDIUM/LOW severity alerts with color-coded boxes, forecast summary table |

**Data sources (sidebar):**
- Live CIC-IDS Demo (2k-30k rows, real attacks)
- Upload PCAP/CSV
- Pull Loop Live (auto-refreshing from `pull_loop.py`)

---

## Data Pipeline

### CIC-IDS-2018 Dataset

| File | Attack Type | Rows |
|------|-------------|------|
| `02-14-2018.csv` | FTP-BruteForce | ~1M |
| `02-15-2018.csv` | DoS-GoldenEye | ~1M |
| `02-16-2018.csv` | DoS-SlowHTTPTest | ~1M |
| `02-20-2018.csv` | DoS-Slowloris | ~1M |
| `02-21-2018.csv` | Web Attack-Brute Force | ~1M |
| `02-22-2018.csv` | Web Attack-XSS | ~1M |
| `02-23-2018.csv` | Infiltration | ~1M |
| `02-28-2018.csv` | Bot | ~1M |
| `03-01-2018.csv` | Bot (continued) | ~1M |
| `03-02-2018.csv` | PortScan | ~1M |

### Pull Loop (Live Data Ingestion)

```bash
python scripts/pull_loop.py --loop --interval 30
```

- Slides random offset (0-80k) across CIC files each pull
- Randomly selects 1-3 files per pull for temporal diversity
- Global window reindex prevents overlap
- Gaussian noise (σ=0.08) on latent vectors for stochastic forecast
- Logs to `data/pull_logs.csv` (pull#, windows, risk_mean, attack_ratio)

---

## Training Pipeline

```bash
# Step 1: Train encoder (self-supervised + supervised)
python scripts/train_encoder.py
# Output: checkpoints/encoder.pt (DGI 200ep + supervised 100ep, 79% accuracy)

# Step 2: Train world model on encoder's latent sequence
python scripts/train_worldmodel.py
# Output: checkpoints/world_model.pt (100ep MSE, loss: 0.24)

# Step 3: Verify all models
python verify_trained.py
```

### Model Checkpoints

| File | Size | Description |
|------|------|-------------|
| `encoder.pt` | 9.3 KB | Trained GAT encoder (913 params + classifier head) |
| `world_model.pt` | 174 KB | Trained causal transformer (29 state dict keys) |
| `world_model_best.pt` | 174 KB | Best world model (lowest MSE) |

---

## API (FastAPI Backend)

```bash
# Start backend
uvicorn portal.backend.app:app --reload

# Health check
GET http://localhost:8000/health

# Simulate
POST http://localhost:8000/simulate
  - file: <PCAP/CSV file>
  - k_steps: 10
  → { timestamps, infiltration_prob, state_type, mitre_stages, attention }
```

---

## Configuration

```yaml
# config.yaml
pipeline:
  window_duration: 1.0    # Temporal window size (seconds)
  stride: 1.0             # Window overlap stride
  port_collapse_threshold: 50   # Edge collapse when >N connections
  robust_zscore_window: 100     # Rolling window for robust normalization
  micro_buffer_timeout: 0.5     # Fragment reassembly timeout (seconds)

paths:
  pcap_dir: "data/pcaps"
  csv_dir: "data/csv"
  output_dir: "data/processed"
```

---

## Key Equations (from spec)

| Eq. | Formula | Description |
|-----|---------|-------------|
| 1 | `μ_IAT = mean(IAT)`, `σ²_IAT = var(IAT)` | Inter-arrival time statistics per flow |
| 2 | `R_bytes = Σ bytes_fwd / Σ bytes_bwd`, `R_packets = Σ pkts_fwd / Σ pkts_bwd` | Bidirectional ratio per window |
| 3 | `H(Ports) = -Σ p(port_i) · log₂(p(port_i))` | Port entropy for scan detection |
| 4 | `S = {(X₁,A₁,E₁), ..., (X_t,A_t,E_t)}` | Ordered graph sequence output |
| 5 | `h_i^(l+1) = σ(Σ_{j∈N(i)∪{i}} α_ij^(l) · W^(l) · h_j^(l))` | GAT message passing |
| 6 | `α_ij = softmax_j(LeakyReLU(a^T [Wh_i ∥ Wh_j ∥ W_e·e_ij]))` | Edge-aware attention |
| 7 | `z_t = GlobalMeanPool(h_1...h_n)` | Graph-level latent representation |
| 8 | `P(z_{t+1} | z_1...z_t) = Transformer(z_1...z_t)` | World model dynamics |
| 9 | `L = MSE(ẑ_{t+1}, z_{t+1})` | Dynamics loss |

---

## Citation

```bibtex
@article{drishti2026,
  title={DRISHTI: Dynamic Risk and Infiltration Sensing via Heuristic Threat Intelligence},
  author={DRISHTI Team},
  year={2026}
}
```

---

## License

Internal research project. All rights reserved.
