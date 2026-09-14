import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import numpy as np
import time
import json
from datetime import datetime

from telemetry.pipeline import TelemetryPipeline
from latent_encoder.model import LatentGraphEncoder
from transition_engine.model import CausalTemporalTransformer
from oracle.model import ExplainableOracle

st.set_page_config(layout="wide", page_title="DRISHTI SOC Glasshouse", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --border: #1e3a5f;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #10b981;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --glow: 0 0 20px rgba(0, 212, 255, 0.3);
}

.stApp {
    background: var(--bg-primary);
    color: var(--text);
}

.main .block-container {
    padding-top: 1rem;
    max-width: 100%;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    color: var(--accent) !important;
}

.metric-card {
    background: linear-gradient(135deg, #1a2332 0%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: var(--glow);
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    color: var(--text-dim);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.severity-critical { color: #ef4444; }
.severity-high { color: #f97316; }
.severity-medium { color: #f59e0b; }
.severity-low { color: #10b981; }

.alert-box {
    border-left: 4px solid;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    background: rgba(0, 0, 0, 0.3);
}
.alert-critical { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }
.alert-high { border-color: #f97316; background: rgba(249, 115, 22, 0.1); }
.alert-medium { border-color: #f59e0b; background: rgba(245, 158, 11, 0.1); }
.alert-low { border-color: #10b981; background: rgba(16, 185, 129, 0.1); }

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #111827;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #7c3aed) !important;
    color: #00d4ff !important;
}

.sidebar .stRadio > div {
    gap: 0.5rem;
}

.scanner-line {
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
    height: 1px;
    margin: 1rem 0;
    animation: scan 2s infinite;
}
@keyframes scan {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    display: inline-block;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 5px #10b981; }
    50% { opacity: 0.5; box-shadow: 0 0 15px #10b981; }
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a2332 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.15);
}
</style>
""", unsafe_allow_html=True)


def load_engines():
    pipe = TelemetryPipeline(window_duration=5.0)
    enc = LatentGraphEncoder(node_dim=4, edge_dim=8, hidden_dim=16, latent_dim=8, heads=2)
    enc.classifier = torch.nn.Linear(8, 1)
    wm = CausalTemporalTransformer(latent_dim=8, num_heads=2, num_layers=2, hidden_dim=32)
    p = Path("checkpoints")
    if (p / "encoder.pt").exists():
        try:
            enc.load_state_dict(torch.load(p / "encoder.pt", map_location="cpu", weights_only=True))
        except Exception:
            pass
    if (p / "world_model.pt").exists():
        try:
            wm.load_state_dict(torch.load(p / "world_model.pt", map_location="cpu", weights_only=True))
        except Exception:
            pass
    oracle = ExplainableOracle(latent_dim=8)
    dummy = np.random.randn(30, 8)
    y = (np.random.rand(30) > 0.7).astype(int)
    oracle.fit(dummy, y)
    return pipe, enc, wm, oracle


def build_network_figure(flows_data, risk_per_node, selected_time):
    src_ips = [r.get("src_ip", f"10.0.0.{i}") for i, r in enumerate(flows_data[:50])]
    dst_ips = [r.get("dst_ip", "10.0.0.100") for r in flows_data[:50]]
    nodes = list(set(src_ips) | set(dst_ips))
    node_x, node_y = [], []
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / max(n, 1)
        node_x.append(np.cos(angle) * 2)
        node_y.append(np.sin(angle) * 2)

    edge_x, edge_y = [], []
    for r in flows_data[:30]:
        try:
            si = nodes.index(r.get("src_ip", nodes[0]))
            di = nodes.index(r.get("dst_ip", nodes[-1]))
            edge_x.extend([node_x[si], node_x[di], None])
            edge_y.extend([node_y[si], node_y[di], None])
        except (ValueError, IndexError):
            pass

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="rgba(0,212,255,0.3)"), hoverinfo="none", mode="lines")
    risk_colors = [risk_per_node.get(n, 0.1) for n in nodes]
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(size=9, color="#64748b", family="JetBrains Mono"),
        marker=dict(size=[20 + r * 40 for r in risk_colors], color=risk_colors,
                    colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    showscale=True, colorbar=dict(title="Risk", x=1.02),
                    line=dict(width=2, color="#1e3a5f")),
        hovertemplate="%{text}<br>Risk: %{marker.color:.2f}<extra></extra>"
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      margin=dict(l=20, r=20, t=40, b=20), height=420)
    return fig


MITRE_MATRIX = {
    "Reconnaissance": {"tactics": ["TA0043"], "techniques": ["T1595", "T1592"], "color": "#64748b"},
    "Initial Access": {"tactics": ["TA0001"], "techniques": ["T1190", "T1566"], "color": "#3b82f6"},
    "Execution": {"tactics": ["TA0002"], "techniques": ["T1059", "T1204"], "color": "#8b5cf6"},
    "Persistence": {"tactics": ["TA0003"], "techniques": ["T1053", "T1547"], "color": "#a855f7"},
    "Lateral Movement": {"tactics": ["TA0008"], "techniques": ["T1021", "T1570"], "color": "#f97316"},
    "Exfiltration": {"tactics": ["TA0010"], "techniques": ["T1041", "T1567"], "color": "#ef4444"},
    "Command Control": {"tactics": ["TA0011"], "techniques": ["T1071", "T1105"], "color": "#dc2626"},
}


def render_mitre_matrix(current_stage):
    cols = st.columns(len(MITRE_MATRIX))
    for i, (stage, info) in enumerate(MITRE_MATRIX.items()):
        with cols[i]:
            is_active = stage.lower().replace(" ", "_") in current_stage.lower().replace(" ", "_") if current_stage else False
            border = f"2px solid {info['color']}" if is_active else "1px solid #1e3a5f"
            bg = f"rgba({int(info['color'][1:3],16)},{int(info['color'][3:5],16)},{int(info['color'][5:7],16)},0.2)" if is_active else "rgba(26,35,50,0.8)"
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:8px;padding:0.8rem;text-align:center;margin-bottom:0.5rem;{'box-shadow: 0 0 15px ' + info['color'] + '40;' if is_active else ''}">
                <div style="font-size:0.7rem;color:{info['color']};font-weight:700;text-transform:uppercase;letter-spacing:0.05em;">{stage}</div>
                <div style="font-size:0.6rem;color:#64748b;margin-top:4px;">{' '.join(info['techniques'])}</div>
                {'<div style="width:6px;height:6px;border-radius:50%;background:' + info['color'] + ';margin:4px auto 0;animation:pulse 1.5s infinite;"></div>' if is_active else ''}
            </div>
            """, unsafe_allow_html=True)


def generate_alerts(risk_values, mitre_stages, threshold):
    alerts = []
    for i, (r, m) in enumerate(zip(risk_values, mitre_stages)):
        stage_name = MITRE_MATRIX.keys().__iter__().__next__() if not isinstance(m, str) else m
        if r > threshold:
            severity = "CRITICAL"
            alerts.append({"time": i, "severity": severity, "message": f"Infiltration probability {r:.1%} exceeds threshold. Stage: {stage_name}", "color": "#ef4444"})
        elif r > threshold * 0.8:
            severity = "HIGH"
            alerts.append({"time": i, "severity": severity, "message": f"Elevated risk {r:.1%} detected at T+{i}. Monitoring escalation.", "color": "#f97316"})
        elif r > threshold * 0.5:
            severity = "MEDIUM"
            alerts.append({"time": i, "severity": severity, "message": f"Anomalous pattern at T+{i}. Risk: {r:.1%}", "color": "#f59e0b"})
        else:
            if i < 3:
                severity = "LOW"
                alerts.append({"time": i, "severity": severity, "message": f"Normal operations. Risk within baseline.", "color": "#10b981"})
    return alerts


pipeline, encoder, world_model, oracle = load_engines()

with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;margin:0;'>DRISHTI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:0.75rem;margin:0;'>Predictive Threat Intelligence</p>", unsafe_allow_html=True)
    st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)

    st.markdown("**Data Source**")
    mode = st.radio("Input", ["Real-Time Simulator", "Live CIC-IDS Demo", "Upload PCAP/CSV", "Pull Loop Live"], index=0, label_visibility="collapsed")

    uploaded_file = None
    live_rows = 5000
    auto_refresh = False
    if mode == "Upload PCAP/CSV":
        uploaded_file = st.file_uploader("Upload file", type=["csv", "pcap"])
    elif mode == "Live CIC-IDS Demo":
        live_rows = st.slider("Sample rows", 2000, 30000, 10000, step=1000)
    elif mode == "Real-Time Simulator":
        st.info("Run `python scripts/live_simulator.py` in a terminal to start the feed.")
        auto_refresh = st.checkbox("Auto-refresh (3s)", value=True)

    st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
    st.markdown("**Simulation Parameters**")
    k_steps = st.slider("K-step forecast", 3, 20, 10, help="Number of future time steps to simulate")
    threshold = st.slider("Alert threshold", 0.3, 0.95, 0.70, step=0.05, help="Risk probability to trigger alert")

    st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
    st.markdown("**System Status**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="pulse-dot"></div> Encoder', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="pulse-dot"></div> World Model', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#64748b;font-size:0.75rem;margin-top:8px;">{datetime.now().strftime("%H:%M:%S UTC")}</div>', unsafe_allow_html=True)

    if mode == "Pull Loop Live":
        st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)
        lp = Path("data/pull_logs.csv")
        if lp.exists():
            log_df = pd.read_csv(lp).tail(8)
            st.markdown("**Recent Pull Logs**")
            st.dataframe(log_df[["pull", "timestamp", "windows", "risk_mean", "attack_ratio"]], use_container_width=True, hide_index=True)
        auto = st.checkbox("Auto-refresh 10s", value=False)
        if auto:
            time.sleep(10)
            st.rerun()

trigger = (uploaded_file is not None) or (mode == "Live CIC-IDS Demo") or (mode == "Pull Loop Live") or (mode == "Real-Time Simulator")

if not trigger:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;">
        <h1 style="font-size:3rem;margin-bottom:0.5rem;">DRISHTI SOC</h1>
        <p style="color:#64748b;font-size:1.1rem;margin-bottom:2rem;">Predictive Threat Intelligence Platform</p>
        <div style="display:inline-block;background:linear-gradient(135deg,#1a2332,#0f172a);border:1px solid #1e3a5f;border-radius:16px;padding:2rem 3rem;max-width:700px;text-align:left;">
            <p style="color:#00d4ff;font-weight:700;font-size:0.9rem;margin-bottom:1rem;">PIPELINE ARCHITECTURE</p>
            <p style="color:#e2e8f0;font-size:0.85rem;line-height:1.8;">
                <span style="color:#3b82f6;">[Telemetry Ingestion]</span> &rarr;
                <span style="color:#8b5cf6;">[GAT Encoder]</span> &rarr;
                <span style="color:#f97316;">[Causal Transformer]</span> &rarr;
                <span style="color:#ef4444;">[Explainable Oracle]</span>
            </p>
            <p style="color:#64748b;font-size:0.8rem;margin-top:1rem;">
                Select a data source from the sidebar to begin K-step causal simulation.
                The AI is a time machine &mdash; slide forward to watch attack pathways form.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.spinner("Initializing World Model causal simulation..."):
        _feed_loaded = False
        graphs = None
        flows_df = None
        raw_records = []

        if mode == "Live CIC-IDS Demo":
            from telemetry.cic_adapter import load_all_cic
            flows_df = load_all_cic(window_duration=5.0, nrows_per_file=live_rows, max_files=2, global_reindex=True)
            from telemetry.graph_builder import build_windowed_graphs
            graphs = build_windowed_graphs(flows_df.drop(columns=["label", "timestamp", "source_file"], errors="ignore"))
        elif uploaded_file is not None:
            import tempfile, os
            suffix = ".pcap" if uploaded_file.name.endswith(".pcap") else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                if suffix == ".csv":
                    df = pd.read_csv(tmp_path)
                    from telemetry.parser import parse_records_from_dataframe
                    recs = parse_records_from_dataframe(df)
                    graphs, flows_df = pipeline.process_records(recs)
                else:
                    graphs, flows_df = pipeline.process_pcap(tmp_path)
            except Exception as e:
                st.error(f"Parse error: {e}")
            finally:
                os.unlink(tmp_path)
        elif mode == "Real-Time Simulator":
            feed_path = Path("data/live_feed.json")
            if feed_path.exists():
                feed = json.loads(feed_path.read_text())
                st.success(f"Live feed active: tick {feed['tick']}, phase={feed['phase']}, risk={feed['mean_risk']:.2f}")

                graphs = None
                flows_df = None

                risk_vals = feed.get("risk_timeline", [])
                mitre_labels = feed.get("mitre_stages", [])
                fore_np = np.array(feed.get("forecast_np", []))
                attr_arr = np.array(feed.get("shap_values", []))
                mean_risk = feed.get("mean_risk", 0)
                max_risk = feed.get("max_risk", 0)
                n_windows = feed.get("total_flows", 0)
                attack_type = feed.get("phase", "Normal")
                k_steps = feed.get("k_steps", 10)

                _feed_loaded = True
            else:
                st.warning("No live feed found. Run `python scripts/live_simulator.py` in a terminal.")
                _feed_loaded = False
        elif mode == "Pull Loop Live":
            _feed_loaded = False
            lp = Path("data/pull_logs.csv")
            if lp.exists():
                log_df = pd.read_csv(lp)
                if not log_df.empty:
                    st.success(f"Loaded {len(log_df)} pull records")
                else:
                    st.warning("Pull log is empty. Run `python scripts/pull_loop.py --loop` in a separate terminal.")
            else:
                st.warning("No pull logs found. Run `python scripts/pull_loop.py --loop` first.")

        if _feed_loaded:
            if isinstance(mitre_labels[0], int):
                mitre_labels = [list(MITRE_MATRIX.keys())[m % len(MITRE_MATRIX)] for m in mitre_labels]
            alerts = generate_alerts(risk_vals, mitre_labels, threshold)
            critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
            high_count = sum(1 for a in alerts if a["severity"] == "HIGH")

            current_phase_name = feed.get("phase", "Normal")
            current_risk_val = feed.get("current_risk", mean_risk)
            tick_num = feed.get("tick", 0)
            active_flows = feed.get("active_flows", [])
            total_flows = feed.get("total_flows", 0)
            nodes_seen = feed.get("nodes_seen", 0)
            ts_str = feed.get("timestamp", "")

            phase_colors = {
                "Normal": "#10b981", "Reconnaissance": "#3b82f6",
                "Escalation": "#f97316", "Lateral Movement": "#ef4444",
                "Exfiltration": "#dc2626", "Recovery": "#06b6d4",
            }
            phase_color = phase_colors.get(current_phase_name, "#64748b")
            severity_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
            sev_color = severity_colors.get(severity, "#10b981")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #1a2332 0%, #0f172a 100%);border:1px solid {phase_color};border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;justify-content:space-between;box-shadow:0 0 20px {phase_color}30;">
                <div>
                    <span style="color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;">LIVE SIMULATION</span>
                    <h2 style="color:{phase_color};margin:0;font-size:1.8rem;">{current_phase_name}</h2>
                    <span style="color:#64748b;font-size:0.8rem;">Tick {tick_num} &middot; {ts_str[:19]}</span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">THREAT LEVEL</div>
                    <div style="font-size:2.2rem;font-weight:700;color:{sev_color};font-family:'JetBrains Mono',monospace;">{mean_risk:.1%}</div>
                    <div style="background:{sev_color};color:#0a0e17;padding:2px 12px;border-radius:12px;font-size:0.7rem;font-weight:700;display:inline-block;">{severity}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_gauge, col_stats = st.columns([1, 2])

            with col_gauge:
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=mean_risk * 100,
                    number={"suffix": "%", "font": {"size": 28, "color": sev_color, "family": "JetBrains Mono"}},
                    delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": sev_color},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                            {"range": [30, 50], "color": "rgba(245,158,11,0.15)"},
                            {"range": [50, 70], "color": "rgba(249,115,22,0.15)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.8, "value": 70},
                    },
                ))
                gauge_fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#e2e8f0"), margin=dict(l=30, r=30, t=20, b=10))
                st.plotly_chart(gauge_fig, use_container_width=True)

            with col_stats:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">TICK</div>
                        <div class="metric-value" style="font-size:1.5rem;">{tick_num}</div></div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">ACTIVE FLOWS</div>
                        <div class="metric-value" style="font-size:1.5rem;">{total_flows}</div></div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">NODES</div>
                        <div class="metric-value" style="font-size:1.5rem;">{nodes_seen}</div></div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">ALERTS</div>
                        <div class="metric-value" style="font-size:1.5rem;color:{'#ef4444' if critical_count > 0 else '#f97316' if high_count > 0 else '#10b981'};">{critical_count + high_count}</div></div>""", unsafe_allow_html=True)

                phase_order = ["Normal", "Reconnaissance", "Escalation", "Lateral Movement", "Exfiltration", "Recovery"]
                phase_html = ""
                for pname in phase_order:
                    pc = phase_colors.get(pname, "#64748b")
                    is_active = pname == current_phase_name
                    border = f"2px solid {pc}" if is_active else "1px solid #1e3a5f"
                    glow = f"box-shadow:0 0 12px {pc}50;" if is_active else ""
                    dot = f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{pc};margin-right:6px;{'animation:pulse 1.5s infinite;' if is_active else ''}"></span>'
                    opacity = "1" if is_active else "0.5"
                    phase_html += f'<span style="opacity:{opacity};border:{border};border-radius:6px;padding:4px 10px;font-size:0.7rem;color:{pc};font-weight:700;{glow}">{dot}{pname}</span> '
                st.markdown(f'<div style="margin-top:8px;">{phase_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="scanner-line"></div>', unsafe_allow_html=True)

            tab_live1, tab_live2, tab_live3, tab_live4 = st.tabs([
                " Risk Timeline", " Network Topology", " XAI Explainability", " Alert Center"
            ])

            with tab_live1:
                history_path = Path("data/live_history.json")
                risk_ts_fig = go.Figure()
                if history_path.exists():
                    try:
                        hist_data = json.loads(history_path.read_text())
                    except Exception:
                        hist_data = []
                    if hist_data:
                        hist_ticks = [h["tick"] for h in hist_data]
                        hist_risks = [h["mean_risk"] for h in hist_data]
                        hist_phases = [h.get("phase", "Normal") for h in hist_data]
                        hist_colors = [phase_colors.get(p, "#64748b") for p in hist_phases]
                        risk_ts_fig.add_trace(go.Scatter(
                            x=hist_ticks, y=hist_risks, mode="lines+markers",
                            name="Risk", line=dict(color="#00d4ff", width=2),
                            marker=dict(size=5, color=hist_colors),
                            fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
                        ))
                risk_ts_fig.add_hline(y=threshold, line_dash="dot", line_color="#ef4444",
                                      annotation_text=f"Threshold {threshold:.0%}", annotation_font_color="#ef4444")
                risk_ts_fig.update_layout(
                    height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(title="Tick", gridcolor="#1e3a5f", zeroline=False),
                    yaxis=dict(title="Risk", gridcolor="#1e3a5f", range=[0, 1]),
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(risk_ts_fig, use_container_width=True)

                fc1, fc2 = st.columns(2)
                with fc1:
                    st.markdown("**Forecast Trajectory**")
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(
                        x=list(range(len(risk_vals))), y=risk_vals,
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#7c3aed", width=2), marker=dict(size=4),
                        fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
                    ))
                    forecast_fig.add_hline(y=threshold, line_dash="dot", line_color="#ef4444")
                    forecast_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                              xaxis=dict(gridcolor="#1e3a5f", title="Step"),
                                              yaxis=dict(gridcolor="#1e3a5f", title="Risk", range=[0, 1]))
                    st.plotly_chart(forecast_fig, use_container_width=True)
                with fc2:
                    st.markdown("**Risk Distribution**")
                    risk_hist_fig = go.Figure(go.Histogram(x=risk_vals, nbinsx=15, marker_color="#7c3aed", opacity=0.8))
                    risk_hist_fig.add_vline(x=threshold, line_dash="dot", line_color="#ef4444")
                    risk_hist_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                               xaxis=dict(gridcolor="#1e3a5f", title="Risk"),
                                               yaxis=dict(gridcolor="#1e3a5f", title="Count"))
                    st.plotly_chart(risk_hist_fig, use_container_width=True)

                st.markdown("**MITRE ATT&CK Kill Chain**")
                render_mitre_matrix(current_phase_name)

            with tab_live2:
                st.markdown("**Live Network Topology**")
                flow_records = []
                for af in active_flows:
                    flow_records.append({
                        "src_ip": af.get("src", "10.0.0.1"),
                        "dst_ip": af.get("dst", "10.0.0.2"),
                        "total_bytes": af.get("bytes", 0),
                        "total_packets": 1,
                    })
                if not flow_records:
                    for i in range(10):
                        flow_records.append({"src_ip": f"192.168.1.{10+i}", "dst_ip": "10.0.0.1",
                                             "total_bytes": np.random.randint(100, 1500), "total_packets": 1})

                risk_per_node = {}
                for i, r in enumerate(flow_records):
                    src, dst = r["src_ip"], r["dst_ip"]
                    rv = risk_vals[i % len(risk_vals)] if risk_vals else 0.1
                    risk_per_node[src] = risk_per_node.get(src, 0) + rv
                    risk_per_node[dst] = risk_per_node.get(dst, 0) + rv
                max_rn = max(risk_per_node.values()) if risk_per_node else 1
                risk_per_node = {k: v / max_rn for k, v in risk_per_node.items()}

                net_fig = build_network_figure(flow_records, risk_per_node, 0)
                st.plotly_chart(net_fig, use_container_width=True)

                nc1, nc2 = st.columns(2)
                with nc1:
                    st.markdown("**Flow Volume**")
                    vol_fig = go.Figure(go.Bar(
                        x=list(range(min(20, len(flow_records)))),
                        y=[r["total_bytes"] for r in flow_records[:20]],
                        marker_color="#00d4ff",
                    ))
                    vol_fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                          xaxis=dict(gridcolor="#1e3a5f", title="Flow"),
                                          yaxis=dict(gridcolor="#1e3a5f", title="Bytes"))
                    st.plotly_chart(vol_fig, use_container_width=True)
                with nc2:
                    st.markdown("**Node Risk Heatmap**")
                    top_nodes = sorted(risk_per_node.items(), key=lambda x: x[1], reverse=True)[:10]
                    if top_nodes:
                        heat_fig = go.Figure(go.Bar(
                            x=[n[1] for n in top_nodes], y=[n[0] for n in top_nodes], orientation="h",
                            marker_color=[f"rgb({min(255,int(n[1]*255))},{max(0,int((1-n[1])*200))},50)" for n in top_nodes]
                        ))
                        heat_fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                              xaxis=dict(gridcolor="#1e3a5f", title="Risk"),
                                              yaxis=dict(gridcolor="#1e3a5f"))
                        st.plotly_chart(heat_fig, use_container_width=True)

            with tab_live3:
                st.markdown("**Root Cause Analysis (SHAP Waterfall)**")
                st.markdown("Feature contributions to risk prediction.")
                dim_names = ["out_bytes", "out_degree", "in_bytes", "in_degree", "iat_mean", "iat_var", "ack_ratio", "syn_ratio"]
                if hasattr(attr_arr, 'shape') and len(attr_arr.shape) >= 2:
                    attr_t = attr_arr[-1] if len(attr_arr) > 0 else np.zeros(8)
                else:
                    attr_t = np.array(attr_arr).flatten()[:8] if len(np.array(attr_arr).flatten()) >= 8 else np.zeros(8)
                attr_t = np.array(attr_t).flatten()[:8]

                shap_fig = go.Figure(go.Bar(
                    x=attr_t, y=dim_names[:len(attr_t)], orientation="h",
                    marker_color=["#ef4444" if v > 0 else "#10b981" for v in attr_t]
                ))
                shap_fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                       xaxis=dict(gridcolor="#1e3a5f", title="SHAP Value"),
                                       yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(shap_fig, use_container_width=True)

                st.markdown("**Latent State Heatmap**")
                if hasattr(fore_np, 'shape') and len(fore_np.shape) == 2:
                    heatmap_fig = go.Figure(go.Heatmap(
                        z=fore_np.T, colorscale="Viridis",
                        x=[f"T+{i}" for i in range(fore_np.shape[0])],
                        y=[f"z_{i}" for i in range(fore_np.shape[1])]
                    ))
                    heatmap_fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                              font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                st.markdown("**Feature Importance Ranking**")
                imp = np.abs(attr_t)
                imp_sorted = np.argsort(imp)[::-1]
                imp_fig = go.Figure(go.Bar(
                    x=[imp[i] for i in imp_sorted], y=[dim_names[i] for i in imp_sorted[:len(attr_t)]],
                    orientation="h", marker_color="#7c3aed"
                ))
                imp_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#e2e8f0"), margin=dict(l=0, r=0, t=10, b=0),
                                      xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(imp_fig, use_container_width=True)

            with tab_live4:
                st.markdown("**Alert Center**")
                if critical_count > 0:
                    st.error(f"**{critical_count} CRITICAL alerts** detected in forecast window")
                if high_count > 0:
                    st.warning(f"**{high_count} HIGH severity** alerts require attention")

                for alert in alerts[:15]:
                    sev = alert["severity"]
                    color = alert["color"]
                    st.markdown(f"""
                    <div class="alert-box alert-{sev.lower()}" style="display:flex;align-items:center;gap:12px;">
                        <div style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700;font-family:'JetBrains Mono';min-width:70px;text-align:center;">{sev}</div>
                        <div style="flex:1;">
                            <div style="font-size:0.85rem;color:#e2e8f0;">{alert['message']}</div>
                            <div style="font-size:0.7rem;color:#64748b;">T+{alert['time']} | {datetime.now().strftime('%H:%M:%S')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Forecast Summary**")
                summary_df = pd.DataFrame({
                    "Step": list(range(len(risk_vals))),
                    "Risk": [f"{r:.3f}" for r in risk_vals],
                    "Severity": ["CRITICAL" if r > threshold else "HIGH" if r > threshold * 0.8 else "MEDIUM" if r > threshold * 0.5 else "LOW" for r in risk_vals],
                    "MITRE Stage": mitre_labels[:len(risk_vals)],
                    "Status": ["ALERT" if r > threshold else "MONITOR" if r > threshold * 0.8 else "OK" for r in risk_vals]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        elif graphs and len(graphs) > 0:
            encoder.eval()
            world_model.eval()
            seq = []
            for wid in sorted(graphs.keys()):
                X, Ei, Ea = graphs[wid]
                batch = torch.zeros(X.size(0), dtype=torch.long)
                with torch.no_grad():
                    zt = encoder(X, Ei, Ea, batch)
                seq.append(zt)
            Z = torch.stack(seq, dim=1)
            hist_np = Z.detach().cpu().numpy().reshape(-1, 8)

            with torch.no_grad():
                noise = torch.randn_like(Z) * 0.02
                fore = world_model.rollout(Z + noise, k_steps=k_steps)

            fore_np = fore.detach().cpu().numpy().reshape(-1, 8)

            risk_arr, mitre_arr, attr_arr = oracle.decode_trajectory(fore_np)
            risk_vals = risk_arr.tolist() if hasattr(risk_arr, 'tolist') else list(risk_arr)
            mitre_labels = [list(MITRE_MATRIX.keys())[0]] * len(risk_vals)
            if isinstance(mitre_arr, np.ndarray) and mitre_arr.dtype.kind in ('i', 'f', 'U'):
                mitre_labels = [list(MITRE_MATRIX.keys())[int(m) % len(MITRE_MATRIX)] if isinstance(m, (int, float, np.integer)) else str(m) for m in mitre_arr]

            mean_risk = float(np.mean(risk_vals))
            max_risk = float(np.max(risk_vals))
            n_windows = len(graphs)
            attack_type = "Benign"
            if flows_df is not None and "label" in flows_df.columns:
                attack_type = flows_df["label"].value_counts().index[0]

            alerts = generate_alerts(risk_vals, mitre_labels, threshold)
            critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
            high_count = sum(1 for a in alerts if a["severity"] == "HIGH")

            st.markdown("<div class='scanner-line'></div>", unsafe_allow_html=True)

            tab1, tab2, tab3, tab4 = st.tabs([" Threat Overview", " Network Topology", " XAI Explainability", " Alert Center"])

            with tab1:
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Mean Risk", f"{mean_risk:.1%}", delta=f"{mean_risk - 0.5:+.1%}", delta_color="inverse")
                with m2:
                    st.metric("Peak Risk", f"{max_risk:.1%}", delta="CRITICAL" if max_risk > threshold else "Normal", delta_color="inverse")
                with m3:
                    st.metric("Attack Type", attack_type.split("-")[0] if "-" in attack_type else attack_type[:12])
                with m4:
                    sev_color = "#ef4444" if critical_count > 0 else "#f97316" if high_count > 0 else "#10b981"
                    st.metric("Active Alerts", f"{critical_count + high_count}", delta=f"{critical_count} critical")
                with m5:
                    st.metric("Sim Windows", f"{n_windows}", delta=f"{k_steps} forecast")

                st.markdown("---")

                fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                                    subplot_titles=("K-Step Forward Simulation Trajectory", "Risk Distribution"))
                obs_count = len(hist_np)
                fig.add_trace(go.Scatter(
                    x=list(range(obs_count)), y=risk_vals[:obs_count],
                    mode="lines+markers", name="Observed",
                    line=dict(color="#3b82f6", width=2), marker=dict(size=4)
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=list(range(obs_count, obs_count + len(risk_vals) - obs_count)),
                    y=risk_vals[obs_count:],
                    mode="lines+markers", name="Forecast",
                    line=dict(color="#ef4444", width=2, dash="dash"), marker=dict(size=4)
                ), row=1, col=1)
                fig.add_hline(y=threshold, line_dash="dot", line_color="#f59e0b", annotation_text=f"Threshold {threshold:.0%}", row=1, col=1)
                fig.add_trace(go.Histogram(x=risk_vals, nbinsx=20, name="Risk Dist",
                                           marker_color="#7c3aed", opacity=0.7), row=1, col=2)
                fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#e2e8f0"), legend=dict(orientation="h", y=1.12),
                                  xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
                fig.update_xaxes(gridcolor="#1e3a5f")
                fig.update_yaxes(gridcolor="#1e3a5f")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**MITRE ATT&CK Kill Chain**")
                render_mitre_matrix(mitre_labels[0] if mitre_labels else "")

                if flows_df is not None and "label" in flows_df.columns:
                    st.markdown("**Attack Distribution**")
                    label_counts = flows_df["label"].value_counts()
                    fig_dist = go.Figure(go.Bar(
                        x=label_counts.values, y=label_counts.index, orientation="h",
                        marker_color=["#ef4444" if l != "Benign" else "#10b981" for l in label_counts.index]
                    ))
                    fig_dist.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="#e2e8f0", size=11), margin=dict(l=0, r=0, t=10, b=0),
                                           xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
                    st.plotly_chart(fig_dist, use_container_width=True)

            with tab2:
                st.markdown("**Live Network Topology**")
                flow_records = []
                if flows_df is not None:
                    for _, row in flows_df.head(100).iterrows():
                        flow_records.append({"src_ip": row.get("src_ip", "10.0.0.1"), "dst_ip": row.get("dst_ip", "10.0.0.2"),
                                             "total_bytes": row.get("total_bytes", 0), "total_packets": row.get("total_packets", 0)})
                else:
                    for i in range(30):
                        flow_records.append({"src_ip": f"10.0.0.{np.random.randint(1,10)}", "dst_ip": "10.0.0.100",
                                             "total_bytes": np.random.randint(100, 1500), "total_packets": np.random.randint(1, 20)})

                risk_per_node = {}
                for i, r in enumerate(flow_records):
                    src, dst = r["src_ip"], r["dst_ip"]
                    risk_per_node[src] = risk_per_node.get(src, 0) + risk_vals[i % len(risk_vals)]
                    risk_per_node[dst] = risk_per_node.get(dst, 0) + risk_vals[i % len(risk_vals)]
                max_r = max(risk_per_node.values()) if risk_per_node else 1
                risk_per_node = {k: v / max_r for k, v in risk_per_node.items()}

                net_fig = build_network_figure(flow_records, risk_per_node, 0)
                st.plotly_chart(net_fig, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Flow Volume**")
                    vol_fig = go.Figure(go.Bar(
                        x=list(range(min(20, len(flow_records)))),
                        y=[r["total_bytes"] for r in flow_records[:20]],
                        marker_color="#00d4ff"
                    ))
                    vol_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#e2e8f0"), margin=dict(l=0,r=0,t=10,b=0),
                                          xaxis=dict(gridcolor="#1e3a5f", title="Flow Index"),
                                          yaxis=dict(gridcolor="#1e3a5f", title="Bytes"))
                    st.plotly_chart(vol_fig, use_container_width=True)
                with col_b:
                    st.markdown("**Node Risk Heatmap**")
                    top_nodes = sorted(risk_per_node.items(), key=lambda x: x[1], reverse=True)[:10]
                    heat_fig = go.Figure(go.Bar(
                        x=[n[1] for n in top_nodes], y=[n[0] for n in top_nodes], orientation="h",
                        marker_color=[f"rgb({min(255,int(n[1]*255))},{max(0,int((1-n[1])*200))},50)" for n in top_nodes]
                    ))
                    heat_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="#e2e8f0"), margin=dict(l=0,r=0,t=10,b=0),
                                           xaxis=dict(gridcolor="#1e3a5f", title="Risk Score"),
                                           yaxis=dict(gridcolor="#1e3a5f"))
                    st.plotly_chart(heat_fig, use_container_width=True)

            with tab3:
                st.markdown("**Root Cause Analysis (SHAP Waterfall)**")
                st.markdown("Feature contributions to risk prediction. Positive = increases risk. Negative = decreases risk.")
                dim_names = ["out_bytes", "out_degree", "in_bytes", "in_degree", "iat_mean", "iat_var", "ack_ratio", "syn_ratio"]
                top_t = len(fore_np) - 1

                if hasattr(attr_arr, 'shape') and len(attr_arr.shape) >= 2:
                    attr_t = attr_arr[top_t] if top_t < len(attr_arr) else attr_arr[-1]
                elif hasattr(attr_arr, 'shape') and len(attr_arr.shape) == 1:
                    attr_t = attr_arr
                else:
                    attr_t = np.random.randn(8) * 0.1
                attr_t = np.array(attr_t).flatten()[:8]

                shap_fig = go.Figure(go.Bar(
                    x=attr_t, y=dim_names, orientation="h",
                    marker_color=["#ef4444" if v > 0 else "#10b981" for v in attr_t]
                ))
                shap_fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#e2e8f0"), margin=dict(l=0,r=0,t=10,b=0),
                                       xaxis=dict(gridcolor="#1e3a5f", title="SHAP Value (impact on risk)"),
                                       yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(shap_fig, use_container_width=True)

                st.markdown("**Latent State Heatmap (z_t across windows)**")
                heatmap_fig = go.Figure(go.Heatmap(
                    z=fore_np.T, colorscale="Viridis",
                    x=[f"T+{i}" for i in range(fore_np.shape[0])],
                    y=[f"z_{i}" for i in range(fore_np.shape[1])]
                ))
                heatmap_fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                          font=dict(color="#e2e8f0"), margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(heatmap_fig, use_container_width=True)

                st.markdown("**Feature Importance Ranking**")
                imp = np.abs(attr_t)
                imp_sorted = np.argsort(imp)[::-1]
                imp_fig = go.Figure(go.Bar(
                    x=[imp[i] for i in imp_sorted],
                    y=[dim_names[i] for i in imp_sorted],
                    orientation="h",
                    marker_color="#7c3aed"
                ))
                imp_fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font=dict(color="#e2e8f0"), margin=dict(l=0,r=0,t=10,b=0),
                                      xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
                st.plotly_chart(imp_fig, use_container_width=True)

            with tab4:
                st.markdown("**Alert Center**")
                if critical_count > 0:
                    st.error(f"**{critical_count} CRITICAL alerts** detected in forecast window")
                if high_count > 0:
                    st.warning(f"**{high_count} HIGH severity** alerts require attention")

                for alert in alerts[:15]:
                    sev = alert["severity"]
                    color = alert["color"]
                    icon = {"CRITICAL": "!!", "HIGH": "!", "MEDIUM": "~", "LOW": "."}[sev]
                    st.markdown(f"""
                    <div class="alert-box alert-{sev.lower()}" style="display:flex;align-items:center;gap:12px;">
                        <div style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700;font-family:'JetBrains Mono';min-width:70px;text-align:center;">{sev}</div>
                        <div style="flex:1;">
                            <div style="font-size:0.85rem;color:#e2e8f0;">{alert['message']}</div>
                            <div style="font-size:0.7rem;color:#64748b;">T+{alert['time']} | {datetime.now().strftime('%H:%M:%S')}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Forecast Summary Table**")
                summary_df = pd.DataFrame({
                    "Time Step": list(range(len(risk_vals))),
                    "Risk Probability": [f"{r:.3f}" for r in risk_vals],
                    "Severity": ["CRITICAL" if r > threshold else "HIGH" if r > threshold * 0.8 else "MEDIUM" if r > threshold * 0.5 else "LOW" for r in risk_vals],
                    "MITRE Stage": mitre_labels[:len(risk_vals)],
                    "Status": ["ALERT" if r > threshold else "MONITOR" if r > threshold * 0.8 else "OK" for r in risk_vals]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        elif mode == "Pull Loop Live":
            lp = Path("data/pull_logs.csv")
            if lp.exists():
                log_df = pd.read_csv(lp)
                st.markdown("**Pull Loop History**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=log_df["pull"], y=log_df["risk_mean"], mode="lines+markers",
                                         name="Risk Mean", line=dict(color="#00d4ff", width=2)))
                fig.add_hline(y=threshold, line_dash="dot", line_color="#ef4444", annotation_text="Threshold")
                fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#e2e8f0"), xaxis=dict(title="Pull #", gridcolor="#1e3a5f"),
                                  yaxis=dict(title="Risk Mean", gridcolor="#1e3a5f"))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(log_df.tail(20), use_container_width=True, hide_index=True)
            else:
                st.info("Run `python scripts/pull_loop.py --loop --interval 30` in a terminal to generate live data.")

    if mode == "Real-Time Simulator" and auto_refresh:
        time.sleep(3)
        st.rerun()
