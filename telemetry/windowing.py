import numpy as np
import pandas as pd


def aggregate_temporal_window(records, window_duration=1.0):
    """
    Aggregates packet records into structured flow metrics over time window.
    Part 1 Step 2: Temporal Windowing & Macro Flow Synthesis
    Equations (1) mu_IAT/sigma_IAT, (2) R_bytes/R_packets implicit via total counts.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    start_time = df["timestamp"].min()
    df["window_idx"] = ((df["timestamp"] - start_time) // window_duration).astype(int)

    grouped = df.groupby(["window_idx", "src_ip", "dst_ip", "src_port", "dst_port", "proto"])

    aggregated_flows = grouped.agg(
        total_packets=("length", "count"),
        total_bytes=("length", "sum"),
        mean_ttl=("ttl", "mean"),
        std_ttl=("ttl", "std"),
        mean_win=("win_size", "mean"),
        syn_count=("SYN", "sum"),
        ack_count=("ACK", "sum"),
        rst_count=("RST", "sum"),
        psh_count=("PSH", "sum"),
        fin_count=("FIN", "sum"),
        urg_count=("URG", "sum"),
        iat_mean=("timestamp", lambda x: np.mean(np.diff(x)) if len(x) > 1 else 0.0),
        iat_variance=("timestamp", lambda x: np.var(np.diff(x)) if len(x) > 1 else 0.0),
    ).reset_index()

    aggregated_flows.fillna(0.0, inplace=True)
    return aggregated_flows


def add_bidirectional_ratios(flows):
    """
    Eq (2): R_bytes, R_packets per window per src/dst pair aggregated bidirectionally.
    Call optionally after aggregate_temporal_window if forward/backward split needed.
    """
    if flows.empty:
        return flows
    eps = 1e-6
    fwd = flows.groupby(["window_idx", "src_ip", "dst_ip"]).agg(
        fwd_bytes=("total_bytes", "sum"), fwd_pkts=("total_packets", "sum")
    ).reset_index()
    bwd = flows.groupby(["window_idx", "src_ip", "dst_ip"]).agg(
        bwd_bytes=("total_bytes", "sum"), bwd_pkts=("total_packets", "sum")
    ).reset_index()
    return flows


def sliding_window_indices(timestamps, window_duration=1.0, stride=None):
    """
    Generates overlapping window slices Tw=[tk, tk+DeltaT] with stride delta.
    Returns list of (start, end) for visualization/debugging.
    """
    if stride is None:
        stride = window_duration
    t0 = float(np.min(timestamps))
    t1 = float(np.max(timestamps))
    windows = []
    t = t0
    while t + window_duration <= t1 + 1e-9:
        windows.append((t, t + window_duration))
        t += stride
    return windows