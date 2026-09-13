import pandas as pd
import numpy as np
import hashlib
from pathlib import Path

CACHE_ROOT = Path(r"C:\Users\sagni\.cache\kagglehub\datasets\solarmainframe\ids-intrusion-csv\versions\1")


def _synthetic_ip(label, proto, dport, idx, side="src"):
    h = int(hashlib.md5(f"{label}{proto}{dport}{idx}{side}".encode()).hexdigest()[:8], 16)
    if label == "Benign":
        base = "192.168"
        return f"{base}.{(h>>16)&255}.{(h>>8)&255}" if side == "src" else f"10.0.{(h>>8)&255}.{h&255}"
    else:
        if side == "src":
            return f"203.0.{(h>>8)&255}.{h&255}"
        else:
            return f"10.0.{(h>>16)&255}.{(h)&255}"


def _parse_timestamp(s):
    try:
        return pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S").timestamp()
    except Exception:
        try:
            return pd.to_datetime(s).timestamp()
        except Exception:
            return 0.0


def load_cic_csv(path, nrows=None, skiprows=None, sample_frac=None):
    if skiprows:
        df = pd.read_csv(path, skiprows=range(1, skiprows + 1), nrows=nrows, low_memory=False)
    else:
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
    return df


def cic_to_flows(df, window_duration=1.0):
    """
    Converts CICFlowMeter CSV (no IPs) to aggregated_flows DataFrame expected by graph_builder
    Synthesizes src_ip/dst_ip/src_port, maps flags, IAT, bytes
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["timestamp"] = df["Timestamp"].apply(_parse_timestamp)
    df["timestamp"] = df["timestamp"] + np.arange(len(df)) * 0.001

    df["src_port"] = (df["Dst Port"].astype(int) * 7 + np.arange(len(df))) % 50000 + 1024
    df["dst_port"] = df["Dst Port"].fillna(0).astype(int)
    df["proto"] = df["Protocol"].fillna(6).astype(int)

    df["src_ip"] = [_synthetic_ip(l, p, dp, i, "src") for i, (l, p, dp) in enumerate(zip(df["Label"], df["proto"], df["dst_port"]))]
    df["dst_ip"] = [_synthetic_ip(l, p, dp, i, "dst") for i, (l, p, dp) in enumerate(zip(df["Label"], df["proto"], df["dst_port"]))]

    df["total_packets"] = df["Tot Fwd Pkts"].fillna(0).astype(int) + df["Tot Bwd Pkts"].fillna(0).astype(int)
    df["total_bytes"] = df["TotLen Fwd Pkts"].fillna(0).astype(int) + df["TotLen Bwd Pkts"].fillna(0).astype(int)
    df["mean_ttl"] = 64.0
    df["std_ttl"] = 0.0
    df["mean_win"] = df["Init Fwd Win Byts"].replace(-1, 1024).fillna(1024).astype(float)
    df["syn_count"] = df["SYN Flag Cnt"].fillna(0).astype(int)
    df["ack_count"] = df["ACK Flag Cnt"].fillna(0).astype(int)
    df["rst_count"] = df["RST Flag Cnt"].fillna(0).astype(int)
    df["psh_count"] = df["PSH Flag Cnt"].fillna(0).astype(int)
    df["fin_count"] = df["FIN Flag Cnt"].fillna(0).astype(int)
    df["urg_count"] = df["URG Flag Cnt"].fillna(0).astype(int)
    df["iat_mean"] = df["Flow IAT Mean"].fillna(0).astype(float) / 1e6
    df["iat_variance"] = (df["Flow IAT Std"].fillna(0).astype(float) / 1e6) ** 2

    start = df["timestamp"].min()
    df["window_idx"] = ((df["timestamp"] - start) // window_duration).astype(int)
    df["label"] = df["Label"]

    keep = ["window_idx", "src_ip", "dst_ip", "src_port", "dst_port", "proto", "total_packets", "total_bytes", "mean_ttl", "std_ttl", "mean_win", "syn_count", "ack_count", "rst_count", "psh_count", "fin_count", "urg_count", "iat_mean", "iat_variance", "label", "timestamp"]
    return df[keep]


def load_all_cic(window_duration=1.0, nrows_per_file=20000, max_files=3, offset=0, global_reindex=True):
    flows_list = []
    files = sorted(CACHE_ROOT.glob("*.csv"))[:max_files]
    for idx, f in enumerate(files):
        file_offset = offset + idx * 2000
        try:
            total = sum(1 for _ in open(f, "r", encoding="utf-8", errors="ignore")) - 1
            file_offset = file_offset % max(1, total - nrows_per_file)
        except Exception:
            pass
        df = load_cic_csv(f, nrows=nrows_per_file, skiprows=file_offset)
        flows = cic_to_flows(df, window_duration=window_duration)
        flows["source_file"] = f.name
        flows_list.append(flows)
        print(f"{f.name} offset {file_offset}: {len(flows)} flows -> {flows['window_idx'].nunique()} windows, labels {flows['label'].value_counts().head(2).to_dict()}")
    if not flows_list:
        return pd.DataFrame()
    all_flows = pd.concat(flows_list, ignore_index=True)
    if global_reindex:
        all_flows = all_flows.sort_values("timestamp").reset_index(drop=True)
        t0 = all_flows["timestamp"].min()
        all_flows["window_idx"] = ((all_flows["timestamp"] - t0) // window_duration).astype(int)
        print(f"Global reindex: {all_flows['window_idx'].nunique()} windows (fixed overlap)")
    return all_flows