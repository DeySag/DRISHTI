import numpy as np
import pandas as pd


def port_entropy(ports):
    """
    Eq (3): H(Ports) = -sum p(port_i) log2 p(port_i)
    Captures scanning behavior without graph explosion.
    """
    if len(ports) == 0:
        return 0.0
    vals, counts = np.unique(ports, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def collapse_port_sweep(aggregated_flows, threshold=50):
    """
    Part 1 Step 4 Mitigation 3: Edge collapse heuristic.
    When (src_ip,dst_ip) opens >N connections within DeltaT, collapse to single
    aggregate edge with port entropy.
    Returns collapsed DataFrame with new column dst_port_entropy.
    """
    if aggregated_flows.empty:
        return aggregated_flows
    collapsed_rows = []
    for (wid, src, dst), group in aggregated_flows.groupby(["window_idx", "src_ip", "dst_ip"]):
        if len(group) > threshold:
            ent = port_entropy(group["dst_port"].values)
            agg = {
                "window_idx": wid,
                "src_ip": src,
                "dst_ip": dst,
                "src_port": int(group["src_port"].mode().iloc[0]) if not group["src_port"].mode().empty else 0,
                "dst_port": -1,
                "proto": int(group["proto"].mode().iloc[0]) if not group["proto"].mode().empty else 6,
                "total_packets": group["total_packets"].sum(),
                "total_bytes": group["total_bytes"].sum(),
                "mean_ttl": group["mean_ttl"].mean(),
                "std_ttl": group["std_ttl"].mean(),
                "mean_win": group["mean_win"].mean(),
                "syn_count": group["syn_count"].sum(),
                "ack_count": group["ack_count"].sum(),
                "rst_count": group["rst_count"].sum(),
                "psh_count": group["psh_count"].sum(),
                "fin_count": group["fin_count"].sum() if "fin_count" in group else 0,
                "urg_count": group["urg_count"].sum() if "urg_count" in group else 0,
                "iat_mean": group["iat_mean"].mean(),
                "iat_variance": group["iat_variance"].mean(),
                "port_entropy": ent,
                "collapsed": True,
                "orig_edge_count": len(group),
            }
            collapsed_rows.append(agg)
        else:
            tmp = group.copy()
            tmp["port_entropy"] = 0.0
            tmp["collapsed"] = False
            tmp["orig_edge_count"] = 1
            collapsed_rows.append(tmp)

    if collapsed_rows and isinstance(collapsed_rows[0], dict):
        # mixed: some collapsed dicts, some DataFrames -> separate handling
        dict_rows = [r for r in collapsed_rows if isinstance(r, dict)]
        df_rows = [r for r in collapsed_rows if isinstance(r, pd.DataFrame)]
        base = pd.concat(df_rows, ignore_index=True) if df_rows else pd.DataFrame()
        extra = pd.DataFrame(dict_rows)
        return pd.concat([base, extra], ignore_index=True) if not base.empty else extra
    return pd.concat(collapsed_rows, ignore_index=True)


def robust_zscore_normalize(df, columns=None, eps=1e-6):
    """
    Part 1 Step 4 Mitigation 3: Robust Z-score using rolling median & IQR
    z = (x - median) / IQR
    """
    if df.empty:
        return df
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c not in ["window_idx", "src_port", "dst_port", "proto"]]

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        median = out[col].median()
        q75, q25 = np.percentile(out[col].dropna(), [75, 25]) if out[col].notna().sum() > 0 else (0, 0)
        iqr = q75 - q25
        if iqr < eps:
            iqr = out[col].std() if out[col].std() > eps else 1.0
        out[col] = (out[col] - median) / (iqr + eps)
    return out


def impute_missing_windows(aggregated_flows, window_duration=1.0):
    """
    Part 1 Step 4 Mitigation 1: Dynamic group-based statistical imputation
    for missing time slices / packet drops. Fills gaps with subnet historical median.
    """
    if aggregated_flows.empty:
        return aggregated_flows
    windows = sorted(aggregated_flows["window_idx"].unique())
    if len(windows) < 2:
        return aggregated_flows.fillna(0.0)

    full_idx = pd.RangeIndex(min(windows), max(windows) + 1, name="window_idx")
    subnets = aggregated_flows["src_ip"].apply(lambda x: ".".join(x.split(".")[:2]) if isinstance(x, str) else "unknown")
    aggregated_flows = aggregated_flows.copy()
    aggregated_flows["_subnet"] = subnets.values

    grouped_median = aggregated_flows.groupby("_subnet").median(numeric_only=True)

    rows = []
    numeric_cols = aggregated_flows.select_dtypes(include=[np.number]).columns
    for wid in full_idx:
        if wid not in windows:
            for subnet, med in grouped_median.iterrows():
                row = {c: med[c] if c in med else 0 for c in numeric_cols}
                row["window_idx"] = wid
                row["src_ip"] = subnet + ".0.1"
                row["dst_ip"] = subnet + ".0.2"
                row["src_port"] = 0
                row["dst_port"] = 0
                row["proto"] = 6
                rows.append(row)

    if rows:
        imputed = pd.DataFrame(rows)
        for c in aggregated_flows.columns:
            if c not in imputed.columns:
                imputed[c] = 0
        aggregated_flows = pd.concat([aggregated_flows, imputed], ignore_index=True)

    aggregated_flows.drop(columns=["_subnet"], errors="ignore", inplace=True)
    aggregated_flows.fillna(0.0, inplace=True)
    return aggregated_flows.sort_values("window_idx").reset_index(drop=True)


class MicroBuffer:
    """
    Part 1 Step 4 Mitigation 2: Short-lived micro-buffer for IP fragmentation
    and out-of-order TCP stream normalization before edge feature vectors.
    """

    def __init__(self, timeout=0.5):
        self.timeout = timeout
        self.buffer = {}

    def add(self, record):
        key = (record["src_ip"], record["dst_ip"], record["src_port"], record["dst_port"], record["proto"])
        self.buffer.setdefault(key, []).append(record)

    def flush(self, current_time=None):
        out = []
        for k, pkts in list(self.buffer.items()):
            pkts_sorted = sorted(pkts, key=lambda x: x["timestamp"])
            if current_time is None or (current_time - pkts_sorted[-1]["timestamp"] > self.timeout):
                out.extend(pkts_sorted)
                del self.buffer[k]
        return out