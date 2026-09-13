import torch
import pandas as pd


def build_graph_tensor(aggregated_flows):
    """
    Transforms windowed flow metrics into PyTorch-ready Graph representations.
    Part 1 Step 3: Graph Construction & Spatial Encoding
    Nodes = unique IPs, Edges = directional flows, Node feat = (out_bytes, out_deg, in_bytes, in_deg)
    Edge feat = [total_packets, total_bytes, iat_mean, iat_variance, syn, ack, rst, mean_ttl]
    """
    if aggregated_flows.empty:
        return torch.zeros((0, 4), dtype=torch.float), torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 8), dtype=torch.float)

    unique_nodes = list(set(aggregated_flows["src_ip"]).union(set(aggregated_flows["dst_ip"])))
    node_to_idx = {ip: idx for idx, ip in enumerate(unique_nodes)}

    num_nodes = len(unique_nodes)
    edge_indices = []
    edge_features = []

    for _, row in aggregated_flows.iterrows():
        u = node_to_idx[row["src_ip"]]
        v = node_to_idx[row["dst_ip"]]
        edge_indices.append([u, v])
        feat = [
            row["total_packets"],
            row["total_bytes"],
            row["iat_mean"],
            row["iat_variance"],
            row["syn_count"],
            row["ack_count"],
            row["rst_count"],
            row["mean_ttl"],
        ]
        edge_features.append(feat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    node_features = torch.zeros((num_nodes, 4), dtype=torch.float)
    for _, row in aggregated_flows.iterrows():
        u = node_to_idx[row["src_ip"]]
        v = node_to_idx[row["dst_ip"]]
        node_features[u, 0] += row["total_bytes"]
        node_features[u, 1] += 1
        node_features[v, 2] += row["total_bytes"]
        node_features[v, 3] += 1

    return node_features, edge_index, edge_attr


def build_windowed_graphs(aggregated_flows):
    """
    Splits flows by window_idx and builds per-window tensors.
    Returns dict {window_idx: (X, edge_index, edge_attr)} and node map per window.
    """
    graphs = {}
    for wid, sub in aggregated_flows.groupby("window_idx"):
        X, Ei, Ea = build_graph_tensor(sub)
        graphs[int(wid)] = (X, Ei, Ea)
    return graphs