import torch
import torch.nn.functional as F


def capped_attention_regularizer(att_weights, cap=0.9, lambda_reg=1e-3):
    """
    Part 2 Section 5: Over-Smoothing & Graph Explosion mitigation
    Caps max attention a single node can receive, isolates volumetric DDoS bursts.
    Loss term: penalize attention > cap
    """
    if att_weights is None:
        return torch.tensor(0.0)
    excess = F.relu(att_weights - cap)
    return lambda_reg * (excess ** 2).mean()


def categorical_embedding(num_categories, embed_dim=8):
    """
    Handles categorical protocol features (HTTP, TCP/IP) that fail numeric normalization.
    Use nn.Embedding instead of one-hot.
    """
    return torch.nn.Embedding(num_categories, embed_dim)


def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, fill_value=0.0):
    """
    Ensures Eq (1) includes self-loop {i} in N(i)U{i}
    """
    device = edge_index.device
    loop = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loop, loop], dim=0)
    loop_attr = torch.full((num_nodes, edge_attr.size(1)), fill_value, device=device, dtype=edge_attr.dtype)
    new_index = torch.cat([edge_index, loop_index], dim=1)
    new_attr = torch.cat([edge_attr, loop_attr], dim=0)
    return new_index, new_attr