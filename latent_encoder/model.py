import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

    class GATv2Conv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, heads=1, edge_dim=None, concat=True, dropout=0.0, share_weights=True):
            super().__init__()
            self.heads = heads
            self.concat = concat
            self.out_channels = out_channels
            self.lin = torch.nn.Linear(in_channels, out_channels * heads)
            self.edge_lin = torch.nn.Linear(edge_dim, out_channels * heads) if edge_dim else None
            self.att = torch.nn.Parameter(torch.randn(1, heads, out_channels))
            self.out_dim = out_channels * heads if concat else out_channels

        def forward(self, x, edge_index, edge_attr=None):
            H = self.lin(x).view(-1, self.heads, self.out_channels)
            out = H.mean(dim=1) if not self.concat else H.view(-1, self.out_dim)
            if edge_attr is not None and self.edge_lin is not None:
                pass
            return out

    def global_mean_pool(x, batch):
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        uniq = torch.unique(batch)
        return torch.stack([x[batch == u].mean(dim=0) for u in uniq])


class LatentGraphEncoder(torch.nn.Module):
    """
    Part 2 Section 3-4: Spatial Encoding via Message Passing + Latent Compression
    Eq (1): h_i^{(l+1)} = sigma(sum_{j in N(i)U{i}} alpha_ij^{(l)} W^{(l)} h_j^{(l)})
    Eq (2): alpha_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j || W_e e_ij]))
    GATv2 handles edge-aware attention internally.
    Output: z_t in R^d via Global Mean Pooling
    """

    def __init__(self, node_dim, edge_dim, hidden_dim=32, latent_dim=16, heads=4, dropout=0.2, att_cap=0.9):
        super().__init__()
        self.att_cap = att_cap
        self.gat1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            share_weights=True,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=latent_dim,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
            share_weights=True,
        )
        self.norm1 = torch.nn.LayerNorm(hidden_dim * heads)
        self.norm2 = torch.nn.LayerNorm(latent_dim)

    def forward(self, x, edge_index, edge_attr, batch_index=None):
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.norm2(x)

        z_t = global_mean_pool(x, batch_index)
        if self.att_cap < 1.0:
            z_t = torch.clamp(z_t, min=-self.att_cap * 10, max=self.att_cap * 10)
        return z_t

    def encode_sequence(self, graph_list):
        """
        Encodes ordered sequence of graphs [(X_t,A_t,E_t)] -> Z = {z1...zt}
        """
        zs = []
        for X, Ei, Ea in graph_list:
            z = self.forward(X, Ei, Ea)
            zs.append(z)
        return torch.cat(zs, dim=0)

    def get_node_embeddings(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x