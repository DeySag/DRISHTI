import torch
import torch.nn.functional as F


class SelfSupervisedPretrainer(torch.nn.Module):
    """
    Part 2 Section 5: Categorical Feature Handling via in-context pre-training
    Maximize mutual information between local node patches and global graph representation
    (Deep Graph Infomax style) to learn protocol semantics before fine-tuning.
    """

    def __init__(self, encoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.discriminator = torch.nn.Bilinear(latent_dim, latent_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        z_graph = self.encoder(x, edge_index, edge_attr, batch)
        node_emb = self.encoder.get_node_embeddings(x, edge_index, edge_attr)

        pos_scores = []
        neg_scores = []
        for b in torch.unique(batch):
            mask = batch == b
            g = z_graph[b if b < z_graph.size(0) else 0]
            pos = node_emb[mask]
            neg = node_emb[torch.randperm(node_emb.size(0))[: pos.size(0)]]

            pos_s = self.discriminator(pos, g.expand(pos.size(0), -1)).squeeze()
            neg_s = self.discriminator(neg, g.expand(neg.size(0), -1)).squeeze()
            pos_scores.append(pos_s)
            neg_scores.append(neg_s)

        pos_scores = torch.cat(pos_scores)
        neg_scores = torch.cat(neg_scores)
        logits = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss

    def pretrain_step(self, optimizer, x, edge_index, edge_attr, batch):
        self.train()
        optimizer.zero_grad()
        loss = self.forward(x, edge_index, edge_attr, batch)
        loss.backward()
        optimizer.step()
        return loss.item()