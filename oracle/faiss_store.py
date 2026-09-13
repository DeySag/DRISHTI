import numpy as np


class SemanticFAISS:
    """
    Lightweight FAISS-like semantic similarity over historical anomalous vectors.
    Fallback to numpy L2 if faiss not installed.
    """

    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.labels = []
        try:
            import faiss
            self._faiss = faiss
            self.index = faiss.IndexFlatL2(dim)
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False
            self.index = None

    def add(self, vecs, labels=None):
        vecs = np.array(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        self.vectors.append(vecs)
        if labels is not None:
            self.labels.extend(list(labels))
        if self._use_faiss:
            self.index.add(vecs)

    def query(self, q, k=5):
        q = np.array(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self._use_faiss and self.index.ntotal > 0:
            D, I = self.index.search(q, min(k, self.index.ntotal))
            return D, I
        if not self.vectors:
            return np.full((q.shape[0], k), np.inf), np.full((q.shape[0], k), -1)
        all_vecs = np.concatenate(self.vectors, axis=0)
        dists = np.linalg.norm(all_vecs[None, :, :] - q[:, None, :], axis=2)
        idx = np.argsort(dists, axis=1)[:, :k]
        d = np.take_along_axis(dists, idx, axis=1)
        return d, idx

    def l2_distance_to_anomaly(self, z):
        d, _ = self.query(z, k=1)
        return d[:, 0]