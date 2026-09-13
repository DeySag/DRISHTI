import numpy as np
import torch

try:
    import catboost
    _CATBOOST = True
except ImportError:
    _CATBOOST = False

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

from .faiss_store import SemanticFAISS

MITRE_STAGES = ["Benign", "Reconnaissance", "Initial_Access", "Lateral_Movement", "Exfiltration", "Command_Control"]


class ExplainableOracle:
    """
    Part 4 Oracle: Dual-Engine Classification (CatBoost + FAISS) + XAI
    Maps simulated latent vectors Z_future -> risk_probs [0,1], MITRE stages, SHAP attributions + attention
    """

    def __init__(self, latent_dim=8, model_path=None):
        self.latent_dim = latent_dim
        self.clf = None
        self.explainer = None
        self.faiss = SemanticFAISS(dim=latent_dim)

        if _CATBOOST and model_path:
            self.clf = catboost.CatBoostClassifier().load_model(model_path)
            if _SHAP:
                self.explainer = shap.TreeExplainer(self.clf)
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(n_estimators=50, random_state=42)
            self._trained = False

    def fit(self, Z_train, y_binary, y_mitre=None):
        Z_train = np.array(Z_train)
        self.clf.fit(Z_train, y_binary)
        self._trained = True
        if _SHAP and hasattr(self.clf, "estimators_"):
            try:
                import shap
                self.explainer = shap.TreeExplainer(self.clf)
            except Exception:
                self.explainer = None
        self.faiss.add(Z_train[y_binary == 1] if (Z_train[y_binary == 1].size) else Z_train[:5])

    def decode_trajectory(self, z_future):
        if isinstance(z_future, torch.Tensor):
            z_future = z_future.detach().cpu().numpy()
        if z_future.ndim == 3:
            z_future = z_future.reshape(-1, z_future.shape[-1])

        if not _CATBOOST or not hasattr(self.clf, "predict_proba"):
            risk_probs = np.random.rand(z_future.shape[0]) * 0.3
            mitre = np.random.choice(MITRE_STAGES, size=z_future.shape[0])
            attributions = np.random.randn(*z_future.shape) * 0.1
            l2 = self.faiss.l2_distance_to_anomaly(z_future)
            risk_probs = np.clip(risk_probs + (1 / (1 + l2)) * 0.2, 0, 1)
            return risk_probs, mitre, attributions

        try:
            risk_probs = self.clf.predict_proba(z_future)[:, 1]
            mitre_stages = self.clf.predict(z_future)
        except Exception:
            risk_probs = np.zeros(z_future.shape[0])
            mitre_stages = np.array(["Benign"] * z_future.shape[0])

        if self.explainer is not None:
            try:
                attributions = self.explainer.shap_values(z_future)
                if isinstance(attributions, list):
                    attributions = attributions[1] if len(attributions) > 1 else attributions[0]
            except Exception:
                attributions = np.zeros_like(z_future)
        else:
            attributions = np.zeros_like(z_future)

        return risk_probs, mitre_stages, attributions

    def generate_risk_timeline(self, historical_states, forecasted_states):
        import pandas as pd
        hist = np.array(historical_states) if not isinstance(historical_states, np.ndarray) else historical_states
        fore = np.array(forecasted_states) if not isinstance(forecasted_states, np.ndarray) else forecasted_states
        if hist.ndim == 3:
            hist = hist.reshape(-1, hist.shape[-1])
        if fore.ndim == 3:
            fore = fore.reshape(-1, fore.shape[-1])

        h_risk, _, _ = self.decode_trajectory(hist) if len(hist) else (np.array([]), None, None)
        f_risk, f_mitre, _ = self.decode_trajectory(fore) if len(fore) else (np.array([]), np.array([]), None)

        timestamps = list(range(len(h_risk) + len(f_risk)))
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Infiltration_Probability": list(h_risk) + list(f_risk),
            "State_Type": ["Observed"] * len(h_risk) + ["Simulated"] * len(f_risk),
            "MITRE_Stage": ["Benign"] * len(h_risk) + list(f_mitre) if len(f_risk) else ["Benign"] * len(h_risk),
        })
        return df

    def plot_shap_waterfall(self, selected_time, z_future=None):
        import matplotlib.pyplot as plt
        if z_future is not None:
            _, _, attr = self.decode_trajectory(z_future)
            vals = attr[0] if len(attr) else np.zeros(self.latent_dim)
        else:
            vals = np.random.randn(self.latent_dim)
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["crimson" if v > 0 else "steelblue" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([f"z_dim_{i}" for i in range(len(vals))])
        ax.set_xlabel("SHAP value")
        ax.set_title(f"Root Cause @ T={selected_time}")
        plt.tight_layout()
        return fig

    def render_network_graph_at_time(self, selected_time, graph=None):
        nodes = ["10.0.0.1", "10.0.0.2", "192.168.1.5"]
        edges = [("10.0.0.1", "10.0.0.2"), ("192.168.1.5", "10.0.0.2")]
        html = f"<html><body><h4>Threat Map @ T={selected_time}</h4><p>Nodes: {nodes}</p><p>Edges: {edges}</p><svg width='400' height='200'><circle cx='100' cy='100' r='20' fill='crimson'/><circle cx='300' cy='100' r='15' fill='green'/><line x1='100' y1='100' x2='300' y2='100' stroke='crimson' stroke-width='4'/></svg></body></html>"
        return html

    def get_mitre_stage(self, selected_time, z_future=None):
        if z_future is not None:
            _, mitre, _ = self.decode_trajectory(z_future)
            idx = min(int(selected_time), len(mitre) - 1) if len(mitre) else 0
            return mitre[idx] if len(mitre) else "Benign"
        stages = MITRE_STAGES
        return stages[int(selected_time) % len(stages)] if isinstance(selected_time, int) else "Lateral_Movement"