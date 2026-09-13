import pandas as pd
import torch
from .parser import parse_packet_stream, parse_records_from_dataframe
from .windowing import aggregate_temporal_window
from .graph_builder import build_windowed_graphs, build_graph_tensor
from .mitigations import collapse_port_sweep, impute_missing_windows, robust_zscore_normalize, MicroBuffer


class TelemetryPipeline:
    """
    End-to-End Pipeline Technical Stack (Part 1 Section 4)
    [Packet Capture] -> PyShark/Scapy/eBPF
    [Data Transformation] -> Pandas/NumPy
    [Graph Tensor Generation] -> PyTorch Geometric
    [Offline Dataset Store] -> CIC-IDS-2018 / CTU-13
    Output: S = {(X1,A1,E1),...,(Xt,At,Et)}
    """

    def __init__(self, window_duration=1.0, collapse_threshold=50, normalize=True):
        self.window_duration = window_duration
        self.collapse_threshold = collapse_threshold
        self.normalize = normalize
        self.buffer = MicroBuffer()

    def process_pcap(self, pcap_file):
        records = parse_packet_stream(pcap_file)
        return self.process_records(records)

    def process_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        records = parse_records_from_dataframe(df)
        return self.process_records(records)

    def process_records(self, records):
        if not records:
            return {}, pd.DataFrame()

        for r in records:
            self.buffer.add(r)
        ordered = self.buffer.flush()

        flows = aggregate_temporal_window(ordered if ordered else records, window_duration=self.window_duration)
        if flows.empty:
            return {}, flows

        flows = collapse_port_sweep(flows, threshold=self.collapse_threshold)
        flows = impute_missing_windows(flows, window_duration=self.window_duration)
        if self.normalize:
            flows = robust_zscore_normalize(flows)

        graphs = build_windowed_graphs(flows)
        sequence = []
        for wid in sorted(graphs.keys()):
            X, Ei, Ea = graphs[wid]
            sequence.append((X, Ei, Ea))

        return graphs, flows

    def process_and_simulate(self, filepath, k_steps=10):
        import pathlib
        p = pathlib.Path(filepath)
        if p.suffix.lower() == ".pcap":
            graphs, flows = self.process_pcap(str(p))
        else:
            graphs, flows = self.process_csv(str(p))
        return graphs, flows

    @staticmethod
    def to_pyg_data(node_features, edge_index, edge_attr):
        try:
            from torch_geometric.data import Data
            return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        except ImportError:
            return {"x": node_features, "edge_index": edge_index, "edge_attr": edge_attr}