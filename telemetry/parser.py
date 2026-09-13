import numpy as np

try:
    from scapy.all import rdpcap, IP, TCP, UDP
    _SCAPY_AVAILABLE = True
except ImportError:
    _SCAPY_AVAILABLE = False


def parse_packet_stream(pcap_file):
    """
    Extracts microsecond packet-level features for sequence modeling.
    Implements Part 1 Step 1: Microsecond Packet Deserialization & Normalization
    Strips L7 payload, extracts L3/L4 headers per 5-tuple.
    """
    if not _SCAPY_AVAILABLE:
        raise ImportError("scapy not installed. pip install scapy")

    packets = rdpcap(pcap_file)
    extracted_records = []

    for pkt in packets:
        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto
            ttl = pkt[IP].ttl
            length = len(pkt)
            timestamp = float(pkt.time)

            flags = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
            src_port, dst_port, win_size = 0, 0, 0

            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                win_size = pkt[TCP].window
                tcp_flags = pkt[TCP].flags
                flags["SYN"] = int(bool(tcp_flags & 0x02))
                flags["ACK"] = int(bool(tcp_flags & 0x10))
                flags["FIN"] = int(bool(tcp_flags & 0x01))
                flags["RST"] = int(bool(tcp_flags & 0x04))
                flags["PSH"] = int(bool(tcp_flags & 0x08))
                flags["URG"] = int(bool(tcp_flags & 0x20))
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            extracted_records.append({
                "timestamp": timestamp,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "proto": proto,
                "ttl": ttl,
                "win_size": win_size,
                "length": length,
                **flags
            })
    return extracted_records


def parse_records_from_dataframe(df):
    """
    Adapter for flow CSVs (NetFlow/CICFlowMeter) already in tabular form.
    Normalizes columns to expected schema.
    """
    required = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "proto"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c}")
    defaults = {"ttl": 64, "win_size": 0, "length": 0, "SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
    for k, v in defaults.items():
        if k not in df.columns:
            df[k] = v
    return df.to_dict(orient="records")