import sys
from pathlib import Path

try:
    import kagglehub
    path = kagglehub.dataset_download("solarmainframe/ids-intrusion-csv")
    print("Path to dataset files:", path)
    import os
    for root, dirs, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for f in files[:20]:
            print(f"{subindent}{f}")
        if level > 2:
            break
except Exception as e:
    print(f"kagglehub failed: {e}")
    print("Fallback: checking local data/")

    import pandas as pd
    local = Path("data")
    if local.exists():
        print(list(local.iterdir())[:10])

try:
    import pandas as pd
    from pathlib import Path
    ds_path = Path(path) if "path" in locals() else Path("data")
    csvs = list(Path(ds_path).rglob("*.csv")) if Path(ds_path).exists() else []
    print(f"Found {len(csvs)} CSVs")
    for p in csvs[:3]:
        print(f"\n--- {p.name} ---")
        try:
            df = pd.read_csv(p, nrows=3)
            print(df.columns.tolist()[:20])
            print(df.head(1).to_string())
            print(f"shape sample: {pd.read_csv(p, usecols=[0]).shape[0]} rows")
        except Exception as e:
            print(f"read error {e}")
except Exception as e:
    print(f"inspect error {e}")