from pathlib import Path

def ensure_dirs(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
