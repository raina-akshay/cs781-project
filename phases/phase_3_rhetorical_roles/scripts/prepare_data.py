#!/usr/bin/env python3
"""Build the full-token cache from the seven official RR files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rr.data import prepare_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "../../assets/foundation/tokenizer.json"
    )
    args = parser.parse_args()
    provenance = json.loads((ROOT / "configs/data-sources.json").read_text())
    for record in provenance["files"]:
        record["local_path"] = str((ROOT / "data/raw" / record["path"]).resolve())
    path = ROOT / "data/provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(provenance, indent=2) + "\n"
    path.write_text(encoded)
    prepare_cache(path, args.tokenizer, ROOT / "data/prepared/sentence-cache.json")


if __name__ == "__main__":
    main()
