#!/usr/bin/env python3
"""Download the three course case files at the selected dataset revision."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import get_token
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "93b0059be756c6c461f3a225de25aabda7d5f84a"
REPOSITORY = "Exploration-Lab/CS-781-Capstone"
FILES = (("train", "Train/train.json"), ("dev", "Train/dev.json"), ("test", "Test/test.json"))


def main():
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    token = get_token()
    headers = {"Authorization": "Bearer " + token} if token else {}
    manifest = {
        "repository": REPOSITORY,
        "repository_type": "dataset",
        "revision": REVISION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for split, relative in FILES:
        path = directory / f"{split}.json"
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/Phase_4_CJPE/{relative}"
        digest, size = hashlib.sha256(), 0
        with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with path.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
        manifest["files"].append(
            {
                "split": split,
                "repository_path": "Phase_4_CJPE/" + relative,
                "project_symlink": str(path),
                "resolve_url": url,
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
        print(f"Downloaded {split}: {size} bytes")
    (directory / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
