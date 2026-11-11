"""JSON output and prediction checks shared by training and inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT
FORMAT = "cs781-phase6-s1-lorma-generation-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def append_event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": now(), **value}, allow_nan=False) + "\n")
        stream.flush()


def verify_metric_assets(snapshot):
    snapshot = Path(snapshot)
    provenance = read_json(PHASE / "metadata/scorer-provenance.json")
    for name, row in provenance["files"].items():
        if sha256(snapshot / name) != row["sha256"]:
            raise ValueError("Scorer weights do not match the selected metric: " + name)


def validate_predictions(predictions, expected_ids):
    if not isinstance(predictions, list) or len(predictions) != len(expected_ids):
        raise ValueError("Prediction count/type differs")
    seen = set()
    for row, expected in zip(predictions, expected_ids):
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "summary"}
            or type(row["id"]) is not str
            or row["id"] != expected
            or row["id"] in seen
            or type(row["summary"]) is not str
            or not row["summary"].strip()
        ):
            raise ValueError("Prediction schema/ID/order/nonempty constraint failed")
        seen.add(row["id"])
    return predictions
