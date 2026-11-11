#!/usr/bin/env python3
"""Prepare the original document split and character-level NER cache."""
import argparse
import json
from pathlib import Path
from shared.ner_contract import load_json
from shared.ner_data import prepare_cache, sha256

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing official Phase_2_LNER Train/ and Test/",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    raw, out = args.raw_dir.resolve(), args.output.resolve()
    split_path = ROOT / "configs/split.json"
    split = load_json(split_path)
    tokenizer = ROOT.parents[1] / "assets/foundation/tokenizer.json"
    if sha256(tokenizer) != split["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the original split")
    texts, gold = {}, {}
    for fold in (1, 2):
        tx = load_json(raw / f"Train/fold{fold}-text.json")
        gd = load_json(raw / f"Train/fold{fold}-gold.json")
        if set(texts) & set(tx) or set(gold) & set(gd):
            raise ValueError("Duplicate document IDs across training folds")
        texts.update(tx)
        gold.update(gd)
    labels_path = raw / "Train/labels.txt"
    if set(gold) != set(texts):
        raise ValueError("Training text and annotation IDs differ")
    if set(texts) != set(split["train_ids"]) | set(split["validation_ids"]):
        raise ValueError("Released training IDs do not match the selected split")
    test_path = raw / "Test/fold3-text.json"
    test = load_json(test_path)
    if set(test) != set(split["test_ids"]):
        raise ValueError("Released test IDs do not match the selected split")
    prepared = out / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    paths = {
        "tokenizer": tokenizer,
        "labels": labels_path,
        "split": split_path,
        "test_text": test_path,
    }
    for part in ("train", "validation"):
        for kind, data in (("text", texts), ("gold", gold)):
            path = prepared / f"{part}-{kind}.json"
            value = {key: data[key] for key in split[part + "_ids"]}
            if path.exists() and load_json(path) != value:
                raise ValueError(f"Existing prepared file differs: {path}")
            path.write_text(
                json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            paths[part + "_" + kind] = path
    print(prepare_cache(out, source_paths=paths))


if __name__ == "__main__":
    main()
