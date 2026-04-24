from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KNOWN_METADATA = [
    "wlasl.json",
    "wlasl-split.json",
    "wlasl_all.json",
    "wlasl.jsonl",
    "labels.json",
]


def guess_label(item: dict[str, Any]) -> str | None:
    for key in ("gloss", "sign", "label", "word", "lexeme"):
        if key in item:
            return str(item[key])
    return None


def guess_video(item: dict[str, Any]) -> str | None:
    for key in ("video", "video_id", "url", "file", "file_name", "file_path"):
        if key in item:
            return str(item[key])
    return None


def guess_split(item: dict[str, Any]) -> str | None:
    for key in ("subset", "split", "partition", "fold"):
        if key in item:
            return str(item[key])
    return None


def load_json_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported metadata format")


def build_manifest(root: Path, metadata: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label_set: set[str] = set()
    rows = []

    for item in metadata:
        label = guess_label(item)
        video = guess_video(item)
        split = guess_split(item) or "train"
        if label is None or video is None:
            continue
        label_set.add(label)
        rows.append((split, video, label))

    labels = sorted(label_set)
    label_map = {label: idx for idx, label in enumerate(labels)}

    with (out_dir / "labels.txt").open("w", encoding="utf-8") as fh:
        for label in labels:
            fh.write(f"{label}\n")

    manifests: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for split, video, label in rows:
        dataset = split.lower()
        if dataset not in manifests:
            dataset = "train"
        manifests[dataset].append((video, label_map[label]))

    for split, entries in manifests.items():
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["video_path", "label_idx"])
            for video, label_idx in entries:
                writer.writerow([video, label_idx])

    print(f"Wrote {len(labels)} labels and manifests to {out_dir}")


def find_metadata(root: Path) -> Path | None:
    for candidate in KNOWN_METADATA:
        path = root / candidate
        if path.exists():
            return path
    for path in root.rglob("*.json"):
        if path.name.lower().startswith("wlasl"):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar manifestos para WLASL")
    parser.add_argument("--root", required=True, help="Carpeta raíz del conjunto WLASL")
    parser.add_argument("--output", required=True, help="Carpeta donde escribir labels y CSV")
    parser.add_argument("--metadata", help="Ruta opcional al archivo de metadatos JSON")
    args = parser.parse_args()

    root = Path(args.root)
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        metadata_path = find_metadata(root) or Path("wlasl.json")

    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontró metadata WLASL en {metadata_path}")

    metadata = load_json_metadata(metadata_path)
    build_manifest(root, metadata, Path(args.output))


if __name__ == "__main__":
    main()
