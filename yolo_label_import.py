
"""
This script converts CVAT XML annotations (with Onion/Weed boxes or Spinach/Weed) from a ZIP to the YOLO format
with optional train/val/test split.

Features:

- Requires ``--zip_path`` (path to the zip file) and optionally ``--out`` for output directory
  (defaults to ``yolo_dataset``).
- Splits and seed use sensible defaults (0.8/0.1/0.1 and 42).
- Extracts the zip to a folder next to the zip (same name without .zip),
  parses the XML, extracts only the classes ``Onion`` (class 0) and ``Weed``
  (class 1) or ``Spinach``(class 0) and ``Weed`` (class 1), converts coordinates to YOLO normalised format, shuffles and
  splits the data, copies images and writes label files under:
  ``<out>/images/{train,val,test}/`` and ``<out>/labels/{train,val,test}/``.
- Writes a ``data.yaml`` file suitable for YOLO training.

-----
Usage examples:

python yolo_label_import.py --zip_path /path/to/archive.zip
python yolo_label_import.py --zip_path data/cvat.zip --out my_dataset
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from shutil import copy2
from typing import List, Tuple
from zipfile import ZipFile, BadZipFile

import xml.etree.ElementTree as ET

#possible crop types (datasets):
ALL_CLASS_MAPS = {
    "onion": {
        "Onion": 0,
        "Weed": 1,
    },
    "spinach": {
        "Spinach": 0,
        "Weed": 1,
    },
}
CLASS_MAP = {}
ID_TO_NAME = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert CVAT XML (with Onion/Weed boxes or Spinach/Weed) from a ZIP → YOLO format "
            "with optional train/val/test split. Requires --zip_path."
        )
    )
    p.add_argument(
        "--zip_path",
        required=True,
        help="Path to the zip file containing CVAT annotations and images.",
    )
    p.add_argument(
        "--dataset",
        choices=sorted(ALL_CLASS_MAPS),
        required=True,
        help="Dataset type. Determines which plant class is used."
    )
    p.add_argument(
        "--out",
        default="yolo_dataset",
        help="Output directory that will contain images/ and labels/ subfolders.",
    )
    p.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.8, 0.1, 0.1],
        help="Fractions for train/val/test (must sum to 1.0). Example: --split 0.7 0.2 0.1",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    total = sum(args.split)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0 (got {total})")
    if any(s < 0 for s in args.split):
        raise ValueError("Split fractions cannot be negative.")


def _find_image_file(
    xml_parent: Path,
    img_name: str,
    img_subdir: str,
    search_root: Path | None = None,
) -> Path:
    """
    Try to locate an image file using several fallback strategies.
    Returns the first existing Path or raises FileNotFoundError.
    Strategies:
        1. XML parent / img_subdir / img_name
        2. XML parent / img_name
        3. Current working directory / img_name (fallback)
        4. If search_root provided: search_root / img_name
        5. If search_root provided: first file with matching basename found anywhere under search_root
        6. First file with matching basename found anywhere under xml_parent
    """
    candidates: List[Path] = []

    # 1) XML parent / img_subdir / img_name
    if img_subdir:
        candidates.append(xml_parent / img_subdir / img_name)
    # 2) XML parent / img_name (same directory as XML)
    candidates.append(xml_parent / img_name)
    # 3) Current working directory / img_name (fallback)
    candidates.append(Path(img_name))
    # 4) If a search_root is provided, try search_root / img_name
    if search_root is not None:
        candidates.append(search_root / img_name)
        # 5) Also try to find the file by basename anywhere under search_root (first match)
        basename = Path(img_name).name
        for cand in search_root.rglob(basename):
            if cand.is_file():
                candidates.append(cand)
                break  # only need one additional candidate
    # 6) Also try to find the file by basename anywhere under xml_parent
    basename = Path(img_name).name
    for cand in xml_parent.rglob(basename):
        if cand.is_file():
            candidates.append(cand)
            break  # only need one additional candidate

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Image not found: {img_name}")


def cvat_to_yolo(
    xml_path: Path,
    img_subdir: str,
    out_dir: Path,
    splits: List[float],
    seed: int,
    img_search_root: Path | None = None,
) -> None:
    """Core conversion routine."""

    # Parse CVAT XML:

    tree = ET.parse(xml_path)
    root = tree.getroot()

    name_to_id = CLASS_MAP

    records: List[Tuple[str, int, int, List[Tuple[int, float, float, float, float]]]] = []
    # each record: (img_name, width, height, [(class_id, xc, yc, w, h), ...])

    for img_elem in root.findall(".//image"):
        img_name = img_elem.get("name")
        width = int(img_elem.get("width"))
        height = int(img_elem.get("height"))
        # Defensive: some XMLs may omit height/width? Assume present.
        boxes: List[Tuple[int, float, float, float, float]] = []

        for box in img_elem.findall("box"):
            label = box.get("label")
            if label not in name_to_id:
                continue  # skip unwanted classes
            class_id = name_to_id[label]

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            # Convert to YOLO normalized format
            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            box_w = (xbr - xtl) / width
            box_h = (ybr - ytl) / height

            # Clamp to [0, 1] to avoid overshoot due to floating point errors
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w = max(0.0, min(1.0, box_w))
            box_h = max(0.0, min(1.0, box_h))

            boxes.append((class_id, x_center, y_center, box_w, box_h))

        records.append((img_name, width, height, boxes))

    # ------------------------------------------------------------------
    # Shuffle & split
    
    random.seed(seed)
    random.shuffle(records)

    n_total = len(records)
    n_train = int(splits[0] * n_total)
    n_val = int(splits[1] * n_total)
    n_test = n_total - n_train - n_val  # remainder

    splits_dict = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    # ------------------------------------------------------------------
    # Create output dirs and copy files + write labels

    for split_name, split_records in splits_dict.items():
        img_out_dir = out_dir / "images" / split_name
        lbl_out_dir = out_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_name, width, height, boxes in split_records:
            # Locate source image (search relative to XML location)
            src_img_path = _find_image_file(xml_path.parent, img_name, img_subdir, img_search_root)
            dst_img_path = img_out_dir / src_img_path.name
            copy2(src_img_path, dst_img_path)

            # Write YOLO label file
            lbl_file = lbl_out_dir / (src_img_path.stem + ".txt")
            with lbl_file.open("w") as f:
                for class_id, xc, yc, bw, bh in boxes:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # ------------------------------------------------------------------
    #print summary
    print("\n=== Conversion summary ===")
    print(f"Source XML          : {xml_path}")
    print(f"Output directory    : {out_dir}")
    print(f"Total images found  : {n_total}")
    for split_name, split_records in splits_dict.items():
        print(
            f"  {split_name:5s}: {len(split_records):4d} images "
            f"({100 * len(split_records) / n_total:5.1f}%)"
        )
    print("Done.\n")


def write_data_yaml(out_dir: Path) -> None:
    """
    Create a data.yaml file for YOLO training inside <out_dir>.
    Assumes two classes: onion (0) and weed (1) or spinach(0) and weed(1).
    """

    class_1 = ID_TO_NAME[0]
    class_2 = ID_TO_NAME[1]
    yaml_content = f"""# Auto‑generated by yolo_label.py
train: {os.path.join(out_dir, 'images', 'train').replace(os.sep, '/')}
val:   {os.path.join(out_dir, 'images', 'val').replace(os.sep, '/')}
test:  {os.path.join(out_dir, 'images', 'test').replace(os.sep, '/')}

# Number of classes
nc: 2

# Class names
names:
  0: {class_1}
  1: {class_2}
"""
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Created data.yaml at {yaml_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    global CLASS_MAP, ID_TO_NAME
    CLASS_MAP = ALL_CLASS_MAPS[args.dataset]
    ID_TO_NAME = {v: k for k, v in CLASS_MAP.items()}

    zip_path = Path(args.zip_path).resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    # Extract zip to a directory next to the zip
    extract_dir = zip_path.with_suffix("")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except BadZipFile as e:
        raise RuntimeError(f"Failed to extract zip {zip_path}: {e}") from e

    #locate the annotation file inside the extracted tree
    xml_candidates = list(extract_dir.rglob("annotations.xml"))
    if not xml_candidates:
        raise FileNotFoundError("annotations.xml not found inside the zip.")
    xml_path = xml_candidates[0]

    #runn conversion
    out_dir = Path(args.out).resolve()
    cvat_to_yolo(
        xml_path=xml_path,
        img_subdir="",   # assume images are next to XML and if not there are fallback strategies in _find_image_file
        out_dir=out_dir,
        splits=args.split,
        seed=args.seed,
        img_search_root=extract_dir,
    )
    write_data_yaml(out_dir)


if __name__ == "__main__":
    main()