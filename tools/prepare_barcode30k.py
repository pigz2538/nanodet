#!/usr/bin/env python
"""
Prepare Barcode-30k merged dataset for NanoDet training.

Steps:
1. Parse three source datasets (two COCO + one VOC/XML).
2. Convert images to grayscale and save to unified_pool/ with prefixes.
3. Unify category ids to: 1=barcode, 2=qr_code.
4. Filter out tiny bboxes (w<4 or h<4).
5. Shuffle globally with fixed seed and split 8:1:1.
6. Move images to barcode30k_final/{train,val,test}/images/.
7. Write COCO JSON annotations.
8. Compute single-channel mean/std on train set.
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
POOL_DIR = DATASET_DIR / "unified_pool"
FINAL_DIR = DATASET_DIR / "barcode30k_final"

# Source dataset paths
DATASET_A = DATASET_DIR / "Barcode and QR code detection.v3i.coco"
DATASET_B = DATASET_DIR / "qr code.v3i.coco"
DATASET_C = DATASET_DIR / "archive"

RANDOM_SEED = 42
SPLIT_RATIOS = [0.8, 0.1, 0.1]  # train, val, test


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_coco_json(json_path: Path, prefix: str, category_map: dict):
    """
    Parse a COCO JSON file and return list of records.
    category_map: {raw_cat_id: new_cat_id}
    """
    with open(json_path) as f:
        data = json.load(f)

    # Build image id -> new file name
    image_records = []
    for img in data["images"]:
        new_file_name = f"{prefix}_{img['file_name']}"
        image_records.append(
            {
                "id": img["id"],
                "new_id": None,  # assigned later
                "file_name": img["file_name"],
                "new_file_name": new_file_name,
                "height": img["height"],
                "width": img["width"],
                "src_image_path": json_path.parent / img["file_name"],
                "pool_image_path": POOL_DIR / new_file_name,
                "annotations": [],
            }
        )

    id_to_record = {r["id"]: r for r in image_records}

    for ann in data["annotations"]:
        raw_cat_id = ann["category_id"]
        if raw_cat_id not in category_map:
            continue
        new_cat_id = category_map[raw_cat_id]

        x, y, w, h = ann["bbox"]
        if w < 4 or h < 4:
            continue

        record = id_to_record.get(ann["image_id"])
        if record is None:
            continue

        record["annotations"].append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "category_id": int(new_cat_id),
                "area": float(ann.get("area", w * h)),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )

    # Drop images with no valid annotations (optional: keep if you want negatives)
    return [r for r in image_records if r["annotations"]]


def find_image_for_xml(xml_path: Path, archive_dir: Path):
    """Find matching image file for a VOC XML (case-insensitive extension)."""
    base = xml_path.stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG"]:
        candidate = archive_dir / (base + ext)
        if candidate.exists():
            return candidate
    return None


def parse_voc_xml(archive_dir: Path, prefix: str):
    """Parse VOC/XML dataset and return list of records."""
    xml_files = sorted(archive_dir.glob("*.xml"))
    records = []

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_elem = root.find("filename")
        if filename_elem is None or not filename_elem.text:
            continue
        original_name = filename_elem.text
        base_name = Path(original_name).stem
        new_file_name = f"{prefix}_{base_name}.jpg"

        src_image_path = find_image_for_xml(xml_path, archive_dir)
        if src_image_path is None:
            print(f"[WARN] No image found for {xml_path.name}, skipping.")
            continue

        size = root.find("size")
        if size is None:
            continue
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        annotations = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            class_name = name_elem.text.lower().strip()

            # Only keep barcode objects in dataset C
            if "barcode" in class_name:
                new_cat_id = 1
            elif "qr" in class_name:
                new_cat_id = 2
            else:
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin

            if w < 4 or h < 4:
                continue

            annotations.append(
                {
                    "bbox": [xmin, ymin, w, h],
                    "category_id": int(new_cat_id),
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )

        if annotations:
            records.append(
                {
                    "id": 0,
                    "new_id": None,
                    "file_name": original_name,
                    "new_file_name": new_file_name,
                    "height": height,
                    "width": width,
                    "src_image_path": src_image_path,
                    "pool_image_path": POOL_DIR / new_file_name,
                    "annotations": annotations,
                }
            )

    return records


def convert_to_grayscale(src_path: Path, dst_path: Path):
    """Read image, convert to grayscale, save as single-channel jpg."""
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {src_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(dst_path), gray)


def build_coco_annotation(records: list, categories: list):
    """Build COCO format dict from records."""
    images = []
    annotations = []
    ann_id = 0

    for rec in records:
        rec["new_id"] = len(images)
        images.append(
            {
                "id": rec["new_id"],
                "file_name": rec["new_file_name"],
                "height": rec["height"],
                "width": rec["width"],
            }
        )

        for ann in rec["annotations"]:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": rec["new_id"],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"],
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": "Barcode-30k merged dataset for NanoDet"},
    }


def split_records(records: list, ratios: list, seed: int):
    """Shuffle records with fixed seed and split into train/val/test."""
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # test gets the rest to avoid rounding issues

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def move_to_final(split_name: str, records: list, categories: list):
    """Move pooled images to final split dir and write COCO JSON."""
    split_dir = FINAL_DIR / split_name
    images_dir = split_dir / "images"
    clean_dir(images_dir)

    for rec in records:
        src = rec["pool_image_path"]
        dst = images_dir / rec["new_file_name"]
        if not src.exists():
            print(f"[WARN] Missing pooled image: {src}")
            continue
        shutil.copy2(str(src), str(dst))

    coco = build_coco_annotation(records, categories)
    with open(split_dir / f"{split_name}.json", "w") as f:
        json.dump(coco, f, indent=2)

    return len(coco["images"]), len(coco["annotations"])


def compute_mean_std(image_dir: Path):
    """Compute single-channel mean and std for grayscale images."""
    files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not files:
        return 128.0, 128.0

    pixels = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pixels.append(img.astype(np.float32).ravel())

    all_pixels = np.concatenate(pixels)
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    return mean, std


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Step 1/5: Parsing source datasets...")
    # Dataset A: 416x416, COCO, cat_id 1=barcode, 2=qr_code
    records_a = []
    for split in ["train", "valid", "test"]:
        json_path = DATASET_A / split / "_annotations.coco.json"
        if json_path.exists():
            records_a.extend(parse_coco_json(json_path, "A", {1: 1, 2: 2}))
    print(f"  Dataset A: {len(records_a)} valid images")

    # Dataset B: 640x640, COCO, only QR code variants
    records_b = []
    for split in ["train", "valid", "test"]:
        json_path = DATASET_B / split / "_annotations.coco.json"
        if json_path.exists():
            # category ids 1,2,3 all map to qr_code (2)
            records_b.extend(parse_coco_json(json_path, "B", {1: 2, 2: 2, 3: 2}))
    print(f"  Dataset B: {len(records_b)} valid images")

    # Dataset C: VOC/XML, only barcode
    records_c = parse_voc_xml(DATASET_C, "C")
    print(f"  Dataset C: {len(records_c)} valid images")

    all_records = records_a + records_b + records_c
    print(f"Total valid images: {len(all_records)}")

    print("\nStep 2/5: Converting images to grayscale...")
    clean_dir(POOL_DIR)
    for rec in all_records:
        convert_to_grayscale(rec["src_image_path"], rec["pool_image_path"])
    print(f"  Saved {len(all_records)} grayscale images to {POOL_DIR}")

    print("\nStep 3/5: Shuffling and splitting 8:1:1...")
    train, val, test = split_records(all_records, SPLIT_RATIOS, RANDOM_SEED)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    categories = [
        {"id": 1, "name": "barcode", "supercategory": "code"},
        {"id": 2, "name": "qr_code", "supercategory": "code"},
    ]

    print("\nStep 4/5: Building final dataset...")
    clean_dir(FINAL_DIR)
    for split_name, records in [("train", train), ("val", val), ("test", test)]:
        n_imgs, n_anns = move_to_final(split_name, records, categories)
        print(f"  {split_name}: {n_imgs} images, {n_anns} annotations")

    print("\nStep 5/5: Computing mean/std on train set...")
    mean, std = compute_mean_std(FINAL_DIR / "train" / "images")
    print(f"  mean: [{mean:.4f}], std: [{std:.4f}]")

    # Write stats for easy copy-paste
    stats = {"mean": [mean], "std": [std]}
    with open(FINAL_DIR / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. Final dataset: {FINAL_DIR}")


if __name__ == "__main__":
    main()
