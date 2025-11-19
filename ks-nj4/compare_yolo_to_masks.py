# compare_yolo_to_masks.py
# So sánh số lượng labels từ YOLO với số annotations từ masks sau convert
# python ks-nj4/compare_yolo_to_masks.py

from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict
import cv2
import numpy as np

# ============================================================================
# CẤU HÌNH
# ============================================================================

# Đường dẫn
YOLO_LABELS_DIR = Path("ks-nj4/data/datasets/labels")
MASKS_DIR = Path("ks-nj4/data/datasets-sam-vit/masks")
IMAGES_DIR = Path("ks-nj4/data/datasets/images")

# Các split
SPLITS = ['train', 'val', 'test']

# Mapping YOLO class → COCO category
# YOLO class 0 → COCO category 2 (whitebacked_planthopper)
# YOLO class 1 → COCO category 3 (rice_leaf_miner)
# YOLO class 2 → COCO category 1 (brown_planthopper)
YOLO_TO_COCO = {
    0: 2,  # whitebacked_planthopper
    1: 3,  # rice_leaf_miner
    2: 1,  # brown_planthopper
}

CLASS_NAMES = {
    0: "whitebacked_planthopper (YOLO) → Category 2 (COCO)",
    1: "rice_leaf_miner (YOLO) → Category 3 (COCO)",
    2: "brown_planthopper (YOLO) → Category 1 (COCO)",
}

COCO_CATEGORY_NAMES = {
    1: "brown_planthopper",
    2: "whitebacked_planthopper",
    3: "rice_leaf_miner",
}

# ============================================================================

def read_yolo_label_file(txt_file: Path) -> list:
    """Đọc file YOLO label và trả về danh sách class IDs"""
    class_ids = []
    try:
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            class_ids.append(class_id)
    except Exception as e:
        print(f"  ⚠️  Lỗi khi đọc {txt_file}: {e}")
    return class_ids

def count_annotations_in_mask(mask_file: Path) -> Dict:
    """Đếm số annotations (connected components) trong mask"""
    if not mask_file.exists():
        return None
    
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    category_annotations = Counter()
    category_images = set()
    
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        if cls > 0:  # Bỏ background
            class_mask = (mask == cls).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(class_mask)
            num_objects = num_labels - 1  # Trừ background label
            if num_objects > 0:
                category_id = int(cls)
                category_annotations[category_id] += num_objects
                category_images.add(category_id)
    
    return {
        'category_annotations': dict(category_annotations),
        'category_images': category_images,
    }

def compare_split(split: str):
    """So sánh YOLO labels với masks cho một split"""
    print(f"\n{'=' * 70}")
    print(f"SO SÁNH SPLIT: {split.upper()}")
    print(f"{'=' * 70}")
    
    yolo_labels_dir = YOLO_LABELS_DIR / split
    masks_dir = MASKS_DIR / split
    images_dir = IMAGES_DIR / split
    
    if not yolo_labels_dir.exists():
        print(f"⚠️  Thư mục YOLO labels {yolo_labels_dir} không tồn tại!")
        return
    
    if not masks_dir.exists():
        print(f"⚠️  Thư mục masks {masks_dir} không tồn tại!")
        return
    
    # Đếm từ YOLO labels
    yolo_label_files = list(yolo_labels_dir.glob("*.txt"))
    yolo_class_counts = Counter()
    yolo_image_counts = defaultdict(set)
    yolo_total_labels = 0
    images_with_yolo_labels = 0
    
    for label_file in yolo_label_files:
        class_ids = read_yolo_label_file(label_file)
        if len(class_ids) > 0:
            images_with_yolo_labels += 1
            yolo_total_labels += len(class_ids)
            unique_classes = set(class_ids)
            for class_id in unique_classes:
                count = class_ids.count(class_id)
                yolo_class_counts[class_id] += count
                yolo_image_counts[class_id].add(label_file.stem)
    
    # Đếm từ masks
    mask_files = [f for f in masks_dir.glob("*.png") if not f.stem.endswith("_vis")]
    mask_category_counts = Counter()
    mask_image_counts = defaultdict(set)
    mask_total_annotations = 0
    images_with_masks = 0
    
    for mask_file in mask_files:
        mask_stats = count_annotations_in_mask(mask_file)
        if mask_stats:
            images_with_masks += 1
            for category_id, count in mask_stats['category_annotations'].items():
                mask_category_counts[category_id] += count
                mask_total_annotations += count
                mask_image_counts[category_id].add(mask_file.stem)
    
    # So sánh
    print(f"\n📊 YOLO LABELS:")
    print(f"  - Tổng số file labels: {len(yolo_label_files):,}")
    print(f"  - Images có labels: {images_with_yolo_labels:,}")
    print(f"  - Tổng số labels: {yolo_total_labels:,}")
    
    print(f"\n📊 MASKS (SAU CONVERT):")
    print(f"  - Tổng số masks: {len(mask_files):,}")
    print(f"  - Images có masks: {images_with_masks:,}")
    print(f"  - Tổng số annotations: {mask_total_annotations:,}")
    
    print(f"\n📊 SO SÁNH CHI TIẾT:")
    print(f"{'=' * 70}")
    
    # So sánh từng class
    for yolo_class in sorted(yolo_class_counts.keys()):
        coco_category = YOLO_TO_COCO.get(yolo_class)
        class_name = CLASS_NAMES.get(yolo_class, f"Class {yolo_class}")
        
        yolo_count = yolo_class_counts[yolo_class]
        yolo_images = len(yolo_image_counts[yolo_class])
        
        mask_count = mask_category_counts.get(coco_category, 0)
        mask_images = len(mask_image_counts.get(coco_category, set()))
        
        diff_count = mask_count - yolo_count
        diff_images = mask_images - yolo_images
        diff_pct = (diff_count / yolo_count * 100) if yolo_count > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  YOLO: {yolo_count:,} labels, {yolo_images:,} images")
        print(f"  Mask: {mask_count:,} annotations, {mask_images:,} images")
        print(f"  Chênh lệch: {diff_count:+,} annotations ({diff_pct:+.2f}%), {diff_images:+,} images")
        
        if diff_count < 0:
            print(f"    ⚠️  Mất {abs(diff_count)} annotations (có thể SAM2 không tạo được mask)")
        elif diff_count > 0:
            print(f"    ℹ️  Tăng {diff_count} annotations (SAM2 tách bbox thành nhiều connected components)")
        
        if diff_images < 0:
            print(f"    ⚠️  Mất {abs(diff_images)} images (có thể không được convert)")
    
    # Tổng hợp
    total_diff = mask_total_annotations - yolo_total_labels
    total_diff_pct = (total_diff / yolo_total_labels * 100) if yolo_total_labels > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"TỔNG HỢP:")
    print(f"  YOLO labels: {yolo_total_labels:,}")
    print(f"  Mask annotations: {mask_total_annotations:,}")
    print(f"  Chênh lệch: {total_diff:+,} ({total_diff_pct:+.2f}%)")
    
    # Tìm images có YOLO labels nhưng không có mask
    yolo_image_names = {f.stem for f in yolo_label_files}
    mask_image_names = {f.stem for f in mask_files}
    missing_masks = yolo_image_names - mask_image_names
    
    if missing_masks:
        print(f"\n⚠️  Images có YOLO labels nhưng KHÔNG có mask ({len(missing_masks)} images):")
        for img_name in sorted(list(missing_masks))[:10]:  # Hiển thị 10 đầu tiên
            print(f"    - {img_name}")
        if len(missing_masks) > 10:
            print(f"    ... và {len(missing_masks) - 10} images khác")

def main():
    """Hàm chính"""
    print("=" * 70)
    print("SO SÁNH YOLO LABELS VỚI MASKS SAU CONVERT")
    print("=" * 70)
    
    for split in SPLITS:
        compare_split(split)
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

