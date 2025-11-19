# debug_convert_issues.py
# Kiểm tra các vấn đề tiềm ẩn trong quá trình convert
# python ks-nj4/debug_convert_issues.py

from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np
from typing import Dict

# ============================================================================
# CẤU HÌNH
# ============================================================================

YOLO_LABELS_DIR = Path("ks-nj4/data/datasets/labels")
MASKS_DIR = Path("ks-nj4/data/datasets-sam-vit/masks")
COCO_JSON_DIR = Path("ks-nj4/data/datasets-sam-vit/labels")
IMAGES_DIR = Path("ks-nj4/data/datasets/images")

SPLITS = ['test', 'train', 'val']

YOLO_TO_COCO = {
    0: 2,  # whitebacked_planthopper
    1: 3,  # rice_leaf_miner
    2: 1,  # brown_planthopper
}

# ============================================================================

def read_yolo_label_file(txt_file: Path) -> list:
    """Đọc file YOLO label"""
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
        pass
    return class_ids

def count_connected_components_in_mask(mask_file: Path) -> Dict:
    """Đếm số connected components trong mask"""
    if not mask_file.exists():
        return None
    
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    category_components = Counter()
    
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        if cls > 0:
            class_mask = (mask == cls).astype(np.uint8)
            num_labels, _ = cv2.connectedComponents(class_mask)
            num_objects = num_labels - 1
            if num_objects > 0:
                category_components[int(cls)] = num_objects
    
    return dict(category_components)

def analyze_convert_issues(split: str):
    """Phân tích các vấn đề trong quá trình convert"""
    print(f"\n{'=' * 70}")
    print(f"PHÂN TÍCH VẤN ĐỀ CONVERT - SPLIT: {split.upper()}")
    print(f"{'=' * 70}")
    
    yolo_labels_dir = YOLO_LABELS_DIR / split
    masks_dir = MASKS_DIR / split
    images_dir = IMAGES_DIR / split
    
    if not yolo_labels_dir.exists():
        print(f"⚠️  Thư mục YOLO labels không tồn tại: {yolo_labels_dir}")
        return
    
    if not masks_dir.exists():
        print(f"⚠️  Thư mục masks chưa tồn tại: {masks_dir}")
        print(f"    → Cần chạy convert script trước: python ks-nj4/convert_to_sam_vit_format.py")
        print(f"\n📊 CHỈ PHÂN TÍCH YOLO LABELS (chưa có masks để so sánh):")
        
        # Chỉ phân tích YOLO labels
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
        
        print(f"  - Tổng file labels: {len(yolo_label_files):,}")
        print(f"  - Images có labels: {images_with_yolo_labels:,}")
        print(f"  - Tổng labels: {yolo_total_labels:,}")
        print(f"\n  Phân bố labels theo class:")
        for class_id in sorted(yolo_class_counts.keys()):
            class_name_map = {
                0: "whitebacked_planthopper",
                1: "rice_leaf_miner",
                2: "brown_planthopper",
            }
            class_name = class_name_map.get(class_id, f"Class {class_id}")
            count = yolo_class_counts[class_id]
            images = len(yolo_image_counts[class_id])
            pct = (count / yolo_total_labels * 100) if yolo_total_labels > 0 else 0
            print(f"    Class {class_id} ({class_name}): {count:,} labels ({pct:.2f}%), {images:,} images")
        return
    
    # Đếm từ YOLO
    yolo_label_files = list(yolo_labels_dir.glob("*.txt"))
    yolo_stats = {
        'total_files': len(yolo_label_files),
        'files_with_labels': 0,
        'total_labels': 0,
        'class_counts': Counter(),
        'images_with_class': defaultdict(set),
    }
    
    for label_file in yolo_label_files:
        class_ids = read_yolo_label_file(label_file)
        if len(class_ids) > 0:
            yolo_stats['files_with_labels'] += 1
            yolo_stats['total_labels'] += len(class_ids)
            for class_id in class_ids:
                yolo_stats['class_counts'][class_id] += 1
                yolo_stats['images_with_class'][class_id].add(label_file.stem)
    
    # Đếm từ masks
    mask_files = [f for f in masks_dir.glob("*.png") if not f.stem.endswith("_vis")]
    mask_stats = {
        'total_files': len(mask_files),
        'files_with_annotations': 0,
        'total_components': 0,
        'category_counts': Counter(),
        'images_with_category': defaultdict(set),
        'images_missing': [],
    }
    
    for mask_file in mask_files:
        components = count_connected_components_in_mask(mask_file)
        if components and sum(components.values()) > 0:
            mask_stats['files_with_annotations'] += 1
            for category_id, count in components.items():
                mask_stats['category_counts'][category_id] += count
                mask_stats['total_components'] += count
                mask_stats['images_with_category'][category_id].add(mask_file.stem)
    
    # Tìm images có YOLO labels nhưng không có mask
    yolo_image_names = {f.stem for f in yolo_label_files}
    mask_image_names = {f.stem for f in mask_files}
    mask_stats['images_missing'] = sorted(list(yolo_image_names - mask_image_names))
    
    # So sánh
    print(f"\n📊 YOLO LABELS:")
    print(f"  - Tổng file labels: {yolo_stats['total_files']:,}")
    print(f"  - Files có labels: {yolo_stats['files_with_labels']:,}")
    print(f"  - Tổng labels: {yolo_stats['total_labels']:,}")
    
    print(f"\n📊 MASKS:")
    print(f"  - Tổng file masks: {mask_stats['total_files']:,}")
    print(f"  - Files có annotations: {mask_stats['files_with_annotations']:,}")
    print(f"  - Tổng connected components: {mask_stats['total_components']:,}")
    
    print(f"\n📊 SO SÁNH CHI TIẾT:")
    print(f"{'=' * 70}")
    
    # So sánh từng class
    for yolo_class in sorted(yolo_stats['class_counts'].keys()):
        coco_category = YOLO_TO_COCO.get(yolo_class)
        
        yolo_count = yolo_stats['class_counts'][yolo_class]
        yolo_images = len(yolo_stats['images_with_class'][yolo_class])
        
        mask_count = mask_stats['category_counts'].get(coco_category, 0)
        mask_images = len(mask_stats['images_with_category'].get(coco_category, set()))
        
        diff_count = mask_count - yolo_count
        diff_images = mask_images - yolo_images
        
        class_name_map = {
            0: "whitebacked_planthopper",
            1: "rice_leaf_miner",
            2: "brown_planthopper",
        }
        class_name = class_name_map.get(yolo_class, f"Class {yolo_class}")
        
        print(f"\n  Class {yolo_class} ({class_name}) → Category {coco_category}:")
        print(f"    YOLO: {yolo_count:,} labels, {yolo_images:,} images")
        print(f"    Mask: {mask_count:,} components, {mask_images:,} images")
        print(f"    Chênh lệch: {diff_count:+,} components, {diff_images:+,} images")
        
        if diff_count < 0:
            loss_pct = (abs(diff_count) / yolo_count * 100) if yolo_count > 0 else 0
            print(f"    ⚠️  MẤT {abs(diff_count)} labels ({loss_pct:.1f}%) - Có thể SAM2 không tạo được mask")
        elif diff_count > 0:
            gain_pct = (diff_count / yolo_count * 100) if yolo_count > 0 else 0
            print(f"    ℹ️  TĂNG {diff_count} components ({gain_pct:.1f}%) - SAM2 tách bbox thành nhiều objects")
        
        if diff_images < 0:
            print(f"    ⚠️  MẤT {abs(diff_images)} images - Có thể không được convert")
    
    # Tổng hợp
    total_diff = mask_stats['total_components'] - yolo_stats['total_labels']
    total_diff_pct = (total_diff / yolo_stats['total_labels'] * 100) if yolo_stats['total_labels'] > 0 else 0
    
    print(f"\n{'=' * 70}")
    print(f"TỔNG HỢP:")
    print(f"  YOLO labels: {yolo_stats['total_labels']:,}")
    print(f"  Mask components: {mask_stats['total_components']:,}")
    print(f"  Chênh lệch: {total_diff:+,} ({total_diff_pct:+.2f}%)")
    
    # Images missing
    if mask_stats['images_missing']:
        print(f"\n⚠️  IMAGES CÓ YOLO LABELS NHƯNG KHÔNG CÓ MASK ({len(mask_stats['images_missing'])} images):")
        for img_name in mask_stats['images_missing'][:10]:
            yolo_labels = read_yolo_label_file(yolo_labels_dir / f"{img_name}.txt")
            print(f"    - {img_name}: {len(yolo_labels)} YOLO labels")
        if len(mask_stats['images_missing']) > 10:
            print(f"    ... và {len(mask_stats['images_missing']) - 10} images khác")
    
    # Phân tích overlap
    print(f"\n{'=' * 70}")
    print("PHÂN TÍCH OVERLAP MASKS:")
    print(f"{'=' * 70}")
    
    overlap_issues = 0
    for mask_file in mask_files[:100]:  # Kiểm tra 100 masks đầu tiên
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Kiểm tra xem có pixel nào có giá trị > 0 nhưng không phải là một trong các category hợp lệ không
        unique_values = np.unique(mask)
        valid_categories = {0, 1, 2, 3}
        invalid_values = [v for v in unique_values if v not in valid_categories]
        
        if invalid_values:
            overlap_issues += 1
            if overlap_issues <= 5:  # Hiển thị 5 đầu tiên
                print(f"  ⚠️  {mask_file.name}: Có giá trị không hợp lệ: {invalid_values}")
    
    if overlap_issues == 0:
        print("  ✓ Không phát hiện vấn đề overlap")
    else:
        print(f"  ⚠️  Tìm thấy {overlap_issues} masks có vấn đề (kiểm tra 100 masks đầu tiên)")

def main():
    """Hàm chính"""
    print("=" * 70)
    print("KIỂM TRA VẤN ĐỀ TRONG QUÁ TRÌNH CONVERT")
    print("=" * 70)
    
    for split in SPLITS:
        analyze_convert_issues(split)
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

