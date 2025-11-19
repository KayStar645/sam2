# count_yolo_labels.py
# Đếm số lượng labels của từng nhãn trong các tập train, test, val từ YOLO format
# python ks-nj4/count_yolo_labels.py

from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict

# ============================================================================
# CẤU HÌNH
# ============================================================================

# Đường dẫn thư mục labels
LABELS_DIR = Path("ks-nj4/data/datasets/labels")

# Các split cần xử lý
SPLITS = ['train', 'val', 'test']

# Mapping class IDs sang tên
CLASS_NAMES = {
    0: "whitebacked_planthopper",  # Rầy lưng trắng (WBPH)
    1: "rice_leaf_miner",           # Sâu ăn lá lúa (RLM)
    2: "brown_planthopper",          # Rầy nâu (BPH)
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
                            class_id = int(float(parts[0]))  # Class ID là phần tử đầu tiên
                            class_ids.append(class_id)
    except Exception as e:
        print(f"  ⚠️  Lỗi khi đọc {txt_file}: {e}")
    
    return class_ids

def count_labels_in_split(split: str, labels_dir: Path) -> Dict:
    """Đếm số lượng labels của từng class trong một split"""
    split_labels_dir = labels_dir / split
    
    if not split_labels_dir.exists():
        print(f"⚠️  Thư mục {split_labels_dir} không tồn tại, bỏ qua...")
        return None
    
    # Tìm tất cả file .txt
    label_files = list(split_labels_dir.glob("*.txt"))
    
    # Đếm labels
    class_counts = Counter()  # Đếm số labels của từng class
    image_counts = Counter()  # Đếm số images có từng class
    images_with_labels = 0
    images_without_labels = 0
    total_labels = 0
    
    for label_file in label_files:
        class_ids = read_yolo_label_file(label_file)
        
        if len(class_ids) > 0:
            images_with_labels += 1
            total_labels += len(class_ids)
            
            # Đếm từng class
            unique_classes = set(class_ids)
            for class_id in unique_classes:
                count = class_ids.count(class_id)
                class_counts[class_id] += count
                image_counts[class_id] += 1  # Image này có class này
        else:
            images_without_labels += 1
    
    stats = {
        'split': split,
        'total_label_files': len(label_files),
        'images_with_labels': images_with_labels,
        'images_without_labels': images_without_labels,
        'total_labels': total_labels,
        'class_counts': dict(class_counts),
        'image_counts': dict(image_counts),
    }
    
    return stats

def main():
    """Hàm chính"""
    print("=" * 70)
    print("ĐẾM SỐ LƯỢNG LABELS TỪ YOLO FORMAT")
    print("=" * 70)
    print(f"Thư mục labels: {LABELS_DIR}")
    print("=" * 70)
    
    if not LABELS_DIR.exists():
        print(f"⚠️  Thư mục {LABELS_DIR} không tồn tại!")
        return
    
    # Đếm labels cho từng split
    all_stats = {}
    total_labels_all = 0
    total_class_counts_all = Counter()
    total_image_counts_all = Counter()
    
    for split in SPLITS:
        print(f"\n{'=' * 70}")
        print(f"Phân tích split: {split.upper()}")
        print(f"{'=' * 70}")
        
        stats = count_labels_in_split(split, LABELS_DIR)
        if stats is None:
            continue
        
        all_stats[split] = stats
        
        # In kết quả
        print(f"  Tổng số file labels: {stats['total_label_files']:,}")
        print(f"  Images có labels: {stats['images_with_labels']:,}")
        print(f"  Images không có labels: {stats['images_without_labels']:,}")
        print(f"  Tổng số labels: {stats['total_labels']:,}")
        
        print(f"\n  Phân bố labels theo class:")
        print(f"  {'=' * 60}")
        
        # Tính tổng để tính phần trăm
        total_labels_split = stats['total_labels']
        
        for class_id in sorted(stats['class_counts'].keys()):
            class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
            label_count = stats['class_counts'][class_id]
            image_count = stats['image_counts'][class_id]
            percentage = (label_count / total_labels_split * 100) if total_labels_split > 0 else 0
            
            print(f"    Class {class_id} ({class_name}):")
            print(f"      - Số labels: {label_count:,} ({percentage:.2f}%)")
            print(f"      - Số images có class này: {image_count:,}")
        
        # Cộng vào tổng
        total_labels_all += stats['total_labels']
        for class_id, count in stats['class_counts'].items():
            total_class_counts_all[class_id] += count
        for class_id, count in stats['image_counts'].items():
            total_image_counts_all[class_id] += count
    
    # Tổng hợp
    print(f"\n{'=' * 70}")
    print("TỔNG HỢP TẤT CẢ SPLITS")
    print(f"{'=' * 70}")
    print(f"Tổng số labels: {total_labels_all:,}")
    print(f"\nPhân bố labels theo class (tổng hợp):")
    print(f"{'=' * 60}")
    
    for class_id in sorted(total_class_counts_all.keys()):
        class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
        label_count = total_class_counts_all[class_id]
        image_count = total_image_counts_all[class_id]
        percentage = (label_count / total_labels_all * 100) if total_labels_all > 0 else 0
        
        print(f"  Class {class_id} ({class_name}):")
        print(f"    - Tổng số labels: {label_count:,} ({percentage:.2f}%)")
        print(f"    - Tổng số images có class này: {image_count:,}")
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

