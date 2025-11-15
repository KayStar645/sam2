# copy_and_count_images.py
# Đọc data từ datasets-sam-vit, copy và đếm hình ảnh từng loại sang ks-nj4/data
# Chỉ copy những hình ảnh có labels (có annotations)
# python ks-nj4/copy_and_count_images.py

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ============================================================================
# CẤU HÌNH - THAY ĐỔI CÁC THAM SỐ Ở ĐÂY
# ============================================================================

# Đường dẫn thư mục dữ liệu
SOURCE_DIR = Path("data/datasets-sam-vit")  # Thư mục nguồn
OUTPUT_DIR = Path("ks-nj4/data/processed")      # Thư mục đích

# Các split cần xử lý
SPLITS = ['test', 'train', 'val']

# Mapping category ID sang tên
CATEGORY_NAMES = {
    1: "brown_planthopper",           # Rầy nâu
    2: "whitebacked_planthopper",     # Rầy lưng trắng
    3: "rice_leaf_miner",            # Sâu ăn lúa
}

# ============================================================================

def load_coco_json(json_file: Path) -> Dict:
    """Đọc file COCO JSON"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Lỗi khi đọc {json_file}: {e}")
        return None

def get_images_with_labels(coco_data: Dict) -> Tuple[Set[int], Dict[int, List[int]]]:
    """
    Lấy danh sách image_id có annotations và mapping category_id -> số lượng
    
    Returns:
        - images_with_labels: Set các image_id có annotations
        - category_counts: Dict {category_id: [image_ids]} - danh sách image_id cho mỗi category
    """
    images_with_labels = set()
    category_to_images = defaultdict(set)  # category_id -> set(image_ids)
    
    if 'annotations' not in coco_data:
        return images_with_labels, {}
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        images_with_labels.add(image_id)
        category_to_images[category_id].add(image_id)
    
    # Chuyển set thành list để dễ xử lý
    category_counts = {cat_id: list(img_ids) for cat_id, img_ids in category_to_images.items()}
    
    return images_with_labels, category_counts

def get_image_info_by_id(coco_data: Dict, image_id: int) -> Dict:
    """Lấy thông tin image từ image_id"""
    for img in coco_data.get('images', []):
        if img['id'] == image_id:
            return img
    return None

def copy_file_safe(source: Path, dest: Path) -> bool:
    """Copy file an toàn, tạo thư mục nếu cần"""
    try:
        if not source.exists():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return True
    except Exception as e:
        print(f"  ⚠️  Lỗi khi copy {source} -> {dest}: {e}")
        return False

def process_split(split: str, source_dir: Path, output_dir: Path) -> Dict:
    """
    Xử lý một split: copy images có labels và đếm
    
    Returns:
        Dict với thống kê: {
            'total_images': int,
            'images_with_labels': int,
            'images_copied': int,
            'category_counts': {category_id: count},
            'category_image_counts': {category_id: count_images}
        }
    """
    print(f"\n{'=' * 70}")
    print(f"Xử lý split: {split.upper()}")
    print(f"{'=' * 70}")
    
    # Đọc COCO JSON
    json_file = source_dir / "labels" / f"{split}.json"
    if not json_file.exists():
        print(f"⚠️  File {json_file} không tồn tại, bỏ qua...")
        return None
    
    print(f"Đọc {json_file}...")
    coco_data = load_coco_json(json_file)
    if coco_data is None:
        return None
    
    # Lấy danh sách images có labels
    images_with_labels, category_to_images = get_images_with_labels(coco_data)
    
    print(f"Tổng số images trong JSON: {len(coco_data.get('images', []))}")
    print(f"Số images có labels: {len(images_with_labels)}")
    
    # Đếm số lượng annotations theo category
    category_annotation_counts = defaultdict(int)
    category_image_counts = {}
    
    for ann in coco_data.get('annotations', []):
        category_id = ann['category_id']
        category_annotation_counts[category_id] += 1
    
    for category_id, image_ids in category_to_images.items():
        category_image_counts[category_id] = len(image_ids)
    
    # Tạo thư mục output
    output_images_dir = output_dir / "images" / split
    output_images_processed_dir = output_dir / "images_processed" / split
    output_masks_dir = output_dir / "masks" / split
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_images_processed_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images và masks
    images_copied = 0
    images_skipped = 0
    
    # Tạo mapping image_id -> image_info
    image_id_to_info = {}
    for img in coco_data.get('images', []):
        image_id_to_info[img['id']] = img
    
    # Copy từng image có labels
    for image_id in sorted(images_with_labels):
        img_info = image_id_to_info.get(image_id)
        if img_info is None:
            continue
        
        file_name = img_info['file_name']
        # file_name có format: "split/filename.jpg"
        # Lấy tên file không có split prefix
        if '/' in file_name:
            filename = file_name.split('/', 1)[1]
        else:
            filename = file_name
        
        # Đường dẫn source
        source_image = source_dir / "images" / split / filename
        source_image_processed = source_dir / "images_processed" / split / filename
        source_mask = source_dir / "masks" / split / f"{Path(filename).stem}.png"
        source_mask_vis = source_dir / "masks" / split / f"{Path(filename).stem}_vis.png"
        
        # Đường dẫn đích
        dest_image = output_images_dir / filename
        dest_image_processed = output_images_processed_dir / filename
        dest_mask = output_masks_dir / f"{Path(filename).stem}.png"
        dest_mask_vis = output_masks_dir / f"{Path(filename).stem}_vis.png"
        
        # Copy files
        success = True
        success &= copy_file_safe(source_image, dest_image)
        success &= copy_file_safe(source_image_processed, dest_image_processed)
        success &= copy_file_safe(source_mask, dest_mask)
        success &= copy_file_safe(source_mask_vis, dest_mask_vis)
        
        if success:
            images_copied += 1
        else:
            images_skipped += 1
            print(f"  ⚠️  Không copy được {filename}")
    
    # Tạo COCO JSON mới chỉ với images có labels
    new_images = [img for img in coco_data.get('images', []) if img['id'] in images_with_labels]
    new_annotations = [ann for ann in coco_data.get('annotations', []) if ann['image_id'] in images_with_labels]
    
    # Cập nhật image_id và annotation_id để liên tục (bắt đầu từ 1)
    image_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(images_with_labels), 1)}
    
    # Cập nhật image_id trong images
    for img in new_images:
        old_id = img['id']
        img['id'] = image_id_mapping[old_id]
    
    # Cập nhật image_id và annotation_id trong annotations
    new_annotation_id = 1
    for ann in new_annotations:
        old_image_id = ann['image_id']
        ann['image_id'] = image_id_mapping[old_image_id]
        ann['id'] = new_annotation_id
        new_annotation_id += 1
    
    # Lưu COCO JSON mới
    new_coco_data = {
        "info": coco_data.get('info', {}).copy(),
        "licenses": coco_data.get('licenses', []),
        "categories": coco_data.get('categories', []),
        "images": new_images,
        "annotations": new_annotations
    }
    
    # Cập nhật info
    new_coco_data['info']['description'] = f"Rice Planthopper Dataset for SAM+ViT (filtered - only images with labels) - {split}"
    
    output_labels_dir = output_dir / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_json_file = output_labels_dir / f"{split}.json"
    
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Đã lưu COCO JSON: {output_json_file}")
    print(f"  - Images: {len(new_images)}")
    print(f"  - Annotations: {len(new_annotations)}")
    
    # Thống kê
    stats = {
        'total_images': len(coco_data.get('images', [])),
        'images_with_labels': len(images_with_labels),
        'images_copied': images_copied,
        'images_skipped': images_skipped,
        'category_annotation_counts': dict(category_annotation_counts),
        'category_image_counts': category_image_counts,
        'total_annotations': len(new_annotations)
    }
    
    return stats

def print_statistics(all_stats: Dict[str, Dict]):
    """In thống kê tổng hợp"""
    print(f"\n{'=' * 70}")
    print("THỐNG KÊ TỔNG HỢP")
    print(f"{'=' * 70}")
    
    # Tổng hợp theo split
    total_images_all = 0
    total_images_with_labels = 0
    total_images_copied = 0
    total_annotations_all = 0
    total_category_annotations = defaultdict(int)
    total_category_images = defaultdict(int)  # Tổng số images có category đó (có thể trùng nếu image có nhiều categories)
    
    for split, stats in all_stats.items():
        if stats is None:
            continue
        
        total_images_all += stats['total_images']
        total_images_with_labels += stats['images_with_labels']
        total_images_copied += stats['images_copied']
        total_annotations_all += stats['total_annotations']
        
        for cat_id, count in stats['category_annotation_counts'].items():
            total_category_annotations[cat_id] += count
        
        for cat_id, count in stats['category_image_counts'].items():
            total_category_images[cat_id] += count  # Cộng số images từ mỗi split
    
    print(f"\n📊 TỔNG QUAN:")
    print(f"  Tổng số images trong dataset: {total_images_all}")
    print(f"  Số images có labels: {total_images_with_labels}")
    print(f"  Số images đã copy: {total_images_copied}")
    print(f"  Tổng số annotations: {total_annotations_all}")
    
    print(f"\n📊 THEO TỪNG SPLIT:")
    for split in SPLITS:
        if split not in all_stats or all_stats[split] is None:
            continue
        
        stats = all_stats[split]
        print(f"\n  {split.upper()}:")
        print(f"    - Tổng images: {stats['total_images']}")
        print(f"    - Images có labels: {stats['images_with_labels']}")
        print(f"    - Images đã copy: {stats['images_copied']}")
        print(f"    - Tổng annotations: {stats['total_annotations']}")
    
    print(f"\n📊 THEO TỪNG LOẠI (CATEGORY):")
    for category_id in sorted(CATEGORY_NAMES.keys()):
        cat_name = CATEGORY_NAMES[category_id]
        ann_count = total_category_annotations.get(category_id, 0)
        img_count = total_category_images.get(category_id, 0)
        
        print(f"\n  Category {category_id} - {cat_name}:")
        print(f"    - Số annotations: {ann_count}")
        print(f"    - Số images: {img_count}")
    
    print(f"\n{'=' * 70}")

def save_report_to_txt(all_stats: Dict[str, Dict], output_dir: Path):
    """Lưu báo cáo chi tiết vào file TXT"""
    from datetime import datetime
    
    report_file = output_dir / "report.txt"
    
    # Tổng hợp dữ liệu
    total_images_all = 0
    total_images_with_labels = 0
    total_images_no_labels = 0
    total_images_copied = 0
    total_images_skipped = 0
    total_annotations_all = 0
    total_category_annotations = defaultdict(int)
    total_category_images = defaultdict(int)
    
    # Thống kê theo split
    split_stats = {}
    
    for split, stats in all_stats.items():
        if stats is None:
            continue
        
        total_images_all += stats['total_images']
        total_images_with_labels += stats['images_with_labels']
        total_images_no_labels += (stats['total_images'] - stats['images_with_labels'])
        total_images_copied += stats['images_copied']
        total_images_skipped += stats.get('images_skipped', 0)
        total_annotations_all += stats['total_annotations']
        
        for cat_id, count in stats['category_annotation_counts'].items():
            total_category_annotations[cat_id] += count
        
        for cat_id, count in stats['category_image_counts'].items():
            total_category_images[cat_id] += count
        
        split_stats[split] = stats
    
    # Tạo nội dung báo cáo
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BÁO CÁO COPY VÀ ĐẾM HÌNH ẢNH TỪ DATASETS-SAM-VIT")
    report_lines.append("=" * 80)
    report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Thư mục nguồn: {SOURCE_DIR}")
    report_lines.append(f"Thư mục đích: {OUTPUT_DIR}")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("1. TỔNG QUAN")
    report_lines.append("=" * 80)
    report_lines.append(f"Tổng số hình ảnh trong dataset: {total_images_all:,}")
    report_lines.append(f"Số hình ảnh CÓ nhãn (có annotations): {total_images_with_labels:,}")
    report_lines.append(f"Số hình ảnh KHÔNG có nhãn: {total_images_no_labels:,}")
    report_lines.append(f"Số hình ảnh đã COPY thành công: {total_images_copied:,}")
    report_lines.append(f"Số hình ảnh BỎ QUA (không copy được): {total_images_skipped:,}")
    report_lines.append(f"Tổng số annotations: {total_annotations_all:,}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("2. THỐNG KÊ THEO TỪNG SPLIT")
    report_lines.append("=" * 80)
    for split in SPLITS:
        if split not in split_stats or split_stats[split] is None:
            continue
        
        stats = split_stats[split]
        images_no_labels = stats['total_images'] - stats['images_with_labels']
        
        report_lines.append(f"\n{split.upper()}:")
        report_lines.append(f"  - Tổng số hình ảnh: {stats['total_images']:,}")
        report_lines.append(f"  - Số hình ảnh CÓ nhãn: {stats['images_with_labels']:,}")
        report_lines.append(f"  - Số hình ảnh KHÔNG có nhãn: {images_no_labels:,}")
        report_lines.append(f"  - Số hình ảnh đã COPY: {stats['images_copied']:,}")
        report_lines.append(f"  - Số hình ảnh BỎ QUA: {stats.get('images_skipped', 0):,}")
        report_lines.append(f"  - Tổng số annotations: {stats['total_annotations']:,}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("3. THỐNG KÊ THEO TỪNG LOẠI RẦY (CATEGORY)")
    report_lines.append("=" * 80)
    for category_id in sorted(CATEGORY_NAMES.keys()):
        cat_name = CATEGORY_NAMES[category_id]
        ann_count = total_category_annotations.get(category_id, 0)
        img_count = total_category_images.get(category_id, 0)
        
        # Tính số hình ảnh có loại này trong từng split
        report_lines.append(f"\nCategory {category_id} - {cat_name}:")
        report_lines.append(f"  - Tổng số annotations: {ann_count:,}")
        report_lines.append(f"  - Tổng số hình ảnh có loại này: {img_count:,}")
        
        # Chi tiết theo split
        report_lines.append(f"  - Chi tiết theo split:")
        for split in SPLITS:
            if split not in split_stats or split_stats[split] is None:
                continue
            split_ann_count = split_stats[split]['category_annotation_counts'].get(category_id, 0)
            split_img_count = split_stats[split]['category_image_counts'].get(category_id, 0)
            if split_ann_count > 0 or split_img_count > 0:
                report_lines.append(f"    + {split.upper()}: {split_ann_count:,} annotations, {split_img_count:,} images")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("4. TỈ LỆ PHẦN TRĂM")
    report_lines.append("=" * 80)
    if total_images_all > 0:
        pct_with_labels = (total_images_with_labels / total_images_all) * 100
        pct_no_labels = (total_images_no_labels / total_images_all) * 100
        pct_copied = (total_images_copied / total_images_all) * 100
        
        report_lines.append(f"Tỉ lệ hình ảnh CÓ nhãn: {pct_with_labels:.2f}%")
        report_lines.append(f"Tỉ lệ hình ảnh KHÔNG có nhãn: {pct_no_labels:.2f}%")
        report_lines.append(f"Tỉ lệ hình ảnh đã COPY: {pct_copied:.2f}%")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("5. PHÂN BỐ ANNOTATIONS THEO CATEGORY")
    report_lines.append("=" * 80)
    if total_annotations_all > 0:
        for category_id in sorted(CATEGORY_NAMES.keys()):
            cat_name = CATEGORY_NAMES[category_id]
            ann_count = total_category_annotations.get(category_id, 0)
            pct = (ann_count / total_annotations_all) * 100
            report_lines.append(f"Category {category_id} - {cat_name}: {ann_count:,} annotations ({pct:.2f}%)")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("KẾT THÚC BÁO CÁO")
    report_lines.append("=" * 80)
    
    # Ghi file
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✓ Đã lưu báo cáo: {report_file}")
        return report_file
    except Exception as e:
        print(f"⚠️  Lỗi khi lưu báo cáo: {e}")
        return None

def main():
    """Hàm chính"""
    print("=" * 70)
    print("COPY VÀ ĐẾM HÌNH ẢNH TỪ DATASETS-SAM-VIT")
    print("=" * 70)
    print(f"Đọc từ: {SOURCE_DIR}")
    print(f"Ghi vào: {OUTPUT_DIR}")
    print(f"Splits: {', '.join(SPLITS)}")
    print("=" * 70)
    
    # Kiểm tra thư mục nguồn
    if not SOURCE_DIR.exists():
        print(f"⚠️  Thư mục nguồn {SOURCE_DIR} không tồn tại!")
        return
    
    # Tạo thư mục đích
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Xử lý từng split
    all_stats = {}
    
    for split in SPLITS:
        stats = process_split(split, SOURCE_DIR, OUTPUT_DIR)
        all_stats[split] = stats
    
    # In thống kê tổng hợp
    print_statistics(all_stats)
    
    # Lưu báo cáo TXT
    save_report_to_txt(all_stats, OUTPUT_DIR)
    
    # Tạo file annotations.json tổng hợp (nếu cần)
    print(f"\n{'=' * 70}")
    print("TẠO FILE ANNOTATIONS.JSON TỔNG HỢP")
    print(f"{'=' * 70}")
    
    all_images = []
    all_annotations = []
    all_categories = []
    image_id_offset = 0
    annotation_id_offset = 0
    
    for split in SPLITS:
        json_file = OUTPUT_DIR / "labels" / f"{split}.json"
        if not json_file.exists():
            continue
        
        coco_data = load_coco_json(json_file)
        if coco_data is None:
            continue
        
        # Lấy categories (chỉ lấy một lần)
        if not all_categories:
            all_categories = coco_data.get('categories', [])
        
        # Cập nhật image_id và annotation_id
        for img in coco_data.get('images', []):
            img['id'] = img['id'] + image_id_offset
            all_images.append(img)
        
        for ann in coco_data.get('annotations', []):
            ann['id'] = ann['id'] + annotation_id_offset
            ann['image_id'] = ann['image_id'] + image_id_offset
            all_annotations.append(ann)
        
        image_id_offset += len(coco_data.get('images', []))
        annotation_id_offset += len(coco_data.get('annotations', []))
    
    # Tạo file tổng hợp
    combined_coco_data = {
        "info": {
            "description": "Rice Planthopper Dataset for SAM+ViT (filtered - only images with labels)",
            "version": "1.0",
            "year": 2025,
            "contributor": "SAM+ViT Research",
            "date_created": "2025-01-01"
        },
        "licenses": [{
            "id": 0,
            "name": "Unknown",
            "url": ""
        }],
        "categories": all_categories,
        "images": all_images,
        "annotations": all_annotations
    }
    
    output_json_file = OUTPUT_DIR / "labels" / "annotations.json"
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Đã lưu COCO JSON tổng hợp: {output_json_file}")
    print(f"  - Images: {len(all_images)}")
    print(f"  - Annotations: {len(all_annotations)}")
    print(f"  - Categories: {len(all_categories)}")
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

