# analyze_data_sam_vit.py
# Đọc và phân tích dữ liệu từ ks-nj4/data/datasets-sam-vit, báo cáo số lượng từng loại và nhãn
# python ks-nj4/analyze_data_sam_vit.py

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI
import cv2
import numpy as np

# ============================================================================
# CẤU HÌNH - THAY ĐỔI CÁC THAM SỐ Ở ĐÂY
# ============================================================================

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path("ks-nj4/data/datasets-sam-vit")  # Thư mục dữ liệu cần phân tích

# Các split cần xử lý
SPLITS = ['test', 'train', 'val']

# Mapping category ID sang tên (mapping cố định)
# YOLO class 0 → whitebacked_planthopper (Rầy lưng trắng)
# YOLO class 1 → rice_leaf_miner (Sâu ăn lá lúa)
# YOLO class 2 → brown_planthopper (Rầy nâu)
CATEGORY_NAMES = {
    1: "brown_planthopper",           # COCO category 1 - Rầy nâu (BPH) - YOLO class 2
    2: "whitebacked_planthopper",     # COCO category 2 - Rầy lưng trắng (WBPH) - YOLO class 0
    3: "rice_leaf_miner",            # COCO category 3 - Sâu ăn lá lúa (RLM) - YOLO class 1
}

# ============================================================================

def count_category_occurrences(all_stats: Dict[str, Dict]) -> Dict[int, int]:
    """
    Đếm số lần xuất hiện của mỗi category_id trong toàn bộ dataset
    
    Returns:
        Dict {category_id: số_lần_xuất_hiện}
    """
    category_counts = defaultdict(int)
    
    for split, stats in all_stats.items():
        if stats is None:
            continue
        
        for cat_id, count in stats['category_annotations'].items():
            category_counts[cat_id] += count
    
    return dict(category_counts)

def determine_category_mapping(category_counts: Dict[int, int]) -> Dict[int, str]:
    """
    Xác định mapping category_id sang tên (mapping cố định, không tự động)
    
    Mapping cố định:
    - COCO category 1 → brown_planthopper (Rầy nâu - BPH) - YOLO class 2
    - COCO category 2 → whitebacked_planthopper (Rầy lưng trắng - WBPH) - YOLO class 0
    - COCO category 3 → rice_leaf_miner (Sâu ăn lá lúa - RLM) - YOLO class 1
    
    Returns:
        Dict {category_id: tên_category}
    """
    # Sử dụng mapping cố định từ CATEGORY_NAMES
    mapping = {}
    for cat_id in category_counts.keys():
        mapping[cat_id] = CATEGORY_NAMES.get(cat_id, f"category_{cat_id}")
    
    return mapping

# ============================================================================


def analyze_split_from_images_and_masks(split: str, data_dir: Path) -> Dict:
    """
    Phân tích một split từ images và masks PNG (đếm annotations từ masks bằng connected components)
    """
    print(f"\n{'=' * 70}")
    print(f"Phân tích split: {split.upper()} (từ images và masks)")
    print(f"{'=' * 70}")
    
    # Đường dẫn thư mục
    images_dir = data_dir / "images" / split
    masks_dir = data_dir / "masks" / split
    
    if not images_dir.exists():
        print(f"⚠️  Thư mục {images_dir} không tồn tại, bỏ qua...")
        return None
    
    if not masks_dir.exists():
        print(f"⚠️  Thư mục {masks_dir} không tồn tại, bỏ qua...")
        return None
    
    # Đếm files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
                  list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
    # Loại bỏ file _vis.png
    mask_files = [f for f in masks_dir.glob("*.png") if not f.stem.endswith("_vis")]
    
    # Tạo set tên file (không có extension) để matching
    image_names = {f.stem for f in image_files}
    mask_names = {f.stem for f in mask_files}
    
    # Đếm từ masks (connected components)
    category_annotations = defaultdict(int)  # category_id -> số annotations
    category_images = defaultdict(set)  # category_id -> set(image_names)
    image_to_categories = defaultdict(set)  # image_name -> set(category_ids)
    images_with_masks = set()  # Images có mask tương ứng
    images_without_masks = set()  # Images không có mask
    total_annotations = 0
    
    # Phân tích từ masks
    for img_file in image_files:
        image_name = img_file.stem
        mask_file = masks_dir / f"{image_name}.png"
        
        if mask_file.exists() and mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                images_with_masks.add(image_name)
                unique_classes = np.unique(mask)
                
                for cls in unique_classes:
                    if cls > 0:  # Bỏ background
                        # Đếm số annotations (connected components) cho class này
                        class_mask = (mask == cls).astype(np.uint8)
                        num_labels, _ = cv2.connectedComponents(class_mask)
                        num_objects = num_labels - 1  # Trừ background label
                        
                        if num_objects > 0:
                            category_id = int(cls)  # Mask value = COCO category_id
                            category_annotations[category_id] += num_objects
                            category_images[category_id].add(image_name)
                            image_to_categories[image_name].add(category_id)
                            total_annotations += num_objects
            else:
                images_without_masks.add(image_name)
        else:
            images_without_masks.add(image_name)
    
    # Category mapping
    category_id_to_name = {
        1: "brown_planthopper",
        2: "whitebacked_planthopper",
        3: "rice_leaf_miner"
    }
    
    total_images = len(image_files)
    images_with_multiple_categories = sum(1 for cats in image_to_categories.values() if len(cats) > 1)
    
    stats = {
        'split': split,
        'total_images': total_images,
        'total_masks': len(mask_files),
        'images_with_masks': len(images_with_masks),
        'images_without_masks': len(images_without_masks),
        'total_annotations': total_annotations,
        'category_annotations': dict(category_annotations),
        'category_images': {cat_id: len(img_set) for cat_id, img_set in category_images.items()},
        'images_with_multiple_categories': images_with_multiple_categories,
        'category_id_to_name': category_id_to_name,
    }
    
    print(f"  - Tổng số hình ảnh: {stats['total_images']:,}")
    print(f"  - Tổng số masks: {stats['total_masks']:,}")
    print(f"  - Hình ảnh có mask: {stats['images_with_masks']:,}")
    print(f"  - Hình ảnh không có mask: {stats['images_without_masks']:,}")
    print(f"  - Tổng annotations: {stats['total_annotations']:,}")
    print(f"  - Số category_ids tìm thấy: {len(category_annotations)}")
    for category_id in sorted(category_annotations.keys()):
        cat_name = category_id_to_name.get(category_id, f"category_{category_id}")
        print(f"    Category {category_id} ({cat_name}): {category_annotations[category_id]:,} annotations, {len(category_images[category_id]):,} images")
    
    return stats

def analyze_split(split: str, data_dir: Path) -> Dict:
    """
    Phân tích một split từ images và masks
    
    Returns:
        Dict với thống kê chi tiết
    """
    # Chỉ đọc từ images và masks
    return analyze_split_from_images_and_masks(split, data_dir)

def generate_report(all_stats: Dict[str, Dict], data_dir: Path):
    """Tạo báo cáo chi tiết và lưu vào file TXT"""
    from datetime import datetime
    
    report_file = data_dir / "analysis_report.txt"
    
    # Tổng hợp dữ liệu
    total_images = 0
    total_masks = 0
    total_images_with_masks = 0
    total_images_without_masks = 0
    total_annotations = 0
    total_category_annotations = defaultdict(int)
    total_category_images = defaultdict(int)
    total_images_multiple_categories = 0
    
    # Thống kê theo split
    split_stats = {}
    
    # Lấy category mapping từ stats đầu tiên
    category_id_to_name = {}
    for stats in all_stats.values():
        if stats and 'category_id_to_name' in stats:
            category_id_to_name.update(stats['category_id_to_name'])
            break
    
    for split, stats in all_stats.items():
        if stats is None:
            continue
        
        # Chỉ xử lý từ images/masks
        total_images += stats['total_images']
        total_masks += stats.get('total_masks', 0)
        total_images_with_masks += stats.get('images_with_masks', 0)
        total_images_without_masks += stats.get('images_without_masks', 0)
        
        total_annotations += stats['total_annotations']
        total_images_multiple_categories += stats.get('images_with_multiple_categories', 0)
        
        for cat_id, count in stats['category_annotations'].items():
            total_category_annotations[cat_id] += count
        
        for cat_id, count in stats['category_images'].items():
            total_category_images[cat_id] += count  # Cộng số images từ mỗi split (có thể trùng nếu image có nhiều categories)
        
        split_stats[split] = stats
    
    # Đếm số lần xuất hiện và xác định mapping tự động
    category_counts = count_category_occurrences(all_stats)
    category_mapping = determine_category_mapping(category_counts)
    
    # Tạo nội dung báo cáo
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BÁO CÁO PHÂN TÍCH DỮ LIỆU TỪ KS-NJ4/DATA/DATASETS-SAM-VIT")
    report_lines.append("=" * 80)
    report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Thư mục dữ liệu: {DATA_DIR}")
    report_lines.append("")
    
    # Hiển thị mapping cố định
    report_lines.append("=" * 80)
    report_lines.append("MAPPING CỐ ĐỊNH")
    report_lines.append("=" * 80)
    report_lines.append("Số lần xuất hiện của mỗi category_id (COCO) trong toàn bộ dataset:")
    if len(category_counts) == 0:
        report_lines.append("  Không tìm thấy dữ liệu!")
    else:
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for cat_id, count in sorted_counts:
            category_name = category_mapping.get(cat_id, category_id_to_name.get(cat_id, f"category_{cat_id}"))
            # Mapping cố định: category 1 = YOLO 2 (Rầy nâu), category 2 = YOLO 0 (Lưng trắng), category 3 = YOLO 1 (Sâu ăn lúa)
            if cat_id == 1:
                yolo_info = "YOLO class 2 → Rầy nâu (BPH)"
            elif cat_id == 2:
                yolo_info = "YOLO class 0 → Rầy lưng trắng (WBPH)"
            elif cat_id == 3:
                yolo_info = "YOLO class 1 → Sâu ăn lá lúa (RLM)"
            else:
                yolo_info = f"Category {cat_id}"
            report_lines.append(f"  Category {cat_id} (COCO): {count:,} lần xuất hiện - {yolo_info} - {category_name}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("1. TỔNG QUAN")
    report_lines.append("=" * 80)
    report_lines.append(f"📊 TỔNG SỐ HÌNH ẢNH: {total_images:,}")
    report_lines.append(f"  - Hình ảnh có mask: {total_images_with_masks:,}")
    report_lines.append(f"  - Hình ảnh không có mask: {total_images_without_masks:,}")
    report_lines.append(f"  - Số hình ảnh có nhiều loại rầy: {total_images_multiple_categories:,}")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ MASKS: {total_masks:,}")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ ANNOTATIONS: {total_annotations:,}")
    report_lines.append("")
    
    # Tỉ lệ
    if total_images > 0:
        pct_with_masks = (total_images_with_masks / total_images) * 100
        pct_without_masks = (total_images_without_masks / total_images) * 100
        report_lines.append("Tỉ lệ hình ảnh:")
        report_lines.append(f"  - Có mask: {pct_with_masks:.2f}%")
        report_lines.append(f"  - Không có mask: {pct_without_masks:.2f}%")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("2. THỐNG KÊ THEO TỪNG SPLIT")
    report_lines.append("=" * 80)
    for split in SPLITS:
        if split not in split_stats or split_stats[split] is None:
            continue
        
        stats = split_stats[split]
        report_lines.append(f"\n{split.upper()}:")
        report_lines.append(f"  - Tổng số hình ảnh: {stats['total_images']:,}")
        report_lines.append(f"  - Tổng số masks: {stats.get('total_masks', 0):,}")
        report_lines.append(f"  - Hình ảnh có mask: {stats.get('images_with_masks', 0):,}")
        report_lines.append(f"  - Hình ảnh không có mask: {stats.get('images_without_masks', 0):,}")
        report_lines.append(f"  - Tổng số annotations: {stats['total_annotations']:,}")
        
        # Hiển thị số hình ảnh cho từng loại nhãn trong split này
        report_lines.append(f"  - Số hình ảnh theo từng loại nhãn:")
        for category_id in sorted(stats['category_images'].keys()):
            cat_name = category_mapping.get(category_id, category_id_to_name.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}")))
            img_count = stats['category_images'].get(category_id, 0)
            if img_count > 0:
                report_lines.append(f"    + Category {category_id} ({cat_name}): {img_count:,} images")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("3. THỐNG KÊ THEO TỪNG LOẠI RẦY (CATEGORY)")
    report_lines.append("=" * 80)
    # Sử dụng category_mapping, nếu không có thì dùng category_id có trong dữ liệu
    all_category_ids = set(total_category_annotations.keys())
    for category_id in sorted(all_category_ids):
        cat_name = category_mapping.get(category_id, category_id_to_name.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}")))
        ann_count = total_category_annotations.get(category_id, 0)
        img_count = total_category_images.get(category_id, 0)
        
        report_lines.append(f"\nCategory {category_id} (COCO) - {cat_name}:")
        report_lines.append(f"  - Tổng số annotations: {ann_count:,}")
        report_lines.append(f"  - Tổng số hình ảnh có loại này: {img_count:,}")
        
        if total_annotations > 0:
            pct_ann = (ann_count / total_annotations) * 100
            report_lines.append(f"  - Tỉ lệ annotations: {pct_ann:.2f}%")
        
        # Chi tiết theo split
        report_lines.append(f"  - Chi tiết theo split:")
        for split in SPLITS:
            if split not in split_stats or split_stats[split] is None:
                continue
            split_ann_count = split_stats[split]['category_annotations'].get(category_id, 0)
            split_img_count = split_stats[split]['category_images'].get(category_id, 0)
            if split_ann_count > 0 or split_img_count > 0:
                report_lines.append(f"    + {split.upper()}: {split_ann_count:,} annotations, {split_img_count:,} images")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("4. PHÂN BỐ ANNOTATIONS THEO CATEGORY")
    report_lines.append("=" * 80)
    if total_annotations > 0:
        for category_id in sorted(all_category_ids):
            cat_name = category_mapping.get(category_id, category_id_to_name.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}")))
            ann_count = total_category_annotations.get(category_id, 0)
            pct = (ann_count / total_annotations) * 100
            report_lines.append(f"Category {category_id} (COCO) - {cat_name}: {ann_count:,} annotations ({pct:.2f}%)")
    report_lines.append("")
    
    # Tính lại tỉ lệ cho phần tóm tắt
    pct_with_masks_final = (total_images_with_masks / total_images * 100) if total_images > 0 else 0
    pct_without_masks_final = (total_images_without_masks / total_images * 100) if total_images > 0 else 0
    
    report_lines.append("=" * 80)
    report_lines.append("5. TÓM TẮT")
    report_lines.append("=" * 80)
    report_lines.append(f"📊 TỔNG SỐ HÌNH ẢNH: {total_images:,}")
    report_lines.append(f"  - Có mask: {total_images_with_masks:,} ({pct_with_masks_final:.2f}%)")
    report_lines.append(f"  - Không có mask: {total_images_without_masks:,} ({pct_without_masks_final:.2f}%)")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ MASKS: {total_masks:,}")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ ANNOTATIONS: {total_annotations:,}")
    report_lines.append("")
    report_lines.append("Số lượng annotations theo từng loại:")
    for category_id in sorted(all_category_ids):
        cat_name = category_mapping.get(category_id, category_id_to_name.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}")))
        ann_count = total_category_annotations.get(category_id, 0)
        img_count = total_category_images.get(category_id, 0)
        if total_annotations > 0:
            pct = (ann_count / total_annotations) * 100
            report_lines.append(f"  - {cat_name}: {ann_count:,} annotations ({pct:.2f}%), {img_count:,} images")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("KẾT THÚC BÁO CÁO")
    report_lines.append("=" * 80)
    
    # Ghi file
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✓ Đã lưu báo cáo phân tích: {report_file}")
        return report_file
    except Exception as e:
        print(f"⚠️  Lỗi khi lưu báo cáo: {e}")
        return None

def create_charts(all_stats: Dict[str, Dict], category_mapping: Dict[int, str], 
                  total_images_with_masks: int, total_images_without_masks: int,
                  total_annotations: int, total_category_annotations: Dict[int, int],
                  data_dir: Path, category_id_to_name: Dict[int, str]):
    """Tạo các biểu đồ báo cáo"""
    try:
        # Thiết lập font để hiển thị tiếng Việt (nếu có)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Tạo figure với nhiều subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Biểu đồ số lượng annotations theo từng category (Bar chart)
        ax1 = plt.subplot(2, 2, 1)
        sorted_categories = sorted(total_category_annotations.items(), key=lambda x: x[1], reverse=True)
        category_ids = [cat_id for cat_id, _ in sorted_categories]
        ann_counts = [count for _, count in sorted_categories]
        category_names = [category_mapping.get(cat_id, category_id_to_name.get(cat_id, f"Category {cat_id}")) for cat_id in category_ids]
        
        bars = ax1.bar(range(len(category_ids)), ann_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_xlabel('Category (COCO)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Số lượng Annotations', fontsize=12, fontweight='bold')
        ax1.set_title('Số lượng Annotations theo từng Category', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(category_ids)))
        ax1.set_xticklabels([f"Category {cid}\n({name})" for cid, name in zip(category_ids, category_names)], 
                           rotation=0, ha='center', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị lên mỗi cột
        for i, (bar, count) in enumerate(zip(bars, ann_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Pie chart phân bố annotations theo category
        ax2 = plt.subplot(2, 2, 2)
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        labels_pie = [f"Category {cid}\n{category_mapping.get(cid, category_id_to_name.get(cid, f'Category {cid}'))}" 
                     for cid in category_ids]
        sizes_pie = ann_counts
        
        wedges, texts, autotexts = ax2.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%',
                                           colors=colors_pie[:len(sizes_pie)],
                                           startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Phân bố Annotations theo Category (%)', fontsize=14, fontweight='bold')
        
        # 3. Biểu đồ số lượng hình ảnh có/không có mask
        ax3 = plt.subplot(2, 2, 3)
        labels_img = ['Có mask', 'Không có mask']
        values_img = [total_images_with_masks, total_images_without_masks]
        colors_img = ['#4ECDC4', '#FF6B6B']
        
        bars2 = ax3.bar(labels_img, values_img, color=colors_img)
        ax3.set_ylabel('Số lượng Hình ảnh', fontsize=12, fontweight='bold')
        ax3.set_title('Số lượng Hình ảnh Có/Không có Mask', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị và phần trăm
        total_images = total_images_with_masks + total_images_without_masks
        for bar, val in zip(bars2, values_img):
            height = bar.get_height()
            pct = (val / total_images * 100) if total_images > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Biểu đồ số lượng annotations theo split
        ax4 = plt.subplot(2, 2, 4)
        splits = []
        ann_counts_by_split = []
        for split in SPLITS:
            if split in all_stats and all_stats[split] is not None:
                splits.append(split.upper())
                ann_counts_by_split.append(all_stats[split]['total_annotations'])
        
        bars3 = ax4.bar(splits, ann_counts_by_split, color=['#95E1D3', '#F38181', '#AA96DA'])
        ax4.set_xlabel('Split', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Số lượng Annotations', fontsize=12, fontweight='bold')
        ax4.set_title('Số lượng Annotations theo Split', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị lên mỗi cột
        for bar, count in zip(bars3, ann_counts_by_split):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_file = data_dir / "analysis_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Đã lưu biểu đồ báo cáo: {chart_file}")
        return chart_file
        
    except Exception as e:
        print(f"⚠️  Lỗi khi tạo biểu đồ: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary(all_stats: Dict[str, Dict]):
    """In tóm tắt ra console"""
    print(f"\n{'=' * 70}")
    print("TÓM TẮT PHÂN TÍCH")
    print(f"{'=' * 70}")
    
    # Tính tổng từ stats
    total_images = 0
    total_masks = 0
    total_images_with_masks = 0
    total_annotations = 0
    
    for stats in all_stats.values():
        if stats:
            if 'total_images' in stats:
                total_images += stats['total_images']
                total_masks += stats.get('total_masks', 0)
                total_images_with_masks += stats.get('images_with_masks', 0)
            total_annotations += stats.get('total_annotations', 0)
    
    total_category_annotations = defaultdict(int)
    for stats in all_stats.values():
        if stats:
            for cat_id, count in stats['category_annotations'].items():
                total_category_annotations[cat_id] += count
    
    # Lấy category_id_to_name từ stats đầu tiên
    category_id_to_name = {}
    for stats in all_stats.values():
        if stats and 'category_id_to_name' in stats:
            category_id_to_name.update(stats['category_id_to_name'])
            break
    
    # Xác định mapping tự động
    category_counts = count_category_occurrences(all_stats)
    category_mapping = determine_category_mapping(category_counts)
    
    print(f"\n📊 TỔNG QUAN:")
    print(f"  Tổng số hình ảnh: {total_images:,}")
    print(f"  Tổng số masks: {total_masks:,}")
    print(f"  Hình ảnh có mask: {total_images_with_masks:,}")
    print(f"  Tổng số annotations: {total_annotations:,}")
    
    print(f"\n📊 MAPPING CỐ ĐỊNH:")
    sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, (cat_id, count) in enumerate(sorted_counts, 1):
        cat_name = category_mapping.get(cat_id, category_id_to_name.get(cat_id, f"category_{cat_id}"))
        # Mapping cố định: category 1 = YOLO 2 (Rầy nâu), category 2 = YOLO 0 (Lưng trắng), category 3 = YOLO 1 (Sâu ăn lúa)
        if cat_id == 1:
            yolo_info = "YOLO class 2 → Rầy nâu (BPH)"
        elif cat_id == 2:
            yolo_info = "YOLO class 0 → Rầy lưng trắng (WBPH)"
        elif cat_id == 3:
            yolo_info = "YOLO class 1 → Sâu ăn lá lúa (RLM)"
        else:
            yolo_info = f"Category {cat_id}"
        print(f"  Category {cat_id} (COCO) ({cat_name}): {count:,} lần - {yolo_info}")
    
    print(f"\n📊 THEO TỪNG LOẠI:")
    for category_id in sorted(total_category_annotations.keys()):
        cat_name = category_mapping.get(category_id, category_id_to_name.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}")))
        ann_count = total_category_annotations.get(category_id, 0)
        if total_annotations > 0:
            pct = (ann_count / total_annotations) * 100
            print(f"  Category {category_id} (COCO) - {cat_name}: {ann_count:,} annotations ({pct:.2f}%)")
    
    print(f"\n{'=' * 70}")

def main():
    """Hàm chính"""
    print("=" * 70)
    print("PHÂN TÍCH DỮ LIỆU TỪ KS-NJ4/DATA/DATASETS-SAM-VIT")
    print("=" * 70)
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(f"Splits: {', '.join(SPLITS)}")
    print("=" * 70)
    
    # Kiểm tra thư mục dữ liệu
    if not DATA_DIR.exists():
        print(f"⚠️  Thư mục dữ liệu {DATA_DIR} không tồn tại!")
        return
    
    # Phân tích từng split
    all_stats = {}
    
    for split in SPLITS:
        stats = analyze_split(split, DATA_DIR)
        all_stats[split] = stats
    
    # In tóm tắt
    print_summary(all_stats)
    
    # Tạo báo cáo chi tiết
    report_file = generate_report(all_stats, DATA_DIR)
    
    # Tạo biểu đồ báo cáo
    if report_file:
        # Tính toán lại các giá trị cần thiết cho biểu đồ
        total_images = 0
        total_images_with_masks = 0
        total_images_without_masks = 0
        total_annotations = 0
        
        for stats in all_stats.values():
            if stats:
                if 'total_images' in stats:
                    total_images += stats['total_images']
                    total_images_with_masks += stats.get('images_with_masks', 0)
                    total_images_without_masks += stats.get('images_without_masks', 0)
                total_annotations += stats.get('total_annotations', 0)
        
        total_category_annotations = defaultdict(int)
        category_id_to_name = {}
        for stats in all_stats.values():
            if stats:
                for cat_id, count in stats['category_annotations'].items():
                    total_category_annotations[cat_id] += count
                if 'category_id_to_name' in stats:
                    category_id_to_name.update(stats['category_id_to_name'])
        
        category_counts = count_category_occurrences(all_stats)
        category_mapping = determine_category_mapping(category_counts)
        
        create_charts(all_stats, category_mapping, total_images_with_masks, 
                     total_images_without_masks, total_annotations, 
                     dict(total_category_annotations), DATA_DIR, category_id_to_name)
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")
    print(f"Báo cáo đã được lưu tại: {DATA_DIR / 'analysis_report.txt'}")
    print(f"Biểu đồ đã được lưu tại: {DATA_DIR / 'analysis_charts.png'}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

