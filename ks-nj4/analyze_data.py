# analyze_data.py
# Đọc và phân tích dữ liệu từ ks-nj4/data, báo cáo số lượng từng loại và nhãn
# python ks-nj4/analyze_data.py

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không cần GUI

# ============================================================================
# CẤU HÌNH - THAY ĐỔI CÁC THAM SỐ Ở ĐÂY
# ============================================================================

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path("ks-nj4/data/datasets")  # Thư mục dữ liệu cần phân tích

# Các split cần xử lý
SPLITS = ['test', 'train', 'val']

# Mapping category ID sang tên (sẽ được tự động xác định dựa trên số lần xuất hiện)
CATEGORY_NAMES = {
    1: "brown_planthopper",           # Rầy nâu (BPH) - Class nhiều nhất
    2: "whitebacked_planthopper",     # Rầy lưng trắng (WBPH) - Class thứ 2
    3: "rice_leaf_miner",            # Sâu ăn lá lúa (RLM) - Class ít nhất
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
    Tự động xác định mapping category_id sang tên dựa trên số lần xuất hiện
    
    Logic:
    - Category xuất hiện nhiều nhất → brown_planthopper (Rầy nâu - BPH)
    - Category xuất hiện thứ 2 → whitebacked_planthopper (Rầy lưng trắng - WBPH)
    - Category xuất hiện ít nhất → rice_leaf_miner (Sâu ăn lá lúa - RLM)
    
    Returns:
        Dict {category_id: tên_category}
    """
    if len(category_counts) == 0:
        return {}
    
    # Sắp xếp theo số lần xuất hiện (giảm dần)
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Mapping theo thứ tự
    mapping = {}
    expected_names = [
        "brown_planthopper",        # Nhiều nhất
        "whitebacked_planthopper",  # Thứ 2
        "rice_leaf_miner"           # Ít nhất
    ]
    
    for idx, (cat_id, count) in enumerate(sorted_categories):
        if idx < len(expected_names):
            mapping[cat_id] = expected_names[idx]
        else:
            # Nếu có nhiều hơn 3 categories, giữ nguyên tên từ CATEGORY_NAMES
            mapping[cat_id] = CATEGORY_NAMES.get(cat_id, f"category_{cat_id}")
    
    return mapping

# ============================================================================

def read_yolo_label_file(txt_file: Path) -> List[List[float]]:
    """Đọc file YOLO label (.txt) và trả về danh sách annotations"""
    labels = []
    try:
        if txt_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 5:
                            labels.append(coords[:5])  # class_id, x_center, y_center, width, height
    except Exception as e:
        print(f"  ⚠️  Lỗi khi đọc {txt_file}: {e}")
    return labels

def count_files_in_directory(directory: Path, extensions: List[str]) -> int:
    """Đếm số file trong thư mục với các extension cho trước"""
    if not directory.exists():
        return 0
    
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f"*.{ext}"))) + len(list(directory.glob(f"*.{ext.upper()}")))
    
    return count

def analyze_split(split: str, data_dir: Path) -> Dict:
    """
    Phân tích một split từ YOLO format
    
    Returns:
        Dict với thống kê chi tiết
    """
    print(f"\n{'=' * 70}")
    print(f"Phân tích split: {split.upper()}")
    print(f"{'=' * 70}")
    
    # Đếm images thực tế trong thư mục
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split
    
    if not images_dir.exists():
        print(f"⚠️  Thư mục {images_dir} không tồn tại, bỏ qua...")
        return None
    
    if not labels_dir.exists():
        print(f"⚠️  Thư mục {labels_dir} không tồn tại, bỏ qua...")
        return None
    
    # Đếm images thực tế
    image_count_actual = count_files_in_directory(images_dir, ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'])
    
    # Tìm tất cả file ảnh (bao gồm cả .jpeg, .JPEG)
    image_files = (list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) +
                   list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPEG")) +
                   list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG")))
    
    # Loại bỏ trùng lặp (nếu có)
    image_files = list(set(image_files))
    
    # Đếm số lần xuất hiện của mỗi class_id
    class_annotations = defaultdict(int)  # class_id (YOLO) -> số annotations
    class_images = defaultdict(set)  # class_id -> set(image_names)
    images_with_labels = set()
    images_without_labels = set()
    images_processed = set()  # Để đảm bảo mỗi ảnh chỉ được đếm một lần
    image_to_classes = defaultdict(set)  # image_name -> set(class_ids)
    total_annotations = 0
    
    # Đọc từng file label
    for image_file in image_files:
        image_name = image_file.stem
        # Tránh đếm trùng nếu có file cùng tên với extension khác
        if image_name in images_processed:
            continue
        images_processed.add(image_name)
        
        label_file = labels_dir / f"{image_name}.txt"
        
        if label_file.exists():
            labels = read_yolo_label_file(label_file)
            if labels:
                images_with_labels.add(image_name)
                for label in labels:
                    class_id = int(label[0])  # YOLO class_id
                    class_annotations[class_id] += 1
                    class_images[class_id].add(image_name)
                    image_to_classes[image_name].add(class_id)
                    total_annotations += 1
            else:
                images_without_labels.add(image_name)
        else:
            images_without_labels.add(image_name)
    
    # Đảm bảo tổng số images = có nhãn + không có nhãn
    total_images_counted = len(images_with_labels) + len(images_without_labels)
    
    # Thống kê
    stats = {
        'split': split,
        'total_images_json': total_images_counted,  # Tổng số images đã xử lý (có nhãn + không có nhãn)
        'total_images_actual': image_count_actual,
        'images_with_labels': len(images_with_labels),
        'images_without_labels': len(images_without_labels),
        'total_annotations': total_annotations,
        'category_annotations': dict(class_annotations),  # Lưu dưới tên category_annotations để tương thích
        'category_images': {cat_id: len(img_set) for cat_id, img_set in class_images.items()},
        'images_with_multiple_categories': sum(1 for classes in image_to_classes.values() if len(classes) > 1),
    }
    
    # Kiểm tra tính nhất quán
    if total_images_counted != len(images_with_labels) + len(images_without_labels):
        print(f"  ⚠️  Warning: Tổng số images không khớp! Đã xử lý: {total_images_counted}, Có nhãn: {len(images_with_labels)}, Không có nhãn: {len(images_without_labels)}")
    
    print(f"  - Tổng images đã xử lý: {stats['total_images_json']}")
    print(f"  - Tổng images thực tế trong thư mục: {stats['total_images_actual']}")
    print(f"  - Images có nhãn: {stats['images_with_labels']}")
    print(f"  - Images không có nhãn: {stats['images_without_labels']}")
    print(f"  - Tổng annotations: {stats['total_annotations']}")
    print(f"  - Số class_ids tìm thấy: {len(class_annotations)}")
    for class_id in sorted(class_annotations.keys()):
        print(f"    Class {class_id}: {class_annotations[class_id]:,} annotations, {len(class_images[class_id]):,} images")
    
    return stats

def generate_report(all_stats: Dict[str, Dict], data_dir: Path):
    """Tạo báo cáo chi tiết và lưu vào file TXT"""
    from datetime import datetime
    
    report_file = data_dir / "analysis_report.txt"
    
    # Tổng hợp dữ liệu
    total_images_json = 0
    total_images_actual = 0
    total_images_with_labels = 0
    total_images_without_labels = 0
    total_annotations = 0
    total_category_annotations = defaultdict(int)
    total_category_images = defaultdict(int)
    total_images_multiple_categories = 0
    
    # Thống kê theo split
    split_stats = {}
    
    for split, stats in all_stats.items():
        if stats is None:
            continue
        
        total_images_json += stats['total_images_json']
        total_images_actual += stats['total_images_actual']
        total_images_with_labels += stats['images_with_labels']
        total_images_without_labels += stats['images_without_labels']
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
    report_lines.append("BÁO CÁO PHÂN TÍCH DỮ LIỆU TỪ KS-NJ4/DATA")
    report_lines.append("=" * 80)
    report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Thư mục dữ liệu: {DATA_DIR}")
    report_lines.append("")
    
    # Hiển thị mapping tự động
    report_lines.append("=" * 80)
    report_lines.append("MAPPING TỰ ĐỘNG DỰA TRÊN SỐ LẦN XUẤT HIỆN")
    report_lines.append("=" * 80)
    report_lines.append("Số lần xuất hiện của mỗi class_id (YOLO) trong toàn bộ dataset:")
    if len(category_counts) == 0:
        report_lines.append("  Không tìm thấy dữ liệu!")
    else:
        sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for cat_id, count in sorted_counts:
            category_name = category_mapping.get(cat_id, f"category_{cat_id}")
            rank = sorted_counts.index((cat_id, count)) + 1
            if rank == 1:
                rank_text = "NHIỀU NHẤT → Rầy nâu (BPH)"
            elif rank == 2:
                rank_text = "THỨ 2 → Rầy lưng trắng (WBPH)"
            elif rank == 3:
                rank_text = "ÍT NHẤT → Sâu ăn lá lúa (RLM)"
            else:
                rank_text = f"XẾP HẠNG {rank}"
            report_lines.append(f"  Class {cat_id} (YOLO): {count:,} lần xuất hiện - {rank_text} - {category_name}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("1. TỔNG QUAN")
    report_lines.append("=" * 80)
    report_lines.append(f"📊 TỔNG SỐ HÌNH ẢNH: {total_images_json:,}")
    report_lines.append(f"  - Số hình ảnh CÓ nhãn (có annotations): {total_images_with_labels:,}")
    report_lines.append(f"  - Số hình ảnh KHÔNG có nhãn: {total_images_without_labels:,}")
    report_lines.append(f"  - Số hình ảnh có nhiều loại rầy: {total_images_multiple_categories:,}")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ ANNOTATIONS: {total_annotations:,}")
    report_lines.append("")
    report_lines.append(f"📁 Tổng số file ảnh thực tế trong thư mục: {total_images_actual:,}")
    report_lines.append("")
    
    # Tỉ lệ
    if total_images_json > 0:
        pct_with_labels = (total_images_with_labels / total_images_json) * 100
        pct_without_labels = (total_images_without_labels / total_images_json) * 100
        report_lines.append("Tỉ lệ hình ảnh:")
        report_lines.append(f"  - CÓ nhãn: {pct_with_labels:.2f}%")
        report_lines.append(f"  - KHÔNG có nhãn: {pct_without_labels:.2f}%")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("2. THỐNG KÊ THEO TỪNG SPLIT")
    report_lines.append("=" * 80)
    for split in SPLITS:
        if split not in split_stats or split_stats[split] is None:
            continue
        
        stats = split_stats[split]
        report_lines.append(f"\n{split.upper()}:")
        report_lines.append(f"  - Tổng số hình ảnh: {stats['total_images_json']:,}")
        report_lines.append(f"  - Tổng số hình ảnh thực tế: {stats['total_images_actual']:,}")
        report_lines.append(f"  - Số hình ảnh CÓ nhãn: {stats['images_with_labels']:,}")
        report_lines.append(f"  - Số hình ảnh KHÔNG có nhãn: {stats['images_without_labels']:,}")
        report_lines.append(f"  - Tổng số annotations: {stats['total_annotations']:,}")
        
        if stats['total_images_json'] > 0:
            pct_with = (stats['images_with_labels'] / stats['total_images_json']) * 100
            pct_without = (stats['images_without_labels'] / stats['total_images_json']) * 100
            report_lines.append(f"  - Tỉ lệ CÓ nhãn: {pct_with:.2f}%")
            report_lines.append(f"  - Tỉ lệ KHÔNG có nhãn: {pct_without:.2f}%")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("3. THỐNG KÊ THEO TỪNG LOẠI RẦY (CLASS)")
    report_lines.append("=" * 80)
    # Sử dụng category_mapping, nếu không có thì dùng category_id có trong dữ liệu
    all_category_ids = set(total_category_annotations.keys())
    for category_id in sorted(all_category_ids):
        cat_name = category_mapping.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}"))
        ann_count = total_category_annotations.get(category_id, 0)
        img_count = total_category_images.get(category_id, 0)
        
        report_lines.append(f"\nClass {category_id} (YOLO) - {cat_name}:")
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
    report_lines.append("4. PHÂN BỐ ANNOTATIONS THEO CLASS")
    report_lines.append("=" * 80)
    if total_annotations > 0:
        for category_id in sorted(all_category_ids):
            cat_name = category_mapping.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}"))
            ann_count = total_category_annotations.get(category_id, 0)
            pct = (ann_count / total_annotations) * 100
            report_lines.append(f"Class {category_id} (YOLO) - {cat_name}: {ann_count:,} annotations ({pct:.2f}%)")
    report_lines.append("")
    
    # Tính lại tỉ lệ cho phần tóm tắt
    pct_with_labels_final = (total_images_with_labels / total_images_json * 100) if total_images_json > 0 else 0
    pct_without_labels_final = (total_images_without_labels / total_images_json * 100) if total_images_json > 0 else 0
    
    report_lines.append("=" * 80)
    report_lines.append("5. TÓM TẮT")
    report_lines.append("=" * 80)
    report_lines.append(f"📊 TỔNG SỐ HÌNH ẢNH: {total_images_json:,}")
    report_lines.append(f"  - Có nhãn: {total_images_with_labels:,} ({pct_with_labels_final:.2f}%)")
    report_lines.append(f"  - Không có nhãn: {total_images_without_labels:,} ({pct_without_labels_final:.2f}%)")
    report_lines.append("")
    report_lines.append(f"📊 TỔNG SỐ ANNOTATIONS: {total_annotations:,}")
    report_lines.append("")
    report_lines.append("Số lượng annotations theo từng loại:")
    for category_id in sorted(all_category_ids):
        cat_name = category_mapping.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}"))
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
                  total_images_with_labels: int, total_images_without_labels: int,
                  total_annotations: int, total_category_annotations: Dict[int, int],
                  data_dir: Path):
    """Tạo các biểu đồ báo cáo"""
    try:
        # Thiết lập font để hiển thị tiếng Việt (nếu có)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Tạo figure với nhiều subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Biểu đồ số lượng annotations theo từng class (Bar chart)
        ax1 = plt.subplot(2, 2, 1)
        sorted_categories = sorted(total_category_annotations.items(), key=lambda x: x[1], reverse=True)
        class_ids = [cat_id for cat_id, _ in sorted_categories]
        ann_counts = [count for _, count in sorted_categories]
        class_names = [category_mapping.get(cat_id, f"Class {cat_id}") for cat_id in class_ids]
        
        bars = ax1.bar(range(len(class_ids)), ann_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_xlabel('Class (YOLO)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Số lượng Annotations', fontsize=12, fontweight='bold')
        ax1.set_title('Số lượng Annotations theo từng Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(class_ids)))
        ax1.set_xticklabels([f"Class {cid}\n({name})" for cid, name in zip(class_ids, class_names)], 
                           rotation=0, ha='center', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị lên mỗi cột
        for i, (bar, count) in enumerate(zip(bars, ann_counts)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Pie chart phân bố annotations theo class
        ax2 = plt.subplot(2, 2, 2)
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        labels_pie = [f"Class {cid}\n{category_mapping.get(cid, f'Class {cid}')}" 
                     for cid in class_ids]
        sizes_pie = ann_counts
        
        wedges, texts, autotexts = ax2.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%',
                                           colors=colors_pie[:len(sizes_pie)],
                                           startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Phân bố Annotations theo Class (%)', fontsize=14, fontweight='bold')
        
        # 3. Biểu đồ số lượng hình ảnh có/không có nhãn
        ax3 = plt.subplot(2, 2, 3)
        labels_img = ['Có nhãn', 'Không có nhãn']
        values_img = [total_images_with_labels, total_images_without_labels]
        colors_img = ['#4ECDC4', '#FF6B6B']
        
        bars2 = ax3.bar(labels_img, values_img, color=colors_img)
        ax3.set_ylabel('Số lượng Hình ảnh', fontsize=12, fontweight='bold')
        ax3.set_title('Số lượng Hình ảnh Có/Không có Nhãn', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị và phần trăm
        total_images = total_images_with_labels + total_images_without_labels
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
    
    total_images_json = sum(s['total_images_json'] for s in all_stats.values() if s)
    total_images_with_labels = sum(s['images_with_labels'] for s in all_stats.values() if s)
    total_images_without_labels = sum(s['images_without_labels'] for s in all_stats.values() if s)
    total_annotations = sum(s['total_annotations'] for s in all_stats.values() if s)
    
    total_category_annotations = defaultdict(int)
    for stats in all_stats.values():
        if stats:
            for cat_id, count in stats['category_annotations'].items():
                total_category_annotations[cat_id] += count
    
    # Xác định mapping tự động
    category_counts = count_category_occurrences(all_stats)
    category_mapping = determine_category_mapping(category_counts)
    
    print(f"\n📊 TỔNG QUAN:")
    print(f"  Tổng số hình ảnh: {total_images_json:,}")
    print(f"  - Có nhãn: {total_images_with_labels:,}")
    print(f"  - Không có nhãn: {total_images_without_labels:,}")
    print(f"  Tổng số annotations: {total_annotations:,}")
    
    print(f"\n📊 MAPPING TỰ ĐỘNG (dựa trên số lần xuất hiện):")
    sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, (cat_id, count) in enumerate(sorted_counts, 1):
        cat_name = category_mapping.get(cat_id, f"category_{cat_id}")
        if idx == 1:
            rank_text = "NHIỀU NHẤT → Rầy nâu (BPH)"
        elif idx == 2:
            rank_text = "THỨ 2 → Rầy lưng trắng (WBPH)"
        elif idx == 3:
            rank_text = "ÍT NHẤT → Sâu ăn lá lúa (RLM)"
        else:
            rank_text = f"XẾP HẠNG {idx}"
        print(f"  Class {cat_id} (YOLO) ({cat_name}): {count:,} lần - {rank_text}")
    
    print(f"\n📊 THEO TỪNG LOẠI:")
    for category_id in sorted(total_category_annotations.keys()):
        cat_name = category_mapping.get(category_id, CATEGORY_NAMES.get(category_id, f"category_{category_id}"))
        ann_count = total_category_annotations.get(category_id, 0)
        if total_annotations > 0:
            pct = (ann_count / total_annotations) * 100
            print(f"  Class {category_id} (YOLO) - {cat_name}: {ann_count:,} annotations ({pct:.2f}%)")
    
    print(f"\n{'=' * 70}")

def main():
    """Hàm chính"""
    print("=" * 70)
    print("PHÂN TÍCH DỮ LIỆU TỪ KS-NJ4/DATA")
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
        total_images_json = sum(s['total_images_json'] for s in all_stats.values() if s)
        total_images_with_labels = sum(s['images_with_labels'] for s in all_stats.values() if s)
        total_images_without_labels = sum(s['images_without_labels'] for s in all_stats.values() if s)
        total_annotations = sum(s['total_annotations'] for s in all_stats.values() if s)
        
        total_category_annotations = defaultdict(int)
        for stats in all_stats.values():
            if stats:
                for cat_id, count in stats['category_annotations'].items():
                    total_category_annotations[cat_id] += count
        
        category_counts = count_category_occurrences(all_stats)
        category_mapping = determine_category_mapping(category_counts)
        
        create_charts(all_stats, category_mapping, total_images_with_labels, 
                     total_images_without_labels, total_annotations, 
                     dict(total_category_annotations), DATA_DIR)
    
    print(f"\n{'=' * 70}")
    print("HOÀN THÀNH!")
    print(f"{'=' * 70}")
    print(f"Báo cáo đã được lưu tại: {DATA_DIR / 'analysis_report.txt'}")
    print(f"Biểu đồ đã được lưu tại: {DATA_DIR / 'analysis_charts.png'}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()

