# convert_to_sam_vit_format.py
# Đọc ảnh và labels YOLO, dùng SAM2 để detect chính xác rầy, tạo nhãn mask cho SAM+ViT
# python ks-nj4/convert_to_sam_vit_format.py
import os
import cv2
import numpy as np
import json
import torch
import warnings
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================================
# CẤU HÌNH - THAY ĐỔI CÁC THAM SỐ Ở ĐÂY
# ============================================================================

# Đường dẫn model SAM2
CHECKPOINT = "./checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEVICE = "cpu"  # "cpu" hoặc "cuda"

# Đường dẫn thư mục dữ liệu
BASE_DIR = Path("ks-nj4/data/datasets")           # Thư mục chứa dataset gốc (YOLO format)
OUTPUT_DIR = Path("ks-nj4/data/datasets-sam-vit")  # Thư mục ghi kết quả (SAM+ViT format)

# Các split cần xử lý
SPLITS = ['test', 'train', 'val']

# Format output
OUTPUT_FORMATS = {
    'coco_json': True,    # Lưu COCO JSON format
    'png_masks': True,    # Lưu PNG mask files
}

# Mapping class IDs (từ YOLO sang COCO) - Mapping cố định
# Dataset có 3 classes: 0, 1, 2
# 
# Mapping cố định:
# - YOLO class 0 → whitebacked_planthopper (Rầy lưng trắng - WBPH) → COCO category 2
# - YOLO class 1 → rice_leaf_miner (Sâu ăn lá lúa - RLM) → COCO category 3
# - YOLO class 2 → brown_planthopper (Rầy nâu - BPH) → COCO category 1
#
# Mapping YOLO class IDs sang COCO category IDs:
CLASS_MAPPING = {
    0: {"id": 2, "name": "whitebacked_planthopper", "supercategory": "planthopper"}, # YOLO class 0 → COCO category 2: Whitebacked planthopper (Rầy lưng trắng - WBPH)
    1: {"id": 3, "name": "rice_leaf_miner", "supercategory": "planthopper"},      # YOLO class 1 → COCO category 3: Rice leaf miner (Sâu ăn lá lúa - RLM)
    2: {"id": 1, "name": "brown_planthopper", "supercategory": "planthopper"},      # YOLO class 2 → COCO category 1: Brown planthopper (Rầy nâu - BPH)
}

# Color mapping cho visualization (BGR format cho OpenCV)
# Category 1: brown_planthopper - màu đỏ
# Category 2: whitebacked_planthopper - màu xanh lá
# Category 3: rice_leaf_miner - màu xanh dương
COLOR_MAP = {
    1: (255, 0, 0),    # Brown Planthopper - đỏ (BGR)
    2: (0, 255, 0),    # White-Backed Planthopper - xanh lá (BGR)
    3: (0, 0, 255),    # Rice Leaf Miner - xanh dương (BGR)
}

# Color map cho mask visualization PNG (BGR format cho OpenCV cv2.imwrite)
MASK_COLOR_MAP = {
    0: (0, 0, 0),       # Background - đen
    1: (0, 0, 255),     # Brown Planthopper - đỏ (BGR)
    2: (0, 255, 0),     # White-Backed Planthopper - xanh lá (BGR)
    3: (255, 0, 0),     # Rice Leaf Miner - xanh dương (BGR)
}

# ============================================================================

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """Chuyển đổi YOLO format sang bounding box"""
    class_id, x_center, y_center, width, height = yolo_coords
    
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    return int(class_id), x1, y1, x2, y2

def bbox_to_mask(x1, y1, x2, y2, img_height, img_width):
    """Tạo binary mask từ bounding box (fallback khi SAM2 không tạo được mask)"""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    # Đảm bảo tọa độ hợp lệ
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    
    return mask.astype(bool)

def mask_to_polygon(mask):
    """Chuyển đổi binary mask sang polygon (contour)"""
    # Tìm contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return []
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour (giảm số điểm)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Chuyển sang format [x1, y1, x2, y2, ...]
    polygon = []
    for point in approx:
        polygon.extend([int(point[0][0]), int(point[0][1])])
    
    return [polygon] if len(polygon) >= 6 else []  # Ít nhất 3 điểm (6 coordinates)

def mask_to_coco_annotation(mask, image_id, annotation_id, category_id, img_width, img_height):
    """Chuyển mask sang COCO annotation format"""
    # Chuyển mask sang polygon
    polygon = mask_to_polygon(mask)
    
    if len(polygon) == 0:
        return None
    
    # Tính area từ mask
    area = float(np.sum(mask))
    
    # Tính bbox từ mask
    mask_y, mask_x = np.where(mask)
    if len(mask_y) == 0:
        return None
    
    x1 = float(np.min(mask_x))
    y1 = float(np.min(mask_y))
    x2 = float(np.max(mask_x))
    y2 = float(np.max(mask_y))
    width = x2 - x1
    height = y2 - y1
    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygon,  # Polygon format
        "area": area,
        "bbox": [x1, y1, width, height],
        "iscrowd": 0
    }
    
    return annotation

def read_yolo_labels(txt_file):
    """Đọc file YOLO labels"""
    labels = []
    try:
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        coords = [float(x) for x in line.split()]
                        if len(coords) >= 5:
                            labels.append(coords[:5])
    except Exception as e:
        print(f"Error reading {txt_file}: {e}")
    
    return labels

def generate_conversion_report(stats: Dict, output_dir: Path):
    """Tạo báo cáo chi tiết về quá trình convert"""
    report_file = output_dir / "conversion_report.txt"
    
    CLASS_NAMES = {
        0: "whitebacked_planthopper (YOLO) → Category 2 (COCO)",
        1: "rice_leaf_miner (YOLO) → Category 3 (COCO)",
        2: "brown_planthopper (YOLO) → Category 1 (COCO)",
    }
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BÁO CÁO CHUYỂN ĐỔI DATASET TỪ YOLO SANG SAM+ViT FORMAT")
    report_lines.append("=" * 80)
    report_lines.append(f"Ngày tạo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Thư mục output: {output_dir}")
    report_lines.append("")
    
    # Tổng quan
    report_lines.append("=" * 80)
    report_lines.append("1. TỔNG QUAN")
    report_lines.append("=" * 80)
    report_lines.append(f"Tổng số ảnh đã xử lý: {stats['total_images_processed']:,}")
    report_lines.append(f"Tổng số YOLO labels: {stats['total_labels_processed']:,}")
    report_lines.append(f"Tổng số annotations: {stats['total_annotations']:,}")
    report_lines.append(f"Tổng SAM2 masks: {stats['total_sam2_masks']:,}")
    report_lines.append(f"Tổng Fallback masks (bbox): {stats['total_fallback_masks']:,}")
    
    if stats['total_labels_processed'] > 0:
        sam2_pct = (stats['total_sam2_masks'] / stats['total_labels_processed']) * 100
        fallback_pct = (stats['total_fallback_masks'] / stats['total_labels_processed']) * 100
        report_lines.append(f"Tỉ lệ SAM2 masks: {sam2_pct:.2f}%")
        report_lines.append(f"Tỉ lệ Fallback masks: {fallback_pct:.2f}%")
    report_lines.append("")
    
    # Thống kê theo từng split
    report_lines.append("=" * 80)
    report_lines.append("2. THỐNG KÊ THEO TỪNG SPLIT")
    report_lines.append("=" * 80)
    
    for split in ['train', 'val', 'test']:
        if split not in stats['split_stats']:
            continue
        
        split_stat = stats['split_stats'][split]
        report_lines.append(f"\n{split.upper()}:")
        report_lines.append(f"  - Tổng số ảnh: {split_stat['total_images']:,}")
        report_lines.append(f"  - Số ảnh đã xử lý: {split_stat['images_processed']:,}")
        report_lines.append(f"  - Tổng YOLO labels: {split_stat['total_yolo_labels']:,}")
        report_lines.append(f"  - Labels đã xử lý: {split_stat['labels_processed']:,}")
        report_lines.append(f"  - SAM2 masks: {split_stat['sam2_masks']:,}")
        report_lines.append(f"  - Fallback masks: {split_stat['fallback_masks']:,}")
        report_lines.append(f"  - Tổng annotations: {split_stat['total_annotations']:,}")
        
        if split_stat['total_yolo_labels'] > 0:
            sam2_pct = (split_stat['sam2_masks'] / split_stat['total_yolo_labels']) * 100
            fallback_pct = (split_stat['fallback_masks'] / split_stat['total_yolo_labels']) * 100
            report_lines.append(f"  - Tỉ lệ SAM2: {sam2_pct:.2f}%, Fallback: {fallback_pct:.2f}%")
        
        # Thống kê theo class
        report_lines.append(f"\n  Phân bố theo từng class:")
        for class_id in sorted(split_stat['class_stats'].keys()):
            class_stat = split_stat['class_stats'][class_id]
            class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            report_lines.append(f"    {class_name}:")
            report_lines.append(f"      - YOLO labels: {class_stat['yolo_count']:,}")
            report_lines.append(f"      - SAM2 masks: {class_stat['sam2_count']:,}")
            report_lines.append(f"      - Fallback masks: {class_stat['fallback_count']:,}")
            report_lines.append(f"      - Số images: {len(class_stat['images']):,}")
            
            if class_stat['yolo_count'] > 0:
                sam2_pct = (class_stat['sam2_count'] / class_stat['yolo_count']) * 100
                fallback_pct = (class_stat['fallback_count'] / class_stat['yolo_count']) * 100
                report_lines.append(f"      - Tỉ lệ SAM2: {sam2_pct:.2f}%, Fallback: {fallback_pct:.2f}%")
    
    # Thống kê theo từng class (tổng hợp)
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("3. THỐNG KÊ THEO TỪNG LOẠI NHÃN (TỔNG HỢP)")
    report_lines.append("=" * 80)
    
    for class_id in sorted(stats['class_stats'].keys()):
        class_stat = stats['class_stats'][class_id]
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        report_lines.append(f"\n{class_name}:")
        report_lines.append(f"  - Tổng YOLO labels: {class_stat['yolo_count']:,}")
        report_lines.append(f"  - Tổng SAM2 masks: {class_stat['sam2_count']:,}")
        report_lines.append(f"  - Tổng Fallback masks: {class_stat['fallback_count']:,}")
        report_lines.append(f"  - Tổng số images: {len(class_stat['images']):,}")
        
        if class_stat['yolo_count'] > 0:
            sam2_pct = (class_stat['sam2_count'] / class_stat['yolo_count']) * 100
            fallback_pct = (class_stat['fallback_count'] / class_stat['yolo_count']) * 100
            report_lines.append(f"  - Tỉ lệ SAM2: {sam2_pct:.2f}%, Fallback: {fallback_pct:.2f}%")
    
    # Lỗi và cảnh báo
    if stats['errors']:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("4. LỖI")
        report_lines.append("=" * 80)
        report_lines.append(f"Tổng số lỗi: {len(stats['errors'])}")
        for i, error in enumerate(stats['errors'][:50], 1):  # Hiển thị 50 lỗi đầu tiên
            report_lines.append(f"  {i}. {error}")
        if len(stats['errors']) > 50:
            report_lines.append(f"  ... và {len(stats['errors']) - 50} lỗi khác")
    
    if stats['warnings']:
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("5. CẢNH BÁO")
        report_lines.append("=" * 80)
        report_lines.append(f"Tổng số cảnh báo: {len(stats['warnings'])}")
        for i, warning in enumerate(stats['warnings'][:50], 1):  # Hiển thị 50 cảnh báo đầu tiên
            report_lines.append(f"  {i}. {warning}")
        if len(stats['warnings']) > 50:
            report_lines.append(f"  ... và {len(stats['warnings']) - 50} cảnh báo khác")
    
    # Ghi file
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n✓ Đã lưu báo cáo: {report_file}")
    except Exception as e:
        print(f"⚠️  Lỗi khi lưu báo cáo: {e}")

def convert_dataset():
    """Chuyển đổi dataset từ YOLO sang SAM+ViT format bằng SAM2"""
    print("=" * 70)
    print("CHUYỂN ĐỔI DATASET TỪ YOLO SANG SAM+ViT FORMAT (DÙNG SAM2)")
    print("=" * 70)
    print(f"Đọc từ: {BASE_DIR}")
    print(f"Ghi vào: {OUTPUT_DIR}")
    print(f"Formats: COCO JSON={OUTPUT_FORMATS['coco_json']}, PNG Masks={OUTPUT_FORMATS['png_masks']}")
    print("=" * 70)
    
    # Load SAM2
    print("Loading SAM2 Image Predictor...")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Model config: {MODEL_CFG}")
    print(f"Device: {DEVICE}")
    
    model = build_sam2(MODEL_CFG, CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(model)
    print("SAM2 loaded!")
    print("=" * 70)
    
    # Tạo thư mục output (tạo cả thư mục cha nếu chưa có)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tạo thư mục labels tổng hợp
    output_labels_dir = OUTPUT_DIR / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Xử lý từng split
    all_images = []
    all_annotations = []
    annotation_id_counter = 1
    split_data = {}  # Lưu dữ liệu từng split
    
    # Thống kê tổng hợp cho báo cáo
    overall_stats = {
        'total_images_processed': 0,
        'total_labels_processed': 0,
        'total_sam2_masks': 0,
        'total_fallback_masks': 0,
        'total_annotations': 0,
        'split_stats': {},
        'class_stats': {
            0: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
            1: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
            2: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
        },
        'errors': [],
        'warnings': [],
    }
    
    for split in SPLITS:
        print(f"\n{'=' * 70}")
        print(f"Xử lý split: {split.upper()}")
        print(f"{'=' * 70}")
        
        images_dir = BASE_DIR / "images" / split
        labels_dir = BASE_DIR / "labels" / split
        
        if not images_dir.exists():
            print(f"⚠️  Thư mục {images_dir} không tồn tại, bỏ qua...")
            continue
        
        if not labels_dir.exists():
            print(f"⚠️  Thư mục {labels_dir} không tồn tại, bỏ qua...")
            continue
        
        # Tạo thư mục output cho split này
        output_images_original_dir = OUTPUT_DIR / "images" / split  # Ảnh gốc
        output_images_processed_dir = OUTPUT_DIR / "images_processed" / split  # Ảnh đã xử lý SAM2
        output_masks_dir = OUTPUT_DIR / "masks" / split if OUTPUT_FORMATS['png_masks'] else None
        
        output_images_original_dir.mkdir(parents=True, exist_ok=True)
        output_images_processed_dir.mkdir(parents=True, exist_ok=True)
        if output_masks_dir:
            output_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Tìm tất cả file ảnh
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")) + \
                     list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
        
        split_annotations = []
        split_images_for_split = []  # Lưu images cho split này
        processed = 0
        
        # Thống kê cho split này
        split_stats = {
            'split': split,
            'total_images': 0,
            'images_processed': 0,
            'total_yolo_labels': 0,
            'labels_processed': 0,
            'sam2_masks': 0,
            'fallback_masks': 0,
            'total_annotations': 0,
            'class_stats': {
                0: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
                1: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
                2: {'yolo_count': 0, 'sam2_count': 0, 'fallback_count': 0, 'images': set()},
            },
            'errors': [],
            'warnings': [],
        }
        
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            # Chỉ xử lý những hình ảnh có file label tương ứng
            if not label_file.exists():
                # Bỏ qua hình ảnh không có file label
                continue
            
            # Đọc YOLO labels
            yolo_labels = read_yolo_labels(label_file)
            split_stats['total_images'] += 1
            
            # Chỉ xử lý những hình ảnh có nhãn (có annotations)
            if not yolo_labels:
                # Bỏ qua hình ảnh có file label nhưng rỗng
                split_stats['warnings'].append(f"{image_file.name}: File label rỗng")
                continue
            
            # Đọc ảnh
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"  ⚠️  Không đọc được ảnh: {image_file.name}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Tạo image_id
            image_id = len(all_images) + 1
            
            # Lưu ảnh gốc vào output (copy)
            output_image_original_file = output_images_original_dir / image_file.name
            cv2.imwrite(str(output_image_original_file), image)
            
            # Tạo ảnh đã xử lý (có vẽ mask)
            result_image = image.copy()
            
            # Thông tin image cho COCO
            image_info = {
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": f"{split}/{image_file.name}",  # Bao gồm split trong file_name
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            all_images.append(image_info)
            split_images_for_split.append(image_info)
            
            # Tạo combined mask cho PNG
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            masks_found_count = 0  # Đếm số mask đã tìm được từ SAM2
            masks_fallback_count = 0  # Đếm số mask dùng bbox fallback
            labels_processed = 0  # Đếm số labels đã xử lý (phải bằng len(yolo_labels))
            
            # Đếm YOLO labels cho split và theo từng class
            split_stats['total_yolo_labels'] += len(yolo_labels)
            class_counts_per_image = {0: 0, 1: 0, 2: 0}  # Đếm labels theo class cho image này
            class_sam2_counts = {0: 0, 1: 0, 2: 0}  # Đếm SAM2 masks theo class
            class_fallback_counts = {0: 0, 1: 0, 2: 0}  # Đếm fallback masks theo class
            
            for yolo_label in yolo_labels:
                class_id_yolo = int(yolo_label[0])
                if class_id_yolo in split_stats['class_stats']:
                    split_stats['class_stats'][class_id_yolo]['yolo_count'] += 1
                    split_stats['class_stats'][class_id_yolo]['images'].add(image_file.stem)
                    class_counts_per_image[class_id_yolo] += 1
            
            # Dùng SAM2 để detect chính xác từ YOLO bbox
            with torch.inference_mode():
                predictor.set_image(image_rgb)
                
                # Xử lý từng annotation - ĐẢM BẢO KHÔNG BỎ SÓT BẤT KỲ LABEL NÀO
                for yolo_label in yolo_labels:
                    class_id_yolo, x1, y1, x2, y2 = yolo_to_bbox(yolo_label, w, h)
                    
                    # Chuyển class_id (YOLO) sang COCO category_id
                    if class_id_yolo not in CLASS_MAPPING:
                        print(f"  ⚠️  Class ID {class_id_yolo} không có trong mapping, bỏ qua...")
                        continue
                    
                    category_id = CLASS_MAPPING[class_id_yolo]["id"]
                    
                    # Bỏ qua category_id = 0 (background) - không xử lý background trong mask
                    if category_id == 0:
                        print(f"  ⚠️  {image_file.name} - Bỏ qua category_id=0 (background)")
                        continue
                    
                    # Đảm bảo tọa độ hợp lệ
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    if x2 <= x1 or y2 <= y1:
                        print(f"  ⚠️  {image_file.name} - Bbox không hợp lệ: ({x1}, {y1}, {x2}, {y2})")
                        continue
                    
                    # Dùng SAM2 để predict mask từ bbox
                    input_box = np.array([x1, y1, x2, y2])
                    sam_mask = None
                    score = 0.0
                    use_fallback = False
                    
                    try:
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )
                        
                        # Lấy mask tốt nhất
                        if len(masks.shape) == 3:
                            sam_mask_raw = masks[0]
                        else:
                            sam_mask_raw = masks
                        
                        # Chuyển sang boolean (threshold = 0.0)
                        sam_mask = (sam_mask_raw > 0.0).astype(bool)
                        score = scores[0].item()
                        
                        # Kiểm tra mask hợp lệ
                        mask_pixel_count_raw = np.sum(sam_mask)
                        if mask_pixel_count_raw == 0:
                            # SAM2 không tạo được mask, dùng bbox fallback
                            sam_mask = bbox_to_mask(x1, y1, x2, y2, h, w)
                            use_fallback = True
                            masks_fallback_count += 1
                            class_fallback_counts[class_id_yolo] += 1
                            split_stats['class_stats'][class_id_yolo]['fallback_count'] += 1
                            warning_msg = f"{image_file.name}: SAM2 không tạo được mask cho class {class_id_yolo} (category {category_id}), dùng bbox fallback"
                            print(f"  ⚠️  {warning_msg}")
                            split_stats['warnings'].append(warning_msg)
                        else:
                            masks_found_count += 1
                            class_sam2_counts[class_id_yolo] += 1
                            split_stats['class_stats'][class_id_yolo]['sam2_count'] += 1
                        
                        # Kiểm tra shape và resize nếu cần
                        if sam_mask.shape != (h, w):
                            sam_mask = cv2.resize(sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                    except Exception as e:
                        # Nếu có lỗi, dùng bbox fallback
                        error_msg = f"{image_file.name}: Lỗi khi predict mask cho class {class_id_yolo} (category {category_id}): {str(e)}, dùng bbox fallback"
                        print(f"  ⚠️  {error_msg}")
                        sam_mask = bbox_to_mask(x1, y1, x2, y2, h, w)
                        use_fallback = True
                        masks_fallback_count += 1
                        class_fallback_counts[class_id_yolo] += 1
                        split_stats['class_stats'][class_id_yolo]['fallback_count'] += 1
                        split_stats['errors'].append(error_msg)
                    
                    # Đảm bảo có mask (từ SAM2 hoặc fallback)
                    if sam_mask is None or np.sum(sam_mask) == 0:
                        error_msg = f"{image_file.name}: Không thể tạo mask cho class {class_id_yolo} (category {category_id}), bỏ qua label này"
                        print(f"  ⚠️  {error_msg}")
                        split_stats['errors'].append(error_msg)
                        continue
                    
                    # Gán mask vào combined mask
                    category_id_uint8 = np.uint8(category_id)
                    
                    # Sử dụng np.where để gán mask
                    # Nếu sam_mask[i,j] == True, gán category_id_uint8, ngược lại giữ nguyên combined_mask[i,j]
                    combined_mask_new = np.where(sam_mask, category_id_uint8, combined_mask)
                    
                    # Đảm bảo dtype là uint8
                    if combined_mask_new.dtype != np.uint8:
                        combined_mask_new = combined_mask_new.astype(np.uint8)
                    
                    combined_mask = combined_mask_new
                    labels_processed += 1
                    split_stats['labels_processed'] += 1
                    
                    # Vẽ mask lên ảnh đã xử lý
                    color = COLOR_MAP.get(category_id, (255, 255, 255))
                    overlay = result_image.copy()
                    overlay[sam_mask.astype(bool)] = color
                    result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
                    
                    # Vẽ score và class info
                    mask_y, mask_x = np.where(sam_mask)
                    if len(mask_y) > 0:
                        text_x = int(np.min(mask_x))
                        text_y = int(np.min(mask_y))
                        class_name = CLASS_MAPPING[class_id_yolo]["name"]
                        label_text = f"C{category_id}:{score:.3f}" if not use_fallback else f"C{category_id}:BBOX"
                        cv2.putText(result_image, label_text, 
                                   (text_x, text_y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Tạo COCO annotation từ mask
                    coco_ann = mask_to_coco_annotation(
                        sam_mask, image_id, annotation_id_counter, 
                        category_id, w, h
                    )
                    
                    if coco_ann is not None:
                        split_annotations.append(coco_ann)
                        all_annotations.append(coco_ann)
                        annotation_id_counter += 1
                    else:
                        # Nếu không tạo được polygon, vẫn giữ mask trong combined_mask
                        print(f"  ⚠️  Không tạo được polygon cho {image_file.name}, class {category_id}, nhưng mask đã được lưu")
            
            # Lưu ảnh đã xử lý SAM2
            output_image_processed_file = output_images_processed_dir / image_file.name
            cv2.imwrite(str(output_image_processed_file), result_image)
            
            # Kiểm tra số lượng labels đã xử lý
            if labels_processed != len(yolo_labels):
                warning_msg = f"{image_file.name}: Đã xử lý {labels_processed}/{len(yolo_labels)} labels"
                print(f"  ⚠️  Warning: {warning_msg}")
                split_stats['warnings'].append(warning_msg)
            
            # Cập nhật thống kê
            split_stats['images_processed'] += 1
            split_stats['sam2_masks'] += masks_found_count
            split_stats['fallback_masks'] += masks_fallback_count
            split_stats['total_annotations'] += len([ann for ann in split_annotations if ann['image_id'] == image_id])
            
            # Lưu combined mask PNG (nếu cần)
            if output_masks_dir:
                # Kiểm tra mask có dữ liệu không
                mask_pixel_count = np.sum(combined_mask > 0)
                if mask_pixel_count == 0:
                    print(f"  ⚠️  Warning: {image_file.name} - Mask rỗng (0 pixels) dù đã xử lý {len(yolo_labels)} labels!")
                    print(f"      SAM2 masks: {masks_found_count}, Fallback masks: {masks_fallback_count}")
                    print(f"      Combined_mask shape: {combined_mask.shape}, dtype: {combined_mask.dtype}")
                    print(f"      Combined_mask min/max: {combined_mask.min()}/{combined_mask.max()}")
                    print(f"      Combined_mask unique values: {np.unique(combined_mask)}")
                else:
                    unique_categories = np.unique(combined_mask[combined_mask > 0])
                    
                    # Tạo log chi tiết theo từng YOLO class
                    class_details = []
                    CLASS_NAMES_MAP = {
                        0: "whitebacked_planthopper",
                        1: "rice_leaf_miner",
                        2: "brown_planthopper",
                    }
                    
                    # Hiển thị chi tiết theo từng YOLO class với cả YOLO và COCO
                    for class_id in sorted(class_counts_per_image.keys()):
                        if class_counts_per_image[class_id] > 0:
                            class_name = CLASS_NAMES_MAP.get(class_id, f"Class{class_id}")
                            category_id = CLASS_MAPPING[class_id]["id"]
                            sam2_count = class_sam2_counts[class_id]
                            fallback_count = class_fallback_counts[class_id]
                            total_count = class_counts_per_image[class_id]
                            
                            # Tạo chuỗi hiển thị với cả YOLO class ID và COCO category ID
                            if sam2_count > 0 and fallback_count > 0:
                                detail = f"YOLO_{class_id}({class_name})→COCO_{category_id}: {total_count} labels (SAM2: {sam2_count}, Fallback: {fallback_count})"
                            elif sam2_count > 0:
                                detail = f"YOLO_{class_id}({class_name})→COCO_{category_id}: {total_count} labels (SAM2: {sam2_count})"
                            elif fallback_count > 0:
                                detail = f"YOLO_{class_id}({class_name})→COCO_{category_id}: {total_count} labels (Fallback: {fallback_count})"
                            else:
                                detail = f"YOLO_{class_id}({class_name})→COCO_{category_id}: {total_count} labels"
                            
                            class_details.append(detail)
                    
                    details_str = " | ".join(class_details) if class_details else "No classes"
                    print(f"  ✓ {image_file.name} - Mask: {mask_pixel_count} pixels | {details_str}")
                
                # Lưu mask với giá trị class_id (để train - format chính xác)
                mask_file = output_masks_dir / f"{image_file.stem}.png"
                cv2.imwrite(str(mask_file), combined_mask)
                
                # Tạo mask visualization (màu) để dễ xem hơn
                mask_vis_file = output_masks_dir / f"{image_file.stem}_vis.png"
                mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Vẽ mask với màu sử dụng MASK_COLOR_MAP
                for class_id, color in MASK_COLOR_MAP.items():
                    mask_vis[combined_mask == class_id] = color
                
                cv2.imwrite(str(mask_vis_file), mask_vis)
            
            processed += 1
            if processed % 50 == 0:
                print(f"  ... Đã xử lý {processed}/{len(image_files)} ảnh...")
        
        print(f"✓ Split {split.upper()}: Xử lý {processed} ảnh, {len(split_annotations)} annotations")
        print(f"  - SAM2 masks: {split_stats['sam2_masks']}, Fallback masks: {split_stats['fallback_masks']}")
        print(f"  - Labels processed: {split_stats['labels_processed']}/{split_stats['total_yolo_labels']}")
        
        # Lưu dữ liệu split
        split_data[split] = {
            'images': split_images_for_split,
            'annotations': split_annotations
        }
        
        # Cập nhật thống kê tổng hợp
        overall_stats['split_stats'][split] = split_stats
        overall_stats['total_images_processed'] += split_stats['images_processed']
        overall_stats['total_labels_processed'] += split_stats['labels_processed']
        overall_stats['total_sam2_masks'] += split_stats['sam2_masks']
        overall_stats['total_fallback_masks'] += split_stats['fallback_masks']
        overall_stats['total_annotations'] += len(split_annotations)
        
        # Cập nhật thống kê theo class
        for class_id in [0, 1, 2]:
            overall_stats['class_stats'][class_id]['yolo_count'] += split_stats['class_stats'][class_id]['yolo_count']
            overall_stats['class_stats'][class_id]['sam2_count'] += split_stats['class_stats'][class_id]['sam2_count']
            overall_stats['class_stats'][class_id]['fallback_count'] += split_stats['class_stats'][class_id]['fallback_count']
            overall_stats['class_stats'][class_id]['images'].update(split_stats['class_stats'][class_id]['images'])
        
        overall_stats['errors'].extend(split_stats['errors'])
        overall_stats['warnings'].extend(split_stats['warnings'])
    
    # Lưu COCO JSON cho từng split và tổng hợp (nếu cần)
    if OUTPUT_FORMATS['coco_json']:
        # Lưu từng split riêng
        for split in SPLITS:
            if split not in split_data or len(split_data[split]['annotations']) == 0:
                continue
                
            coco_format_split = {
                "info": {
                    "description": f"Rice Planthopper Dataset for SAM+ViT (converted with SAM2) - {split}",
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
                "categories": [v for k, v in CLASS_MAPPING.items() if k > 0],  # Bỏ background
                "images": split_data[split]['images'],
                "annotations": split_data[split]['annotations']
            }
            
            coco_json_file_split = output_labels_dir / f"{split}.json"
            with open(coco_json_file_split, 'w', encoding='utf-8') as f:
                json.dump(coco_format_split, f, indent=2, ensure_ascii=False)
            print(f"✓ Đã lưu COCO JSON cho split {split}: {coco_json_file_split}")
    
    # Lưu COCO JSON tổng hợp (nếu cần)
    if OUTPUT_FORMATS['coco_json']:
        coco_format = {
            "info": {
                "description": "Rice Planthopper Dataset for SAM+ViT (converted with SAM2)",
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
            "categories": [v for k, v in CLASS_MAPPING.items() if k > 0],  # Bỏ background
            "images": all_images,
            "annotations": all_annotations
        }
        
        coco_json_file = OUTPUT_DIR / "labels" / "annotations.json"
        coco_json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(coco_json_file, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Đã lưu COCO JSON tổng hợp: {coco_json_file}")
    
    # Tóm tắt
    print(f"\n{'=' * 70}")
    print("TỔNG KẾT")
    print(f"{'=' * 70}")
    print(f"Tổng số ảnh: {len(all_images)}")
    print(f"Tổng số annotations: {len(all_annotations)}")
    print(f"Tổng SAM2 masks: {overall_stats['total_sam2_masks']}")
    print(f"Tổng Fallback masks: {overall_stats['total_fallback_masks']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    
    # Tạo báo cáo chi tiết
    generate_conversion_report(overall_stats, OUTPUT_DIR)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHUYỂN ĐỔI DATASET TỪ YOLO SANG SAM+ViT FORMAT (DÙNG SAM2)")
    print("=" * 70)
    convert_dataset()
    print("\n✓ Hoàn thành!")
