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

# Mapping class IDs (từ YOLO sang COCO)
# Dataset có 3 classes: 0, 1, 2
# 
# Dựa vào phân tích thực tế từ dataset (analysis_report.txt):
# - Class 2 (YOLO): NHIỀU NHẤT → brown_planthopper (Rầy nâu - BPH)
# - Class 0 (YOLO): THỨ 2 → whitebacked_planthopper (Rầy lưng trắng - WBPH)
# - Class 1 (YOLO): ÍT NHẤT → rice_leaf_miner (Sâu ăn lá lúa - RLM)
#
# Mapping YOLO class IDs sang COCO category IDs:
CLASS_MAPPING = {
    0: {"id": 2, "name": "whitebacked_planthopper", "supercategory": "planthopper"}, # Class 0: Whitebacked planthopper (Rầy lưng trắng - WBPH) - THỨ 2
    1: {"id": 3, "name": "rice_leaf_miner", "supercategory": "planthopper"},      # Class 1: Rice leaf miner (Sâu ăn lá lúa - RLM) - ÍT NHẤT
    2: {"id": 1, "name": "brown_planthopper", "supercategory": "planthopper"},      # Class 2: Brown planthopper (Rầy nâu - BPH) - NHIỀU NHẤT
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

# Color map cho mask visualization PNG (RGB format)
MASK_COLOR_MAP = {
    0: (0, 0, 0),       # Background - đen
    1: (255, 0, 0),     # Brown Planthopper - đỏ
    2: (0, 255, 0),     # White-Backed Planthopper - xanh lá
    3: (0, 0, 255),     # Rice Leaf Miner - xanh dương
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
        
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"
            
            # Chỉ xử lý những hình ảnh có file label tương ứng
            if not label_file.exists():
                # Bỏ qua hình ảnh không có file label
                continue
            
            # Đọc YOLO labels
            yolo_labels = read_yolo_labels(label_file)
            
            # Chỉ xử lý những hình ảnh có nhãn (có annotations)
            if not yolo_labels:
                # Bỏ qua hình ảnh có file label nhưng rỗng
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
            masks_found_count = 0  # Đếm số mask đã tìm được
            
            # Dùng SAM2 để detect chính xác từ YOLO bbox
            with torch.inference_mode():
                predictor.set_image(image_rgb)
                
                # Xử lý từng annotation
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
                        continue
                    
                    # Dùng SAM2 để predict mask từ bbox
                    input_box = np.array([x1, y1, x2, y2])
                    
                    try:
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )
                        
                        # Lấy mask tốt nhất
                        # masks có shape (C, H, W) hoặc (H, W) nếu multimask_output=False
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
                            print(f"  ⚠️  SAM2 không tạo được mask cho {image_file.name}, class {category_id}")
                            continue
                        
                        # Kiểm tra shape và resize nếu cần
                        if sam_mask.shape != (h, w):
                            print(f"  ⚠️  Warning: {image_file.name} - sam_mask shape {sam_mask.shape} không khớp với image shape ({h}, {w}), sẽ resize")
                            # Resize mask nếu cần
                            sam_mask = cv2.resize(sam_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        # Gán mask vào combined mask
                        # Đảm bảo category_id là uint8
                        category_id_uint8 = np.uint8(category_id)
                        
                        # Sử dụng np.where để gán mask
                        # Nếu sam_mask[i,j] == True, gán category_id_uint8, ngược lại giữ nguyên combined_mask[i,j]
                        combined_mask_new = np.where(sam_mask, category_id_uint8, combined_mask)
                        
                        # Đảm bảo dtype là uint8
                        if combined_mask_new.dtype != np.uint8:
                            combined_mask_new = combined_mask_new.astype(np.uint8)
                        
                        combined_mask = combined_mask_new
                        masks_found_count += 1
                        
                        # Vẽ mask lên ảnh đã xử lý (giống sam2_process.py)
                        color = COLOR_MAP.get(category_id, (255, 255, 255))  # Mặc định màu trắng nếu không tìm thấy
                        overlay = result_image.copy()
                        overlay[sam_mask.astype(bool)] = color
                        result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
                        
                        # Vẽ score và class info (tùy chọn)
                        mask_y, mask_x = np.where(sam_mask)
                        if len(mask_y) > 0:
                            text_x = int(np.min(mask_x))
                            text_y = int(np.min(mask_y))
                            class_name = CLASS_MAPPING[class_id_yolo]["name"]
                            cv2.putText(result_image, f"C{category_id}:{score:.3f}", 
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
                        
                    except Exception as e:
                        print(f"  ⚠️  Lỗi khi predict mask cho {image_file.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Lưu ảnh đã xử lý SAM2
            output_image_processed_file = output_images_processed_dir / image_file.name
            cv2.imwrite(str(output_image_processed_file), result_image)
            
            # Lưu combined mask PNG (nếu cần)
            if output_masks_dir:
                # Kiểm tra mask có dữ liệu không
                mask_pixel_count = np.sum(combined_mask > 0)
                if mask_pixel_count == 0:
                    print(f"  ⚠️  Warning: {image_file.name} - Mask rỗng (0 pixels) dù đã xử lý {len(yolo_labels)} labels, tìm được {masks_found_count} masks từ SAM2!")
                    print(f"      Combined_mask shape: {combined_mask.shape}, dtype: {combined_mask.dtype}")
                    print(f"      Combined_mask min/max: {combined_mask.min()}/{combined_mask.max()}")
                    print(f"      Combined_mask unique values: {np.unique(combined_mask)}")
                else:
                    unique_classes = np.unique(combined_mask[combined_mask > 0])
                    print(f"  ✓ {image_file.name} - Mask có {mask_pixel_count} pixels, classes: {unique_classes.tolist()}")
                
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
        
        # Lưu dữ liệu split
        split_data[split] = {
            'images': split_images_for_split,
            'annotations': split_annotations
        }
    
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
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHUYỂN ĐỔI DATASET TỪ YOLO SANG SAM+ViT FORMAT (DÙNG SAM2)")
    print("=" * 70)
    convert_dataset()
    print("\n✓ Hoàn thành!")
