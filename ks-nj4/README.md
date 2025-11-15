# Hướng dẫn sử dụng convert_to_sam_vit_format.py

## 📋 Mô tả tổng quan

Script `convert_to_sam_vit_format.py` là công cụ chuyển đổi dataset từ định dạng YOLO sang định dạng SAM+ViT (COCO format) sử dụng mô hình SAM2 (Segment Anything Model 2) để tạo mask segmentation chính xác cho các đối tượng rầy lúa.

### Mục đích

- **Đọc dataset YOLO format**: Đọc ảnh và nhãn bounding box từ dataset YOLO
- **Sử dụng SAM2 để tạo mask**: Dùng SAM2 để tạo mask segmentation chính xác từ bounding box YOLO
- **Chuyển đổi sang COCO format**: Tạo file JSON theo chuẩn COCO với polygon segmentation
- **Tạo mask PNG**: Lưu mask dưới dạng PNG để training SAM+ViT models
- **Visualization**: Tạo ảnh đã xử lý với mask được vẽ lên để kiểm tra

## 🎯 Dataset được hỗ trợ

Script này được thiết kế cho **Rice Planthopper Dataset** với 3 loại rầy:

1. **Class 0 (YOLO) → Category 1 (COCO)**: Brown Planthopper (Rầy nâu) - BPH
2. **Class 1 (YOLO) → Category 2 (COCO)**: White-Backed Planthopper (Rầy lưng trắng) - WBPH  
3. **Class 2 (YOLO) → Category 3 (COCO)**: Green Leafhopper (Rầy xanh) - GLH

**Tài liệu tham khảo:**
- Bài báo MDPI: ["Driving by a Publicly Available RGB Image Dataset for Rice Planthopper Detection and Counting by Fusing Swin Transformer and YOLOv8-p2 Architectures in Field Landscapes"](https://www.mdpi.com/2077-0472/15/13/1366)
- Dataset Kaggle: [Planthopper Dataset](https://www.kaggle.com/datasets/xushengji/planthopper)

## 📦 Yêu cầu hệ thống

### Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pathlib
```

### SAM2 Model

Cần có checkpoint và config file của SAM2:
- Checkpoint: `./checkpoints/sam2.1_hiera_tiny.pt`
- Config: `configs/sam2.1/sam2.1_hiera_t.yaml`

### Cấu trúc thư mục input

Dataset YOLO format cần có cấu trúc như sau:

```
datasets/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

**Format file label YOLO** (`.txt`):
```
class_id x_center y_center width height
```
- Tất cả giá trị là normalized (0.0 - 1.0)
- Mỗi dòng là một object

**Ví dụ:**
```
0 0.5 0.5 0.1 0.1
1 0.3 0.7 0.15 0.12
2 0.8 0.2 0.08 0.09
```

## ⚙️ Cấu hình

Tất cả các tham số cấu hình nằm ở đầu file `convert_to_sam_vit_format.py`:

### Model SAM2

```python
CHECKPOINT = "./checkpoints/sam2.1_hiera_tiny.pt"  # Đường dẫn checkpoint
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"   # Đường dẫn config
DEVICE = "cpu"  # "cpu" hoặc "cuda"
```

### Đường dẫn dữ liệu

```python
BASE_DIR = Path("datasets")           # Thư mục dataset YOLO gốc
OUTPUT_DIR = Path("datasets-sam-vit")  # Thư mục output
```

### Splits cần xử lý

```python
SPLITS = ['test', 'train', 'val']  # Các split cần chuyển đổi
```

### Format output

```python
OUTPUT_FORMATS = {
    'coco_json': True,    # Lưu COCO JSON format
    'png_masks': True,    # Lưu PNG mask files
}
```

### Class Mapping

Mapping từ YOLO class IDs sang COCO category IDs:

```python
CLASS_MAPPING = {
    0: {"id": 1, "name": "brown_planthopper", "supercategory": "planthopper"},
    1: {"id": 2, "name": "whitebacked_planthopper", "supercategory": "planthopper"},
    2: {"id": 3, "name": "green_leafhopper", "supercategory": "planthopper"},
}
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu

Đảm bảo dataset YOLO đã được tổ chức đúng cấu trúc thư mục như mô tả ở trên.

### Bước 2: Kiểm tra SAM2 model

Đảm bảo có checkpoint và config file của SAM2:
```bash
ls ./checkpoints/sam2.1_hiera_tiny.pt
ls configs/sam2.1/sam2.1_hiera_t.yaml
```

### Bước 3: Chạy script

```bash
python ks-nj4/convert_to_sam_vit_format.py
```

Script sẽ:
1. Load SAM2 model
2. Xử lý từng split (train/val/test)
3. Đọc ảnh và labels YOLO
4. Dùng SAM2 để tạo mask từ bounding box
5. Tạo COCO annotations
6. Lưu kết quả

### Bước 4: Kiểm tra kết quả

Sau khi chạy xong, kiểm tra thư mục output:

```bash
ls datasets-sam-vit/
```

## 📁 Cấu trúc output

Sau khi chạy, thư mục `datasets-sam-vit` sẽ có cấu trúc:

```
datasets-sam-vit/
├── images/
│   ├── train/          # Ảnh gốc (copy từ input)
│   ├── val/
│   └── test/
├── images_processed/
│   ├── train/          # Ảnh đã xử lý (có vẽ mask)
│   ├── val/
│   └── test/
├── masks/
│   ├── train/          # Mask PNG files
│   │   ├── image1.png          # Mask với giá trị class_id
│   │   ├── image1_vis.png      # Mask visualization (màu)
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train.json      # COCO JSON cho train split
    ├── val.json        # COCO JSON cho val split
    ├── test.json       # COCO JSON cho test split
    └── annotations.json # COCO JSON tổng hợp (tất cả splits)
```

## 📊 Format output

### 1. COCO JSON Format

File JSON tuân theo chuẩn COCO với các trường:

```json
{
  "info": {
    "description": "Rice Planthopper Dataset for SAM+ViT (converted with SAM2)",
    "version": "1.0",
    "year": 2025
  },
  "licenses": [...],
  "categories": [
    {
      "id": 1,
      "name": "brown_planthopper",
      "supercategory": "planthopper"
    },
    {
      "id": 2,
      "name": "whitebacked_planthopper",
      "supercategory": "planthopper"
    },
    {
      "id": 3,
      "name": "green_leafhopper",
      "supercategory": "planthopper"
    }
  ],
  "images": [
    {
      "id": 1,
      "width": 1920,
      "height": 1080,
      "file_name": "train/image1.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],  # Polygon format
      "area": 1234.5,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ]
}
```

### 2. PNG Mask Format

- **Mask file** (`image1.png`): Mask với giá trị pixel = category_id
  - 0: Background
  - 1: Brown Planthopper
  - 2: White-Backed Planthopper
  - 3: Green Leafhopper

- **Visualization file** (`image1_vis.png`): Mask với màu để dễ xem
  - Background: Đen (0, 0, 0)
  - Brown Planthopper: Xanh lá (0, 255, 0)
  - White-Backed Planthopper: Đỏ (255, 0, 0)
  - Green Leafhopper: Xanh dương (0, 0, 255)

## 🔧 Các hàm chính

### `yolo_to_bbox(yolo_coords, img_width, img_height)`

Chuyển đổi tọa độ YOLO (normalized) sang bounding box (pixel).

**Input:**
- `yolo_coords`: `[class_id, x_center, y_center, width, height]` (normalized)
- `img_width`, `img_height`: Kích thước ảnh

**Output:**
- `(class_id, x1, y1, x2, y2)`: Bounding box trong pixel

### `mask_to_polygon(mask)`

Chuyển đổi binary mask sang polygon (contour) để lưu trong COCO format.

**Input:**
- `mask`: Binary mask (numpy array, bool)

**Output:**
- `[polygon]`: List chứa polygon `[x1, y1, x2, y2, ...]`

### `mask_to_coco_annotation(mask, image_id, annotation_id, category_id, img_width, img_height)`

Tạo COCO annotation từ mask.

**Input:**
- `mask`: Binary mask
- `image_id`: ID của ảnh
- `annotation_id`: ID của annotation
- `category_id`: Category ID (1, 2, hoặc 3)
- `img_width`, `img_height`: Kích thước ảnh

**Output:**
- COCO annotation dictionary hoặc `None` nếu không hợp lệ

### `read_yolo_labels(txt_file)`

Đọc file label YOLO.

**Input:**
- `txt_file`: Path đến file `.txt`

**Output:**
- List các `[class_id, x_center, y_center, width, height]`

### `convert_dataset()`

Hàm chính thực hiện chuyển đổi dataset.

## 🔍 Quy trình xử lý

1. **Load SAM2 model**: Khởi tạo SAM2 Image Predictor
2. **Vòng lặp qua các splits**: Xử lý train/val/test
3. **Vòng lặp qua các ảnh**: 
   - Đọc ảnh và labels YOLO
   - Copy ảnh gốc vào output
   - Khởi tạo combined mask
4. **Vòng lặp qua các YOLO labels**:
   - Chuyển YOLO bbox sang pixel coordinates
   - Map class_id sang category_id
   - Dùng SAM2 predict mask từ bbox
   - Gán mask vào combined mask với category_id
   - Tạo COCO annotation
   - Vẽ mask lên ảnh processed
5. **Lưu kết quả**:
   - Lưu ảnh processed
   - Lưu mask PNG
   - Lưu mask visualization
6. **Tạo COCO JSON**: Lưu cho từng split và tổng hợp

## ⚠️ Lưu ý quan trọng

### 1. Device

- Mặc định sử dụng `CPU` (`DEVICE = "cpu"`)
- Nếu có GPU, đổi thành `DEVICE = "cuda"` để tăng tốc
- Script tự động set `CUDA_VISIBLE_DEVICES=''` để force CPU

### 2. Memory

- SAM2 model có thể tốn nhiều RAM/VRAM
- Với dataset lớn, nên xử lý từng split riêng
- Có thể giảm batch size hoặc xử lý ít ảnh một lúc

### 3. Mask overlap

- Nếu nhiều objects overlap, mask sau sẽ ghi đè mask trước
- Script sử dụng `np.where()` để gán mask, không merge

### 4. Polygon conversion

- Nếu mask quá nhỏ hoặc không tìm được contour, polygon sẽ là `[]`
- Annotation vẫn được tạo nhưng `segmentation` có thể rỗng
- Mask PNG vẫn được lưu đầy đủ

## 🐛 Troubleshooting

### Lỗi: "Không đọc được ảnh"

**Nguyên nhân:** File ảnh bị hỏng hoặc format không hỗ trợ

**Giải pháp:**
- Kiểm tra file ảnh có tồn tại không
- Kiểm tra format ảnh (chỉ hỗ trợ JPG, PNG)
- Thử mở ảnh bằng image viewer

### Lỗi: "SAM2 không tạo được mask"

**Nguyên nhân:** 
- Bounding box quá nhỏ hoặc không hợp lệ
- SAM2 không detect được object trong bbox

**Giải pháp:**
- Kiểm tra YOLO labels có hợp lệ không
- Kiểm tra bbox có nằm trong ảnh không
- Có thể cần điều chỉnh threshold hoặc dùng model SAM2 lớn hơn

### Lỗi: "Mask rỗng (0 pixels)"

**Nguyên nhân:** 
- SAM2 không tạo được mask cho bất kỳ object nào
- Có lỗi trong quá trình gán mask

**Giải pháp:**
- Kiểm tra debug output trong console
- Kiểm tra `combined_mask` có được gán đúng không
- Xem file `_vis.png` để kiểm tra mask visualization

### Lỗi: "Không tạo được polygon"

**Nguyên nhân:** 
- Mask quá nhỏ hoặc không có contour
- Contour có ít hơn 3 điểm

**Giải pháp:**
- Đây là warning, không phải lỗi nghiêm trọng
- Mask PNG vẫn được lưu đầy đủ
- Có thể bỏ qua hoặc điều chỉnh `epsilon` trong `mask_to_polygon()`

### Performance chậm

**Nguyên nhân:** 
- Sử dụng CPU thay vì GPU
- Dataset quá lớn
- SAM2 model quá lớn

**Giải pháp:**
- Chuyển sang GPU nếu có: `DEVICE = "cuda"`
- Sử dụng model nhỏ hơn (như `sam2.1_hiera_tiny`)
- Xử lý từng split riêng
- Giảm số ảnh xử lý một lúc

## 📈 Output và Logs

Script sẽ in ra console:

- Thông tin cấu hình
- Tiến trình xử lý từng split
- Số lượng ảnh và annotations đã xử lý
- Các warning và error (nếu có)
- Tổng kết cuối cùng

**Ví dụ output:**
```
======================================================================
CHUYỂN ĐỔI DATASET TỪ YOLO SANG SAM+ViT FORMAT (DÙNG SAM2)
======================================================================
Đọc từ: datasets
Ghi vào: datasets-sam-vit
Formats: COCO JSON=True, PNG Masks=True
======================================================================
Loading SAM2 Image Predictor...
SAM2 loaded!
======================================================================

======================================================================
Xử lý split: TRAIN
======================================================================
  ✓ image1.jpg - Mask có 1234 pixels, classes: [1, 2]
  ... Đã xử lý 50/1000 ảnh...
✓ Split TRAIN: Xử lý 1000 ảnh, 2500 annotations
✓ Đã lưu COCO JSON cho split train: datasets-sam-vit/labels/train.json

======================================================================
TỔNG KẾT
======================================================================
Tổng số ảnh: 3000
Tổng số annotations: 7500
Output directory: datasets-sam-vit
======================================================================
```

## 📚 Tài liệu tham khảo

- **SAM2 Paper**: [Segment Anything 2.0](https://arxiv.org/abs/2311.15796)
- **COCO Format**: [COCO Dataset Format](https://cocodataset.org/#format-data)
- **YOLO Format**: [YOLO Format Documentation](https://docs.ultralytics.com/datasets/)
- **Dataset Paper**: [MDPI Paper on Rice Planthopper Dataset](https://www.mdpi.com/2077-0472/15/13/1366)

## 📝 Changelog

### Version 1.0
- Hỗ trợ chuyển đổi YOLO → COCO format
- Tích hợp SAM2 để tạo mask segmentation
- Hỗ trợ 3 classes: Brown Planthopper, White-Backed Planthopper, Green Leafhopper
- Tạo mask PNG và visualization
- Lưu COCO JSON cho từng split và tổng hợp

## 👤 Tác giả

Script được phát triển cho dự án SAM+ViT research về phát hiện và đếm rầy lúa.

## 📄 License

Xem file LICENSE trong repository chính.

---

**Lưu ý:** Đọc kỹ phần cấu hình trước khi chạy script. Đảm bảo đường dẫn model và dataset đúng với cấu hình của bạn.

