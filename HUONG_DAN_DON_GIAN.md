# SAM 2: Hướng dẫn cài đặt đơn giản

## 🚀 Cài đặt SAM2 trên máy Windows của bạn

### Bước 1: Kiểm tra Python
```bash
python --version
```
*Kết quả: Python 3.12.0 ✅*

### Bước 2: Tạo môi trường ảo
```bash
python -m venv sam2_env
source sam2_env/Scripts/activate
```

### Bước 3: Cài đặt PyTorch (CPU version)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python.exe -m pip install --upgrade pip
```

### Bước 4: Cài đặt SAM2
```bash
pip install -e ".[notebooks]"
```

### Bước 5: Tải model nhỏ
```bash
cd checkpoints
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
cd ..
```

### Bước 6: Test cài đặt
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"

pip install hydra-core

python -c "import sam2; print('SAM2 installed successfully')"
```

### Test
```
pip install opencv-python
pip install matplotlib
```

## 🎯 Sử dụng SAM2

### Phân đoạn hình ảnh
```python
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load hình ảnh
image = cv2.imread("your_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Phân đoạn
with torch.inference_mode():
    predictor.set_image(image_rgb)
    
    # Click vào đối tượng muốn phân đoạn (x, y)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Lấy mask tốt nhất
    best_mask = masks[np.argmax(scores)]
    print(f"Score: {scores[np.argmax(scores)]:.3f}")
```

### Chạy notebook ví dụ
```bash
jupyter notebook notebooks/image_predictor_example.ipynb
```

## 📊 Các model có sẵn

| Model | Kích thước | Tốc độ | Chất lượng |
|-------|------------|--------|------------|
| sam2.1_hiera_tiny | 38.9M | ⚡⚡⚡ | ⭐⭐⭐ |
| sam2.1_hiera_small | 46M | ⚡⚡ | ⭐⭐⭐⭐ |
| sam2.1_hiera_base_plus | 80.8M | ⚡ | ⭐⭐⭐⭐⭐ |
| sam2.1_hiera_large | 224.4M | 🐌 | ⭐⭐⭐⭐⭐ |

**Khuyến nghị:** Bắt đầu với `tiny` để test nhanh.

## 🔧 Xử lý lỗi

### Lỗi CUDA extension
```bash
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

### Lỗi ImportError
```bash
pip uninstall -y SAM-2
pip install -e ".[notebooks]"
```

## 🎉 Hoàn thành!

Bây giờ bạn có thể:
- ✅ Phân đoạn đối tượng trong hình ảnh
- ✅ Chạy các notebook ví dụ
- ✅ Sử dụng SAM2 cho dự án của mình

**Lưu ý:** Chạy trên CPU sẽ chậm (30-60s/hình ảnh) nhưng ổn định.

---
**Chúc bạn thành công! 🚀**
