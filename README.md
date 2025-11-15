# SAM 2: Hướng dẫn chạy project

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

SAM 2 (Segment Anything Model 2) là một mô hình AI mạnh mẽ để phân đoạn đối tượng trong hình ảnh và video. Đây là hướng dẫn chi tiết để thiết lập và chạy project này.

![SAM 2 architecture](assets/model_diagram.png?raw=true)

## 📋 Yêu cầu hệ thống

### Phần cứng:
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị)
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB+
- **Ổ cứng**: Ít nhất 10GB dung lượng trống

### Phần mềm:
- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.5.1
- **CUDA**: Phiên bản tương thích với PyTorch (thường là CUDA 12.1)
- **Hệ điều hành**: Linux (khuyến nghị), Windows với WSL

## 🚀 Cài đặt nhanh

### Bước 1: Thiết lập môi trường Python

```bash
# Tạo môi trường conda mới
conda create --name sam2 python=3.12 --yes

# Kích hoạt môi trường
conda activate sam2

# Cài đặt PyTorch với CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Bước 2: Cài đặt SAM2

```bash
# Di chuyển vào thư mục project
cd sam2

# Cài đặt SAM2 với notebooks
pip install -e ".[notebooks]"
```

**Nếu gặp lỗi CUDA extension:**
```bash
# Bỏ qua CUDA extension nếu cần thiết
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

### Bước 3: Tải xuống model checkpoints

```bash
# Tải xuống tất cả checkpoints
cd checkpoints
./download_ckpts.sh
cd ..
```

**Hoặc tải xuống từng model riêng lẻ:**
- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) (38.9M) - Nhanh nhất
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) (46M) - Cân bằng
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) (80.8M) - Khuyến nghị
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) (224.4M) - Chất lượng cao nhất

## 🎯 Cách sử dụng

### Kiểm tra cài đặt

Trước khi bắt đầu, hãy kiểm tra xem SAM2 đã được cài đặt đúng chưa:

```bash
# Kiểm tra Python và PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra SAM2
python -c "import sam2; print('SAM2 installed successfully!')"
```

### 1. Phân đoạn hình ảnh

**Chạy notebook ví dụ:**
```bash
jupyter notebook notebooks/image_predictor_example.ipynb
```

**Code Python cơ bản:**
```python
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Thiết lập model
checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load hình ảnh
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sử dụng model
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_rgb)
    
    # Click point (x, y) và label (1 = foreground, 0 = background)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    
    # Predict mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Lấy mask tốt nhất
    best_mask = masks[np.argmax(scores)]
    print(f"Mask score: {scores[np.argmax(scores)]:.3f}")
```

**Ví dụ với box prompt:**
```python
# Sử dụng box thay vì point
input_box = np.array([x1, y1, x2, y2])  # [left, top, right, bottom]

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
```

### 2. Phân đoạn video

```python
import torch
import cv2
from sam2.build_sam import build_sam2_video_predictor

# Thiết lập video predictor
checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Load video
video_path = "path/to/your/video.mp4"

# Initialize state
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path)
    
    # Thêm prompt ban đầu
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state, 
        points=[[x, y]], 
        labels=[1]
    )
    
    # Propagate qua toàn bộ video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        print(f"Frame {frame_idx}: {len(object_ids)} objects tracked")
        # Xử lý masks cho từng frame
        for obj_id, mask in zip(object_ids, masks):
            # Visualize hoặc lưu mask
            pass
```

### 3. Tự động tạo mask

```bash
jupyter notebook notebooks/automatic_mask_generator_example.ipynb
```

**Code tự động tạo mask:**
```python
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Thiết lập automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=build_sam2(model_cfg, checkpoint),
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Tạo masks tự động
masks = mask_generator.generate(image_rgb)
print(f"Generated {len(masks)} masks")
```

## 🖥️ Chạy trên máy của bạn

### Thông tin hệ thống hiện tại
- **Python**: 3.12.0 (64-bit) ✅
- **Platform**: Windows ✅  
- **PyTorch**: Chưa cài đặt ❌
- **SAM2**: Chưa cài đặt ❌

### Hướng dẫn cài đặt từng bước cho máy của bạn

**Phương án 1: Cài đặt Anaconda (Khuyến nghị)**
```bash
# Tải xuống từ: https://www.anaconda.com/products/distribution
# Cài đặt và mở Anaconda Prompt
# Sau đó chạy:
conda create --name sam2 python=3.10 --yes
conda activate sam2
```

**Phương án 2: Sử dụng Python có sẵn (Cho máy không có Conda)**
```bash
# Kiểm tra Python hiện tại
python --version

# Nâng cấp pip
python -m pip install --upgrade pip

# Tạo virtual environment
python -m venv sam2_env

# Kích hoạt virtual environment
# Trên Windows:
sam2_env\Scripts\activate
# Trên Linux/Mac:
source sam2_env/bin/activate
```

**Bước tiếp theo: Cài đặt PyTorch**
```bash
# Cài đặt PyTorch với CUDA (nếu có GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hoặc cài đặt CPU version (khuyến nghị cho máy của bạn)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Kiểm tra PyTorch:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Cài đặt SAM2:**
```bash
# Di chuyển vào thư mục project
cd D:\3.Research\3.VisionTransformer\3.Project\sam2

# Cài đặt SAM2
pip install -e ".[notebooks]"
```

**Nếu gặp lỗi CUDA extension:**
```bash
# Bỏ qua CUDA extension nếu cần thiết
SAM2_BUILD_CUDA=0 pip install -e ".[notebooks]"
```

**Tải xuống model:**
```bash
# Tải xuống model nhỏ để test
cd checkpoints
curl -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
cd ..
```

**Test cài đặt:**
```bash
# Kiểm tra PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra SAM2
python -c "import sam2; print('SAM2 installed successfully!')"
```

### Chạy ví dụ đầu tiên

**Tạo file test đơn giản:**
```bash
# Tạo file test_sam2.py
echo 'import torch
import sys
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    import sam2
    print("✅ SAM2 installed successfully!")
except ImportError as e:
    print("❌ SAM2 not installed:", e)
' > test_sam2.py
```

**Chạy test:**
```bash
python test_sam2.py
```

**Nếu SAM2 đã cài đặt thành công, tạo file demo:**
```python
# Tạo file demo_sam2.py
import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Thiết lập model
checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

try:
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    print("✅ SAM2 model loaded successfully!")
    print("Bạn có thể chạy các notebook ví dụ:")
    print("- notebooks/image_predictor_example.ipynb")
    print("- notebooks/video_predictor_example.ipynb")
    print("- notebooks/automatic_mask_generator_example.ipynb")
except Exception as e:
    print("❌ Error loading model:", e)
```

**Chạy demo:**
```bash
python demo_sam2.py
```

### 🚨 Lưu ý quan trọng cho máy của bạn

**Vì máy của bạn không có Conda:**
1. **Sử dụng virtual environment** để tránh xung đột packages
2. **Cài đặt CPU version** của PyTorch (phù hợp với máy không có GPU)
3. **Bắt đầu với model tiny** để test nhanh
4. **Chạy trên CPU** sẽ chậm nhưng ổn định

**Thời gian dự kiến:**
- Cài đặt PyTorch: 5-10 phút
- Cài đặt SAM2: 10-15 phút  
- Phân đoạn 1 hình ảnh: 30-60 giây

## 🌐 Chạy Web Demo

### Cách 1: Sử dụng Docker (Khuyến nghị)

```bash
# Chạy cả frontend và backend
docker compose up --build
```

**Truy cập:**
- Frontend: http://localhost:7262
- Backend: http://localhost:7263/graphql

### Cách 2: Chạy local

**Cài đặt dependencies cho demo:**
```bash
pip install -e '.[interactive-demo]'
conda install -c conda-forge ffmpeg
```

**Chạy backend:**
```bash
cd demo/backend/server/

PYTORCH_ENABLE_MPS_FALLBACK=1 \
APP_ROOT="$(pwd)/../../../" \
API_URL=http://localhost:7263 \
MODEL_SIZE=base_plus \
DATA_PATH="$(pwd)/../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn \
    --worker-class gthread app:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:7263 \
    --timeout 60
```

**Chạy frontend:**
```bash
cd demo/frontend
yarn install
yarn dev --port 7262
```

## 📊 Hiệu suất các model

| Model | Kích thước | Tốc độ (FPS) | SA-V test (J&F) | MOSE val (J&F) |
|-------|------------|--------------|------------------|-----------------|
| sam2.1_hiera_tiny | 38.9M | 91.2 | 76.5 | 71.8 |
| sam2.1_hiera_small | 46M | 84.8 | 76.6 | 73.5 |
| sam2.1_hiera_base_plus | 80.8M | 64.1 | 78.2 | 73.7 |
| sam2.1_hiera_large | 224.4M | 39.5 | 79.5 | 74.6 |

## 🔧 Xử lý lỗi thường gặp

### 1. Lỗi CUDA
```bash
# Kiểm tra CUDA
python -c 'import torch; print(torch.cuda.is_available())'

# Thiết lập CUDA_HOME nếu cần
export CUDA_HOME=/usr/local/cuda
```

### 2. Lỗi ImportError
```bash
# Cài đặt lại SAM2
pip uninstall -y SAM-2
pip install -e ".[notebooks]"
```

### 3. Lỗi thiếu config
```bash
# Thiết lập PYTHONPATH
export SAM2_REPO_ROOT=/path/to/sam2
export PYTHONPATH="${SAM2_REPO_ROOT}:${PYTHONPATH}"
```

### 4. Lỗi Visual Studio trên Windows
Nếu gặp lỗi Visual Studio không tương thích, thêm flag `-allow-unsupported-compiler` vào file `setup.py` tại dòng 48.

## 🎮 Ví dụ sử dụng

### Phân đoạn đối tượng từ điểm click

```python
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load image
image = cv2.imread("path/to/your/image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set image
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image_rgb)
    
    # Click point (x, y) và label (1 = foreground, 0 = background)
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    
    # Predict mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Hiển thị kết quả
    best_mask = masks[np.argmax(scores)]
    # Visualize mask...
```

### Tracking đối tượng trong video

```python
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor

# Load video predictor
checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Load video
video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize state
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path)
    
    # Add initial prompt
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state, 
        points=[[x, y]], 
        labels=[1]
    )
    
    # Propagate through video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # Process each frame
        print(f"Frame {frame_idx}: {len(object_ids)} objects tracked")
        # Visualize masks...
```

## 📚 Tài liệu tham khảo

- [Paper chính thức](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- [Demo online](https://sam2.metademolab.com/)
- [Dataset SA-V](https://ai.meta.com/datasets/segment-anything-video)
- [Blog Meta AI](https://ai.meta.com/blog/segment-anything-2)

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) để biết cách đóng góp.

## 🔧 Xử lý lỗi thường gặp

### 1. Lỗi CUDA
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Thiết lập CUDA_HOME nếu cần (Windows)
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

### 2. Lỗi ImportError
```bash
# Cài đặt lại SAM2
pip uninstall -y SAM-2
pip install -e ".[notebooks]"
```

### 3. Lỗi thiếu config
```bash
# Thiết lập PYTHONPATH (Windows)
set SAM2_REPO_ROOT=D:\3.Research\3.VisionTransformer\3.Project\sam2
set PYTHONPATH=%SAM2_REPO_ROOT%;%PYTHONPATH%
```

### 4. Lỗi Visual Studio trên Windows
Nếu gặp lỗi Visual Studio không tương thích, thêm flag `-allow-unsupported-compiler` vào file `setup.py` tại dòng 48.

## 📊 So sánh các model

| Model | Kích thước | Tốc độ | Chất lượng | RAM cần | GPU cần |
|-------|------------|--------|------------|---------|---------|
| sam2.1_hiera_tiny | 38.9M | ⚡⚡⚡ | ⭐⭐⭐ | 4GB | GTX 1060+ |
| sam2.1_hiera_small | 46M | ⚡⚡ | ⭐⭐⭐⭐ | 6GB | GTX 1070+ |
| sam2.1_hiera_base_plus | 80.8M | ⚡ | ⭐⭐⭐⭐⭐ | 8GB | RTX 2060+ |
| sam2.1_hiera_large | 224.4M | 🐌 | ⭐⭐⭐⭐⭐ | 12GB | RTX 3070+ |

**Khuyến nghị:**
- **Bắt đầu**: sam2.1_hiera_tiny (nhanh, ít RAM)
- **Sản xuất**: sam2.1_hiera_base_plus (cân bằng tốt nhất)
- **Nghiên cứu**: sam2.1_hiera_large (chất lượng cao nhất)

## 📄 Giấy phép

SAM 2 được phát hành dưới giấy phép [Apache 2.0](LICENSE).

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra [INSTALL.md](INSTALL.md) để xem các lỗi thường gặp
2. Tạo issue trên GitHub repository
3. Tham khảo [FAQ](https://github.com/facebookresearch/sam2/issues)

## 🎉 Kết luận

Bạn đã có hướng dẫn chi tiết để:
- ✅ Thiết lập môi trường Python và cài đặt dependencies
- ✅ Cài đặt SAM2 package
- ✅ Tải xuống các model checkpoints
- ✅ Chạy các ví dụ demo
- ✅ Thiết lập và chạy web demo

SAM2 là một công cụ mạnh mẽ cho việc phân đoạn đối tượng trong hình ảnh và video. Bạn có thể bắt đầu với các notebook ví dụ để làm quen với API, sau đó tích hợp vào các dự án của mình.

**Lưu ý quan trọng:**
- Sử dụng GPU để có hiệu suất tốt nhất
- Model `tiny` phù hợp cho thử nghiệm nhanh
- Model `large` cho chất lượng cao nhất
- Web demo cung cấp giao diện trực quan để test

---

**Chúc bạn thành công với SAM2! 🚀**

*Hướng dẫn này được tùy chỉnh cho máy Windows của bạn với Python 3.12.0*
