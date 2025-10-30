# DeepLabv3+ on Jetson Orin NX - Setup Guide

## Tổng quan
Notebook này đã được tối ưu hóa để chạy trên **Jetson Orin NX ARM64** với các cấu hình sau:
- GPU: 8GB VRAM
- Architecture: ARM64
- CUDA: Jetson native CUDA
- Memory optimization: FP16 mixed precision

## Các thay đổi chính so với phiên bản gốc

### 1. **Cấu hình bộ nhớ (Memory Configuration)**
- **Batch size**: Giảm từ 4 → **2** (phù hợp với 8GB GPU)
- **Input size**: Giảm từ 384 → **320** (tiết kiệm bộ nhớ)
- **Gradient accumulation**: Tăng từ 2 → **4** steps (mô phỏng batch size = 8)
- **Workers**: 2 (tối ưu cho ARM64)

### 2. **Tối ưu hóa Jetson**
- **Mixed Precision (FP16)**: Bắt buộc - tiết kiệm ~50% bộ nhớ
- **Memory fraction**: 85% GPU memory
- **TF32**: Enabled cho hiệu năng tốt hơn
- **CUDA allocation**: `max_split_size_mb:128`

### 3. **Đường dẫn dữ liệu**
- Thay đổi từ Kaggle paths → **local filesystem**
- Data root: `./VOCdevkit/VOC2012/`
- Model save: `./models/best_deeplabv3plus_jetson.pth`

## Yêu cầu hệ thống

### Phần cứng
- Jetson Orin NX (hoặc tương đương)
- 8GB GPU RAM tối thiểu
- 32GB System RAM khuyến nghị
- 50GB dung lượng ổ cứng trống

### Phần mềm
```bash
# JetPack 5.x hoặc mới hơn
# Python 3.8+
# PyTorch for Jetson (ARM64)
# CUDA 11.4+ (Jetson native)
```

## Cài đặt

### 1. Cài đặt PyTorch cho Jetson
```bash
# Tải PyTorch wheel cho Jetson từ NVIDIA
# Ví dụ cho JetPack 5.x:
wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# Cài đặt
pip3 install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```

### 2. Cài đặt dependencies
```bash
pip3 install torchvision albumentations opencv-python-headless
pip3 install numpy matplotlib seaborn Pillow tqdm
```

### 3. Tải dataset PASCAL VOC 2012
```bash
# Tải dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Giải nén
tar -xf VOCtrainval_11-May-2012.tar

# Cấu trúc thư mục nên là:
# ./VOCdevkit/VOC2012/
#   ├── JPEGImages/
#   ├── SegmentationClass/
#   └── ImageSets/Segmentation/
```

## Chạy notebook

### 1. Kết nối SSH đến Jetson
```bash
ssh user@jetson-ip-address
```

### 2. Khởi động Jupyter
```bash
# Cài đặt Jupyter nếu chưa có
pip3 install jupyter

# Khởi động Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Truy cập từ máy tính của bạn:
# http://jetson-ip-address:8888
```

### 3. Chạy từng cell
- Chạy tuần tự từ cell đầu tiên
- Kiểm tra GPU memory sau mỗi cell quan trọng
- Cell 2 sẽ hiển thị thông tin Jetson và cấu hình

## Giám sát hiệu năng

### Kiểm tra GPU usage
```bash
# Terminal riêng để giám sát
watch -n 1 tegrastats
```

### Kiểm tra nhiệt độ
```bash
# Xem nhiệt độ GPU
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### Trong notebook
```python
# Kiểm tra memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

## Tối ưu hóa thêm (nếu cần)

### Nếu gặp lỗi Out of Memory:
1. **Giảm batch size** xuống 1:
   ```python
   CFG['BATCH_SIZE'] = 1
   CFG['GRADIENT_ACCUMULATION_STEPS'] = 8
   ```

2. **Giảm input size** xuống 256:
   ```python
   CFG['CROP_SIZE'] = 256
   ```

3. **Giảm workers**:
   ```python
   CFG['NUM_WORKERS'] = 1
   ```

### Tăng tốc độ training:
1. **Tăng workers** (nếu có đủ RAM):
   ```python
   CFG['NUM_WORKERS'] = 4
   ```

2. **Enable CUDA graphs** (PyTorch 2.0+):
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

## Thông số hiệu năng dự kiến

### Training speed (Jetson Orin NX)
- **~2-3 iterations/second** với batch size 2
- **~15-20 giờ** để hoàn thành 15,000 iterations
- **Memory usage**: ~5-6 GB / 8 GB GPU

### Accuracy
- **mIoU dự kiến**: ~70-75% trên validation set
- Thấp hơn một chút so với paper gốc do:
  - Batch size nhỏ hơn
  - Input resolution thấp hơn
  - Ít iterations hơn

## Troubleshooting

### Lỗi CUDA Out of Memory
```python
# Thêm vào đầu notebook
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

### Lỗi DataLoader workers
```python
# Đặt num_workers = 0 nếu gặp lỗi
CFG['NUM_WORKERS'] = 0
```

### Lỗi Mixed Precision
```python
# Tắt mixed precision nếu không tương thích
CFG['MIXED_PRECISION'] = False
```

### Jetson bị quá nóng
```bash
# Tăng tốc độ quạt
sudo jetson_clocks
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

## Lưu model

Model sẽ tự động được lưu tại:
```
./models/best_deeplabv3plus_jetson.pth
```

Để load model:
```python
model.load_state_dict(torch.load('./models/best_deeplabv3plus_jetson.pth'))
```

## Tài liệu tham khảo

- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [DeepLabv3+ Paper](https://arxiv.org/abs/1802.02611)

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra JetPack version: `sudo apt-cache show nvidia-jetpack`
2. Kiểm tra CUDA version: `nvcc --version`
3. Kiểm tra PyTorch: `python3 -c "import torch; print(torch.__version__)"`
4. Kiểm tra CUDA available: `python3 -c "import torch; print(torch.cuda.is_available())"`

---
**Lưu ý**: Notebook này đã được tối ưu hóa đặc biệt cho Jetson Orin NX ARM64. Các thông số có thể cần điều chỉnh cho các thiết bị Jetson khác.
