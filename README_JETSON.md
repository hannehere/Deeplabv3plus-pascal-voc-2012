# DeepLabv3+ PASCAL VOC 2012 - Jetson Orin NX Edition

## 📋 Tổng quan

Notebook này đã được **tối ưu hóa đặc biệt** cho **Jetson Orin NX ARM64** với GPU 8GB. Tất cả các cấu hình đã được điều chỉnh để đảm bảo training ổn định và hiệu quả trên nền tảng Jetson.

## 🚀 Các tối ưu hóa chính

### Memory Optimization
- ✅ **Batch size: 2** (thay vì 4-8)
- ✅ **Input resolution: 320x320** (thay vì 384 hoặc 513)
- ✅ **Mixed Precision FP16** (tiết kiệm ~50% memory)
- ✅ **Gradient Accumulation: 4 steps** (effective batch = 8)

### Jetson-Specific Settings
- ✅ **ARM64 optimized workers: 2**
- ✅ **Memory fraction: 85%** (an toàn cho Jetson)
- ✅ **TF32 enabled** (tăng tốc độ)
- ✅ **CUDA allocation optimized**

### Data Configuration
- ✅ **Local filesystem paths** (không dùng Kaggle)
- ✅ **Prefetch factor: 2** (tối ưu I/O)
- ✅ **Pin memory enabled**

## 📦 Yêu cầu

### Hardware
- Jetson Orin NX (hoặc tương đương)
- 8GB GPU RAM minimum
- 32GB System RAM (khuyến nghị)
- 50GB storage space

### Software
```bash
JetPack 5.x hoặc mới hơn
Python 3.8+
PyTorch for Jetson (ARM64 build)
CUDA 11.4+ (Jetson native)
```

## 🔧 Cài đặt nhanh

### 1. Trên Jetson của bạn (qua SSH)

```bash
# Cài đặt PyTorch cho Jetson
# Download từ: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# Hoặc sử dụng pip wheel phù hợp với JetPack version

# Cài đặt dependencies
pip3 install albumentations opencv-python-headless
pip3 install matplotlib seaborn Pillow tqdm jupyter
```

### 2. Tải dataset

```bash
# Tải PASCAL VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Giải nén
tar -xf VOCtrainval_11-May-2012.tar

# Kiểm tra cấu trúc
ls -la VOCdevkit/VOC2012/
```

### 3. Chạy setup script

```bash
# Trên Jetson (Linux)
chmod +x run_on_jetson.sh
./run_on_jetson.sh
```

## 📓 Sử dụng Notebook

### Khởi động Jupyter

```bash
# Trên Jetson
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# Truy cập từ máy tính của bạn
# http://<jetson-ip>:8888
```

### Chạy Training

1. Mở notebook: `deeplabv3plus_complete_reproduction.ipynb`
2. Chạy tuần tự từ cell đầu tiên
3. Cell 2 sẽ hiển thị cấu hình Jetson
4. Cell 3 sẽ kiểm tra dataset
5. Tiếp tục chạy các cell theo thứ tự

### Giám sát Training

Mở terminal riêng để giám sát:

```bash
# Xem GPU stats
watch -n 1 tegrastats

# Xem nhiệt độ
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
```

## 📊 Hiệu năng dự kiến

### Training Speed
- **2-3 iterations/second** (batch size 2)
- **~15-20 giờ** cho 15,000 iterations
- **Memory usage**: 5-6 GB / 8 GB

### Accuracy
- **mIoU**: ~70-75% (validation set)
- Thấp hơn paper gốc do constraints của Jetson

## 🔍 Cấu trúc thư mục

```
.
├── deeplabv3plus_complete_reproduction.ipynb  # Main notebook (Jetson optimized)
├── JETSON_SETUP_GUIDE.md                      # Chi tiết setup
├── README_JETSON.md                           # File này
├── run_on_jetson.sh                           # Setup script
├── VOCdevkit/                                 # Dataset
│   └── VOC2012/
│       ├── JPEGImages/
│       ├── SegmentationClass/
│       └── ImageSets/
├── models/                                    # Saved models
│   └── best_deeplabv3plus_jetson.pth
└── results/                                   # Training results
```

## ⚙️ Các thông số có thể điều chỉnh

### Nếu gặp Out of Memory:

```python
# Trong cell 2 của notebook
CFG['BATCH_SIZE'] = 1                    # Giảm xuống 1
CFG['GRADIENT_ACCUMULATION_STEPS'] = 8   # Tăng lên 8
CFG['CROP_SIZE'] = 256                   # Giảm resolution
CFG['NUM_WORKERS'] = 1                   # Giảm workers
```

### Để tăng tốc độ (nếu có đủ memory):

```python
CFG['BATCH_SIZE'] = 4                    # Tăng lên 4
CFG['CROP_SIZE'] = 384                   # Tăng resolution
CFG['NUM_WORKERS'] = 4                   # Tăng workers
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

### Jetson quá nóng
```bash
# Tăng tốc độ quạt
sudo jetson_clocks
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

### DataLoader errors
```python
# Đặt num_workers = 0
CFG['NUM_WORKERS'] = 0
```

### PyTorch không nhận GPU
```bash
# Kiểm tra
python3 -c "import torch; print(torch.cuda.is_available())"

# Nếu False, cài lại PyTorch cho Jetson
```

## 📈 Monitoring Commands

```bash
# GPU usage
tegrastats

# Temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Memory
free -h

# Disk space
df -h

# CPU usage
htop
```

## 💾 Lưu và Load Model

### Lưu model
```python
# Tự động lưu trong training loop
# Path: ./models/best_deeplabv3plus_jetson.pth
```

### Load model
```python
import torch
model.load_state_dict(torch.load('./models/best_deeplabv3plus_jetson.pth'))
model.eval()
```

## 📚 Tài liệu tham khảo

- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [DeepLabv3+ Paper](https://arxiv.org/abs/1802.02611)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

## ✅ Checklist trước khi training

- [ ] JetPack đã cài đặt đúng version
- [ ] PyTorch for Jetson đã cài đặt
- [ ] CUDA available = True
- [ ] Dataset đã download và giải nén
- [ ] Đủ dung lượng ổ cứng (>50GB)
- [ ] Fan đã set maximum
- [ ] `jetson_clocks` đã chạy
- [ ] Jupyter notebook đã khởi động
- [ ] Có thể truy cập qua browser

## 🎯 Quick Start Commands

```bash
# 1. SSH vào Jetson
ssh user@jetson-ip

# 2. Clone/copy project
cd /path/to/project

# 3. Download dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

# 4. Setup performance
sudo jetson_clocks
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

# 5. Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

# 6. Truy cập từ browser
# http://jetson-ip:8888
```

## 📞 Support

Nếu gặp vấn đề:

1. Kiểm tra JetPack: `sudo apt-cache show nvidia-jetpack`
2. Kiểm tra CUDA: `nvcc --version`
3. Kiểm tra PyTorch: `python3 -c "import torch; print(torch.__version__)"`
4. Xem logs trong notebook output
5. Tham khảo `JETSON_SETUP_GUIDE.md` để biết thêm chi tiết

---

**Lưu ý**: Notebook này được tối ưu hóa đặc biệt cho **Jetson Orin NX ARM64 8GB**. Các thiết bị Jetson khác có thể cần điều chỉnh thông số.

**Tác giả**: Optimized for Jetson Orin NX  
**Version**: 1.0 - Jetson Edition  
**Last Updated**: 2025
