# 🏙️ DeepLabv3+ Semantic Segmentation - Multi-Dataset Implementation

Complete implementation of **DeepLabv3+ semantic segmentation** for both **PASCAL VOC 2012** and **Cityscapes** datasets, optimized for different hardware configurations.

## 📊 **Supported Datasets**

### 🎯 PASCAL VOC 2012
- **Classes**: 21 semantic classes (20 objects + background)
- **Resolution**: 513×513 (optimized for T4 GPU)
- **Target**: Academic research and prototyping
- **Hardware**: T4 GPU with memory optimization

### 🏙️ Cityscapes
- **Classes**: 19 urban scene classes
- **Resolution**: 769×769 (high-detail urban scenes)  
- **Target**: Production-grade urban scene understanding
- **Hardware**: TPU v5e-8 with massive parallelization

## 🚀 **Quick Start**

### Option 1: PASCAL VOC 2012 (T4 GPU)
```bash
# Open the PASCAL VOC notebook
jupyter notebook deeplabv3plus_complete_reproduction.ipynb
```

### Option 2: Cityscapes (TPU v5e-8)
```bash
# Open the Cityscapes notebook (for Kaggle TPU)
jupyter notebook deeplabv3plus_cityscapes_reproduction.ipynb
```

### Option 3: Standalone Training
```bash
# Run standalone training script
python main.py
```

## 📁 **Project Structure**

```
Deeplabv3plus-pascal-voc-2012/
├── 🎯 PASCAL VOC Implementation
│   ├── deeplabv3plus_complete_reproduction.ipynb    # Complete PASCAL VOC notebook
│   └── main.py                                      # Standalone training script
├── 🏙️ Cityscapes Implementation  
│   └── deeplabv3plus_cityscapes_reproduction.ipynb  # TPU-optimized Cityscapes notebook
├── 📚 Documentation
│   ├── README.md                                    # This file
│   └── requirements.txt                             # Dependencies
└── 📄 Project Files
    └── LICENSE                                      # MIT License
```

## ⚡ **Key Features**

### 🔥 **Hardware Optimizations**
- **T4 GPU**: Memory-efficient training with gradient accumulation and mixed precision (FP16)
- **TPU v5e-8**: Massive batch training with bfloat16 and distributed processing
- **CPU Fallback**: Full compatibility for development without accelerators

### 🧠 **Model Architecture**
- **Backbone**: ResNet-101 with pretrained ImageNet weights
- **Decoder**: DeepLabv3+ ASPP + decoder architecture  
- **Output**: Pixel-wise classification for semantic segmentation
- **Loss**: CrossEntropyLoss with ignore_index for unlabeled pixels

### 📊 **Data Pipeline**
- **Automatic Download**: kagglehub integration for seamless dataset access
- **Smart Augmentation**: Albumentations with dataset-specific strategies
- **Label Remapping**: Critical Cityscapes labelIds→trainIds conversion
- **Memory Optimization**: Efficient data loading for large-scale training

### 📈 **Training Features**
- **Learning Rate**: Polynomial decay with warmup for large batch training
- **Mixed Precision**: FP16 (GPU) and bfloat16 (TPU) for memory efficiency
- **Evaluation**: Standard mIoU metrics with per-class analysis
- **Checkpointing**: Automatic best model saving and emergency recovery

## 🎯 **Expected Results**

| Dataset | Resolution | Hardware | Expected mIoU | Training Time |
|---------|------------|----------|---------------|---------------|
| PASCAL VOC 2012 | 513×513 | T4 GPU | 70-75% | ~6-8 hours |
| Cityscapes | 769×769 | TPU v5e-8 | 70-75% | ~2-3 hours |

## 📦 **Installation**

### 1. Clone Repository
```bash
git clone https://github.com/hannehere/Deeplabv3plus-pascal-voc-2012.git
cd Deeplabv3plus-pascal-voc-2012
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Additional TPU Requirements (for Cityscapes)
```bash
# For Kaggle TPU environment
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## 🔧 **Configuration**

### PASCAL VOC 2012 Settings
```python
CFG = {
    'NUM_CLASSES': 21,
    'CROP_SIZE': 513, 
    'BATCH_SIZE': 4,          # T4 GPU optimized
    'MAX_ITERATIONS': 30000,
    'MIXED_PRECISION': True,   # FP16
    'GRADIENT_ACCUMULATION': 2
}
```

### Cityscapes Settings  
```python
CFG = {
    'NUM_CLASSES': 19,
    'CROP_SIZE': 769,
    'BATCH_SIZE': 32,         # TPU v5e-8 optimized  
    'MAX_ITERATIONS': 40000,
    'MIXED_PRECISION': True,   # bfloat16
    'TPU_CORES': 8
}
```

## 📊 **Datasets**

### PASCAL VOC 2012
- **Auto-download**: Via kagglehub integration
- **Classes**: `background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor`
- **Format**: JPG images + PNG segmentation masks

### Cityscapes  
- **Auto-download**: Via kagglehub integration
- **Classes**: `road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`
- **Format**: PNG images + PNG labelIds (requires remapping)

## 🎨 **Visualization**

Both notebooks include comprehensive visualization:
- **Training Progress**: Loss curves, mIoU progression, learning rate schedules
- **Inference Results**: Side-by-side comparison of predictions vs ground truth
- **Color-coded Masks**: Beautiful semantic segmentation visualization
- **Per-class Analysis**: Detailed IoU breakdown for each semantic class

## 🔬 **Advanced Features**

### Memory Optimization
- **Gradient Accumulation**: Simulate larger batch sizes on limited memory
- **Mixed Precision**: Reduce memory usage while maintaining accuracy
- **Dynamic Batch Sizing**: Automatic adjustment based on available memory

### TPU Optimization
- **Distributed Training**: Multi-core TPU parallelization
- **XLA Compilation**: Optimized computation graphs
- **bfloat16 Native**: TPU-native mixed precision

### Error Handling
- **Emergency Checkpoints**: Automatic recovery from training interruptions
- **Memory Monitoring**: OOM detection and graceful degradation
- **Comprehensive Logging**: Detailed progress tracking and debugging

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **DeepLabv3+**: Original paper by Chen et al.
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities  
- **Albumentations**: Advanced image augmentation
- **kagglehub**: Seamless dataset integration
- **PASCAL VOC**: Classic semantic segmentation benchmark
- **Cityscapes**: Urban scene understanding dataset

## 📞 **Support**

For questions and support:
- 📧 Create an issue in this repository
- 📚 Check the comprehensive notebook documentation
- 🔍 Review the inline code comments and explanations

---

**⭐ Star this repository if it helps your semantic segmentation projects!** 🚀
