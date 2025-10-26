# DeepLabv3+ for Pascal VOC 2012 Semantic Segmentation

PyTorch implementation of DeepLabv3+ for semantic segmentation on Pascal VOC 2012 dataset.

## Project Setup

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Configuration
All configuration parameters are centralized in the `CFG` dictionary in `main.py`:
- **Device**: Auto-detects CUDA/CPU
- **Dataset**: Pascal VOC 2012 paths and parameters
- **Model**: ResNet101 backbone with output stride 16
- **Training**: Batch size, learning rate, and optimization parameters

### Quick Start
```bash
python main.py
```

This will display the current configuration and confirm GPU availability.
Implementing DeepLabv3+ on PASCAL VOC 2012 for Deep Learning course
