#!/bin/bash

# DeepLabv3+ Training Script for Jetson Orin NX
# This script helps set up and run the training on Jetson

echo "=========================================="
echo "DeepLabv3+ on Jetson Orin NX - Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Jetson device${NC}"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Display Jetson info
echo -e "\n${GREEN}Jetson Information:${NC}"
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
fi

# Check CUDA
echo -e "\n${GREEN}Checking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo -e "${RED}CUDA not found!${NC}"
    exit 1
fi

# Check Python
echo -e "\n${GREEN}Checking Python...${NC}"
python3 --version

# Check PyTorch
echo -e "\n${GREEN}Checking PyTorch...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo -e "${RED}PyTorch not installed or not working!${NC}"
    echo "Please install PyTorch for Jetson first."
    exit 1
}

# Check GPU
echo -e "\n${GREEN}GPU Information:${NC}"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU')"
python3 -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB') if torch.cuda.is_available() else None"

# Check dataset
echo -e "\n${GREEN}Checking dataset...${NC}"
if [ -d "./VOCdevkit/VOC2012" ]; then
    echo -e "${GREEN}✓ Dataset found${NC}"
    
    # Count images
    if [ -d "./VOCdevkit/VOC2012/JPEGImages" ]; then
        img_count=$(ls -1 ./VOCdevkit/VOC2012/JPEGImages/*.jpg 2>/dev/null | wc -l)
        echo "  Images: $img_count"
    fi
    
    # Count masks
    if [ -d "./VOCdevkit/VOC2012/SegmentationClass" ]; then
        mask_count=$(ls -1 ./VOCdevkit/VOC2012/SegmentationClass/*.png 2>/dev/null | wc -l)
        echo "  Masks: $mask_count"
    fi
else
    echo -e "${RED}✗ Dataset not found at ./VOCdevkit/VOC2012${NC}"
    echo -e "${YELLOW}Download instructions:${NC}"
    echo "  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    echo "  tar -xf VOCtrainval_11-May-2012.tar"
    exit 1
fi

# Create necessary directories
echo -e "\n${GREEN}Creating directories...${NC}"
mkdir -p ./models
mkdir -p ./results
echo -e "${GREEN}✓ Directories created${NC}"

# Check dependencies
echo -e "\n${GREEN}Checking dependencies...${NC}"
python3 -c "import albumentations" 2>/dev/null && echo "✓ albumentations" || echo "✗ albumentations (pip3 install albumentations)"
python3 -c "import cv2" 2>/dev/null && echo "✓ opencv" || echo "✗ opencv (pip3 install opencv-python-headless)"
python3 -c "import matplotlib" 2>/dev/null && echo "✓ matplotlib" || echo "✗ matplotlib (pip3 install matplotlib)"
python3 -c "import seaborn" 2>/dev/null && echo "✓ seaborn" || echo "✗ seaborn (pip3 install seaborn)"
python3 -c "import PIL" 2>/dev/null && echo "✓ Pillow" || echo "✗ Pillow (pip3 install Pillow)"
python3 -c "import tqdm" 2>/dev/null && echo "✓ tqdm" || echo "✗ tqdm (pip3 install tqdm)"

# Performance settings
echo -e "\n${GREEN}Setting Jetson performance mode...${NC}"
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    echo -e "${GREEN}✓ Performance mode enabled${NC}"
else
    echo -e "${YELLOW}jetson_clocks not found, skipping${NC}"
fi

# Fan control (optional)
echo -e "\n${YELLOW}Do you want to set fan to maximum? (recommended for training) (y/n)${NC}"
read -r fan_response
if [[ "$fan_response" =~ ^[Yy]$ ]]; then
    if [ -f /sys/devices/pwm-fan/target_pwm ]; then
        sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
        echo -e "${GREEN}✓ Fan set to maximum${NC}"
    else
        echo -e "${YELLOW}Fan control not available${NC}"
    fi
fi

# Display current stats
echo -e "\n${GREEN}Current system stats:${NC}"
tegrastats --interval 1000 --logfile /dev/null &
TEGRA_PID=$!
sleep 2
kill $TEGRA_PID 2>/dev/null

# Ask to start Jupyter
echo -e "\n${GREEN}=========================================="
echo "Setup complete!"
echo "==========================================${NC}"
echo -e "\nOptions:"
echo "  1. Start Jupyter Notebook"
echo "  2. Run training script (if available)"
echo "  3. Exit"
echo -n "Choose option (1-3): "
read -r option

case $option in
    1)
        echo -e "\n${GREEN}Starting Jupyter Notebook...${NC}"
        echo "Access at: http://$(hostname -I | awk '{print $1}'):8888"
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
        ;;
    2)
        echo -e "\n${YELLOW}Training script not implemented yet${NC}"
        echo "Please use Jupyter Notebook to run the training"
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac
