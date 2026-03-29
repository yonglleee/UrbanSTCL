#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== DeiT 模型下载和本地使用 =====${NC}"

# 检查 models 目录是否存在
if [ ! -d "models" ]; then
    echo -e "${YELLOW}创建 models 目录${NC}"
    mkdir -p models
fi

# 检查模型文件是否已下载
if [ ! -f "models/deit_tiny_patch16_224.pth" ]; then
    echo -e "${YELLOW}下载 DeiT 模型到本地...${NC}"
    python download_deit.py
else
    echo -e "${GREEN}模型文件已存在: models/deit_tiny_patch16_224.pth${NC}"
fi

# 运行本地模型示例
echo -e "${BLUE}运行使用本地 DeiT 模型的 Grad-CAM 示例...${NC}"
python vit_local_example.py --image-path ./examples/both.png --method gradcam --device cuda

echo -e "${GREEN}完成! 结果保存为: gradcam_cam.jpg${NC}"
