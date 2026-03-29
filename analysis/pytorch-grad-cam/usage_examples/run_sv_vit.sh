#!/bin/bash

# 设置默认值
DEFAULT_IMAGE_PATH="/home/liyong/code/svpretrain/cl-vs-mim/data/0/n01440764/Boston_0008555_2017_10_015_42.39519505016474_-71.12174207176281_7vdq9ZeNrrSboVEpeSp1AA.jpg"
DEFAULT_METHOD="gradcam"
DEFAULT_DEVICE="cuda"  # 默认使用 GPU
# DEFAULT_MODEL_PATH="/home/liyong/code/svpretrain/output_dir/checkpoint/mocov3_paper/vit-b-300ep.pth.tar"

#self 
# DEFAULT_MODEL_PATH="/home/liyong/code/svpretrain/output_dir/checkpoint/GSV/new/Mocov3VITB-lr-self-cities10-1m/checkpoint_299.pth.tar" 
#tenporal
#DEFAULT_MODEL_PATH="/home/liyong/code/svpretrain/output_dir/checkpoint/GSV/new/Mocov3VITB-lr-temporal-random-cities10t-1m/checkpoint_299.pth.tar" 
#spatial
# DEFAULT_MODEL_PATH="/home/liyong/code/svpretrain/output_dir/checkpoint/GSV/new/Mocov3VITB-lr-spatial-5k100t-cities10-1m-new/checkpoint_299.pth.tar" 
#la cbg
DEFAULT_MODEL_PATH="/home/huangyj/representation/models/contrastive/GSV/Mocov3VITB-lr-spatial-cbg-LosAngeles/checkpoint_299.pth.tar"

DEFAULT_OUTPUT_PATH="/home/liyong/code/svpretrain/pytorch-grad-cam/outputs/moco1/cbg"  # 输出路径，包含 {method} 占位符

# 解析参数
IMAGE_PATH=${1:-$DEFAULT_IMAGE_PATH}
METHOD=${2:-$DEFAULT_METHOD}
DEVICE=${3:-$DEFAULT_DEVICE}
MODEL_PATH=${4:-$DEFAULT_MODEL_PATH}
OUTPUT_PATH=${5:-$DEFAULT_OUTPUT_PATH}

# 打印参数信息
echo "运行 sv_vit.py 使用以下参数:"
echo "图像路径: $IMAGE_PATH"
echo "方法: $METHOD"
echo "设备: $DEVICE"
echo "模型路径: $MODEL_PATH"
echo "输出路径: $OUTPUT_PATH"

cd /home/liyong/code/svpretrain/pytorch-grad-cam
# 设置PYTHONPATH以解决模块导入问题
export PYTHONPATH=$PYTHONPATH:/home/liyong/code/svpretrain/pytorch-grad-cam

# 执行脚本
python usage_examples/sv_vit.py \
  --image-path "$IMAGE_PATH" \
  --method "$METHOD" \
  --device "$DEVICE" \
  --model-path "$MODEL_PATH" \
  --output-path "$OUTPUT_PATH"

# 检查执行状态
if [ $? -eq 0 ]; then
    # 将占位符替换为实际方法名称
    ACTUAL_OUTPUT_PATH=${OUTPUT_PATH//\{method\}/$METHOD}
    echo "完成! 结果保存在目录: $ACTUAL_OUTPUT_PATH/"
else
    echo "执行过程中出现错误!"
fi
