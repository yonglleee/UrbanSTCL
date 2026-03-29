import argparse
import cv2
import numpy as np
import torch
# mocov3_paper
# import os
# import sys
# import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')

    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='/home/liyong/code/svpretrain/output_dir/checkpoint/mocov3_paper/vit-b-300ep.pth.tar',
        help='Path to model checkpoint file')
        
    parser.add_argument(
        '--output-path',
        type=str,
        default='{method}_cam_results',
        help='Output directory path for saving CAM visualization images. Use {method} as placeholder for the method name.')

    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")


    
    # 导入本地moco_model模块
    from usage_examples.moco_model import vit_base, load_state_dict
    
    # 创建一个包装类来处理模型输出是元组的情况
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
        
        def forward(self, x):
            output = self.model(x)
            # 如果输出是元组，返回第一个元素
            if isinstance(output, tuple):
                return output[0]
            return output
    
    model = vit_base()
    model = model.to(torch.device(args.device))
    model = model.eval()

    # 加载模型权重
    print(f"Loading model from: {args.model_path}")
    state_dict = load_state_dict(args.model_path)
    _ = model.load_state_dict(state_dict, strict=False)
    
    # 使用包装器包装模型
    model = ModelWrapper(model)
    model_mocov3_imagenet = model
    print(model_mocov3_imagenet)

    # 获取模型中的块数量
    import os
    num_blocks = len(model.model.blocks)
    # 决定我们要可视化的块索引
    # 这里我们选择所有块 (从0到num_blocks-1)
    block_indices = list(range(num_blocks))  # 所有块 (从0到11)
    
    print(f"模型共有 {num_blocks} 个块，将可视化以下块的norm1层:")
    for idx in block_indices:
        actual_idx = idx if idx >= 0 else num_blocks + idx
        if 0 <= actual_idx < num_blocks:
            print(f"- 块 {idx} (实际索引: {actual_idx})")

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    # 读取图像并预处理
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]).to(args.device)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None
    
    # 处理每个块
    for block_idx in block_indices:
        # 确保索引在有效范围内
        actual_idx = block_idx if block_idx >= 0 else num_blocks + block_idx
        if actual_idx < 0 or actual_idx >= num_blocks:
            continue
            
        # 为当前块设置目标层
        target_layers = [model.model.blocks[block_idx].norm1]
        print(f"正在处理块 {block_idx} (实际索引: {actual_idx}) 的norm1层")
        
        # 为当前块创建CAM对象
        if args.method == "ablationcam":
            cam = methods[args.method](model=model,
                                     target_layers=target_layers,
                                     reshape_transform=reshape_transform,
                                     ablation_layer=AblationLayerVit())
        else:
            cam = methods[args.method](model=model,
                                     target_layers=target_layers,
                                     reshape_transform=reshape_transform)
                                     
        # 设置批处理大小
        cam.batch_size = 32
        
        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor,
                          targets=targets,
                          eigen_smooth=args.eigen_smooth,
                          aug_smooth=args.aug_smooth)
                          
        # 这里grayscale_cam只有批处理中的一个图像
        grayscale_cam = grayscale_cam[0, :]
        
        # 生成可视化图像
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        
        # 生成输出目录路径
        output_dir = args.output_path.format(method=args.method)
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成包含块信息的文件名
        block_name = f"block{actual_idx}"
        output_filename = f"{args.method}_{block_name}.jpg"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # 保存图像
        cv2.imwrite(output_filepath, cam_image)
        print(f"可视化结果已保存为: {output_filepath}")
