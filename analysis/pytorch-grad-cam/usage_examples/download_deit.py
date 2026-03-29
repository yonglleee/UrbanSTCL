import torch
import os

# 创建保存目录
os.makedirs('models', exist_ok=True)

# 下载模型
print("正在下载 DeiT tiny 模型...")
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

# 保存模型
save_path = 'models/deit_tiny_patch16_224.pth'
torch.save(model.state_dict(), save_path)
print(f"模型已保存到: {save_path}")

# 验证模型结构
print("\n模型结构:")
print(model)
