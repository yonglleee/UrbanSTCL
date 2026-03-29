import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

# 定义计算 MAPE 的函数
def compute_mape_scores(y_true, y_pred, label_columns):
    """
    计算每个指标的 MAPE，忽略 NaN。

    Args:
        y_true (np.ndarray): 真实值，形状为 (n_samples, n_targets)。
        y_pred (np.ndarray): 预测值，形状为 (n_samples, n_targets)。
        label_columns (list): 每个目标列的名称列表。

    Returns:
        dict: 每个指标对应的 MAPE。
    """
    mape_scores = {}
    for i, label in enumerate(label_columns):
        y_true_col = y_true[:, i]
        y_pred_col = y_pred[:, i]

        # 忽略 NaN 值
        mask = ~np.isnan(y_true_col) & ~np.isnan(y_pred_col)
        y_true_cleaned = y_true_col[mask]
        y_pred_cleaned = y_pred_col[mask]

        # 确保数据点足够计算 MAPE
        if len(y_true_cleaned) > 0:  # 至少一个点才能计算 MAPE
            mape_scores[label] = np.mean(np.abs((y_true_cleaned - y_pred_cleaned) / y_true_cleaned))
        else:
            mape_scores[label] = np.nan  # 数据不足时返回 NaN

    return mape_scores


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim):
        super(TransformerRegressor, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 定义 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, 18)  # 回归头，输出 1 个值

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # 映射到 Transformer 的隐藏维度
        x = self.transformer(x)  # Transformer 编码
        x = x.mean(dim=1)  # 平均池化
        x = self.fc(x)  # 回归输出
        return x

# 自定义损失函数，忽略 NaN
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 忽略 NaN 值
        mask = ~torch.isnan(y_true)
        return ((y_pred[mask] - y_true[mask]) ** 2).mean()



def main(args):
    # 设置 GPU ID
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    merged_df = pd.read_csv(args.data_path)

    # 提取特征和目标
    label_columns = [
        "logcrime", "logpetty", "walkbike_per_cbg", "publictrans_per_cbg",
        "drove_alone_per_cbg", "estvmiles", "estpmiles", "estvtrp",
        "estptrp", "obesitycru", "diabetescr", "lpacrudepr", "mhlthcrude",
        "phlthcrude", "cancercrud", "mhincome_cbg", "povertyline_below100",
        "povertyline_below200"
    ]

    feature_columns = [str(i) for i in range(args.num_features)]

    X = merged_df[feature_columns].values  # 特征
    y = merged_df[label_columns].values  # 多目标，可能包含 NaN

    # 数据归一化
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加一个时间维度
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)

    # K-Fold 交叉验证
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_mape_scores = {label: [] for label in label_columns}

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{args.k_folds}")

        # 构建训练集和测试集
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

        # 初始化模型并移动到设备
        model = TransformerRegressor(
            input_dim=X_tensor.size(2),
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim
        ).to(device)

        # 使用自定义损失函数
        criterion = MaskedMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # 训练循环
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                # 将数据移动到设备
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        print(f"Fold {fold+1}, Training Loss: {train_loss / len(train_loader)}")

        # 测试模型
        model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 将数据移动到设备
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                predictions = model(X_batch).cpu().numpy()
                y_true_all.append(y_batch.cpu().numpy())
                y_pred_all.append(predictions)

        # 合并所有批次的结果
        y_true_all = np.vstack(y_true_all)
        y_pred_all = np.vstack(y_pred_all)

        # 计算当前折的 MAPE
        fold_mape = compute_mape_scores(y_true_all, y_pred_all, label_columns)
        for label, mape in fold_mape.items():
            fold_mape_scores[label].append(mape)
            print(f"Fold {fold+1}, {label} MAPE: {mape:.4f}" if not np.isnan(mape) else f"Fold {fold+1}, {label}: NaN")

    # 输出每个目标的平均 MAPE
    print("\nAverage MAPE Scores across all folds:")
    avg_mape_scores = {}
    for label in label_columns:
        valid_scores = [mape for mape in fold_mape_scores[label] if not np.isnan(mape)]
        avg_mape = np.mean(valid_scores) if valid_scores else np.nan
        avg_mape_scores[label] = avg_mape
        print(f"{label}: {avg_mape:.4f}" if not np.isnan(avg_mape) else f"{label}: NaN")

    # 返回总的平均 MAPE
    overall_avg_mape = np.mean([m for m in avg_mape_scores.values() if not np.isnan(m)])
    print(f"\nOverall Average MAPE Score: {overall_avg_mape:.4f}")
    return overall_avg_mape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer for Regression Task with K-Fold Cross Validation")
    parser.add_argument("--data_path", type=str, default='/home/liyong/code/svpretrain/economic/merged_output.csv', help="Path to the dataset CSV file")
    parser.add_argument("--label_column", type=str, default='logpetty', help="Name of the label column")
    parser.add_argument("--num_features", type=int, default=768, help="Number of feature columns")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of the Transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of Transformer encoder layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for K-Fold Cross Validation")
    parser.add_argument("--gpu_id", type=int, default=1, help="ID of the GPU to use")
    args = parser.parse_args()

    avg_mape_score = main(args)
