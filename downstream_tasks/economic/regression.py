import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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
        self.fc = nn.Linear(hidden_dim, 1)  # 回归头，输出 1 个值

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # 映射到 Transformer 的隐藏维度
        x = self.transformer(x)  # Transformer 编码
        x = x.mean(dim=1)  # 平均池化
        x = self.fc(x)  # 回归输出
        return x


# 主函数
def main(args):
    # 设置 GPU ID
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 加载数据
    merged_df = pd.read_csv(args.data_path)

    # 提取特征和目标
    label_column = args.label_column
    print(label_column)

    # 检查并移除目标列中的 NaN
    if merged_df[label_column].isna().any():
        print(f"The column '{label_column}' contains NaN values. Removing rows with NaN...")
        merged_df = merged_df.dropna(subset=[label_column])
        print(f"After removing NaN, the dataset has {len(merged_df)} rows.")
    else:
        print(f"The column '{label_column}' does not contain NaN values.")



    feature_columns = [str(i) for i in range(args.num_features)]

    X = merged_df[feature_columns].values  # 转换为 NumPy 数组
    y = merged_df[label_column].values.reshape(-1, 1)

    # # # 数据归一化
    # scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    # X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # 添加一个时间维度
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)

    # K-Fold 交叉验证
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    r2_scores = []

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

        criterion = nn.MSELoss()
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
                # print(y_pred)

        print(f"Fold {fold+1}, Training Loss: {train_loss / len(train_loader)}")

        # 测试模型
        model.eval()
        y_true, y_pred_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 将数据移动到设备
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                predictions = model(X_batch).squeeze().cpu().numpy()
                y_true.extend(y_batch.squeeze().cpu().numpy())
                y_pred_list.extend(predictions)

        # 计算 R^2
        r2 = r2_score(y_true, y_pred_list)
        r2_scores.append(r2)
        print(f"Fold {fold+1}, R^2 Score: {r2}")

    # 输出平均 R^2
    print(f"Average R^2 Score across {args.k_folds} folds: {sum(r2_scores) / len(r2_scores)}")
    return sum(r2_scores) / len(r2_scores)

# 设置命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer for Regression Task with K-Fold Cross Validation")
    parser.add_argument("--data_path", type=str, default='/home/liyong/code/svpretrain/economic/merged_output.csv', help="Path to the dataset CSV file")
    parser.add_argument("--label_column", type=str, default='logpetty', help="Name of the label column")
    parser.add_argument("--num_features", type=int, default=768, help="Number of feature columns")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of the Transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of Transformer encoder layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for K-Fold Cross Validation")
    parser.add_argument("--gpu_id", type=int, default=1, help="ID of the GPU to use")
    args = parser.parse_args()

    # main(args)

        # 多目标列训练
    label_columns = [
        "logcrime", "logpetty", "walkbike_per_cbg", "publictrans_per_cbg",
        "drove_alone_per_cbg", "estvmiles", "estpmiles", "estvtrp",
        "estptrp", "obesitycru", "diabetescr", "lpacrudepr", "mhlthcrude",
        "phlthcrude", "cancercrud", "mhincome_cbg", "povertyline_below100",
        "povertyline_below200"
    ]
    label_columns = [
        "lpacrudepr"
    ]
    results = []
    for label in label_columns:
        print(f"\n=== Training for label: {label} ===")
        args.label_column = label
        avg_r2_score = main(args)
        results.append({"Label": label, "Average R²": avg_r2_score})

        # 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('output_csv.scv', index=False)
    # print(f"Results saved to {}")
