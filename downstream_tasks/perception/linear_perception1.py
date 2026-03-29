import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# 假设 df 是你的 DataFrame，包含 768 维特征 (列名 0-767) 和 'label' 列
# 示例 df，可以替换为实际数据
data = {
    **{str(i): range(100) for i in range(768)},  # 768 维特征
    'label': [0, 1] * 50  # 100 行标签，0 和 1 交替
}
model_name = 'urban2vec_vit_epoch11'
df = pd.read_csv(f'/home/liyong/code/svpretrain/Perception/feature/{model_name}.csv')

# 1. 提取特征和标签
features = df.loc[:, '0':'49'].values  # 特征列 (768 维)
labels = df['label'].values  # 标签列 (0 或 1)

# 2. 使用 train_test_split 将数据划分为训练集和测试集，37% 作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.37, random_state=42
)

# 3. 将数据转换为 PyTorch 的 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 转为列向量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 4. 使用 TensorDataset 打包特征和标签，设置训练集和测试集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 5. 使用 DataLoader 加载数据，设置批量大小 (batch_size)，并打乱训练集
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 6. 定义简单的二分类神经网络模型
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 全连接层，输出1个节点

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(x)  # 二分类的 sigmoid 激活函数

# 7. 初始化模型、损失函数和优化器
input_dim = 50  # 输入维度为 768
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 验证函数
def validate(model, test_loader, epoch, model_name):
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # 转换为 0 或 1 的二分类预测
            all_preds.append(preds)
            all_labels.append(labels)

    # 拼接所有批次的预测值和标签
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    # 打印一行输出
    print(f"Epoch {epoch}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC Score: {auc:.4f}")

    # 创建一个字典保存结果
    results = {
        'Model': [model_name],
        'Epoch': [epoch],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'AUC Score': [auc]
    }

    # 转换为 DataFrame 并追加到 CSV 文件中
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)
    if epoch >15:
        df_results.to_csv('/home/liyong/code/svpretrain/Perception/feature/evaluation_metrics.csv', mode='a', header=False, index=False)

# 训练步骤 (5 个 epoch)
num_epochs = 20

model.train()

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    model.train()  # 切换到训练模式

    # 8. 使用 DataLoader 进行批次训练
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")
    
    # 每个 epoch 后验证模型
    validate(model, test_loader, epoch, model_name)
