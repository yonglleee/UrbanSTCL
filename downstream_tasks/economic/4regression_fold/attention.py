import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os, shutil
from tqdm import tqdm
import time, math
import neptune

def main(args):
    feature_name = os.path.basename(args.feature_path).split('.')[0]
    output_dir = f'{args.output_dir}/Attn_{feature_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(args.label_path, args.feature_path, output_dir, args.feature_len)
    targets = args.targets.split(',')

    paramerters = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'warmup_epochs': args.warmup_epochs,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'device': args.device,
        'feature_len': args.feature_len,
        'weight_decay': args.weight_decay
    }

    print(paramerters)
    
    for target in tqdm(targets):
        if target not in df.columns:
            print(f'{target} not in the label file')
            continue
        paramerters['target'] = target
        df_target = df.dropna(subset=[target])
        df_target = df_target.reset_index(drop=True)

        regression(df_target, output_dir, paramerters)

############## modules ##############

def load_data(label_file, feature_file, output_dir, feature_len):
    df = pd.read_pickle(label_file)
    features = pd.read_pickle(feature_file)
    df['GEOID'] = df['GEOID'].astype(int)
    features['GEOID'] = features['GEOID'].astype(int)

    df = df.merge(features, on='GEOID', how='inner')
    print(f'Loaded {len(df)} samples')
    X = df[list(range(feature_len))].values
    # Standardize the features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
        # Apply PCA to retain 99% of variance
   
    df.drop(list(range(feature_len)), axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(X_scaled)], axis=1)

    # scaler2 = StandardScaler()
    # X_scaled2 = scaler2.fit_transform(X_pca)

    return df

def regression(df, output_dir, paramerters, n_splits=5):
    # Perform K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'r2': [],
        'mse': [],
        'mae': []
    }

    for i, (train_index, test_index) in enumerate(kf.split(df)):

        X_train, X_test = df.loc[train_index, list(range(paramerters['feature_len']))].values, df.loc[test_index, list(range(paramerters['feature_len']))].values
        y_train, y_test = df.loc[train_index][paramerters['target']].values, df.loc[test_index][paramerters['target']].values

        train_dataset = FeatureDataset(X_train, y_train)
        test_dataset = FeatureDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=paramerters['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=paramerters['batch_size'], shuffle=False)

        # Track the indices of the training samples
        run = neptune.init_run(
            project="huangyj/Regression-SDG",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYmUwMGUzNy0wMmEzLTQ1NmEtOTQyYy1jZmJiNTM3OTFkZTcifQ==",
        )
        paramerters['fold'] = i
        paramerters['train_samples'] = df.loc[train_index, "GEOID"].tolist()
        run["parameters"] = paramerters

        # Train attention model
        # model = SimpleAttentionModel(paramerters['feature_len']).to(paramerters['device'])
        model = TransformerModel(paramerters['feature_len']).to(paramerters['device'])
        from torchsummary import summary
        summary(model, input_size=(1, 768))
        input("Press Enter to continue...")
        # 选择损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=paramerters['lr'], weight_decay=paramerters['weight_decay'])

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=paramerters['patience'], delta=0.005)
        save_path = f'{output_dir}/{paramerters["target"]}_fold_{i}.pth.tar'

        for epoch in range(paramerters['num_epochs']):
            train(model, train_loader, criterion, optimizer, paramerters['device'], epoch, run, paramerters)
            y_true, y_pred, val_loss = evaluate(model, test_loader, criterion, paramerters['device'], run)
            mae, mse, r2 = metrics(y_true, y_pred, run)

            # earlsy stopping
            early_stopping(val_loss, model, save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        results['r2'].append(r2)
        results['mse'].append(mse)
        results['mae'].append(mae)
    save_results(results, output_dir, paramerters)
    
def save_results(results, output_dir, paramerters):
    res_ls = [paramerters['target'], np.mean(results['r2']), np.mean(results['mse']), np.mean(results['mae'])] + results['r2'] + results['mse'] + results['mae']
    res_ls = np.array(res_ls).reshape(1, -1)
    results = pd.DataFrame(res_ls, columns=[
        'target', 'r2', 'mse', 'mae', 
        'r2_1', 'r2_2', 'r2_3', 'r2_4', 'r2_5',
        'mse_1', 'mse_2', 'mse_3', 'mse_4', 'mse_5',
        'mae_1', 'mae_2', 'mae_3', 'mae_4', 'mae_5'
        ])
    result_path = f'{output_dir}/results.csv'
    if os.path.exists(result_path):
        results.to_csv(result_path, mode='a', header=False, index=False)
    else:
        results.to_csv(result_path, index=False)

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, save_path='checkpoint.pth.tar'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

def metrics(y_true, y_pred, run):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    run['test/mae'].append(mae)
    run['test/mse'].append(mse)
    run['test/r2'].append(r2)
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')
    return mae, mse, r2

def evaluate(model, test_loader, criterion, device, run):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            loss = criterion(output.squeeze(), labels.float())
            losses.update(loss.item(), features.size(0))
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(output.squeeze(1).cpu().numpy())
    run['test/loss'].append(losses.avg)
    return y_true, y_pred, losses.avg

def train(model, train_loader, criterion, optimizer, device, epoch, run, paramerters):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, (features, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, paramerters)
        learning_rates.update(lr)

        features, labels = features.to(device), labels.to(device)
        output = model(features)
        
        optimizer.zero_grad()
        loss = criterion(output.squeeze(), labels.float())

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), features.size(0))
        learning_rates.update(optimizer.param_groups[0]['lr'])

        if i % 10 == 0:
            progress.display(i)

    if run is not None:
        run['train/loss'].append(losses.avg)
        run['train/lr'].append(learning_rates.avg)

def adjust_learning_rate(optimizer, epoch, paramerters):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < paramerters['warmup_epochs']:
        lr = paramerters['lr'] * epoch / paramerters['warmup_epochs']
    else:
        lr = paramerters['lr'] * 0.5 * (1. + math.cos(math.pi * (epoch - paramerters['warmup_epochs']) / (paramerters['num_epochs'] - paramerters['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class FeatureDataset(Dataset):
    def __init__(self, features, labels=None):
        """
        :param features: 特征数据，形状应为 (N, D)，N是样本数，D是特征维度
        :param labels: 可选标签数据，形状应为 (N,)，如果没有标签，可以设置为 None
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]

# Custom Attention Layer (Optional for this case)
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def forward(self, x):
        # Input x shape: [batch_size, 1, feature_dim]
        # Attention weights (not necessary for single vector, but kept for illustration)
        attention_weights = torch.softmax(torch.matmul(x, x.transpose(-1, -2)), dim=-1)  # Attention weights
        output = torch.matmul(attention_weights, x)  # Weighted sum of the input
        return output

class SimpleAttentionModel(nn.Module):
    def __init__(self, feature_dim=768, hideen_dim=256, dropout_rate=0.5):
        super(SimpleAttentionModel, self).__init__()
        # Define Attention Layer (optional for single vector inputs)
        self.attention = AttentionLayer()

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(feature_dim, hideen_dim)  # Adjust based on the input feature_dim
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
        self.fc2 = nn.Linear(hideen_dim, 1)  # Output layer for regression

    def forward(self, x):
        # Forward through the attention layer (optional, as it's not very useful with seq_len=1)
        attention_output = self.attention(x)  # Shape: [batch_size, 1, feature_dim]

        # If attention is skipped, you can directly use `x` as the input
        attention_output = attention_output.squeeze(1)  # Shape: [batch_size, feature_dim]

        # Forward through fully connected layers with Dropout
        x = F.relu(self.fc1(attention_output))  # Shape: [batch_size, 64]
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x) 
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, timesteps=64, num_heads=6, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        
        self.timesteps = timesteps
        self.feature_dim = input_dim // timesteps
        
        # Transformer 层：多头自注意力
        self.attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True)
        
        # 后续的全连接层
        self.fc1 = nn.Linear(self.feature_dim * timesteps, 256)
        # self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # BatchNormalization 和 Dropout
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 处理输入数据
        x = x.view(-1, self.timesteps, self.feature_dim)  # 调整输入形状
        
        # Transformer 注意力层
        attn_output, _ = self.attention(x, x, x)  # 这里不使用注意力输出的第三个参数
        
        # 将 attention 输出展平
        x = attn_output.flatten(start_dim=1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        # 全连接层及其后处理
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the features pickle file')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the y-labels pickle file')
    parser.add_argument('--targets', type=str, required=True, help='target columns')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--feature_len', type=int, default=768, help='feature length')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
    parser.add_argument('--weight_decay', default=1e-3, type=float, metavar='W',)
    args = parser.parse_args()
    main(args)