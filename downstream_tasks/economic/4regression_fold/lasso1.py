import numpy as np
import pandas as pd
import os
import shutil
import joblib
import argparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

def regression(df_target, pca_num, target, output_dir, n_splits=5):
    # Perform K-Fold Cross Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    reg_results = {
        'target': target,
        'alpha': None,
        'avg_mse': None,
        'avg_rmse': None,
        'avg_mae': None,
        'avg_r2': None,
        'avg_overall_r2': None,
        'avg_overall_mse': None,
        'avg_overall_rmse': None,
        'avg_overall_mae': None,
        'avg_train_r2': None,
        'best_mse': None,
        'best_rmse': None,
        'best_mae': None,
        'best_r2': None,
        'mse_scores': [],
        'rmse_scores': [],
        'mae_scores': [],
        'r2_scores': [],
        'train_r2': [],
        'overall_r2': [],
        'overall_mse': [],
        'overall_rmse': [],
        'overall_mae': [],
        'train_samples': [], # Track train sample indices
    }

    best_model = None
    best_r2 = -float('inf')
    best_mse = float('inf')
    best_mae = float('inf')
    best_alpha = None

    for train_index, test_index in kf.split(df_target):
        X_train, X_test = df_target.loc[train_index, range(pca_num)].values, df_target.loc[test_index, range(pca_num)].values
        y_train, y_test = df_target.loc[train_index][target].values, df_target.loc[test_index][target].values

        # Track the indices of the training samples
        train_indices = df_target.index[train_index].tolist()
        reg_results['train_samples'].append(train_indices)

        # Train LassoCV model to find the best alpha
        model = LassoCV(cv=5, random_state=42, precompute=False, max_iter=5000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, model.predict(X_train))
        
        # Overall metrics
        all_y_true = list(y_train) + list(y_test)
        all_y_pred = list(model.predict(X_train)) + list(y_pred)
        r2_overall = r2_score(all_y_true, all_y_pred)
        mse_overall = mean_squared_error(all_y_true, all_y_pred)
        rmse_overall = np.sqrt(mse_overall)
        mae_overall = mean_absolute_error(all_y_true, all_y_pred)

        reg_results['mse_scores'].append(mse)
        reg_results['rmse_scores'].append(rmse)
        reg_results['mae_scores'].append(mae)
        reg_results['r2_scores'].append(r2)
        reg_results['train_r2'].append(train_r2)
        reg_results['overall_r2'].append(r2_overall)
        reg_results['overall_mse'].append(mse_overall)
        reg_results['overall_rmse'].append(rmse_overall)
        reg_results['overall_mae'].append(mae_overall)

        # Save the best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_r2 = r2
            best_mae = mae
            best_alpha = model.alpha_
    
    # Calculate average metrics
    reg_results['alpha'] = best_alpha
    reg_results['best_mse'] = best_mse
    reg_results['best_rmse'] = np.sqrt(best_mse)
    reg_results['best_mae'] = best_mae
    reg_results['best_r2'] = best_r2
    reg_results['avg_mse'] = np.mean(reg_results['mse_scores'])
    reg_results['avg_rmse'] = np.sqrt(reg_results['avg_mse'])
    reg_results['avg_mae'] = np.mean(reg_results['mae_scores'])
    reg_results['avg_r2'] = np.mean(reg_results['r2_scores'])
    reg_results['avg_train_r2'] = np.mean(reg_results['train_r2'])
    reg_results['avg_overall_r2'] = np.mean(reg_results['overall_r2'])
    reg_results['avg_overall_mse'] = np.mean(reg_results['overall_mse'])
    reg_results['avg_overall_rmse'] = np.mean(reg_results['overall_rmse'])
    reg_results['avg_overall_mae'] = np.mean(reg_results['overall_mae'])

    # Save the model
    joblib.dump(best_model, f'{output_dir}/{target}.pkl')

    # Save detailed results
    reg_result_df = pd.DataFrame([reg_results])
    result_path = f'{output_dir}/results.pkl'
    if not os.path.exists(result_path):
        reg_result_df.to_pickle(result_path)
    else:
        existing_data = pd.read_pickle(result_path)
        reg_result_df = pd.concat([existing_data, reg_result_df], ignore_index=True)
        reg_result_df.to_pickle(result_path)

    # Save summary results to CSV
    summary_results = {k: v for k, v in reg_results.items() if not isinstance(v, list)}
    summary_df = pd.DataFrame([summary_results])
    result_path = f'{output_dir}/results.csv'
    if not os.path.exists(result_path):
        summary_df.to_csv(result_path, index=False)
    else:
        summary_df.to_csv(result_path, mode='a', header=False, index=False)

def load_data(label_file, feature_file, output_dir, feature_len):
    # 加载CSV文件
    try:
        df_labels = pd.read_csv(label_file)
        df_features = pd.read_csv(feature_file)
        print(f"成功加载标签文件: {label_file}")
        print(f"成功加载特征文件: {feature_file}")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        raise

    # 确保GEOID列存在
    id_col = 'GEOID'
    if id_col not in df_labels.columns:
        raise ValueError(f"标签文件中找不到{id_col}列")
    if id_col not in df_features.columns:
        raise ValueError(f"特征文件中找不到{id_col}列")
    
    # 转换GEOID为字符串，确保合并时类型匹配
    df_labels[id_col] = df_labels[id_col].astype(str)
    df_features[id_col] = df_features[id_col].astype(str)
    
    # 获取特征列（除了GEOID和path之外的所有列）
    feature_cols = [col for col in df_features.columns if col != id_col and col != 'path']
    print(f"特征文件中包含以下列: {df_features.columns.tolist()}")
    print(f"识别出的特征列: {feature_cols}")
    
    # 限制特征数量
    if len(feature_cols) > feature_len:
        feature_cols = feature_cols[:feature_len]
        print(f"使用前{feature_len}个特征列")
    else:
        print(f"使用所有{len(feature_cols)}个特征列")
    
    # 合并特征和标签
    df = pd.merge(df_labels, df_features[feature_cols + [id_col]], on=id_col, how='inner')
    print(f"合并后的数据集包含{len(df)}行")
    
    # 提取特征矩阵
    X = df[feature_cols].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    print("保存标准化模型")
    
    # PCA降维，保留99%方差
    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X_scaled)
    pca_num = X_pca.shape[1]
    joblib.dump(pca, f'{output_dir}/pca.pkl')
    print(f"应用PCA降维，保留{pca_num}个主成分，解释{pca.explained_variance_ratio_.sum()*100:.2f}%的方差")
    
    # 创建包含PCA结果的DataFrame
    pca_df = pd.DataFrame(X_pca, columns=range(pca_num))
    
    # 移除原始特征列并添加PCA特征
    df = df.drop(columns=feature_cols)
    df = pd.concat([df, pca_df], axis=1)
    
    return df, pca_num

def main(args):
    # 创建输出目录
    feature_name = os.path.basename(args.feature_path).split('.')[0]
    output_dir = f'{args.output_dir}/{feature_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载并处理数据
    df, pca_num = load_data(args.label_path, args.feature_path, output_dir, args.feature_len)
    
    # 解析目标变量
    targets = args.targets.split(',')
    print(f"将对以下{len(targets)}个目标变量进行回归分析: {targets}")
    
    # 对每个目标变量进行回归分析
    for target in tqdm(targets, desc="处理目标变量"):
        # 检查目标变量是否存在
        if target not in df.columns:
            print(f"警告: 标签文件中找不到目标变量 '{target}', 跳过")
            continue
        
        # 删除包含目标变量的缺失值的行
        df_target = df.dropna(subset=[target])
        if len(df_target) < len(df):
            print(f"为目标变量'{target}'删除了{len(df) - len(df_target)}行缺失值")
        
        # 重置索引
        df_target = df_target.reset_index(drop=True)
        
        # 运行回归
        print(f"开始对'{target}'进行回归分析...")
        regression(df_target, pca_num, target, output_dir)
        print(f"已完成'{target}'的回归分析并保存结果")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对多个目标变量进行LASSO回归并计算交叉验证指标')
    parser.add_argument('--feature_path', type=str, required=True, help='特征CSV文件的路径')
    parser.add_argument('--label_path', type=str, required=True, help='标签CSV文件的路径')
    parser.add_argument('--targets', type=str, required=True, help='目标变量列名（用逗号分隔）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--feature_len', type=int, default=50, help='使用的特征数量')
    args = parser.parse_args()
    main(args)