import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os, shutil
from tqdm import tqdm
import argparse

def regression(df_target, pca_num, target, output_dir, n_splits=5):
    # Perform K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
        train_indices = df_target.loc[train_index, "GEOID"].tolist()
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
        # overall metrics
        all_y_true = list(y_train) + list(y_test)
        all_y_pred = list(y_train) + list(y_pred)
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

    reg_result_df = pd.DataFrame(reg_results)
    result_path = f'{output_dir}/results.pkl'
    if not os.path.exists(result_path):
        reg_result_df.to_pickle(result_path)
    else:
        existing_data = pd.read_pickle(result_path)
        reg_result_df = pd.concat([existing_data, reg_result_df], ignore_index=True)
        reg_result_df.to_pickle(result_path)

    reg_results.pop('mse_scores')
    reg_results.pop('rmse_scores')
    reg_results.pop('mae_scores')
    reg_results.pop('r2_scores')
    reg_results.pop('train_r2')
    reg_results.pop('train_samples')
    reg_results.pop('overall_r2')
    reg_results.pop('overall_mse')
    reg_results.pop('overall_rmse')
    reg_results.pop('overall_mae')
    reg_result_df = pd.DataFrame({key: [value] for key, value in reg_results.items()})
    result_path = f'{output_dir}/results.csv'
    if not os.path.exists(result_path):
        reg_result_df.to_csv(result_path, index=False)
    else:
        reg_result_df.to_csv(result_path, mode='a', header=False, index=False)

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
    pca = PCA(n_components=0.99)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(pca, f'{output_dir}/pca.pkl')
    pca_num = X_pca.shape[1]
    df.drop(list(range(feature_len)), axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(X_pca)], axis=1)

    # scaler2 = StandardScaler()
    # X_scaled2 = scaler2.fit_transform(X_pca)

    return df, pca_num

def main(args):
    feature_name = os.path.basename(args.feature_path).split('.')[0]
    output_dir = f'{args.output_dir}/{feature_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df, pca_num = load_data(args.label_path, args.feature_path, output_dir, args.feature_len)

    targets = args.targets.split(',')
    
    for target in tqdm(targets):

        if target not in df.columns:
            print(f'{target} not in the label file')
            continue

        df_target = df.dropna(subset=[target])
        df_target = df_target.reset_index(drop=True)

        regression(df_target, pca_num, target, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True, help='Path to the features pickle file')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the y-labels pickle file')
    parser.add_argument('--targets', type=str, required=True, help='target columns')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--feature_len', type=int, default=768, help='feature length')
    args = parser.parse_args()
    main(args)