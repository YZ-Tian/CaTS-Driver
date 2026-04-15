import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
import os
import sys
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import MutationDataset
from model.cats_model import CaTS_Driver

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_FOLDS = 5

TISSUE_NAMES = ['BLCA', 'BRCA', 'CM', 'COREAD', 'DLBC', 'ESCA', 'GBM', 'HC', 'HNSC', 'LGG',
                'LUAD', 'LUSC', 'OV', 'PA', 'PAAD', 'PRAD', 'RCCC', 'STAD', 'THCA', 'UCEC']


# --- 1. 获取模型预测结果 (为集成做准备) ---
def get_single_model_predictions(model, loader):
    model.eval()
    d_preds, t_preds = [], []
    with torch.no_grad():
        for ex, mx, _, _ in loader: # 不需要标签，只需要预测
            d_logits, t_logits = model(ex.to(DEVICE), mx.to(DEVICE))
            d_preds.extend(torch.sigmoid(d_logits).cpu().numpy())
            t_preds.extend(torch.sigmoid(t_logits).cpu().numpy())
    return np.array(d_preds), np.array(t_preds)

# --- 2. 绘图函数 ---
def plot_roc_pr_curves(d_true, d_pred, save_dir="./results/test_results"):
    os.makedirs(save_dir, exist_ok=True)

    # ROC 曲线
    fpr, tpr, _ = roc_curve(d_true, d_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Driver Prediction')
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box') # 确保x轴和y轴比例一致
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "driver_roc_curve_ensemble.png"), dpi=300)
    plt.close()

    # PR 曲线
    precision, recall, _ = precision_recall_curve(d_true, d_pred)
    pr_auc = average_precision_score(d_true, d_pred)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve for Driver Prediction')
    plt.legend(loc="lower left", frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.gca().set_aspect('equal', adjustable='box') # 确保x轴和y轴比例一致
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "driver_pr_curve_ensemble.png"), dpi=300)
    plt.close()
    print(f"ROC and PR curves saved to {save_dir}/")


# --- 3. 主评估流程 (集成) ---
def run_evaluation_ensemble():
    # 使用新的、未经重采样的数据集
    csv_path = "./data/proper_split/test_data_unbalanced.csv"
    esm_path = "./data/proper_split/test_esm2_embeddings_unbalanced.npy"
    macro_path = "./data/proper_split/test_macro_features_unbalanced.npy" 
    
    test_dataset = MutationDataset(csv_path, esm_path, macro_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_d_preds = []
    all_t_preds = []
    all_best_thresholds_per_fold = [] # 存储每个模型的最佳阈值

    for fold in range(NUM_FOLDS):
        model = CaTS_Driver(tissue_num=test_dataset.tissue_num).to(DEVICE)
        model_path = f"./best_models/best_model_fold{fold}.pth" 
        thresholds_path = f"./best_thresholds/best_thresholds_fold{fold}.npy" # 阈值路径
            
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"成功加载模型权重: {model_path}")
        
        d_preds_fold, t_preds_fold = get_single_model_predictions(model, test_loader)
        all_d_preds.append(d_preds_fold)
        all_t_preds.append(t_preds_fold)

        best_thresholds_fold = np.load(thresholds_path)
        all_best_thresholds_per_fold.append(best_thresholds_fold)
        print(f"成功加载最佳阈值: {thresholds_path}")


    # 对所有模型的预测结果进行平均
    ensemble_d_pred = np.mean(all_d_preds, axis=0)
    ensemble_t_pred = np.mean(all_t_preds, axis=0)

    ensemble_optimal_thresholds = np.mean(all_best_thresholds_per_fold, axis=0)

    d_true = test_dataset.driver_labels
    t_true = test_dataset.tissue_labels

    # 打印致病性预测指标
    d_auc = roc_auc_score(d_true, ensemble_d_pred)
    d_aupr = average_precision_score(d_true, ensemble_d_pred)
    print(f"\n--- 致病性预测 (Driver Prediction) (Ensemble) ---")
    print(f"Test Driver AUC: {d_auc:.4f}")
    print(f"Test Driver AUPR: {d_aupr:.4f}")

    # 绘制致病性 ROC 和 PR 曲线
    plot_roc_pr_curves(d_true, ensemble_d_pred)

    # 打印组织特异性预测指标
    tissue_aucs = []
    ensemble_t_pred_binary = (ensemble_t_pred >= ensemble_optimal_thresholds).astype(int)
    t_f1_macro = f1_score(t_true, ensemble_t_pred_binary, average='macro', zero_division=0)


    for i in range(t_true.shape[1]):
        if len(np.unique(t_true[:, i])) > 1:
            auc_score = roc_auc_score(t_true[:, i], ensemble_t_pred[:, i])
            tissue_aucs.append(auc_score)
    t_auc_mean = np.mean(tissue_aucs) if len(tissue_aucs) > 0 else 0.5
    
    print(f"\n--- 组织特异性预测 (Tissue Specificity Prediction) (Ensemble) ---")
    print(f"Test Tissue Macro AUC (using raw probabilities): {t_auc_mean:.4f}")
    print(f"Test Tissue Macro F1 (using optimized thresholds): {t_f1_macro:.4f}")


if __name__ == "__main__":
    run_evaluation_ensemble()
