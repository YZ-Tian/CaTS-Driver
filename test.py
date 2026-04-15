import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# 设置全局绘图参数以生成精美可发表的图表
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
import os
import sys
import ast

# 确保路径配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import MutationDataset
from model.cats_model import CaTS_Driver

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_FOLDS = 5 # 交叉验证的折数

# 假设您的癌种名称列表 (顺序要和 tissue_vector 匹配)
# 请替换为您的实际癌种名称，确保与训练时一致
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


def plot_tissue_heatmap(t_true, t_pred_binary, tissue_names, save_dir="./results/test_results"):
    os.makedirs(save_dir, exist_ok=True)

    # 注意：传入的 t_pred_binary 已经是应用了在验证集上找到的优化阈值之后的结果
    # 这样可以确保评估的公平性，避免在测试集上泄露信息

    tissue_recall = np.zeros(t_true.shape[1])
    tissue_precision = np.zeros(t_true.shape[1])
    tissue_f1 = np.zeros(t_true.shape[1])
    tissue_support = np.sum(t_true, axis=0) # 每个癌种的真实正样本数
    
    for i in range(t_true.shape[1]):
        tp = np.sum((t_true[:, i] == 1) & (t_pred_binary[:, i] == 1))
        fp = np.sum((t_true[:, i] == 0) & (t_pred_binary[:, i] == 1))
        fn = np.sum((t_true[:, i] == 1) & (t_pred_binary[:, i] == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        tissue_recall[i] = recall
        tissue_precision[i] = precision
        tissue_f1[i] = f1
    
    # 将 Support 添加到行标签中
    heatmap_labels = [f"{name} (n={int(sup)})" for name, sup in zip(tissue_names, tissue_support)]

    metrics_df = pd.DataFrame({
        'Tissue': heatmap_labels,
        'Recall': tissue_recall,
        'Precision': tissue_precision,
        'F1-Score': tissue_f1
    })
    metrics_df = metrics_df.set_index('Tissue')

    plt.figure(figsize=(15, 8)) # 调整图表大小
    sns.heatmap(metrics_df[['Recall', 'Precision', 'F1-Score']].T, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5, cbar_kws={'label': 'Score'})
    plt.title('Tissue-Specific Performance on Test Set (Ensemble)') # 简化标题
    plt.ylabel('Metric')
    plt.xlabel('Tissue (Number of Positive Samples)') # 简化标签
    plt.xticks(rotation=60, ha='right') # 调整标签旋转角度
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tissue_performance_heatmap_ensemble_optimized_thresholds.png"), dpi=300)
    plt.close()
    print(f"Tissue performance heatmap saved to {save_dir}/")

    # 绘制真实 vs 预测的混淆热力图
    confusion_matrix_per_class = np.zeros((t_true.shape[1], t_true.shape[1]))
    for i in range(t_true.shape[1]): 
        true_indices_for_i = np.where(t_true[:, i] == 1)[0]
        if len(true_indices_for_i) > 0:
            predicted_for_i = t_pred_binary[true_indices_for_i, :]
            for j in range(t_true.shape[1]):
                confusion_matrix_per_class[i, j] = np.sum(predicted_for_i[:, j] == 1) / len(true_indices_for_i)

    plt.figure(figsize=(16, 14)) # 调整图表大小
    sns.heatmap(confusion_matrix_per_class, annot=True, cmap="Blues", fmt=".2f", linewidths=.5,
                xticklabels=heatmap_labels, yticklabels=heatmap_labels, cbar_kws={'label': 'Proportion of Predicted Tissue'})
    plt.title('Normalized Confusion Matrix for Tissue Specificity (Ensemble)') # 简化标题
    plt.xlabel('Predicted Tissue')
    plt.ylabel('Actual Tissue')
    plt.xticks(rotation=60, ha='right') # 调整标签旋转角度
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tissue_actual_vs_predicted_heatmap_ensemble_optimized_thresholds.png"), dpi=300)
    plt.close()
    print(f"Tissue actual vs. predicted heatmap saved to {save_dir}/")


def plot_tissue_metrics_charts(t_true, t_pred_binary, tissue_names, save_dir="./results/test_results"):
    os.makedirs(save_dir, exist_ok=True)

    tissue_recall = np.zeros(t_true.shape[1])
    tissue_precision = np.zeros(t_true.shape[1])
    tissue_f1 = np.zeros(t_true.shape[1])
    tissue_support = np.sum(t_true, axis=0)

    for i in range(t_true.shape[1]):
        tp = np.sum((t_true[:, i] == 1) & (t_pred_binary[:, i] == 1))
        fp = np.sum((t_true[:, i] == 0) & (t_pred_binary[:, i] == 1))
        fn = np.sum((t_true[:, i] == 1) & (t_pred_binary[:, i] == 0))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        tissue_recall[i] = recall
        tissue_precision[i] = precision
        tissue_f1[i] = f1

    metrics_df = pd.DataFrame({
        'Tissue': tissue_names,
        'Recall': tissue_recall,
        'Precision': tissue_precision,
        'F1-Score': tissue_f1,
        'Support': tissue_support
    })

    # Sort by Support in descending order
    metrics_df = metrics_df.sort_values(by='Support', ascending=False).reset_index(drop=True)
    
    # 将Support添加到X轴标签中
    metrics_df['Tissue_Label'] = metrics_df.apply(lambda row: f"{row['Tissue']} (n={int(row['Support'])})", axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True) # 3 rows, 1 column, shared X-axis

    # Plot Recall
    sns.barplot(x='Tissue_Label', y='Recall', data=metrics_df, palette='viridis', ax=axes[0])
    axes[0].set_title('(A) Recall')
    axes[0].set_ylabel('Recall')
    axes[0].set_ylim(0, 1) # Recall values are between 0 and 1

    # Plot Precision
    sns.barplot(x='Tissue_Label', y='Precision', data=metrics_df, palette='magma', ax=axes[1])
    axes[1].set_title('(B) Precision')
    axes[1].set_ylabel('Precision')
    axes[1].set_ylim(0, 1) # Precision values are between 0 and 1

    # Plot F1-Score
    sns.barplot(x='Tissue_Label', y='F1-Score', data=metrics_df, palette='cividis', ax=axes[2])
    axes[2].set_title('(C) F1-Score')
    axes[2].set_xlabel('Tissue (Sample Size)')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim(0, 1)

    # Rotate x-axis labels for better readability on the last subplot
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('') # Remove x-label from individual subplots

    plt.xlabel('Tissue (Sample Size)') # Set a common x-label
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tissue_metrics_bar_charts.pdf"), dpi=300)
    plt.close()
    print(f"Combined tissue metrics bar charts saved to {save_dir}/tissue_metrics_bar_charts.pdf")



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
        
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型权重文件 '{model_path}'。请确保已运行 train.py 并保存了所有折的模型。")
            sys.exit(1)
        if not os.path.exists(thresholds_path):
            print(f"错误: 找不到阈值文件 '{thresholds_path}'。请确保已运行 train.py 并保存了所有折的最佳阈值。")
            sys.exit(1)
            
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

    # 对所有模型的最佳阈值进行平均 (作为集成模型的最终阈值)
    ensemble_optimal_thresholds = np.mean(all_best_thresholds_per_fold, axis=0)


    # 获取真实标签
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
    # 这里我们使用集成后的概率和平均优化阈值来计算 F1-macro
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


    # 绘制组织特异性热力图
    # 现在传入的是经过阈值处理后的二分类结果，而不是原始概率
    plot_tissue_metrics_charts(t_true, ensemble_t_pred_binary, TISSUE_NAMES)

if __name__ == "__main__":
    run_evaluation_ensemble()