import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import os
import sys
import ast 

# 确保路径配置
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import MutationDataset
from model.cats_model import CaTS_Driver

# --- 超参数与环境设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 100        
PATIENCE = 20       

# --- 1. 权重计算函数 ---
def get_driver_pos_weight(subset_dataset):
    print("正在计算当前训练集的 Driver 任务的 pos_weight...")
    all_driver_labels = subset_dataset.dataset.driver_labels[subset_dataset.indices]
    num_positive = np.sum(all_driver_labels == 1)
    num_negative = np.sum(all_driver_labels == 0)
    pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    print(f"Driver 任务中, Neg/Pos 比例: {pos_weight:.2f}")
    return torch.tensor([pos_weight], device=DEVICE)

def get_tissue_pos_weights(subset_dataset, tissue_num):
    print("正在计算当前训练集的每个癌种的 pos_weight (多标签 BCEWithLogitsLoss 策略)...")
    all_tissue_labels = subset_dataset.dataset.tissue_labels[subset_dataset.indices]
    
    pos_weights = torch.zeros(tissue_num)

    for i in range(tissue_num):
        num_positive_for_tissue = np.sum(all_tissue_labels[:, i] == 1)
        num_negative_for_tissue = np.sum(all_tissue_labels[:, i] == 0)

        if num_positive_for_tissue == 0:
            pos_weights[i] = 1.0 # 如果没有正样本，设为 1.0
        else:
            pos_weights[i] = num_negative_for_tissue / num_positive_for_tissue
    
    return pos_weights.to(DEVICE)

# --- 2. 评估函数 ---
def evaluate_metrics(model, loader):
    model.eval()
    d_true, d_pred, t_true, t_pred = [], [], [], []
    with torch.no_grad():
        for ex, mx, yd, yt in loader:
            d_logits, t_logits = model(ex.to(DEVICE), mx.to(DEVICE))
            
            p_d = torch.sigmoid(d_logits)
            d_true.extend(yd.numpy())
            d_pred.extend(p_d.cpu().numpy())
            
            p_t = torch.sigmoid(t_logits)
            t_true.extend(yt.numpy())
            t_pred.extend(p_t.cpu().numpy())
    
    d_true, d_pred = np.array(d_true), np.array(d_pred)
    t_true, t_pred = np.array(t_true), np.array(t_pred)

    d_auc = roc_auc_score(d_true, d_pred)
    d_aupr = average_precision_score(d_true, d_pred)

    tissue_aucs = []
    for i in range(t_true.shape[1]):
        if len(np.unique(t_true[:, i])) > 1:
            auc_score = roc_auc_score(t_true[:, i], t_pred[:, i])
            tissue_aucs.append(auc_score)
    t_auc_mean = np.mean(tissue_aucs) if len(tissue_aucs) > 0 else 0.5
    
    return d_auc, d_aupr, t_auc_mean, d_true, d_pred, t_true, t_pred # 返回预测结果以便后续阈值优化

# --- 3. 寻找每个癌种的最佳阈值 ---
def find_best_thresholds(t_true, t_pred, tissue_num):
    best_thresholds = np.zeros(tissue_num)
    for i in range(tissue_num):
        precision, recall, thresholds = precision_recall_curve(t_true[:, i], t_pred[:, i])
        
        # 计算每个阈值对应的 F1 分数
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-10) # 加一个小 epsilon 避免除以零
        
        if len(f1_scores) > 0:
            # 找到 F1 分数最大的阈值
            best_f1_idx = np.argmax(f1_scores)
            best_thresholds[i] = thresholds[best_f1_idx]
        else:
            best_thresholds[i] = 0.5 # 默认阈值
    return best_thresholds

# --- 4. 主训练流程 ---
def run_cv():
    # 使用新的、未经重采样的数据集
    csv_path = "./data/proper_split/train_data_unbalanced.csv"
    esm_path = "./data/proper_split/train_esm2_embeddings_unbalanced.npy"
    macro_path = "./data/proper_split/train_macro_features_unbalanced.npy" 
    
    full_train_dataset = MutationDataset(csv_path, esm_path, macro_path)
    
    gkf = GroupKFold(n_splits=5)
    groups = full_train_dataset.df['gene'] 
    
    cv_results = {"d_auc": [], "d_aupr": [], "t_auc": []}

    os.makedirs("./best_models", exist_ok=True)
    os.makedirs("./best_thresholds", exist_ok=True) # 新增：保存最佳阈值的目录

    for fold, (train_idx, val_idx) in enumerate(gkf.split(full_train_dataset.df, groups=groups)):
        print(f"\n" + "="*15 + f" Fold {fold+1}/5 (Multi-label BCE MTL) " + "="*15)
        
        current_train_subset = Subset(full_train_dataset, train_idx)
        current_val_subset = Subset(full_train_dataset, val_idx)

        train_loader = DataLoader(current_train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(current_val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 分别计算两个任务的 pos_weight
        driver_pos_weight = get_driver_pos_weight(current_train_subset)
        tissue_pos_weights = get_tissue_pos_weights(current_train_subset, full_train_dataset.tissue_num)
        
        model = CaTS_Driver(tissue_num=full_train_dataset.tissue_num).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        
        criterion_driver = nn.BCEWithLogitsLoss(pos_weight=driver_pos_weight) 
        criterion_tissue = nn.BCEWithLogitsLoss(pos_weight=tissue_pos_weights) 
        
        best_overall_val_score_this_fold = -1 
        patience_counter = 0
        best_metrics_this_fold = (0, 0, 0) 
        best_thresholds_this_fold = None # 新增：保存当前折的最佳阈值

        current_fold_model_path = f"./best_models/best_model_fold{fold}.pth" 
        current_fold_thresholds_path = f"./best_thresholds/best_thresholds_fold{fold}.npy" # 新增：阈值保存路径
        
        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            
            for ex, mx, yd, yt in train_loader:
                optimizer.zero_grad()
                d_logits, t_logits = model(ex.to(DEVICE), mx.to(DEVICE))
                
                loss_d = criterion_driver(d_logits, yd.to(DEVICE).float().view(-1, 1))
                loss_t = criterion_tissue(t_logits, yt.to(DEVICE).float())
                
                total_loss = model.get_weighted_loss(loss_d, loss_t)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(total_loss.item())
            
            d_auc, d_aupr, t_auc, _, _, t_true_val, t_pred_val = evaluate_metrics(model, val_loader)
            
            # 在每个 epoch 评估后，计算当前验证集上的最佳阈值
            current_best_thresholds = find_best_thresholds(t_true_val, t_pred_val, full_train_dataset.tissue_num)
            
            # 使用这些阈值重新计算一个 F1-score 作为综合分数的一部分
            t_pred_val_binary = (t_pred_val >= current_best_thresholds).astype(int)
            # 计算平均 F1 分数作为验证分数的一部分，这里使用 macro F1
            val_f1_macro = f1_score(t_true_val, t_pred_val_binary, average='macro', zero_division=0)

            with torch.no_grad():
                w_d = torch.exp(-model.log_var_driver).item()
                w_t = torch.exp(-model.log_var_tissue).item()
                eff_lambda = w_t / w_d 

            print(f"Epoch {epoch+1:02d} | MTL Loss: {np.mean(train_losses):.4f} | "
                  f"Lambda (w_t/w_d): {eff_lambda:.3f} | Val Driver AUC: {d_auc:.4f} | Val Tissue Macro AUC: {t_auc:.4f} | Val Tissue F1-Macro: {val_f1_macro:.4f}")
            
            # 使用 Driver AUC 和 Tissue F1-Macro 作为综合指标来判断最佳模型
            current_val_score = d_auc + val_f1_macro # 综合考虑 AUC 和 F1
            if current_val_score > best_overall_val_score_this_fold: 
                best_overall_val_score_this_fold = current_val_score
                best_metrics_this_fold = (d_auc, d_aupr, t_auc) # 这里记录的是基于原始AUC的指标，F1用于选择
                best_thresholds_this_fold = current_best_thresholds # 保存最佳阈值
                torch.save(model.state_dict(), current_fold_model_path) 
                np.save(current_fold_thresholds_path, best_thresholds_this_fold) # 保存最佳阈值
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"早停触发！在 Epoch {epoch+1} 停止训练。")
                break

        cv_results["d_auc"].append(best_metrics_this_fold[0])
        cv_results["d_aupr"].append(best_metrics_this_fold[1])
        cv_results["t_auc"].append(best_metrics_this_fold[2])
        
        print(f"--- Fold {fold+1} 最佳验证分数: {best_overall_val_score_this_fold:.4f}. 模型保存至: {current_fold_model_path}, 阈值保存至: {current_fold_thresholds_path} ---")


    print("\n" + "#"*50)
    print("5-Fold 交叉验证汇总 (Multi-label BCE Version):")
    for k, v in cv_results.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
    print("#"*50)
    print(f"\n所有 5 折的最佳模型权重已保存至 './best_models/' 目录下。")
    print(f"所有 5 折的最佳阈值已保存至 './best_thresholds/' 目录下。")

if __name__ == "__main__":
    run_cv()