import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import ast

class MutationDataset(Dataset):
    def __init__(self, csv_path, esm_path, macro_path, ablation_feature_index=None):
        # 1. 加载所有数据到内存
        self.df = pd.read_csv(csv_path)
        self.esm_features = np.load(esm_path).astype(np.float32)
        macro_features_all = np.load(macro_path).astype(np.float32)

        # 消融实验：如果指定了索引，则删除该特征
        if ablation_feature_index is not None:
            self.macro_features = np.delete(macro_features_all, ablation_feature_index, axis=1)
        else:
            self.macro_features = macro_features_all
        
        # 2. 解析 tissue_vector
        # CSV 存的可能是字符串 '[0,0,1...]'，需要转成 list
        self.tissue_labels = np.array([ast.literal_eval(v) for v in self.df['tissue_vector']]).astype(np.float32)
        self.driver_labels = self.df['label'].values.astype(np.float32)

        # 添加 tissue_num 属性
        self.tissue_num = self.tissue_labels.shape[1] if self.tissue_labels.ndim > 1 else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.esm_features[idx]),    # 微观特征 (1280,)
            torch.from_numpy(self.macro_features[idx]),  # 宏观特征 (7,)
            torch.tensor(self.driver_labels[idx]),       # 任务A标签 (0/1)
            torch.from_numpy(self.tissue_labels[idx])    # 任务B标签 (20,)
        )