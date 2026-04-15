import torch
import torch.nn as nn

class CaTS_Driver(nn.Module):
    def __init__(self, seq_dim=1280, macro_dim=7, tissue_num=20):
        super(CaTS_Driver, self).__init__()
        
        # --- 1. 宏观特征编码器 ---
        self.macro_encoder = nn.Sequential(
            nn.Linear(macro_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # --- 2. 多模态融合层 ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(seq_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # --- 3. 任务 A 头：致病性预测 (Driver vs Passenger) ---
        # 输出 Logits，Sigmoid 将在 Loss 函数中处理
        self.driver_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
        # --- 4. 任务 B 头：组织倾向性预测 (20分类，多标签) ---
        # 输出 Logits，Sigmoid 将在 Loss 函数中处理
        self.tissue_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, tissue_num)
        )

        # --- 5. 贝叶斯动态 Loss 权重参数 (可学习) ---
        # 初始化为 0，代表两个任务初始权重比例 1:1
        self.log_var_driver = nn.Parameter(torch.zeros(1))
        self.log_var_tissue = nn.Parameter(torch.zeros(1))

    def forward(self, seq_emb, macro_feat):
        # 1. 提取宏观特征
        macro_emb = self.macro_encoder(macro_feat)
        
        # 2. 拼接微观(ESM)与宏观特征
        combined = torch.cat((seq_emb, macro_emb), dim=1)
        
        # 3. 融合特征
        latent = self.fusion_layer(combined)
        
        # 4. 输出两个分支的 Logits
        driver_logits = self.driver_head(latent)
        tissue_logits = self.tissue_head(latent) # 返回 logits
        
        return driver_logits, tissue_logits

    def get_weighted_loss(self, loss_driver, loss_tissue):
        """
        根据不确定性动态平衡两个任务的 Loss
        公式: L = (1/exp(log_var)) * Loss + log_var
        """
        # 计算致病性预测的动态权重
        precision_d = torch.exp(-self.log_var_driver)
        weighted_loss_d = precision_d * loss_driver + self.log_var_driver
        
        # 计算组织特异性预测的动态权重
        precision_t = torch.exp(-self.log_var_tissue)
        weighted_loss_t = precision_t * loss_tissue + self.log_var_tissue
        
        return weighted_loss_d + weighted_loss_t