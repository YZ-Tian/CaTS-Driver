import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

DATA_DIR = './data'
OUTPUT_DIR = './data/proper_split'

# --- 1. 加载原始数据和特征 ---
df_all_unique_mutations = pd.read_csv(os.path.join(DATA_DIR, 'processed_mutations_with_seq.csv'))
esm2_embeddings_all = np.load(os.path.join(DATA_DIR, 'esm2_delta_embeddings_final.npy'))
macro_features_all = np.load(os.path.join(DATA_DIR, 'macro_context_features_7dim.npy'))

print(f"Loaded all unique mutations DF shape: {df_all_unique_mutations.shape}")
print(f"Loaded ESM-2 embeddings shape: {esm2_embeddings_all.shape}")
print(f"Loaded macro features shape: {macro_features_all.shape}")

# 确保行数一致
assert df_all_unique_mutations.shape[0] == esm2_embeddings_all.shape[0]
assert df_all_unique_mutations.shape[0] == macro_features_all.shape[0]

# --- 2. 根据基因进行 GroupShuffleSplit 划分 ---
# 确保每个基因只出现在训练集或测试集之一
groups_for_split = df_all_unique_mutations['gene']

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df_all_unique_mutations, groups=groups_for_split))

# --- 3. 划分数据和特征---
final_train_df = df_all_unique_mutations.iloc[train_idx].reset_index(drop=True)
final_train_esm2 = esm2_embeddings_all[train_idx]
final_train_macro = macro_features_all[train_idx]

final_test_df = df_all_unique_mutations.iloc[test_idx].reset_index(drop=True)
final_test_esm2 = esm2_embeddings_all[test_idx]
final_test_macro = macro_features_all[test_idx]

print(f"\nAfter GroupShuffleSplit (no resampling):")
print(f"Final train DF shape: {final_train_df.shape}")
print(f"Train label distribution (1/0):\n{final_train_df['label'].value_counts()}\n")
print(f"Final test DF shape: {final_test_df.shape}")
print(f"Test label distribution (1/0):\n{final_test_df['label'].value_counts()}\n")

# 验证基因不重叠
train_genes = set(final_train_df['gene'].unique())
test_genes = set(final_test_df['gene'].unique())
common_genes = train_genes.intersection(test_genes)
if len(common_genes) == 0:
    print("SUCCESS: Genes are mutually exclusive between the new train and test sets.")
else:
    print(f"WARNING: {len(common_genes)} common genes found. This should not happen.")

# --- 4. 保存划分后的原始数据到新目录 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

final_train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_data_unbalanced.csv'), index=False)
final_test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_data_unbalanced.csv'), index=False)

np.save(os.path.join(OUTPUT_DIR, 'train_esm2_embeddings_unbalanced.npy'), final_train_esm2)
np.save(os.path.join(OUTPUT_DIR, 'train_macro_features_unbalanced.npy'), final_train_macro)

np.save(os.path.join(OUTPUT_DIR, 'test_esm2_embeddings_unbalanced.npy'), final_test_esm2)
np.save(os.path.join(OUTPUT_DIR, 'test_macro_features_unbalanced.npy'), final_test_macro)

print(f"\nSuccessfully saved the new, unbalanced data splits to: {OUTPUT_DIR}")
