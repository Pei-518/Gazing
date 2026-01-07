import os
import sys

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Allow importing from Data/session_1_0_2 when running from debug_tools
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SESSION_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SESSION_ROOT not in sys.path:
    sys.path.insert(0, _SESSION_ROOT)

from dataloader_test import EVEyeDataset

# ================= 設定 =================
SESSION_DIR = r"D:\Peggy\Gazing\Data\session_1_0_2"
OUTPUT_DIR = os.path.join(SESSION_DIR, "ground_truth_check")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 載入資料
dataset = EVEyeDataset(session_dir=SESSION_DIR)

# 隨機抽取樣本檢查
import random
num_samples = 10  # 檢查 10 個樣本
indices = random.sample(range(len(dataset)), num_samples)

print(f"🔍 開始檢查 {num_samples} 個樣本的 Ground Truth 準確性")
print(f"結果將存於: {OUTPUT_DIR}")

for i, idx in enumerate(indices):
    img_tensor, evt_tensor, label = dataset[idx]

    # 檢查標籤是否有效
    true_label = label.numpy()
    is_valid = true_label[0] != -1

    # 準備圖片
    img_display = img_tensor.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_display)

    if is_valid:
        # 繪製有效標籤
        plt.scatter(true_label[0], true_label[1], c='green', s=100,
                   marker='+', label='Ground Truth', linewidths=3)
        plt.title(f"Sample {idx} | GT: ({true_label[0]:.1f}, {true_label[1]:.1f})")
        status = "有效"
    else:
        # 無效標籤
        plt.title(f"Sample {idx} | 無效標籤 (-1, -1)")
        status = "無效"

    plt.legend()
    plt.axis('off')

    # 儲存圖片
    save_path = os.path.join(OUTPUT_DIR, f"gt_check_{i}_{idx}_{status}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"   ✅ Sample {idx}: {status} | 座標: ({true_label[0]:.1f}, {true_label[1]:.1f})")

# 統計資訊
print("📊 Ground Truth 統計:")
print(f"總樣本數: {len(dataset)}")
print(f"檢查樣本數: {num_samples}")

# 計算有效標籤統計
all_labels = []
for i in range(len(dataset)):
    _, _, label = dataset[i]
    all_labels.append(label.numpy())

all_labels = np.array(all_labels)
valid_mask = all_labels[:, 0] != -1
valid_labels = all_labels[valid_mask]

print(f"有效標籤數: {len(valid_labels)}")
print(f"無效標籤數: {len(all_labels) - len(valid_labels)}")

if len(valid_labels) > 0:
    print(f"X座標範圍: {valid_labels[:, 0].min():.1f} ~ {valid_labels[:, 0].max():.1f}")
    print(f"Y座標範圍: {valid_labels[:, 1].min():.1f} ~ {valid_labels[:, 1].max():.1f}")
    print(f"X座標平均: {valid_labels[:, 0].mean():.1f}")
    print(f"Y座標平均: {valid_labels[:, 1].mean():.1f}")

print(f"\n🎉 檢查完成！請查看 {OUTPUT_DIR} 資料夾中的圖片，手動驗證標籤位置是否合理。")
