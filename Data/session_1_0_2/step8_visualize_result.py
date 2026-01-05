import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from dataloader_test import EVEyeDataset
from model import DualStreamGazeModel

# ================= 設定 =================
SESSION_DIR = r"D:\Peggy\EV-Eye\Data\session_1_0_2"
MODEL_PATH = os.path.join(SESSION_DIR, "best_model.pth")
OUTPUT_DIR = os.path.join(SESSION_DIR, "results_visualization")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立輸出資料夾
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 載入資料與模型
print("📂 載入資料與模型...")
dataset = EVEyeDataset(session_dir=SESSION_DIR)
model = DualStreamGazeModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 2. 隨機抽取 5 張圖來畫圖
import random
indices = random.sample(range(len(dataset)), 5)

print(f"🎨 開始繪製預測結果，圖片將存於: {OUTPUT_DIR}")

for i, idx in enumerate(indices):
    img_tensor, evt_tensor, label = dataset[idx]
    
    # 預測
    with torch.no_grad():
        input_img = img_tensor.unsqueeze(0).to(DEVICE)
        input_evt = evt_tensor.unsqueeze(0).to(DEVICE)
        pred = model(input_img, input_evt).cpu().numpy()[0]
    
    true_label = label.numpy()
    
    # 計算誤差
    err_px = np.sqrt(np.sum((true_label - pred)**2))
    
    # 繪圖
    img_display = img_tensor.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    
    plt.figure(figsize=(6, 4.5))
    plt.imshow(img_display)
    plt.scatter(true_label[0], true_label[1], c='green', s=80, marker='+', label='Ground Truth', linewidths=2)
    plt.scatter(pred[0], pred[1], c='red', s=80, marker='x', label='Prediction', linewidths=2)
    plt.legend()
    plt.title(f"Sample {idx} | Error: {err_px:.1f} px")
    plt.axis('off')
    
    # 存檔
    save_path = os.path.join(OUTPUT_DIR, f"result_{i}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   ✅ 已儲存: {save_path} (誤差: {err_px:.1f} px)")

print("🎉 視覺化完成！請打開資料夾查看圖片。")