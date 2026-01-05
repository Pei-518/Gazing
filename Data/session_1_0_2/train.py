import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataloader_test import EVEyeDataset
from model import DualStreamGazeModel
import os

# ================= 0. 超參數設定 =================
# 路徑直接指向 session_1_0_2 (因為你的 npy 檔在那裡)
SESSION_DIR = r"D:\Peggy\EV-Eye\Data\session_1_0_2"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 訓練裝置: {DEVICE}")

# ================= 1. 準備資料 =================
print("📂 載入資料集...")
full_dataset = EVEyeDataset(session_dir=SESSION_DIR)

# 簡單切分 80% 訓練, 20% 驗證
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"📊 訓練集: {len(train_dataset)} 筆 | 驗證集: {len(val_dataset)} 筆")

# ================= 2. 建立模型 =================
model = DualStreamGazeModel().to(DEVICE)
criterion = nn.MSELoss() # 預測座標的均方誤差
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= 3. 訓練迴圈 =================
print("🏁 開始正式訓練 (Real Training)...")

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    total_loss = 0
    
    for i, (images, events, labels) in enumerate(train_loader): # 🔥 這裡現在有 labels 了
        images, events, labels = images.to(DEVICE), events.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images, events) # Output: (Batch, 2)
        loss = criterion(outputs, labels) # 與真標籤計算 Loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i+1) % 50 == 0:
            print(f"   [Epoch {epoch+1}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    
    # --- Validation Phase (考試時間) ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, events, labels in val_loader:
            images, events, labels = images.to(DEVICE), events.to(DEVICE), labels.to(DEVICE)
            outputs = model(images, events)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"✨ Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # 儲存最好的模型
    if epoch == 0 or avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(SESSION_DIR, "best_model.pth"))
        print("   💾 模型已儲存 (New Best!)")

print("🎉 訓練結束！模型已存為 best_model.pth")