import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

# ================= 1. 事件處理核心 (修正版) =================
def events_to_voxel_grid(events, num_bins, width, height):
    """
    將事件流轉換為 Voxel Grid (Time_Bins, Height, Width)
    """
    # 1. 如果沒有事件，回傳全零 Tensor
    if len(events) == 0:
        return torch.zeros((num_bins, height, width), dtype=torch.float32)

    # 2. 建立空的 NumPy 陣列 (🔥 修正點：使用 np.float32)
    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32)
    
    t = events[:, 0]
    
    # 避免只有一筆資料導致除以零，或時間長度為 0
    if len(t) < 2 or (t[-1] - t[0] == 0):
         return torch.from_numpy(voxel_grid)

    # 正規化時間戳
    duration = t[-1] - t[0] + 1e-6 # 加一個極小值避免除以零
    t_norm = (t - t[0]) / duration * (num_bins - 1)
    
    x = events[:, 1].astype(int)
    y = events[:, 2].astype(int)
    p = events[:, 3]

    # 累積事件到 Voxel Grid
    for i in range(len(events)):
        t_idx = int(t_norm[i])
        # 邊界檢查
        if 0 <= t_idx < num_bins and 0 <= y[i] < height and 0 <= x[i] < width:
            val = 1.0 if p[i] == 1 else -1.0
            voxel_grid[t_idx, y[i], x[i]] += val
            
    return torch.from_numpy(voxel_grid)

# ================= 2. Dataset 類別 (含標籤讀取) =================
class EVEyeDataset(Dataset):
    def __init__(self, session_dir):
        self.session_path = session_dir
        
        # 設定檔案路徑
        self.events_path = os.path.join(self.session_path, 'events', 'events.txt')
        self.frames_path = os.path.join(self.session_path, 'frames')
        self.labels_path = os.path.join(self.session_path, 'gaze_labels.npy') # 讀取處理好的標籤

        print(f"📂 初始化 Dataset: {self.session_path}")

        # 1. 讀取圖片列表
        if os.path.exists(self.frames_path):
            self.image_files = sorted([f for f in os.listdir(self.frames_path) if f.endswith('.png')])
        else:
            self.image_files = []
            print("❌ 找不到 frames 資料夾")

        # 2. 讀取標籤 & 過濾無效資料
        self.valid_indices = [] # 存放有效資料的索引對照表
        
        if os.path.exists(self.labels_path):
            self.labels = np.load(self.labels_path)
            
            # 篩選出標籤不是 (-1, -1) 的索引
            # 確保圖片數量跟標籤數量一致才能進行對應
            limit = min(len(self.labels), len(self.image_files))
            
            for i in range(limit):
                # 檢查標籤是否有效 (我們在 step7 設定無效值為 -1)
                if self.labels[i][0] != -1:
                    self.valid_indices.append(i)
                    
            print(f"✅ 載入標籤成功！有效資料: {len(self.valid_indices)} / {limit}")
        else:
            print("❌ 找不到 gaze_labels.npy，請先執行 step7！")
            self.labels = None
            # 如果沒有標籤檔，為了測試程式碼，暫時假設所有圖片都有效 (但標籤會錯)
            self.valid_indices = list(range(len(self.image_files)))

        # 3. 讀取事件 (使用 float64 避免精度問題)
        if os.path.exists(self.events_path):
            self.all_events = self._load_events_txt(self.events_path)
        else:
            self.all_events = np.array([])
            print("❌ 找不到 events.txt")

    def _load_events_txt(self, path):
        # 讀取事件文字檔
        events = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(list(map(float, line.split())))
                    except ValueError:
                        continue
        # 重要：使用 float64 保存微秒級時間戳
        return np.array(events, dtype=np.float64)

    def __len__(self):
        # Dataset 的長度是「有效資料」的數量
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 使用有效索引對照表找到真實的 index
        real_idx = self.valid_indices[idx]
        
        # --- A. 讀取圖片 ---
        img_name = self.image_files[real_idx]
        img_path = os.path.join(self.frames_path, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            # 防呆機制
            return torch.zeros((3, 260, 346)), torch.zeros((5, 260, 346)), torch.zeros(2)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 轉 Tensor: (H, W, C) -> (C, H, W) 並正規化到 0~1
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # --- B. 讀取標籤 ---
        if self.labels is not None:
            label = self.labels[real_idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.zeros(2, dtype=torch.float32)

        # --- C. 讀取對應事件 (20ms 視窗) ---
        try:
            # 從檔名解析時間戳
            ts_str = img_name.split('_')[-1].replace('.png', '')
            current_ts = float(ts_str)
        except:
            current_ts = 0.0
            
        window = 20000 # 20ms = 20000us
        local_events = np.array([])

        if len(self.all_events) > 0:
            # 確保搜尋範圍合理
            start_time = current_ts - window
            
            # 使用 Boolean Mask 快速篩選
            # 優化：先檢查是否在整個事件流範圍內
            if current_ts >= self.all_events[0, 0]:
                mask = (self.all_events[:, 0] >= start_time) & (self.all_events[:, 0] <= current_ts)
                local_events = self.all_events[mask]
        
        # 轉換為 Voxel Grid
        event_voxel = events_to_voxel_grid(local_events, num_bins=5, width=346, height=260)
        
        return image_tensor, event_voxel, label_tensor

# ================= 自我測試區塊 =================
if __name__ == "__main__":
    # 簡單測試讀取
    print("🚀 測試 DataLoader...")
    dataset = EVEyeDataset(session_dir=".")
    if len(dataset) > 0:
        img, evt, lbl = dataset[0]
        print(f"✅ 讀取成功！")
        print(f"影像形狀: {img.shape}")
        print(f"事件形狀: {evt.shape}")
        print(f"標籤數值: {lbl}")
    else:
        print("⚠️ 沒有資料可讀取")