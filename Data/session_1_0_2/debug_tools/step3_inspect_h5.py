import h5py
import numpy as np
import os

# ================= 路徑設定 =================
# 請將這裡改成你截圖中那個 .h5 檔案的路徑
h5_path = r"D:\Peggy\Gazing\Data\session_1_0_2\events\user1_session_1_0_2.h5"
# ===========================================

print(f"📦 正在開箱 H5 檔案: {os.path.basename(h5_path)} ...")

if os.path.exists(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            print("\n🔑 檔案裡面的 Keys (資料夾名稱):")
            print(list(f.keys()))
            
            print("\n----------------------------------")
            print("詳細結構分析：")
            
            # 遍歷第一層的所有內容
            for key in f.keys():
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    print(f"📄 Dataset: {key}")
                    print(f"   - 形狀 (Shape): {item.shape}")
                    print(f"   - 類型 (Type) : {item.dtype}")
                    
                    # 如果是標籤數據，偷看一下前幾筆
                    if 'gaze' in key or 'label' in key or 'target' in key:
                        print(f"   - 前 3 筆數據: \n{item[:3]}")
                        
                elif isinstance(item, h5py.Group):
                    print(f"📂 Group: {key} (這是一個資料夾)")
                    print(f"   - 裡面的 Keys: {list(item.keys())}")

    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
else:
    print(f"❌ 找不到檔案: {h5_path}")