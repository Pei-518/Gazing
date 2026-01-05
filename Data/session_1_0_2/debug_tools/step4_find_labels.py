import os
import scipy.io
import numpy as np

# ================= 1. 強制設定絕對路徑 (最穩的做法) =================
# 根據你的錯誤訊息推斷，你的專案根目錄在 D:\Peggy\EV-Eye
# 所以 matlab_processed 應該在這裡：
matlab_folder = r"D:\Peggy\EV-Eye\matlab_processed"
# ================================================================

print(f"🔍 正在搜尋資料夾: {matlab_folder}")

if os.path.exists(matlab_folder):
    files = [f for f in os.listdir(matlab_folder) if f.endswith('.mat')]
    
    if len(files) > 0:
        print(f"✅ 找到 {len(files)} 個 .mat 檔案！")
        # 列出前 3 個檔案確認一下
        print("檔案範例:", files[:3])
            
        # 嘗試讀取第一個檔案，看看裡面有什麼變數
        first_mat = os.path.join(matlab_folder, files[0])
        try:
            mat_data = scipy.io.loadmat(first_mat)
            print(f"\n📦 正在讀取: {files[0]}")
            
            # 尋找像是 'gaze', 'label', 'pupil', 'target' 這樣的關鍵字
            found_keys = []
            print("--- 檔案內的變數 (Keys) ---")
            for key in mat_data.keys():
                if not key.startswith('__'): # 忽略系統變數
                    data = mat_data[key]
                    shape_info = np.shape(data)
                    print(f"   👉 {key}: 形狀 {shape_info}")
                    found_keys.append(key)
            
            print("---------------------------")
            
        except Exception as e:
            print(f"❌ 讀取失敗: {e}")
    else:
        print("❌ 資料夾存在，但裡面沒有 .mat 檔案")
else:
    print(f"❌ 依然找不到資料夾: {matlab_folder}")
    print("請打開檔案總管，確認 D:\\Peggy\\EV-Eye 底下真的有一個叫 matlab_processed 的資料夾嗎？")