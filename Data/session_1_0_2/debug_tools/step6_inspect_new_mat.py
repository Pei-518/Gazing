import scipy.io
import numpy as np
import os

# ================= 設定檔案名稱 =================
# 請確認這跟你在檔案總管看到的名字一模一樣
# 根據你的截圖，它應該就在現在這個資料夾裡
mat_filename = r"D:\Peggy\Gazing\update_20_point_user1_session_1_0_2.mat" 
# ===========================================

print(f"📦 正在嘗試讀取: {mat_filename}")

if os.path.exists(mat_filename):
    try:
        # 讀取 .mat 檔
        mat_data = scipy.io.loadmat(mat_filename)
        
        print("✅ 讀取成功！")
        print("\n--- 檔案內容變數 (Keys) ---")
        
        found_target = False
        
        # 遍歷所有變數
        for key in mat_data.keys():
            if not key.startswith('__'): # 忽略系統變數
                data = mat_data[key]
                shape = np.shape(data)
                print(f"   🔑 Key: {key:20} | 形狀: {shape}")
                
                # 判斷是不是我們要的答案
                # 1. 數量接近圖片數 (2694)
                # 2. 名稱像 label, gaze, target
                if shape[0] > 2000 or 'gaze' in key.lower() or 'target' in key.lower():
                    print(f"      ✨ 疑似目標！數值預覽 (前3筆):")
                    if isinstance(data, np.ndarray):
                         # 為了避免印太多，只印前3行
                         print(data[:3])
                         found_target = True

        if found_target:
            print("\n🎉 太棒了！我們找到標籤了！")
        else:
            print("\n⚠️ 沒看到明顯的標籤，可能要仔細看一下上面的變數名稱。")
            
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")
        print("提示：如果是 .mat 版本過新，可能需要用 h5py 讀取。")
else:
    print(f"❌ 找不到檔案: {mat_filename}")
    print("請確認檔案名稱是否正確 (副檔名是 .mat 嗎?)")