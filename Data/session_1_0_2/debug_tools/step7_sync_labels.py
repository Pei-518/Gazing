import numpy as np
import scipy.io
import os

# ================= 🔥 關鍵路徑設定 (寫死最安全) =================
# 1. 你的資料集資料夾 (這裡面有 frames 和 events 資料夾)
session_dir = r"D:\Peggy\EV-Eye\Data\session_1_0_2"

# 2. 你的標籤檔案 (這裡假設你把它放在專案最外層，因為你上一步 step6 讀到了)
# 如果程式說找不到這個，請確認這個 .mat 檔到底在哪裡
mat_file_path = r"D:\Peggy\EV-Eye\update_20_point_user1_session_1_0_2.mat"

# 3. 自動組裝其他路徑
start_time_file = os.path.join(session_dir, "events", "event_startime.txt")
frames_folder = os.path.join(session_dir, "frames")
# =============================================================

print("🔄 開始執行標籤同步 (Label Synchronization)...")
print(f"📂 資料目錄: {session_dir}")
print(f"📄 標籤檔案: {mat_file_path}")

# 1. 讀取起始時間 (Global Start Time)
if os.path.exists(start_time_file):
    with open(start_time_file, 'r') as f:
        # 有些檔案可能包含多個數值，通常取第一個
        content = f.readline().strip()
        # 處理可能的逗號或空格分隔
        if ',' in content:
            global_start_ts = float(content.split(',')[0])
        else:
            global_start_ts = float(content.split()[0])
    print(f"✅ 讀取起始時間: {global_start_ts:.0f}")
else:
    print(f"❌ 找不到 event_starttime.txt (路徑: {start_time_file})")
    exit()

# 2. 讀取 .mat 標籤檔
if os.path.exists(mat_file_path):
    mat_data = scipy.io.loadmat(mat_file_path)
    # 根據你的 print 結果，變數名稱是 'matcell'
    if 'matcell' in mat_data:
        raw_data = mat_data['matcell']
        
        # 提取關鍵欄位 (根據 step6 的觀察)
        # Col 2 (index 2): 相對時間
        # Col 3 (index 3): Gaze X
        # Col 4 (index 4): Gaze Y
        mat_relative_ts = raw_data[:, 2] 
        mat_gaze_x = raw_data[:, 3]
        mat_gaze_y = raw_data[:, 4]
        
        # 計算絕對時間：起始時間 + 相對時間
        mat_abs_ts = global_start_ts + mat_relative_ts
        
        print(f"✅ 讀取標籤資料: {len(raw_data)} 筆")
        print(f"   標籤時間範圍: {mat_abs_ts[0]:.0f} ~ {mat_abs_ts[-1]:.0f}")
    else:
        print("❌ .mat 檔裡找不到 'matcell' 變數")
        exit()
else:
    print(f"❌ 找不到 .mat 檔案: {mat_file_path}")
    print("請確認你是不是把 update_20_point...mat 放在 D:\\Peggy\\EV-Eye\\ 底下？")
    exit()

# 3. 讀取圖片時間戳
if os.path.exists(frames_folder):
    img_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    img_timestamps = []
    
    for f in img_files:
        # 檔名範例: 000001_1657711084457716.png
        try:
            ts = float(f.split('_')[1].replace('.png', ''))
            img_timestamps.append(ts)
        except:
            pass
            
    img_timestamps = np.array(img_timestamps)
    print(f"✅ 讀取圖片清單: {len(img_timestamps)} 張")
    if len(img_timestamps) > 0:
        print(f"   圖片時間範圍: {img_timestamps[0]:.0f} ~ {img_timestamps[-1]:.0f}")
else:
    print(f"❌ 找不到 frames 資料夾: {frames_folder}")
    exit()

# 4. 核心步驟：時間配對 (Matching)
print("⏳ 正在進行時間對齊 (這可能需要一點時間)...")

aligned_labels = []
valid_count = 0

# 設定容許誤差 (50ms = 50000us)
# 如果圖片時間跟最近的標籤差超過這個值，就當作這張圖沒標籤
MAX_TIME_DIFF = 50000 

for i, img_ts in enumerate(img_timestamps):
    # 尋找最近的時間點索引
    # (這裡用絕對值差最小來找)
    time_diff = np.abs(mat_abs_ts - img_ts)
    min_idx = np.argmin(time_diff)
    min_diff = time_diff[min_idx]
    
    if min_diff < MAX_TIME_DIFF: 
        gaze_x = mat_gaze_x[min_idx]
        gaze_y = mat_gaze_y[min_idx]
        aligned_labels.append([gaze_x, gaze_y])
        valid_count += 1
    else:
        # 標記為無效 (-1, -1)
        aligned_labels.append([-1.0, -1.0])

aligned_labels = np.array(aligned_labels, dtype=np.float32)

# 5. 儲存結果
print("\n=== 📊 對齊結果 ===")
print(f"圖片總數: {len(img_timestamps)}")
print(f"成功配對: {valid_count} 張 ({(valid_count/len(img_timestamps))*100:.1f}%)")
print(f"標籤形狀: {aligned_labels.shape} (N, 2)")

# 儲存成 .npy 檔
output_path = os.path.join(session_dir, "gaze_labels.npy")
np.save(output_path, aligned_labels)
print(f"✅ 已儲存處理好的標籤: {output_path}")

print("\n🔎 前 5 筆對齊的標籤 (X, Y):")
print(aligned_labels[:5])