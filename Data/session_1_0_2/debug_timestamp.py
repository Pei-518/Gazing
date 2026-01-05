import os
import cv2
import numpy as np

# ================= 設定路徑 =================
# 請用你剛剛成功的路徑
session_dir = os.getcwd() # 假設你還是在 session_1_0_2 資料夾
frames_path = os.path.join(session_dir, 'frames')
events_path = os.path.join(session_dir, 'events', 'events.txt')

print("🔍 開始診斷時間戳記 (Timestamp Diagnosis)...")

# 1. 讀取第一張圖片的檔名，解析時間
if os.path.exists(frames_path):
    img_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.png')])
    if img_files:
        first_img = img_files[0]
        # 檔名範例: 000001_1657711084457716.png
        try:
            ts_str = first_img.split('_')[-1].replace('.png', '')
            img_ts = float(ts_str)
            print(f"📸 第一張圖片檔名: {first_img}")
            print(f"⏱️ 圖片時間戳 (Image TS): {img_ts:.0f}")
        except:
            print("❌ 無法解析圖片檔名時間")
            img_ts = 0
    else:
        print("❌ 沒有圖片")
else:
    print("❌ 找不到 frames 資料夾")

# 2. 讀取事件檔的前幾行，解析時間
first_event_ts = 0
last_event_ts = 0
event_count = 0

if os.path.exists(events_path):
    with open(events_path, 'r') as f:
        # 讀第一行
        first_line = f.readline().strip()
        if first_line:
            # 格式: t x y p
            parts = first_line.split()
            first_event_ts = float(parts[0])
            print(f"⚡ 第一筆事件時間 (Event Start): {first_event_ts:.0f}")
        
        # 讀最後一行 (快速跳轉)
        f.seek(0, 2) # 跳到檔尾
        file_size = f.tell()
        # 往回讀一點點找最後一行
        f.seek(max(file_size - 1024, 0)) 
        lines = f.readlines()
        if lines:
            last_line = lines[-1].strip()
            if last_line:
                parts = last_line.split()
                try:
                    last_event_ts = float(parts[0])
                except:
                    pass # 忽略解析錯誤
        
    print(f"⚡ 最後一筆事件時間 (Event End)  : {last_event_ts:.0f}")
else:
    print("❌ 找不到 events.txt")

# 3. 進行比對分析
if img_ts > 0 and first_event_ts > 0:
    diff = img_ts - first_event_ts
    print("\n📊 分析結果:")
    print(f"   圖片時間 - 事件開始時間 = {diff:.0f} (微秒)")
    
    if diff < 0:
        print("⚠️ 警告：圖片時間早於事件開始時間！(怪怪的)")
    elif diff > 0:
        print(f"✅ 圖片在事件開始後約 {diff/1000000:.2f} 秒，看起來正常。")
        
        # 檢查我們之前設定的 20ms (20000微秒) 視窗
        window = 20000
        print(f"   我們尋找的範圍是: [{img_ts - window:.0f}] 到 [{img_ts:.0f}]")
        
        if (img_ts - window) < first_event_ts:
             print("⚠️ 警告：對於第一張圖片，這個時間視窗超出事件資料的開頭了。")
             print("   (這可能是為什麼第一張圖是非零值為 0 的原因)")
        else:
             print("✅ 時間視窗完全落在事件資料範圍內。")
             
             # 如果範圍正確但還是讀不到，可能是資料尚未讀入記憶體
             # 這裡我們不做完整讀取，僅作邏輯判斷
    
    # 檢查是否為單位問題 (例如一個是秒，一個是微秒)
    # 1657711084457716 (16位數) 通常是微秒
    if len(str(int(img_ts))) != len(str(int(first_event_ts))):
        print("🚨 重大警告：圖片與事件的時間戳單位似乎不同！(位數不一樣)")