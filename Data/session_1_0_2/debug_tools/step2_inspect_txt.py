import os
import pandas as pd

# ================= 路徑設定 =================
# 請修改成你電腦裡的路徑
base_path = r"D:\Peggy\EV-Eye\Data\session_1_0_2"
events_folder = os.path.join(base_path, "events")
csv_path = os.path.join(base_path, "user_1.csv")
# ===========================================

print("=== 1. 檢查 Events TXT 格式 ===")
if os.path.exists(events_folder):
    txt_files = [f for f in os.listdir(events_folder) if f.endswith('.txt')]
    
    if len(txt_files) > 0:
        first_txt = os.path.join(events_folder, txt_files[0])
        print(f"找到事件檔: {txt_files[0]}")
        
        # 讀取前 5 行來觀察
        with open(first_txt, 'r') as f:
            lines = [f.readline().strip() for _ in range(5)]
            
        print("檔案內容前 5 行：")
        for i, line in enumerate(lines):
            print(f"Line {i+1}: {line}")
    else:
        print("❌ Events 資料夾裡沒有 .txt 檔！")
else:
    print("❌ 找不到 Events 資料夾")

print("\n=== 2. 尋找真正的 Gaze Labels ===")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"CSV 共有 {len(df.columns)} 個欄位")
    print("欄位名稱:", df.columns.tolist())
    
    # 檢查有沒有看起來像座標的欄位
    potential_labels = [c for c in df.columns if 'x' in c.lower() or 'y' in c.lower() or 'gaze' in c.lower()]
    print("可能是標籤的欄位:", potential_labels)
    
    # 檢查 region_attributes 是否藏有資料 (有些工具會把標籤塞在 JSON 字串裡)
    if 'region_attributes' in df.columns:
        print("region_attributes 第一筆資料:", df['region_attributes'].iloc[0])
else:
    print("❌ 找不到 CSV 檔案")