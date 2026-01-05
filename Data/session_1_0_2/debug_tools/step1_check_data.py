import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# ================= 設定路徑 =================
# 請將這裡改成你截圖中那個資料夾的路徑
# 注意：路徑字串前加 r 可以避免斜線轉義問題
base_path = r"D:\Peggy\EV-Eye\Data\session_1_0_2"  # <-- 請修改這裡

csv_path = os.path.join(base_path, "user_1.csv")
frames_folder = os.path.join(base_path, "frames")
# ===========================================

print(f"正在讀取 CSV: {csv_path} ...")

if os.path.exists(csv_path):
    # 1. 讀取 CSV 標籤檔
    df = pd.read_csv(csv_path)
    print("✅ CSV 讀取成功！前 5 筆資料如下：")
    print(df.head())
    
    # 嘗試取得第一張圖片的檔名
    # 通常 CSV 裡會有一欄叫做 'image_index', 'frame_index' 或直接是檔名
    # 我們假設它是依順序排列，先抓 frames 資料夾裡的第一張圖
    
    if os.path.exists(frames_folder):
        image_files = sorted(os.listdir(frames_folder))
        if len(image_files) > 0:
            first_img_name = image_files[0]
            first_img_path = os.path.join(frames_folder, first_img_name)
            
            # 2. 讀取圖片
            img = cv2.imread(first_img_path)
            if img is not None:
                print(f"✅ 成功讀取第一張圖片: {first_img_name}")
                print(f"   圖片尺寸: {img.shape}")
                
                # 顯示圖片
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"First Frame: {first_img_name}")
                plt.axis('off')
                plt.show()
            else:
                print("❌ 圖片讀取失敗 (格式錯誤?)")
        else:
            print("❌ frames 資料夾是空的")
    else:
        print(f"❌ 找不到 frames 資料夾: {frames_folder}")

else:
    print(f"❌ 找不到 CSV 檔案: {csv_path}")