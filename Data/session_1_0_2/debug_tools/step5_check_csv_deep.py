import pandas as pd
import json
import os

# ================= 路徑設定 =================
# 請確認這是你的 csv 路徑
csv_path = r"D:\Peggy\EV-Eye\Data\session_1_0_2\user_1.csv"
# ===========================================

print(f"🔍 正在深度檢查 CSV: {csv_path}")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ 成功讀取，共 {len(df)} 筆資料")
    
    # 檢查 'region_shape_attributes' 欄位
    if 'region_shape_attributes' in df.columns:
        print("\n🧐 檢查 'region_shape_attributes' 的內容...")
        
        # 隨機抽檢第 0, 10, 100 筆資料 (避免第一筆剛好是空的)
        sample_indices = [0, 10, 100]
        
        for idx in sample_indices:
            if idx < len(df):
                content = df.loc[idx, 'region_shape_attributes']
                print(f"\n--- Row {idx} ---")
                print(f"原始字串: {content}")
                
                # 嘗試解析 JSON
                try:
                    data = json.loads(content)
                    if data:
                        print(f"🎉 解析成功！發現 Key: {list(data.keys())}")
                        if 'name' in data and data['name'] == 'ellipse':
                            print(f"   👉 橢圓中心 (cx, cy): ({data.get('cx')}, {data.get('cy')})")
                            print(f"   👉 橢圓半徑 (rx, ry): ({data.get('rx')}, {data.get('ry')})")
                            print("   ✅ 這就是我們要的幾何標籤！")
                    else:
                        print("⚠️ 內容是空的 JSON {}")
                except:
                    print("❌ JSON 解析失敗")
    else:
        print("❌ 找不到 'region_shape_attributes' 欄位")
else:
    print("❌ 找不到 CSV 檔案")