# EV-Eye: Event-Based Eye Tracking System

## 專案描述

EV-Eye 是一個基於事件相機（Event Camera）的眼動追蹤系統。該系統利用事件相機的高動態範圍和低延遲特性，結合 U-Net 深度學習模型，對瞳孔進行即時檢測和追蹤。主要應用於眼動研究、視覺輔助系統和人機交互。

## 主要功能

- **瞳孔檢測**：使用 U-Net 卷積神經網路進行瞳孔區域分割
- **事件數據處理**：處理來自事件相機的數據流
- **視覺化**：提供結果視覺化和分析工具
- **訓練與測試**：包含完整的訓練腳本和測試數據

## 安裝說明

### 系統需求
- Python 3.7+
- MATLAB (用於數據處理)
- Git

### Python 環境設置
1. 克隆倉庫：
   ```bash
   git clone https://github.com/Pei-518/Gazing.git
   cd Gazing
   ```

2. 創建虛擬環境（推薦使用 conda）：
   ```bash
   conda create -n eveye python=3.8
   conda activate eveye
   ```

3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

### MATLAB 設置
參考 `matlab_processed/README.md` 進行 MATLAB 環境配置。

## 使用說明

### 數據準備
- 將事件相機數據放置在 `Data/session_1_0_2/events/` 目錄
- 幀數據放置在 `Data/session_1_0_2/frames/` 目錄

### 訓練模型
```bash
python train.py
```

### 測試模型
```bash
python Data/session_1_0_2/train.py  # 或其他測試腳本
```

### 視覺化結果
```bash
python Data/session_1_0_2/step8_visualize_result.py
```

## 專案結構

```
EV-Eye/
├── Data/
│   └── session_1_0_2/
│       ├── events/          # 事件相機數據
│       ├── frames/          # 幀數據
│       ├── results_visualization/  # 視覺化結果
│       ├── best_model.pth   # 訓練好的模型
│       ├── model.py         # 模型定義
│       ├── train.py         # 訓練腳本
│       └── step8_visualize_result.py  # 視覺化腳本
├── matlab_processed/        # MATLAB 數據處理腳本
├── unet/                    # U-Net 模型實現
│   ├── unet_model.py
│   ├── unet_parts.py
│   └── __init__.py
├── utils/                   # 工具函數
│   ├── data_loading.py
│   ├── dice_score.py
│   └── utils.py
├── pictures/                # 結果圖片
├── train.py                 # 主訓練腳本
├── requirements.txt         # Python 依賴
├── LICENSE                  # 授權文件
└── README.md               # 本文件
```

## 技術細節

- **模型架構**：U-Net 卷積神經網路
- **數據格式**：事件數據 (.h5, .txt)，幀數據 (.png)
- **評估指標**：Dice 係數，IoU 等
- **框架**：PyTorch

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 授權

本專案採用 MIT 授權。詳見 [LICENSE](LICENSE) 文件。

## 聯繫

如有問題，請聯繫：peggy94.hsu@gmail.com