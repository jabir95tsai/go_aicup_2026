# AI CUP 2026 圍棋競賽

本專案處理兩項任務：依多盤棋譜辨識棋手，以及預測棋力。競賽資料格式、評分與限制見 [AGENTS.md](AGENTS.md)；研究與方法參考見 [reference.md](reference.md)。

## 檔案位置

```text
go_aicup_2026/
├─ AGENTS.md                 # 競賽規則與專案慣例（現有文件）
├─ reference.md              # 研究與實作參考（現有文件）
├─ README.md                 # 專案入口與目錄說明
├─ .gitignore
├─ configs/                  # 版控的實驗設定
│  ├─ identity/              # 棋手辨識設定
│  └─ strength/              # 棋力預測設定
├─ src/go_aicup/             # 可重用的 Python 程式碼
│  ├─ data/                  # CSV/SGF 解析、盤面重建、去重與切分
│  ├─ features/              # 棋譜／棋手特徵與快取格式
│  ├─ models/                # 兩題可共用的編碼器與模型元件
│  ├─ tasks/
│  │  ├─ identity/            # 棋手辨識：候選排序、Top-5 指標與推論
│  │  └─ strength/            # 棋力預測：等級分類、官方指標與推論
│  └─ ...                    # 共用訓練流程與提交檔驗證
├─ scripts/                  # 命令列入口：準備資料、訓練、驗證、推論
├─ tests/                    # 解析、切分、防洩漏、指標與提交格式測試
├─ data/                     # 不進 Git
│  ├─ raw/                  # 官方原始壓縮檔及原樣解壓資料
│  └─ processed/            # 清理後資料、特徵快取、切分索引與清單
├─ outputs/                  # 不進 Git；每次實驗一個 <run-id>/
│  └─ <run-id>/             # config、環境／資料版本、分數、log、checkpoint
└─ submissions/              # 不進 Git；依官方範例產生的最終提交 CSV
```

目前只建立目錄骨架；圖中的模組檔案與實驗檔案會在實作對應功能時加入。尚未取得官方測試資料與提交範例，因此提交檔欄位和命名應以屆時的官方格式為準。

## 放置原則

- 官方下載檔放 `data/raw/`，保留原樣與 SHA-256；任何轉換、去重結果、特徵和切分索引放 `data/processed/`。同一棋局及其衍生片段只能位於同一切分。
- 可重用邏輯放 `src/go_aicup/`，`scripts/` 只負責讀取設定並呼叫程式。兩題共用 SGF 解析、盤面表示、編碼器與訓練入口；專屬的預測頭、損失、官方指標和推論流程分別放 `tasks/identity/`、`tasks/strength/`。
- 實驗參數分別放 `configs/identity/`、`configs/strength/`。每次訓練建立唯一 `outputs/<run-id>/`，建議以任務名稱作為 run-id 前綴，並保存實際設定、seed、資料版本與切分、Git commit、套件及硬體資訊、驗證分數和權重，方便重現。
- 提交檔只由推論程式寫入 `submissions/`，產生後依官方範例檢查欄位、順序、候選與等級值。不要手動修改答案。
- 原始資料、快取、權重、實驗輸出和提交檔由 `.gitignore` 排除；程式、設定、測試及文件進 Git。外部資料或模型的來源與用途須在相關設定或實驗紀錄中留存。

## 建議實作順序

1. CSV/SGF 解析及資料稽核，確認 pass、異常與重複棋局。
2. 建立不洩漏的驗證切分與兩題官方指標。
3. 實作 baseline、推論及提交格式檢查。
4. 再加入可設定的 PyTorch 訓練入口、續訓、smoke test 與 DDP 支援。
