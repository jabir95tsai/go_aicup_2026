# 棋力程式分層

V2 目前只有 [設計計畫](../../../../STRENGTH_V2_PLAN.md)，尚未實作。本次整理既有程式，將 V1 與未來 V2 會共用的基礎設施抽出；沒有新增 V2 head、loss、訓練或評估。

```text
strength/
├── shared/
│   ├── model.py       # 18 層輸入、ResNet blocks、壓縮與隱藏層、初始化
│   ├── dataset.py     # packed board loader、D4、排列、全域 batch 與尾批 mask
│   ├── prepare.py     # split／固定題目／指定玩家逐手索引的產製
│   ├── runtime.py     # manifest、hash、DDP、autocast、rank-zero 執行
│   └── training.py    # seed、RNG、環境紀錄、scheduler、logging
└── v1/
    ├── model.py       # scalar head，輸出 B
    ├── config.py      # V1 config 預設與專用檢查
    ├── losses.py      # 各級平均 scalar score 的排序 loss
    ├── train.py       # V1 訓練、checkpoint 格式、最終產物與清理
    ├── infer.py       # scalar 推論、V1 validation 抽樣 loss、逐手分數
    ├── evaluate.py    # training-only 切點、scalar 聚合與 V1 報表
    └── prepare.py     # 共用資料準備的 V1 CLI
```

`shared/` 不依賴 `v1/`。`v1/` 可使用 `shared/`；未來實作 V2 時放入獨立 `v2/`，直接使用共用層，不從 `v1/` 匯入訓練、排序 loss、切點或報表。V2 專用分類 head、cross entropy、每盤 confidence 聚合、監控子集、產物格式及圖表均待實作，不因抽出基礎設施而視為完成。

其他共用程式維持原位：`go_aicup.data.board_cache`、`data.sgf`、`data.cv`、`features.sgf_planes`、`metrics`、`artifacts`、`paths`。Split 與固定題目仍只有 `data.cv` 產生；其中 `make_strength_v1_*` 函式名稱及 `strength_v1_*.parquet` 資料檔名是沿用的歷史名稱，V2 計畫明訂重用這些資料，不在這次重新命名或複製。現有 config、processed 路徑與產物格式維持原設定。

模型共用主幹保留 `stem`、`blocks`、`head.0..5` 的權重名稱；V1 在 `head.6` 接 scalar 層後才初始化，維持原始參數建立／初始化順序及 state dict 格式。模型實作位於 `v1.model`；舊的 `models.strength_resnet` 轉接檔已移除，請使用 `from go_aicup.tasks.strength.v1.model import StrengthResNet`。

現行入口（明確指定各 seed 的 config）：

```bash
python -m go_aicup.tasks.strength.v1.prepare --config configs/strength/rank_cnn_v1_seed2027.json
python -m go_aicup.tasks.strength.v1.train --config configs/strength/rank_cnn_v1_seed2027.json
python -m go_aicup.tasks.strength.v1.evaluate --run-dir outputs/rank_cnn_v1_seed2027
```

根目錄的舊轉接檔已移除，僅保留套件用的 `__init__.py`。命令與 import 統一使用上述 `shared` 或 `v1` 路徑；例如 `strength.train_rank_v1` 改用 `strength.v1.train`，`strength.dataset` 改用 `strength.shared.dataset`，設定讀取改用 `strength.v1.config`。`scripts/plot_rank_v1.py` 與 `configs/strength/rank_cnn_v1_seed*.json` 皆屬 V1 專用。

程式搬移後，新資料準備紀錄與 checkpoint 的 code fingerprint 會記錄新實作位置。既有已完成資料可由 `load_dataset_manifest` 讀取，既有模型 state dict 與續跑格式保留；若對舊 processed 目錄重新執行 prepare，原有 generator hash 防護仍會要求改用新目錄，不覆寫舊資料。
