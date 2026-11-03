# 棋力程式分層

V2 的 config、資料準備、十類模型、CE loss、每盤推論聚合、訓練與評估已實作，規格見 [設計計畫](../../../../STRENGTH_V2_PLAN.md)；首輪結果見 `RESULTS.md` A-024。

```text
strength/
├── shared/
│   ├── model.py       # 18 層輸入、ResNet blocks、壓縮與隱藏層、初始化
│   ├── dataset.py     # packed board loader、D4、排列、全域 batch 與尾批 mask
│   ├── prepare.py     # split／固定題目／指定玩家逐手索引的產製
│   ├── runtime.py     # manifest、hash、DDP、autocast、rank-zero 執行
│   ├── training.py    # seed、RNG、環境紀錄、scheduler、logging
│   └── checkpoint.py  # checkpoint 存取、latest-only 修剪與解析
├── v2/
│   ├── model.py       # 十類 logits head，輸出 B×10
│   ├── losses.py      # FP32 CE、有效 mask 與 DDP 全域樣本平均
│   ├── config.py      # V2 設定與監控／正式解碼規格檢查
│   ├── prepare.py     # 共用資料準備的 V2 CLI（player-disjoint）
│   ├── aggregate.py   # 每盤／題目聚合、官方分數、CATEGORY_DTYPE（numpy）
│   ├── infer.py       # 凍結模型整盤推論，DDP all_reduce
│   ├── train.py       # V2 訓練、U／S 監控、最終分類檔與 model.pt
│   └── evaluate.py    # CPU 報表與 plot/ 圖表
└── v1/
    ├── model.py       # scalar head，輸出 B
    ├── config.py      # V1 config 預設與專用檢查
    ├── losses.py      # 各級平均 scalar score 的排序 loss
    ├── train.py       # V1 訓練、checkpoint 格式、最終產物與清理
    ├── infer.py       # scalar 推論、V1 validation 抽樣 loss、逐手分數
    ├── evaluate.py    # training-only 切點、scalar 聚合與 V1 報表
    └── prepare.py     # 共用資料準備的 V1 CLI
```

`shared/` 不依賴 `v1/`。`v1/` 可使用 `shared/`；V2 放入獨立 `v2/`，直接使用共用層，不從 `v1/` 匯入訓練、排序 loss、切點或報表。V2 專用分類 head、cross entropy 與監控子集索引已實作；每盤 confidence 聚合、訓練產物格式及圖表仍待實作，不因抽出基礎設施而視為完成。

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


V2 資料準備（只讀既有盤面快取，不解析 SGF、不執行模型）：

```bash
python -m go_aicup.tasks.strength.v2.prepare --config configs/strength/rank_cnn_v2_seed2027.json
```

預設 seed=2027、global batch=65,536、1 epoch、每 200 steps 監控（A-023 由 100 改為 200）；optimizer／scheduler／BN／SyncBN／precision／D4 沿用 V1。`dataset_dir` 為 `data/processed/strength_v2/seed2027/`，模型輸出預定為 `outputs/v2_strength/`。資料準備不建立模型輸出目錄。

切分與固定題目沿用 `data.cv` 的相同 seed 與設定，在 V2 的獨立目錄保存，保留 `strength_v1_games.parquet`、`strength_v1_validation_questions.parquet` 歷史檔名；同 seed／設定／快取的切分、題目與 train／valid 索引一致。V2 不改寫既有 V1 資料。

監控從 validation 每級無放回抽取 5,000 整盤，保存 `monitor_games.npy`（50,000 個唯一 game_index、排序後的 uint32），以及 `monitor_game.npy`／`monitor_ply.npy`（所有抽中棋局的指定玩家逐手索引）。監控不限制手數，黑白依快取顏色選取。可直接將後兩檔傳入 `shared.dataset.PackedBoardDataset`，評估時關閉 D4。

`manifest.json` 的 `sets.monitor` 記錄整盤數、總手數、每級手數，`monitoring_config` 記錄抽樣／單盤 argmax 評分規格；保留原 split 玩家重疊率。準備完成後重跑會讀取保存索引，不重新抽樣；改 seed／監控設定須用另一個 dataset_dir。未完成的準備可依相同 seed 重建相同集合。多盤正式解碼固定為 `score_rank`，不影響監控單盤 argmax。


V2 模型與 loss（第三步）：

```python
from go_aicup.tasks.strength.v2.model import StrengthResNet
from go_aicup.tasks.strength.v2.losses import global_cross_entropy

model = StrengthResNet(**config["model"])
logits = model(planes)  # [B,10]，D、C、B、A、1D..6D；訓練不先做 softmax
loss, loss_sum, valid_count = global_cross_entropy(logits, rank_ids, valid)
loss.backward()
probabilities = logits.detach().softmax(dim=-1)  # 推論端使用
```

`valid` 是與樣本等長、位於同一裝置的 bool mask，可省略。loss 對全部有效指定玩家盤面等權，不要求 batch 包含十級；CE 與總和以 FP32 計算，計數為 int64。回傳的 loss_sum／valid_count 不帶梯度，可作為統計；loss 保留梯度。DDP 預設同步各卡總和與數量，所有程序均須呼叫 loss 及 backward，包含零有效樣本的程序。可微 SUM 與 DDP 的梯度平均相抵，不再手動乘除 world_size。`synchronize=False` 可明確選擇本地計算；全域零有效樣本則拒絕。CUDA／SyncBN 的實際驗收留待後續短測。


V2 訓練與評估（player-disjoint，`RESULTS.md` A-019～A-024）：

```bash
python -m go_aicup.tasks.strength.v2.prepare --config configs/strength/rank_cnn_v2_seed2027.json
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m go_aicup.tasks.strength.v2.train --config configs/strength/rank_cnn_v2_seed2027.json \
    [--resume outputs/v2_strength/last.json]
python -m go_aicup.tasks.strength.v2.evaluate --run-dir outputs/v2_strength
```

切分：`data.cv.make_strength_v2_split` 的 fold 0＝Valid-U（未見玩家），fold 1–4＝train 並每玩家顏色組留出 10%＝Valid-S，fold −1 不使用。每 200 steps 與最後一步監控 `monitor_unseen`／`monitor_seen`（每級各 2,500 整盤）。成功的 run 目錄只留 `model.pt`、`losses.npy`、`validation_unseen_category.npy`、`validation_seen_category.npy`，評估圖在 `plot/`。`evaluate` 的題目規則（每盤平均機率、題內 1/k、`metrics.score_rank`）只作 10/19 舊規則對照；V2 正式解碼已改為 C2（D-002，`predict.py`），競賽二首選已改為 HumanSL AH＋C（A-055／A-056，見 CLAUDE.md 第 11 節第 6 項）。CUDA 驗收測試 `tests/test_strength_v2_cuda.py` 需在配置 2 張 GPU 的計算節點執行。


## SE3 分類實驗

獨立的 18-block／十級 CE＋policy／value、每盤十五位置、固定題組 early stopping 實作與操作見 [reproduce_paper_3/README.md](reproduce_paper_3/README.md)。CPU／Gloo 工程驗收已完成，八卡短測與正式分數尚未取得；使用獨立 `se3_*.json` 與輸出目錄。
