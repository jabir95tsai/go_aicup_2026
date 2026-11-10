# AI CUP 2026 圍棋棋力與風格預測：技術報告（骨架，2026-10-25）

> 本檔為報告骨架。標 **〔待補〕** 的項目在 Private LB 公布後（11/25）或人員確認後填入。所有本地數字皆為合成驗證題目上的結果，不是 LB 分數；細節與完整實驗紀錄見 `RESULTS.md`（條目編號如 A-044）。

## 1. 隊伍與成員

| 姓名 | 學校／系所 | 分工 |
|---|---|---|
| 〔待補〕 | 〔待補〕 | 〔待補〕 |
| 〔待補〕 | 〔待補〕 | 〔待補〕 |

上傳帳號：〔待補：隊伍唯一上傳帳號〕

## 2. 執行環境

| 項目 | 規格 |
|---|---|
| 訓練與推論 | 國網中心晶創 26（Nano4）：NVIDIA H200 SXM5 141GB，Slurm `dev`／`8gpus` 佇列，單節點最多 8 卡；driver 580.65.06 |
| 開發 | 本機 RTX 3060 Ti 8GB（CUDA 12.4） |
| Python 環境 | CPython 3.13（Conda 環境 `go`），`torch==2.6.0+cu124`；直接依賴版本鎖定於 `requirements.txt` |
| 外部執行檔 | KataGo v1.18.2（CUDA 12.8／cuDNN 9.8 版），函式庫取自 NVIDIA PyPI wheel（見 `EXTERNAL_RESOURCES.md`） |
| 隨機種子 | 切分 seed 2027；各 config 檔記錄初始化 seed；B2 合成題目 seed 2027 |

〔待補：提交時的實際套件版本清單（`pip freeze`）與每個最終 run 的執行時間〕

## 3. 資料前處理（原始碼：`src/go_aicup/data/`）

- **SGF 解析**（`sgf.py`）：訓練集用嚴格解析（只接受 `SZ[19]` 與 `;B[..]`／`;W[..]`）；官方測試檔與候選庫用寬鬆解析（pass、setup 石只計數不中斷）。座標轉為 `row*19+col`（0–360）。
- **一次性轉檔**（`preprocess.py`）：CSV → `int16` 落子陣列＋顏色＋metadata parquet（GameStore）；唯一鍵為 `(rank, game_id)`。
- **盤面快取**（`board_cache.py`）：重播每盤棋，打包成每手 2×361 bits 的盤面；測試檔以 `write_store_cache` 用同一套重播。
- **驗證切分**（`cv.py`、`tasks/strength/shared/prepare.py`）：以 `player_id` 分組，Valid-U（訓練完全沒見過的玩家）、Valid-S（訓練玩家的留出盤）、fold −1 保留不用；題目依官方結構以同一玩家的 k∈{5,10,15,20} 盤合成。競賽一以 400 位 held-out 玩家複刻候選庫（每人 100 盤）與互斥的查詢題。
- 稽核結論（`RESULTS.md` A-001、A-002）：訓練集只有 `SZ` 與落子，黑先、黑白交替、無 pass，100–445 手。

## 4. 競賽二：棋力預測

### 4.1 方法

1. **V2 盤面 ResNet**（`tasks/strength/v2/`，A-024）：256 通道 × 20 blocks，輸入雙方 18 層歷史盤面，10 類 softmax，逐手 cross entropy；8×H200 訓練 1 epoch（5% warmup＋cosine）。每盤取指定玩家落子的平均 log-prob（10 維）。
2. **KataGo HumanSL 特徵**（`features/humansl.py`，A-039）：以 HumanSL 網路 `b18c384nbt-humanv0` 為主模型、analysis engine 每個位置 1 visit，取實際著手在 15 個段位 profile（`rank_20k`～`rank_9d`）下的 log-likelihood；每盤取本人與對手各 15 個平均（30 維）。
3. **B2 題目層級模型**（`tasks/strength/humansl.py`，A-041）：每個 k 一個多類 logistic regression；在擬合玩家的合成題目（同一玩家抽 k 盤，40 維特徵取平均，每 k 20,000 題）上擬合；以 Bayes 最佳解碼 `metrics.score_rank`（期望分數 p[r]＋(p[r−1]＋p[r+1])/e 的 argmax）輸出級別。
4. **備援**：任一題沒有 HumanSL 特徵時，改用 V2＋C2（每盤平均 log-prob、題內平均、每 k 溫度、`score_rank`；A-025）。

### 4.2 本地驗證（Valid-U 1,000 題，未見玩家）

| 方法 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| V2 ResNet＋C2 | 0.493 | 0.562 | 0.558 | 0.600 | 0.5532 |
| V2＋HumanSL＋B2（正式） | 0.617 | 0.679 | 0.722 | 0.715 | 0.6833 |

官方 baseline 0.4631。〔待補：Public／Private LB 分數〕

## 5. 競賽一：玩家辨識

### 5.1 方法

1. **開局模型**（`tasks/identity/opening.py`，A-029）：前 60 手做 D4 正規化；prefix back-off＋逐手 slot naive Bayes，以未在候選庫的訓練盤估計族群先驗。
2. **GE2E 保險模型**（`tasks/identity/ge2e.py`、`insurance.py`，A-035、A-053）：沿用 V2 ResNet 主幹，只看 ply ≥20 的黑方手（看不到開局），GE2E loss 訓練；候選人向量為其 100 盤的平均。
3. **融合與指派**：開局分數＋8×w×cosine，Sinkhorn＋Hungarian（L6）產生 top-5。

### 5.2 本地驗證（合成候選庫 split 1／2）

| 方法 | split 1 | split 2 |
|---|---:|---:|
| 開局＋Hungarian | 0.9746 | 0.9913 |
| 開局＋保險＋Hungarian | 1.0 | 1.0 |

合成驗證已飽和；官方 baseline 0.2845。〔待補：LB 分數與真實候選庫稽核結論〕

## 6. 訓練與推論步驟

- 一鍵推論：`bash scripts/release/run_release.sh <官方 CSV 目錄> <輸出目錄>`（A-054），輸出 `submissions/` 與 `SUMMARY.md`。
- 重新訓練：V2（`slurm/v2_train.slurm`、`configs/strength/rank_cnn_v2_seed2027.json`）；HumanSL 擬合特徵抽取（`slurm/humansl.slurm`）與 B2（`python -m go_aicup.tasks.strength.humansl_predict fit --method B2`）；GE2E（`configs/identity/pid_ge2e_v1_short.json`）。〔待補：完整重現指令與時間〕

## 7. 參數與權重

| 檔案 | 內容 |
|---|---|
| `outputs/v2_strength/model.pt` | V2 ResNet 權重與 config（A-024） |
| `outputs/v2_agg_calibration/result.json` | C2 每 k 溫度（A-025） |
| `outputs/humansl_model/model.pkl` | B2 模型（65,223 盤 Valid-S 擬合，A-044） |
| `outputs/pid_ge2e_v1_short/model.pt`、`eval_summary.json` | GE2E 權重、w 與 λ（A-035） |
| `outputs/pid_opening_v1_d4/summary.json` | 開局模型超參（A-029） |

〔待補：最終提交實際使用的檔案 sha256〕

## 8. 外部資源

見 `EXTERNAL_RESOURCES.md`：KataGo v1.18.2（MIT）、HumanSL 網路與 checkpoint、kata1 b18 網路（KataGo Neural Network License）、NVIDIA CUDA／cuDNN 函式庫、scikit-learn 等；生成式 AI 工具的使用範圍亦列於該檔。〔待補：每項的實際貢獻數字（LB）〕
