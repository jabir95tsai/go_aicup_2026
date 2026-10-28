# AI CUP 2026 圍棋棋力與風格預測：技術報告（骨架，2026-10-28 更新）

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
3. **幽玄級別條件向量**（`features/humansl_rankvec.py`，A-055；下稱 C）：凍結 HumanSL 全部權重，以學到的向量取代其 metadata 編碼器輸出。每個幽玄級別學兩個 384 維向量（輪到標註玩家／輪到對手），共 7.7k 個參數，在 train 切分的 15,000 盤上最大化真實級別下實際落子的 log-likelihood。每盤特徵＝10 個級別假設下本人／對手各自的平均 log p（20 維）。
4. **B2 題目層級模型**（`tasks/strength/humansl.py`，A-041）：每個 k 一個多類 logistic regression；在擬合玩家的合成題目（同一玩家抽 k 盤，特徵取平均，每 k 20,000 題）上擬合；以 Bayes 最佳解碼 `metrics.score_rank`（期望分數 p[r]＋(p[r−1]＋p[r+1])/e 的 argmax）輸出級別。正式模型（**AH＋C**）的特徵為 V2（A，10 維）＋HumanSL（H，30 維）＋C（20 維），以全部 Valid-S（65,223 盤、5,000 人）擬合；Valid-U 從不擬合（A-056）。
5. **備援**：題目缺 C 特徵時改用 AH（A＋H，A-044）；缺 HumanSL 特徵時改用 V2＋C2 聚合（每盤平均 log-prob、題內平均、每 k 溫度、`score_rank`；A-025）。

提交流程每份題目檔輸出四個檔案：`rank_<part>.csv`（AH＋C，首選）、`rank_<part>_ah.csv`（AH，對照）、`rank_<part>_allplayers.csv`（AH＋C 改以 Valid-S＋Valid-U 6,442 人擬合，LB 探針；本地交叉擬合增益 +0.0017，在雜訊內，A-058）、`rank_<part>_v2_fallback.csv`（V2＋C2）。

### 4.2 本地驗證（Valid-U 1,000 題，未見玩家）

| 方法 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| V2 ResNet＋C2 | 0.493 | 0.562 | 0.558 | 0.600 | 0.5532 |
| AH：V2＋HumanSL＋B2（對照） | 0.617 | 0.679 | 0.722 | 0.715 | 0.6833 |
| **AH＋C：V2＋HumanSL＋級別向量＋B2（首選）** | 0.624 | 0.698 | 0.730 | 0.721 | **0.6934** |

AH＋C 相對 AH +0.0102，玩家 bootstrap 95% 區間 [−0.0003, +0.0211]，四個 k 桶都上升；在另一組獨立玩家（S-select）上也是 +0.011（A-055）。官方 baseline 0.4631。

**未採用的嘗試（保留程式，T5 依 LB 判斷；D-003）：** 多 seed ensemble（A-033）、14 epochs（A-038）、貼目 0（A-043）、強引擎特徵（A-048）、HumanSL 主幹表示（A-051、A-052）；HumanSL 全網路微調（C4／C4b，A-057）在試跑階段就沒通過 gate。〔待補：Public／Private LB 分數〕

## 5. 競賽一：玩家辨識

### 5.1 方法

1. **開局模型**（`tasks/identity/opening.py`，A-029）：前 60 手做 D4 正規化；prefix back-off＋逐手 slot naive Bayes，以未在候選庫的訓練盤估計族群先驗。
2. **GE2E 保險模型**（`tasks/identity/ge2e.py`、`insurance.py`，A-035、A-053）：沿用 V2 ResNet 主幹，只看 ply ≥20 的黑方手（看不到開局），GE2E loss 訓練；候選人向量為其 100 盤的平均。
3. **融合**：開局分數＋λ·w·cosine（λ＝8，w 為 GE2E 的尺度參數），直接依分數取 top-5，**不做一對一指派**。合成驗證依建構必然一對一，所以 Hungarian 在本地最高；但模擬顯示真實測試若不是一對一，Hungarian 會扣 0.27–0.35、Sinkhorn 扣 0.08–0.20，一對一時兩者最多只加 0.04（A-059）。因此在確認一對一之前，預設不指派（D-004）。

提交流程每個候選庫輸出：`pid_<part>.csv`（融合、不指派，首選）、`pid_<part>_hungarian.csv`／`_sinkhorn.csv`（融合＋指派，用來以 LB 判斷是否一對一）、`pid_<part>_opening_only.csv`（只用開局）、`pid_<part>_style_only.csv`（只用保險模型）。後兩者與首選各只差一個元件，是受控探針。`pid_<part>_robust.csv`：每盤開局後驗與均勻分布混合（ε＝0.1）再加總，λ＝32；開局漂移時不崩，但沒有漂移時約 −0.014（A-062）。

### 5.2 本地驗證（彩排題目：public＝合成 split 1、private＝split 2，各 400 題）

| 方法 | split 1 | split 2 |
|---|---:|---:|
| 只用開局，不指派 | 0.9458 | 0.9530 |
| 只用保險模型，不指派 | 0.9385 | 0.9343 |
| **開局＋保險，不指派（首選）** | **0.9865** | **0.9828** |
| 開局＋保險＋Hungarian | 1.0 | 1.0 |

（D-004；只用保險模型一列為 2026-10-28 實測，分 k 桶 split 1：0.840／0.968／0.952／0.994，split 2：0.779／0.965／0.994／1.000。）合成驗證已飽和；官方 baseline 0.2845。〔待補：LB 分數與真實候選庫稽核結論〕

## 6. 訓練與推論步驟

- 一鍵推論：`bash scripts/run_release.sh <官方 CSV 目錄> <輸出目錄>`（A-054），輸出 `submissions/` 與 `SUMMARY.md`。
- 重新訓練：V2（`slurm/v2_train.slurm`、`configs/strength/rank_cnn_v2_seed2027.json`）；HumanSL 擬合特徵抽取（`slurm/humansl.slurm`）；級別向量學習與抽取（`slurm/humansl_rankvec.slurm`，`MODE=pack`／`train`／`extract`）；B2（AH：`python -m go_aicup.tasks.strength.humansl_predict fit --method B2`，AH＋C 加 `--rankvec`）；GE2E（`configs/identity/pid_ge2e_v1_short.json`）。〔待補：完整重現指令與時間〕

## 7. 參數與權重

| 檔案 | 內容 |
|---|---|
| `outputs/v2_strength/model.pt` | V2 ResNet 權重與 config（A-024） |
| `outputs/v2_agg_calibration/result.json` | C2 每 k 溫度（A-025） |
| `outputs/humansl_rankvec/vectors.pt` | 幽玄級別條件向量（10 級 × 2 × 384，A-055） |
| `outputs/humansl_rankvec_model/model.pkl` | B2 AH＋C 首選模型（65,223 盤 Valid-S 擬合，A-056） |
| `outputs/humansl_model/model.pkl` | B2 AH 對照模型（65,223 盤 Valid-S 擬合，A-044） |
| `outputs/humansl_allplayers_model/model.pkl` | B2 AH＋C 全玩家版（91,917 盤、6,442 人，A-058，LB 探針） |
| `outputs/pid_ge2e_v1_short/model.pt`、`eval_summary.json` | GE2E 權重、w 與 λ（A-035、A-053） |
| `outputs/pid_opening_v1_d4/summary.json` | 開局模型超參（A-029） |

〔待補：最終提交實際使用的檔案 sha256〕

## 8. 外部資源

見 `EXTERNAL_RESOURCES.md`：KataGo v1.18.2（MIT）、HumanSL 網路與 checkpoint、kata1 b18 網路（KataGo Neural Network License）、NVIDIA CUDA／cuDNN 函式庫、scikit-learn 等；生成式 AI 工具的使用範圍亦列於該檔。〔待補：每項的實際貢獻數字（LB）〕
