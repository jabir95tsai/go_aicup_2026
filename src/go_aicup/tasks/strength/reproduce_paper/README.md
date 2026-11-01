# Strength Estimator paper reproduction

分階段實作範圍、進度與驗收目標見 [plan.md](plan.md)。

**2026-11-01 最新決策：逐有效落子前盤面各自分析並取得 v，取代先前末盤共用 target。第三步 script 已撰寫，依要求未執行計算或測試。**

本目錄專用於 Chen、Shih、Wu，*Strength Estimation and Human-Like Strength Adjustment in Games*（ICLR 2025，arXiv:2502.17109v2）的重現。

- 分支：`reproduce/2502.17109v2`
- 起點：`396b71fd741f363b8e0133295094be16658c56f9`（`t1-pipeline-t2-models`）
- 本機論文：`reference/2502.17109v2.pdf`，完整閱讀含附錄 A–C，共 18 頁。
- PDF SHA-256：`83aa786ff7c659f9715fb030f4e9eb0c094b19f0ca6b0229aeeb9ebb37249f87`
- 作者程式：[rlglab/strength-estimator](https://github.com/rlglab/strength-estimator)，查核版本 `d9498964e2ba84b5798c40e3f5b8a91a8791d66c`（`iclr2025`）。本次僅閱讀，未複製或納入作者程式。
- 狀態：實作方向已確認（2026-11-01），可依本計畫進行實作；目前尚未實作模型或啟動訓練。正式長作業仍依 `CLAUDE.md` §9 先回報實測估時。

## 已確認方向（2026-11-01 更新）

使用者最新決定：競賽版每步使用 **128 個獨立排序組**，初始 **LR=0.04（作者 0.01 的四倍）**，總計 **32,500 次 optimizer updates**，取代先前的 512 組／10,000 次更新。此方案以作者 32 組／130,000 次更新為基礎，將四個小 batch 合併為一個大 batch，保持總排序組數 `128 × 32,500 = 32 × 130,000 = 4,160,000`。Optimizer 維持 SGD，momentum=0.9、weight_decay=0.0001；loss 仍按組／樣本平均，不改成加總。每組每級仍抽同盤指定玩家的七個不同位置。

這是線性 LR scaling 的近似方案；momentum、BN 與參數更新時點不同，尚無本版收斂或等效性驗證。未另行加入 warmup 或更動 momentum。正式訓練前須短測穩定性、數值與吞吐；分 k 成績及加速比未知。

Adam、AdamW、Muon 或其他 optimizer 的比較不列入目前重現範圍。訓練採 step-based 隨機抽樣，不以 epoch 或完整走訪所有棋局終止；預設 seed=2026。資料與原論文的差異依下方記錄處理。

**範圍決定（2026-10-30）：SE∞／SE-infinity 移出本次實作與驗收範圍。** 不建立合成最低級、隨機合法著手生成器或其專用 loss、訓練與評估入口。下方 SE-infinity 說明僅保留為論文背景。

### 五項決策已確認（2026-11-01）

使用者同意以下五項建議，並指定不實作 SLsum／SLvote：

| 項目 | 已確認設定 |
|---|---|
| Policy head | 本版加入，使用 strength／policy／value 三頭；實作前核對固定版本 MiniZero 的輸入與 target 時點，驗證無目標落子洩漏 |
| Centroid 資料 | 使用全部 800,000 盤 training 棋局的指定玩家有效位置（`ply >= 30`），按位置等權計算各級平均，不抽一小部分棋局或每盤只取七手 |
| 80/20 split | 每級按棋局隨機切分為 80,000／20,000 盤，split seed 固定為 2026，首次沿用 `data.cv.make_strength_v1_split` 產生；所有本目錄訓練 run 共用同一份切分表，記錄玩家重疊 |
| KataGo 規則／貼目 | 日本規則、貼目 6.5；是 teacher 分析假設，不是原局規則已知值 |
| 本次範圍 | 僅基本 SE 三頭模型、逐有效盤面 soft value 偽標籤與對應評估；不實作／訓練 SLsum、SLvote，也不建立 SL 對照 gate |

本版總 loss 採 `L_BT + L_policy + L_value`，各項初始權重為 1。SE-infinity 維持排除。BF16、BN 行為、teacher visits 與監控頻率等屬原始碼查核／短測項目，並非以上五項決策尚待確認。

### 目前競賽版訓練預算

| 項目 | 設定 |
|---|---:|
| 人類級別數 | 10 |
| 每組每級位置數 | 7 |
| 每次更新的排序組數 | 128 |
| Global batch（SE 位置數） | 8,960 |
| 8×H200 每卡完整組數 | 16 |
| 8×H200 每卡 SE 位置數 | 1,120 |
| Optimizer updates | 32,500 |
| 全程 SE 位置使用量（含重複） | 291,200,000 |

此設定保持十級、32 組／130,000 updates 參考方案的總位置使用量，更新次數為其四分之一；相對舊的 512 組／10,000 updates，總位置使用量減少 18.75%。這不代表每盤都被讀取或原論文 Fox 十一級位置量完全相同。

**已確認 schedule（2026-11-01）：** 完成第 `100,000 / 4 = 25,000` 次更新後，LR 由 `0.01 × 4 = 0.04` 降為 `0.005 × 4 = 0.02`，訓練至 `130,000 / 4 = 32,500` 次更新。本節以下的 32 組、130,000 steps 與 LR 0.01／0.005 為原論文方法說明。

## 訓練流程：論文原版與競賽資料版本

本節分開說明論文原版，以及使用者先前指定的競賽資料改動。取消 epoch 已確認；十級、每級十萬盤、80/20 與排除前 30 手保留為既有資料要求，不能稱為原論文資料規格。設定、split／有效位置索引與樣本組裝已實作；模型與訓練尚未實作。

### 一次 optimizer update 如何產生

作者固定版本的 `se_go.cfg` 設 `bt_num_batch_size=32`、`bt_num_position_per_rank=7`、`bt_use_same_game_per_rank=true`。對每個 step：

1. 建立 32 個獨立排序組。每組包含全部 11 個人類級別，由弱到強排列供程式計算；論文公式則以強到弱表示，兩者等價。
2. 每組每級，從該級訓練棋局均勻隨機選一盤。不同組或不同 step 可重複抽到同一盤，不追蹤「本輪尚未抽過」的棋局清單。
3. 作者程式隨機選黑或白，列出該顏色在這盤的著手位置，打亂後取 7 個。因此同一個七手集合內不重複；跨組／跨 step 可重複。論文文字只說每級抽七個 state-action pairs，同盤同色是作者設定補充。
4. 組合 18-plane 歷史輸入並做空間對稱增強。論文附錄 A 指八步歷史與兩個顏色層；已查核固定版本 MiniZero，實際歷史通道為當手方／對手，輸入位於當手落下前，policy target 為當手落點。
5. 整批 forward 產生逐位置 strength score、policy 與 value。SE 每步位置數為 `32 × 11 × 7 = 2,464`。
6. 每組每級先平均七個 score，分組算排序 loss，再平均全部 32 組；與 policy/value loss 相加，做一次 backward、一次 SGD update，最後推進 scheduler。
7. 重新抽取下一 step 的樣本，持續到 130,000 次 updates。沒有 epoch，也沒有全資料不放回的逐輪 shuffle；步內選位置時的 shuffle 仍保留。

競賽版有一個已知標註差異：一列只標註指定玩家的 rank/color，不能沿用作者「隨機黑白」後把指定玩家的 rank 當成對手 rank。應從該盤指定玩家的有效位置取七手。

### 排序 loss 的意義與公式

網路輸出實數 beta，強度定義為 lambda=exp(beta)。對組 g、級別 i 的七個位置，先計算 `b[g,i] = mean(beta[g,i,1:7])`；相當於對七個 lambda 取幾何平均後再取 log。

若 n 個級別以最強到最弱排列，論文 Eq. (5) 每組的 loss 為：

```text
L_group = sum(i=1..n-1) [logsumexp(b[i], ..., b[n]) - b[i]]
```

它依序要求最強級別勝過其餘級別、次強級別勝過更弱級別，直到只剩最弱一級。作者程式將此和除以 `n-1`，再對 32 組平均；`bt_use_weight=false`，七個位置等權。重現作者訓練尺度時必須保留這個正規化。級別平均與 logsumexp 使用 FP32。

這個 loss 約束跨多個位置的平均表現；不要求每個強者著手的 score 都高於每個弱者著手。七手用於降低單一決策的雜訊；論文沒有提供 m 的消融結果來證明七手最優。

### SE-infinity 的額外最低級（論文背景，本次不實作）

SE-infinity 增加合成最低級 r_infinity，每組另有七個合成位置。由人類棋局位置修改為隨機合法著手，將這一級排在所有人類級別以下，以約束模型對人類資料分布外著手的分數。隨機合法著手不等於隨機空點，須符合作者環境的合法性規則。

原版每步為 `32 × 12 × 7 = 2,688` 個位置；本次競賽版不加入合成最低級。合成樣本不構成另一套有 epoch 的資料集，而是在抽樣時產生。

附錄 A 說合成位置不計 policy/value loss；固定版本作者程式卻將其納入，並設定合成 value=-1。此差異僅記錄為論文查核結果，不再列為本次實作待決策事項。

### 三個 heads 與總 loss

論文使用 20 residual blocks、256 channels 的共享網路，輸出 policy、value 與 strength。作者 SE 訓練總 loss 為等權相加：

```text
L = L_policy + L_value + L_BT
L_policy = mean(cross_entropy(policy_logits, human_policy_target))
L_value  = mean_squared_error(predicted_value, game_outcome_target)
```

SGD：momentum=0.9、weight_decay=0.0001。第 1–100,000 次更新用 LR=0.01，完成第 100,000 次更新後降為 0.005，直到第 130,000 次。這是固定的 optimizer-update schedule，不按抽到多少不同棋局決定。

### 競賽版目前資料要求與偏離

- 十個人類級別：`D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D`，內部編碼 0..9。
- 每級 100,000 盤，training 80,000、validation 20,000；總計 training 800,000、validation 200,000 盤。已確認按棋局分層隨機切分、split seed=2026，使用 `data.cv.make_strength_v1_split`；玩家可能跨集合，須記錄重疊，不能把結果解讀為未見玩家泛化。
- **已確認位置規則（2026-10-30）：training、validation 與 centroid 計算均排除全局第 1–30 手**，零起算 `ply >= 30`，並且須屬於指定玩家。八步歷史保留真實盤面，不把歷史中較早的盤面清空。Training 每次抽七個不同位置前先確認有足夠有效位置，否則報告資料問題，不靜默改成重複抽樣。Validation 與 centroid 按相同有效位置規則推論、聚合，不限每盤七手。
- 本版每步 SE：`128 × 10 × 7 = 8,960` 個位置，32,500 steps 共 `291,200,000` 次位置使用（含重複）。這不是 800,000 盤資料被完整讀取的次數保證。
- 十級、盤數、切分、前 30 手排除及指定玩家取樣均為競賽版差異。原論文用 Fox 十一級，未設定排除前 30 手；完全原版結果重現需撤回這些差異並準備原始資料。
- 已確認使用既有 KataGo 強引擎分析每個有效落子前盤面，產生 soft value 偽標籤監督 value head；不再以 scalar-only 作為本版預設方向。此監督不同於論文的真實勝負，仍須明記為偏離；policy head 的輸入時點與 target 對齊另行驗證。
- Centroid 僅由 training 資料計算，使用相同的 `ply >= 30` 與指定玩家位置規則；已確認使用全部 800,000 盤 training 棋局的有效位置，按位置等權。Validation 不用於抽訓練樣本、擬合 centroid 或選切點。逐位置 teacher 使用該位置之前的完整歷史，不裁切開局。

### 逐有效盤面 soft value（2026-11-01 最新決策）

原末盤共用 target 方案由使用者明確改為逐盤面分析。每個有效 `ply` 對應落子前盤面，由強 KataGo 在日本規則／6.5 貼目下各自估計 `p_black`。黑方 v=`2*p_black-1`，白方取負；不硬分類、不當成原局真實勝負、不寫入 RE。同局不同位置不再共用末盤 v。

原資料無勝負結果，這仍是 teacher 偽標籤，與論文 outcome supervision 不同。規則／貼目是分析假設，visits、品質門檻與成本待 training 小樣本短測，不能依 validation rank 挑選。


## 先盤點目前可以沿用的程式

以下路徑以 `src/go_aicup/` 為根；沿用以 import 為原則，避免複製出另一套共用管線。

| 現有程式與介面 | 可沿用用途 | 重現時的限制／需調整處 |
|---|---|---|
| `paths.py`、`artifacts.py` | 根目錄解析、JSON 與雜湊紀錄 | 新實驗使用獨立資料及輸出目錄 |
| `data/sgf.py`：`parse`、座標轉換 | 官方純落手資料解析 | strict parser 僅接受官方簡化格式；`parse_lenient` 會丟棄 pass，不能直接用於完整 Fox 重現 |
| `data/preprocess.py`、`data/store.py`：`GameStore` | 一次轉檔、memmap 著手序列與 metadata | 不能把官方十級當成 Fox 十一級；外部資料格式擴充仍須留在唯一 parser／資料管線 |
| `features/sgf_planes.py`：`moves_to_planes` | 盤面重播、提子及既有合法性稽核 | 僅有 immediate-ko 稽核，不能直接作為作者 positional-superko 下的隨機合法著手生成器 |
| `data/board_cache.py`：`BoardCache`、`history_batch`、`assemble_inputs` | 壓縮盤面、八步黑白歷史、18-plane 輸入 | 快取包含著手後盤面；本版已在 dataset 讀取前一手歷史、轉為當手方／對手，保留實際當手顏色；rank metadata 固定十級 |
| `data/cv.py`：`make_rank_folds`、`assert_player_disjoint`、`make_rank_questions` | 官方資料的玩家互斥驗證與 k=5/10/15/20 題目 | 所有 split 仍由此模組產生；論文 candidate/query protocol 如需新增，也集中於此，不能在本目錄另造 split |
| `tasks/strength/shared/dataset.py`：`PackedBoardDataset`、`apply_d4` | lazy 讀取、D4 變換、worker 開檔處理 | 新增按 rank/game 抽樣；已確認加入 policy head，policy target 須同步變換；既有 epoch sampler 不符合作者分組抽樣 |
| `tasks/strength/shared/model.py`：`ResidualBlock`、`StrengthResNetBackbone` | 20 blocks、256 channels 與 scalar head 的結構參考 | 目前 block 回傳 residual sum 後沒有 ReLU；需比對作者固定版本及 MiniZero，不能僅因深度／寬度相同便宣稱等價；policy/value heads 尚缺 |
| `tasks/strength/v1/model.py`：`StrengthResNet` | scalar score forward 介面參考 | 僅輸出 scalar；完整重現須補作者三種輸出，並驗證 backbone 與 head 細節 |
| `tasks/strength/v1/losses.py` | FP32、logsumexp 穩定排序 loss、DDP 可微聚合的參考 | 固定十級，且把整個 global batch 合成一個各級平均；須改為每組分別算 loss 後平均，不能直接套用 |
| `tasks/strength/shared/runtime.py`、`shared/training.py` | 單節點 1–8 GPU、AMP、RNG、環境記錄 | 部分 API 依賴現有 manifest/config；`scheduler_for` 為 cosine，論文使用 step schedule |
| `tasks/strength/shared/checkpoint.py` | 原子保存、optimizer/scheduler/RNG 續跑 | manifest、cursor 與 sampler state 須適配本實驗；現有 latest-only 保留策略需明確記錄 |
| `tasks/strength/v1/infer.py`、`v1/evaluate.py` | 分片推論、逐手分數保存及聚合流程參考 | V1 的 training-only cutpoints 與每盤等權聚合不能直接當論文解碼；作者以所有被選位置平均、最近 rank centroid 解碼 |
| `metrics.py`：`rank_report`、`format_report` | 官方十級移植版的分 k 報表、accuracy 與 within1 | 論文十一級報表需獨立支援；`score_rank` 是官方機率解碼，不適用於論文 scalar nearest-centroid |

## 論文規格與目前架構的差異

論文 §3.1–3.2 定義逐手分數 beta，先對每級 m 個位置取算術平均，再對強到弱的級別順序計算 listwise Bradley–Terry 負對數似然。這等價於對 exp(beta) 的幾何平均進行比較，不是逐手 cross entropy。

| 項目 | 論文／作者固定版本 | 現有實作或資料 |
|---|---|---|
| Go 資料 | Fox，11 級：9d 到 1d、1–2k、3–5k | 幽玄，10 級：D 到 6D；不可逐級等同 |
| 訓練／候選／查詢 | 每級 45,000／100／900 盤，共 495,000／1,100／9,900 | 官方一百萬盤與既有 player-disjoint split |
| 輸入 | 18 planes；固定版 MiniZero 為八步當手方／對手歷史與兩個顏色層 | 本版 dataset 已對齊落子前時點、通道及零 padding；未執行作者二進位比對 |
| 網路 | 20 residual blocks、256 channels，policy/value/strength 輸出 | V1 僅 strength，V2 為十類分類 |
| 每步 SE 樣本 | 作者 config：32 組 × 11 級 × 7 手 = 2,464 | V1 以全域級別平均計一次排序 loss |
| 每步 SE-infinity 樣本 | 32 組 × 12 級 × 7 手 = 2,688，含隨機合法著手的最低級 | 目前沒有合成最低級 |
| 每組每級抽樣 | 作者 `bt_use_same_game_per_rank=true`：同盤、同色取 7 個位置 | 既有全量 epoch 順序不是此抽樣方式 |
| 優化 | SGD，LR 0.01，100,000 steps 後 0.005，共 130,000 steps；作者 momentum=0.9、weight_decay=0.0001 | 既有訓練器與 cosine scheduler 需適配 |
| 聚合／解碼 | candidate 各級所有位置平均為 centroid；query 所有選定位置平均後取最近 centroid | V1 cutpoints、V2 C2/B2 均非論文規格 |
| 評估 | 各級、各 N 重複 500 次；exact accuracy、within-one accuracy 與 95% CI | 官方主要指標為 exact + 差一級/e，須另列而不能混稱 |

論文附錄 A 說 SE-infinity 的合成位置不計 policy/value loss；查核的作者 `train.py` 卻對整個 batch 計算這兩項 loss，`st_data_loader.cpp` 同時會將合成樣本 value 設為 -1。這是文件與程式的差異；SE-infinity 已移出本次範圍，因此此項僅保留查核紀錄。

## 已確認實作計畫（2026-11-01）

### 範圍與資料選擇

本次實作 Go strength estimation：基本 SE 三頭模型、逐有效盤面 soft value 偽標籤與 §4.2 的對應棋力評估。SLsum／SLvote 已明確移出範圍。§4.4 的 2-rank／3-rank、附錄 B 的位置消融、SE-MCTS 與 chess 均非本次必要實作，整篇尚不能宣告重現完成。

原始結果重現需要 Fox 原始資料、十一級標註、勝負結果與 candidate/query 清單。作者 README 要求自行準備 Fox SGF，未提供可直接視為同一份實驗資料的固定清單。原始切分與玩家是否互斥仍待確認；不能從本文的 separate dataset 推定 player-disjoint。

使用者已選擇現有官方資料，明確命名為「幽玄十級方法移植」。官方資料沒有勝負結果，無法忠實監督 value head；不得補造勝負或以另一模型的估值冒充論文標籤。本版使用基本 SE 加逐有效盤面 teacher soft value supervision，不安排 SL 對照；teacher 偽標籤取代真實勝負、資料來源與十級設定全部記為偏離。SE-infinity 及其合法著手生成與驗收皆不在本次範圍。

### 本目錄的程式配置

計畫已確認，專用程式集中於本目錄；檔案狀態如下；目前已建立 config／prepare／dataset，其餘為後續實作：

- `config.py`：資料模式、級別順序、模型、分組抽樣、loss 規則、schedule 與 paths。
- `dataset.py`：包裝既有 store/cache，提供每組各級同盤指定玩家的 7 個不同位置，並讀取對齊的每盤 soft value target。
- `label_value.py`：KataGo 強引擎逐有效盤面分析、分片、續跑與偽標籤 manifest；不覆寫原始棋譜或 RE。
- `model.py`：與作者固定版本對齊的 backbone、strength/policy/value 三個 heads。
- `losses.py`：`[groups, ranks, 7]` 分組平均與排序 loss；加入逐有效盤面 soft target 的 value MSE；policy auxiliary loss 於輸入／target 對齊後處理。
- `prepare.py`：呼叫唯一資料與 split 模組，保存本實驗 manifest，不重寫 parser 或 split。
- `train.py`：128 組／step、32,500-step SGD、單卡／DDP、checkpoint、續跑與成本紀錄。
- `infer.py`：保存逐手 score、各盤 score_sum 與 position_count，支援位置加權平均。
- `evaluate.py`：全量 training centroid、nearest-centroid、固定 validation 題組評估及圖表重建；論文式重複抽樣與題組結果需標明各自 protocol。
- 專案根目錄 `configs/strength/`：每個模型與資料模式各自的可重建設定。

專用 processed 與 run 目錄建議為 `data/processed/reproduce_paper/`、`outputs/rank/reproduce_paper/<run-id>/`，路徑皆走 config；完整盤面沿用既有壓縮快取，避免複製全量資料。

### 假設、指標與 gate

假設：跨多位置的排序 supervision 比逐手分類更能辨識棋力順序；本版幽玄資料的分 k 桶分數與對競賽主線的增益均未知；不再安排同規模 SL 對照。

1. 正確性 gate：分組 loss 與直接公式在 FP32 的絕對差 <=1e-6；單卡與兩程序 loss／參數梯度相對差 <=1e-5；固定 seed 續跑的抽樣與 optimizer steps 一致。容差是本計畫提案，並非論文數字。
2. 資料 gate：train/candidate/query 棋局零重疊；本次每級精確 80/20 已確認按棋局切分（seed 2026）並記錄玩家重疊，不要求本模式玩家互斥。記錄來源、級別映射、每級盤數、實際顏色與指定玩家位置；不將 ID 作輸入。
3. 原論文指標目標：N=15 exact accuracy >0.80、N=100 約 0.975；within-one 在 N=6 >0.99（§4.2）。這些是原文目標，尚無本地測量，不承諾移植版達標。圖中非明文數字不編造。
4. 幽玄模式：固定 validation 題目，完整回報 k=5/10/15/20 的 exact、within-one、官方 score 與四桶等權平均。棋局切分結果不能和主線 player-disjoint 分數直接比較；原先相對 SL 至少 +0.01 的 gate 隨 SL 對照移除，不另訓練 baseline 來驗收本次 SE。分數結果仍需如實記錄，不能只憑 training loss 宣稱泛化改善。
5. 正式數字重現的容差、95% CI 算法、原始抽樣 seed 與資料清單尚待查核／確認，不能只達到第 3 點便宣稱原始實驗完全重現。

### 算力與啟動順序

附錄 A 報告 Go 訓練約 242 RTX A5000 GPU-hours；此數字不能直接換算成本機 3060 Ti 或 H200 時間。原先包含 SE-infinity 的三模型 726 GPU-hours 占位估算撤回。本版只規劃基本 SE 三頭模型與 teacher 偽標籤，且 batch／總步數已修改，正式成本以 H200 短測重估；不沿用 A5000 數字作為本版預算。資料處理與推論另計。

先完成 CPU 數值驗證，再於 H200 dev 配額中做單卡 100–500 steps 吞吐與記憶體測量，必要時測單節點 DDP；回報 steps/s、GPU 峰值、主機 RAM、預計訓練／推論 GPU-hours 與儲存量，再依 `CLAUDE.md` §9 確認長作業。不得以 8 卡直接假定八倍加速；8GB 裝置若需 microbatch，需說明 BatchNorm 行為是否偏離作者設定。本次未提交 Slurm 作業。

## 後續查核、短測與正式作業估時

五項模型／資料決策均已確認，不再要求重複確認。實作期間需完成以下工程查核：

- 固定版本 MiniZero 的輸入時點、顏色層、policy target、residual block 與多卡 BN 行為；以具體棋局樣本驗證，避免輸入包含 policy 答案。
- Teacher visits、標籤品質與失敗處理：在日本規則／6.5 下用固定 training 小樣本比較，凍結設定後再處理全量資料。
- FP32 基準、BF16 與 BN／SyncBN 的數值、顯存和吞吐驗證；監控與 checkpoint 頻率依成本決定。
- 偽標籤生成、training centroid 全量推論與正式 32,500-step 訓練的時間、GPU-hours 和磁碟需求；依 CLAUDE.md §9，在長作業前回報估時並確認啟動。

原論文 Fox 資料清單等未知事項僅為原版結果查核限制，不阻擋已確認的幽玄十級實作。

## 查核來源

- 本機 PDF：§3、§4、附錄 A–C。
- [作者 README（固定版本）](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/README.md)。
- [SE config](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg/se_go.cfg)、[SE-infinity config](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg/se_infty_go.cfg)。
- [網路](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/bt_network.py)、[訓練與 loss](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/train.py)、[抽樣](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/st_data_loader.cpp)、[評估](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/misc/evaluator.cpp)。


## 第一、二步操作與驗收（2026-11-01）

依使用者最新指定，`reproduce_paper` 所有訓練共用同一份 training／validation 切分，split seed 固定為 2026；初始化／抽樣 seed 預設 2026，後續可獨立修改，但不得重新切分資料。共用切分表為 `data/processed/reproduce_paper/seed2026/games.parquet`，各 run 應記錄相同的切分雜湊並重用固定 validation 題組。詳細規則與新增驗收見 [plan.md](plan.md)。V1／V2 等其他訓練流程不受此規則影響。

資料準備第二步的固定切分已實作：`split_seed` 固定 2026，與可獨立調整的訓練 `seed` 分開。`prepare.py` 首次使用 split_seed 建立資料，後續驗證並重用同一份 manifest、切分表及位置索引，不重抽或覆寫；初始化 seed、模型 channels、run 輸出目錄與 loader 設定不進入共用資料身分。固定題組由 `questions.py` 讀取同一資料目錄。已驗證初次建置、跨訓練 seed 重用及不同訓練 seed 下獨立重建的切分完全一致。

已建立 `config.py`、`configs/strength/se_seed2026.json`（專案根目錄下）、`prepare.py` 與 `dataset.py`。正式資料準備入口：

```bash
python -m go_aicup.tasks.strength.reproduce_paper.prepare
# 或指定獨立 config
python -m go_aicup.tasks.strength.reproduce_paper.prepare --config path/to/config.json
```

此命令會讀取既有 packed cache；本次尚未對官方全量資料執行。預設輸出為 `data/processed/reproduce_paper/seed2026/`：

- `games.parquet`：唯一 CV 模組產生的 split、原棋局 metadata 及有效位置數。
- `{train,valid}_game.npy`：各集合的棋局索引（uint32）。
- `{train,valid}_offsets.npy`：每盤在位置陣列中的起訖範圍（int64，長度為盤數加一）。
- `{train,valid}_ply.npy`：指定玩家且 `ply >= 30` 的全部有效位置（uint16）；不只保存七手。
- `manifest.json`：固定 split_seed、cache 身分、程式／產物雜湊、每級盤數／位置數與玩家重疊。相同設定可重跑驗證並重用；設定不同、產物損壞或未完成的目錄會明確拒絕覆寫。

每盤少於七個有效位置會報錯，不能靜默重複抽樣。資料準備僅讀取 metadata／顏色，不解壓全量盤面；不建立另一套 parser 或 split。

```python
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset

config = read_config()
dataset = PaperDataset(config['paths']['dataset_dir'], split='train', augment=True,
                       seed=config['seed'])
try:
    sample = dataset[(0, 0)]  # (有效位置序號, augmentation round)
    # planes: uint8[18,19,19]; policy_target: int in 0..361
    # rank_id, player_color, game_index, ply, sample_index, d4: metadata
finally:
    dataset.close()
```

`PaperDataset` 支援批次 `__getitems__`、process-local memmap、LRU 及 spawn workers；增強由 seed／round／sample index 決定。第 4 步的分組 sampler 尚未加入。識別資訊不進入模型通道；teacher 尚未生成，樣本不提供虛構 value label。`soft_value_target(p_black, player_color)` 已提供有限值／範圍檢查及黑白視角轉換，供後續 teacher 接入。

### 已查核的 state/action 契約

固定作者版本 `d9498964e2ba84b5798c40e3f5b8a91a8791d66c`：

- [`strength/misc/game_wrapper.cpp`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/misc/game_wrapper.cpp) 的 `calculateFeatures` 直接呼叫 loader。
- [`minizero/minizero/environment/base/base_env.h`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/environment/base/base_env.h) 的 `getFeatures(pos)` 只重播索引小於 pos 的著手，`getPolicy(pos)` 使用 pos 的 action。
- [`minizero/minizero/environment/go/go.cpp`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/environment/go/go.cpp) 的 `getFeatures` 以最新歷史優先，每對通道為當手方／對手；不足八步補零，最後兩層為黑／白回合。

因此本版對選中的 ply 讀取 cache 的 `ply-1` 及更早盤面，白方交換黑白歷史通道；policy 使用所選 ply 的著手。`ply=0` 歷史全零；`ply >= 30` 不會清空前 30 手的歷史。盤面及 policy 共用相同 D4；本 repo 維持自身 row-major 座標，pass 類別為 361（官方訓練資料無 pass）。此為來源閱讀與合成棋局對照，未執行作者 C++ 二進位逐值比對。

針對本次功能的 CPU 合成測試：

```bash
PYTHONPATH=src python -m pytest tests/test_reproduce_paper_data.py -q
```

本次結果：**7 passed in 6.80s**（CPU、合成資料）。涵蓋十級精確 split／玩家重疊、全部有效位置、重用與設定衝突、有效位置不足報錯、提子／落子前歷史／白方通道／padding、八種 D4 的全部 361 個座標與 pass、soft value 視角與非法值、pickle／spawn DataLoader 一致性。未啟動 teacher、GPU 作業、訓練或正式 centroid 推論。


## 資料準備第五步：固定 validation 題組（已實作）

此處的「第五步」是資料準備流程的固定題組，不是 plan.md 八階段中的第五步三頭模型／loss。

先完成基本資料準備，再以獨立入口生成題組：

```bash
python -m go_aicup.tasks.strength.reproduce_paper.prepare
python -m go_aicup.tasks.strength.reproduce_paper.questions
# 自訂資料目錄／題數時，兩者使用同一 config
python -m go_aicup.tasks.strength.reproduce_paper.questions --config path/to/config.json
```

新增 `questions.py`，只讀取 `dataset_dir/games.parquet` 與既有 manifest，確認來源切分 seed=2026 及棋局表雜湊，再呼叫唯一 CV 模組的 `make_strength_v1_questions`；不讀盤面、不重新切分、不啟動 GPU。

設定 `questions.validation` 預設 **10,000**，必須是 40 的正整數倍。固定題組 seed=2026，與 config 的訓練 `seed` 無關。四個 k 桶各 2,500 題，各 k／rank cell 各 250 題；每題同玩家、同色、k 盤互異，全部來自 validation，與 training 棋局零重疊。不同題目可以重用棋局／玩家，不能將所有題目視為獨立樣本。

同一 prepared 資料目錄新增：

- `validation_questions.parquet`：`question_id`、`player_id`、`rank_id`、`k`、`color`、`games`；games 為既有 cache 的 game_index。
- `validation_eligibility.parquet`：各 k／rank 的可用玩家顏色組與玩家數。任一級別缺少可組成 k 盤的玩家顏色組，直接報錯，不以重複棋局補足。
- `questions_manifest.json`：來源棋局表與生成程式雜湊、固定 seed、題數、各 k／rank 數量、涉及的不同棋局／玩家數與兩個產物的雜湊。

後續所有 run 直接讀取同一份題組，記錄 questions manifest 的 build_id。重跑入口會驗證並重用產物；訓練 seed 改變不重抽題組。題數／來源／程式身分改變、檔案損壞或輸出不完整會報錯，不靜默覆寫。

合成驗收：

```bash
PYTHONPATH=src python -m pytest tests/test_reproduce_paper_questions.py tests/test_reproduce_paper_data.py -q
```

結果 **17 passed in 6.38s**：包含既有七項資料測試，以及題組平衡、train／valid 隔離、同玩家同色、題內不重複、跨訓練 seed 重用、獨立生成結果一致、題數衝突、損壞偵測、來源 seed 拒絕與 eligibility 不足。尚未對官方全量資料實際產生題組。


## 資料準備第二步：固定共用切分（已實作）

```json
{
  "seed": 3030,
  "split_seed": 2026
}
```

以上是更換訓練 seed 的設定片段：保留同一 `paths.dataset_dir`，完整 config 的其他設定照舊。所有 reproduce_paper 訓練共用預設 `data/processed/reproduce_paper/seed2026/`，run 權重與 log 放各自的 output_dir。`split_seed` 只接受整數 2026；訓練 `seed` 不再決定 split 或共用資料 build_id。

執行 `python -m go_aicup.tasks.strength.reproduce_paper.prepare --config path/to/config.json`：首次依固定 split seed 產生每級 80/20 切分與有效位置索引；已存在時驗證資料身分、必要檔案集合與雜湊後直接重用。改變訓練 seed 不重算 split，也不改寫資料檔。資料來源、切分盤數或有效位置規則不同，必須另建明確版本；缺檔、損壞或身分不符會報錯。

Dataset manifest identity 格式更新為 version 2，以 `split_seed` 取代原本混用的 `seed`。舊版 manifest 不會自動覆寫／遷移，須保留原產物並在新版本資料目錄重新準備。`PaperDataset` 的 D4 seed 預設 2026，也可明確傳入訓練 seed；它不從切分 seed 推導訓練隨機性。

第二步與第五步整合驗收：

```bash
PYTHONPATH=src python -m pytest tests/test_reproduce_paper_data.py tests/test_reproduce_paper_questions.py -q
```

結果 **24 passed in 7.17s**。新增驗證包含：更改訓練 seed／模型 channels／output_dir／loader 後不重抽且資料雜湊與修改時間不變；不同訓練 seed 初次建置結果一致；非法 split_seed 拒絕；共用索引損壞偵測；完整 synthetic cache → prepare → questions 在 seed 3030／4040 下共用完全相同產物。未執行官方全量準備或訓練。


## 第三步：準備每個有效盤面的 v（程式已撰寫，未執行）

先具備 `prepare.py` 產物。本次沒有啟動 prepare、teacher、測試、GPU 或訓練。
以下命令供之後使用，`VISITS` 必須先由 training 短測決定；不提供預設正式預算。

```bash
# 固定 training 前 100 局；同一小樣本比較不同 visits 時必須換 out。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out outputs/rank/reproduce_paper/value_pilot_v32 --split train \
  --visits 32 --limit-games 100

# 正式分片範例：每個 worker 各自使用 shard 0..7，指定自己的 GPU。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out data/processed/reproduce_paper/value_v1 --split train \
  --visits "$VISITS" --num-shards 8 --shard 0

# 全部分片成功後匯出一個 sample_index 對齊的 v 向量（此命令不啟動引擎）。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out data/processed/reproduce_paper/value_v1 --split train --export
```

上例 32 visits 僅示範小樣本呼叫，不是已通過品質 gate。完整 valid 也要取得 v 時，用同一組 teacher 設定改 `--split valid` 分片執行與匯出。以 `CUDA_VISIBLE_DEVICES` 配置各 worker；不會自動提交 Slurm 或啟動多 GPU。`--engine-config` 可指定執行配置，變更後須使用新輸出版本。

每個 `<out>/<split>/shard_XXXXX/` 保存與該分片有效位置同序的 `value.npy`、`p_black.npy`、`score_black.npy`、`score_stdev.npy`、`visits.npy`、`status.npy` 及來源 manifest。分片按共用 game array 的 row modulo 分配，局內按原有效 ply 排序。狀態 0=未完成、1=成功、2=失敗；缺失值為 NaN。每局完成先 flush 數值再提交 status，`--retries` 預設 2 次；重跑相同命令只處理非成功位置。錯誤見 `failures.jsonl`／`katago_stderr.log`。硬中斷留下 `.writer.lock` 時先確認原程序已停止再移除鎖。

`--export` 要求每個有效位置成功、分片完整、來源與 teacher 身分一致，才發布 `<out>/<split>/value.npy` ，成功後自動刪除該 split 的所有分片、續跑狀態、log、暫存檔及 manifest，只留 `value.npy`。其 shape 為 `(有效位置數,)`，dtype=float32，與 `{split}_ply.npy`、`PaperDataset` 的 sample_index 完全同序。透過原 game／offsets／ply 索引和已綁定雜湊的 games.parquet 回溯 `(rank, game_id, ply)`，不重複儲存大量字串。生成及匯出驗證期間以 SHA-256 綁定來源、引擎、強網路、設定及程式，完成清理後不保留這些追溯檔；只接受已登記 kata1 強網路，不使用 HumanSL。

```python
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
samples = PaperDataset(
    'data/processed/reproduce_paper/seed2026', split='train',
    label_dir='data/processed/reproduce_paper/value_v1/train',
)
# samples[i]['value_target']: 此盤面、指定玩家視角的 v
# samples[i]['value_weight']: 1.0；未提供 label_dir 時不輸出虛構 v。
```

目前只提供完整標籤模式，失敗位置須重試成功才可匯出；不刪除 ranking 樣本。scoreStdev 如引擎未提供則為 NaN，不能冒充信心。品質遮罩／權重政策仍待短測。

成本以「有效位置數／實測 positions/s」估算；各分片每位置 21 bytes 數值陣列，匯出 v 額外 4 bytes（尚不含來源、log 與暫存匯出檔）。全量磁碟、RAM 與 GPU-hours 未實測。新增 `tests/test_reproduce_paper_value.py` 僅供後續驗收，本次未執行。


### 完成後只保留 value.npy（2026-11-01）

`--export` 完成完整性檢查並原子發布後，自動清除該 split 的 `shard_*`、status、log、暫存檔與 manifest。最終輸出僅為 `<out>/train/value.npy`、`<out>/valid/value.npy`（依實際完成的 split）。原始 cache／prepared 索引與其他 split 不受影響。

匯出失敗時保留續跑資料；清理中斷時重跑相同 `--export` 接續清理。完成後重跑 export 只檢查既有 v 的長度、dtype 與數值範圍，不重新啟動引擎；原標籤生成命令會拒絕重新建立分片。Dataset 僅讀取 value.npy，檢查形狀／dtype 及讀取樣本值。因最終不保留 manifest，無法再驗證同長度檔案的來源或 teacher 身分，須將 label_dir 指向正確的 split／資料版本。

本次僅修改程式、文件與未執行的驗收案例，沒有執行計算、測試或刪除現有產物。

### Nano4 Slurm 執行

腳本：`slurm/label_value.slurm`。預設 `8gpus` partition、單節點 8 張 H200、8 tasks（每 task 1 GPU／12 CPU）、總 RAM 128GB、時限 48 小時。這是資源申請上限，並非實測完成時間；未量測 GPU-hours。執行前須已有 prepare 產物，腳本不自動重新準備資料。

從 repo 根目錄提交，第一個參數必須是已選定的正整數 visits：

```bash
# 先設定 VISITS；此處不替正式預算選值。
sbatch slurm/label_value.slurm "$VISITS"

# 也可指定輸出位置與單一 split。
sbatch slurm/label_value.slurm "$VISITS" \
  data/processed/reproduce_paper/value_v1 train
```

預設依序執行 train、valid，各 split 內啟動八個分片；所有分片成功才執行 `--export`，完成後各目錄只留 `value.npy`。任一分片失敗即停止，保留續跑資料；修正問題後以相同 visits、task 數、設定、輸出目錄重提。遭 Slurm 強制終止時，先確認所有原作業程序停止，再移除殘留 `.writer.lock` 才能續跑。已完成的 split 只重新檢查匯出檔，不啟動 teacher。

預設使用 `$HOME/.conda/envs/go/bin/python`，可透過 `PYTHON_BIN`、`CONFIG` 環境變數覆寫；工作目錄沿用既有腳本固定為 `/work/jason1122/go_aicup_2026`；Slurm 負責每個 task 的 GPU 綁定，不手動覆寫 `CUDA_VISIBLE_DEVICES`。account 預設 `GOV115094`，需變更時用 `sbatch --account=你的account ...` 覆寫。KataGo 的 `LD_LIBRARY_PATH` 沿用 `slurm/humansl.slurm` 的 `data/external/cuda-libs-12.8/{lib,stdcxx}`。若改為單卡，可加 `--ntasks=1 --mem=16G`，但既有分片數不得改變，須使用新輸出目錄。Slurm 自己的 stdout／stderr 合併至 `outputs/slurm-se-value-JOBID.out`（提交前確保 outputs 目錄存在），不混入 label 目錄。

本次僅撰寫 `.slurm`，未提交作業、執行計算或測試。

腳本亦沿用既有環境變數寫法：`VISITS="$VISITS" RUN=data/processed/reproduce_paper/value_v1 SPLIT=both sbatch slurm/label_value.slurm`。位置參數仍可使用且優先於環境變數。`slurm/` 依既有 `.gitignore` 規則為本機腳本目錄，本次未變更忽略規則。
