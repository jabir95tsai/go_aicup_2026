# Strength Estimator paper reproduction

分階段實作範圍、進度與驗收目標見 [plan.md](plan.md)。

**2026-11-01 實作紀錄：** 資料準備、樣本組裝與固定題組已通過合成驗收；逐有效盤面 teacher script、dataset 標籤接線與 Slurm 腳本已撰寫，當時未執行 teacher 測試、標籤生成、GPU 或訓練作業。以下為該日的程式與驗收紀錄，不代表目前作業狀態。

本目錄專用於 Chen、Shih、Wu，*Strength Estimation and Human-Like Strength Adjustment in Games*（ICLR 2025，arXiv:2502.17109v2）的重現。

- 分支：`reproduce/2502.17109v2`
- 起點：`396b71fd741f363b8e0133295094be16658c56f9`（`t1-pipeline-t2-models`）
- 本機論文：`reference/2502.17109v2.pdf`，完整閱讀含附錄 A–C，共 18 頁。
- PDF SHA-256：`83aa786ff7c659f9715fb030f4e9eb0c094b19f0ca6b0229aeeb9ebb37249f87`
- 作者程式：[rlglab/strength-estimator](https://github.com/rlglab/strength-estimator)，查核版本 `d9498964e2ba84b5798c40e3f5b8a91a8791d66c`（`iclr2025`）。前置查核僅閱讀；2026-11-02 已依查核的架構與公式重新撰寫專用模型／loss，來源及差異見下方第五步。
- 後續實作與驗收見 [plan.md](plan.md)；正式長作業依 `CLAUDE.md` §9 先回報實測估時。

## 已確認方向（2026-11-01 更新）

使用者最新決定：競賽版每步使用 **128 個獨立排序組**，初始 **LR=0.04（作者 0.01 的四倍）**，總計 **32,500 次 optimizer updates**，取代先前的 512 組／10,000 次更新。此方案以作者 32 組／130,000 次更新為基礎，將四個小 batch 合併為一個大 batch，保持總排序組數 `128 × 32,500 = 32 × 130,000 = 4,160,000`。Optimizer 維持 SGD，momentum=0.9、weight_decay=0.0001；loss 仍按組／樣本平均，不改成加總。每組每級仍抽同盤指定玩家的七個不同位置。

這是線性 LR scaling 的近似方案；momentum、BN 與參數更新時點不同，尚無本版收斂或等效性驗證。未另行加入 warmup 或更動 momentum。正式訓練前須短測穩定性、數值與吞吐；分 k 成績及加速比未知。

訓練採 step-based 隨機抽樣，不以 epoch 或完整走訪所有棋局終止；預設 seed=2026。資料與原論文的差異依下方記錄處理。

### 已確認設定與範圍（2026-11-01）

已確認設定與實作範圍如下：

| 項目 | 已確認設定 |
|---|---|
| Policy head | 本版加入，使用 strength／policy／value 三頭；實作前核對固定版本 MiniZero 的輸入與 target 時點，驗證無目標落子洩漏 |
| Centroid 資料 | 使用全部 800,000 盤 training 棋局的指定玩家有效位置（`ply >= 30`），按位置等權計算各級平均，不抽一小部分棋局或每盤只取七手 |
| 80/20 split | 每級按棋局隨機切分為 80,000／20,000 盤，split seed 固定為 2026，首次沿用 `data.cv.make_strength_v1_split` 產生；所有本目錄訓練 run 共用同一份切分表，記錄玩家重疊 |
| KataGo 規則／貼目 | 日本規則、貼目 6.5；是 teacher 分析假設，不是原局規則已知值 |
| 本次範圍 | 基本 SE 三頭模型、逐有效盤面 soft value 偽標籤與論文 §4.2 對應評估 |
| 排除項目 | SE-infinity（含合成最低級與合法著手生成器）、SLsum／SLvote（含對照訓練與增益 gate）、SE-MCTS、chess、2-rank／3-rank、附錄 B 位置消融及 Adam／AdamW／Muon 等 optimizer 比較 |

本版總 loss 採 `L_BT + L_policy + L_value`，各項初始權重為 1。BF16、BN 行為、監控頻率等屬原始碼查核／短測項目；teacher visits 已由使用者指定為 1，無須重複確認。

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

本節分開說明論文原版，以及使用者先前指定的競賽資料改動。取消 epoch 已確認；十級、每級十萬盤、80/20 與排除前 30 手保留為既有資料要求，不能稱為原論文資料規格。設定、split／有效位置索引、樣本組裝及三頭模型／loss 已實作；訓練器仍待第六步。

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

### SE-infinity 論文背景

SE-infinity 每組加入七個由人類棋局位置產生的隨機合法著手，作為低於所有人類級別的合成最低級；合法性須符合作者環境，不能只抽空點。作者每步為 `32 × 12 × 7 = 2,688` 個位置，合成樣本在 step 內產生。

附錄 A 說合成位置不計 policy/value loss；固定版本 `train.py` 卻對整個 batch 計算，`st_data_loader.cpp` 並將合成 value 設為 -1。此處保留文件與程式差異的查核紀錄。

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

原資料無勝負結果，這仍是 teacher 偽標籤，與論文 outcome supervision 不同。規則／貼目是分析假設，visits 已由使用者指定為 1；品質門檻與成本仍待 training 小樣本短測，不能依 validation rank 挑選。

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
| 每組每級抽樣 | 作者 `bt_use_same_game_per_rank=true`：同盤、同色取 7 個位置 | 既有全量 epoch 順序不是此抽樣方式 |
| 優化 | SGD，LR 0.01，100,000 steps 後 0.005，共 130,000 steps；作者 momentum=0.9、weight_decay=0.0001 | 既有訓練器與 cosine scheduler 需適配 |
| 聚合／解碼 | candidate 各級所有位置平均為 centroid；query 所有選定位置平均後取最近 centroid | V1 cutpoints、V2 C2/B2 均非論文規格 |
| 評估 | 各級、各 N 重複 500 次；exact accuracy、within-one accuracy 與 95% CI | 官方主要指標為 exact + 差一級/e，須另列而不能混稱 |

## 實作範圍與程式配置

本版為「幽玄十級方法移植」：teacher 偽標籤取代真實勝負，資料來源、級別與訓練預算均偏離原論文，不能宣稱完成 Fox 原始結果或整篇重現。Fox 資料、勝負與 candidate/query 固定清單仍缺，原始切分是否玩家互斥未知。

### 已有程式（2026-11-02 更新）

| 檔案 | 已實作功能 |
|---|---|
| `config.py` | 讀取根目錄 `configs/strength/se_seed2026.json`、檢查設定與解析 paths |
| `prepare.py` | 呼叫共用 CV 模組，保存固定切分、有效位置索引與 manifest |
| `questions.py` | 從共用切分建立固定 validation 題組與 eligibility |
| `dataset.py` | 落子前盤面與 policy target、同步 D4、批次讀取及對齊的逐位置 value 標籤讀取；完整排序組 sampler 與分組 collate |
| `label_value.py` | KataGo 逐有效盤面分析、分片、重試、續跑與匯出；不改寫棋譜或 RE |
| `model.py` | 專用三頭 ResNet；作者版 residual activation／bias／BN／head 結構，支援完整組輸入 |
| `losses.py` | FP32 分組 listwise BT、362 類 policy CE、soft value MSE 與等權總 loss |

訓練器、推論與評估的待實作內容及驗收要求統一見 [plan.md §3](plan.md#3-實作順序與驗收)。專用資料放 `data/processed/reproduce_paper/`，各 run 放 `outputs/rank/reproduce_paper/<run-id>/`；沿用既有壓縮快取，路徑走 config。

### 效果目標與後續驗收

方法假設為跨多位置的排序 supervision 能辨識棋力順序；本版 k=5／10／15／20 的分數、相對競賽主線增益與收斂速度均未知，training loss 降低不足以證明泛化改善。評估須分 k 回報 exact、within-one、官方 score 與四桶等權平均；本版棋局切分的成績不能直接與主線 player-disjoint 成績比較。

原論文 §4.2 的參考數字為 N=15 exact >0.80、N=100 約 0.975、N=6 within-one >0.99；這些不是移植版必過門檻。原始清單、抽樣 seed、95% CI 算法及重現容差仍待查核，不以達到參考數字宣稱原始結果重現。

分組 loss 的 FP32 容差、單卡／DDP 梯度、固定 seed 續跑、資料隔離與分 k 評估等驗收要求統一見 [plan.md §3](plan.md#3-實作順序與驗收)。MiniZero 輸入／target、block／head、BN／SyncBN、BF16、teacher 品質與失敗處理均須依該計畫查核或短測。

附錄 A 的 Go 訓練成本約 242 RTX A5000 GPU-hours，不能直接換算本版 H200 或 3060 Ti 的成本。先做 CPU 驗收，再用 H200 dev 單卡 100–500 steps 及所需 DDP 配置量測吞吐、顯存與 RAM；8GB 裝置若需 microbatch，須說明 BN 偏離。分別估算標籤、訓練與 centroid／validation 推論的 GPU-hours、磁碟和監控／checkpoint 成本，不假定八卡線性加速。正式長作業依 `CLAUDE.md` §9 回報實測估時並確認啟動。

## 查核來源

- 本機 PDF：§3、§4、附錄 A–C。
- [作者 README（固定版本）](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/README.md)。
- [SE config](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg/se_go.cfg)、[SE-infinity config](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg/se_infty_go.cfg)。
- [網路](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/bt_network.py)、[訓練與 loss](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/train.py)、[抽樣](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/st_data_loader.cpp)、[評估](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/misc/evaluator.cpp)。

## 第一、二步：固定共用切分、操作與驗收（2026-11-01）

`reproduce_paper` 所有訓練共用每級 80/20 的 training／validation 切分，`split_seed` 固定為整數 2026。初始化／抽樣 `seed` 預設 2026，可獨立修改；各 run 記錄相同切分雜湊並重用固定 validation 題組。詳細規則見 [plan.md](plan.md)；V1／V2 等其他流程不受影響。

`prepare.py` 首次建立資料，後續驗證身分、必要檔案與雜湊後直接重用，不重抽或覆寫。訓練 seed、模型 channels、run 輸出目錄與 loader 設定不進入共用資料身分；資料來源、切分盤數或有效位置規則改變時，須另建版本目錄。固定題組由 `questions.py` 讀取同一資料目錄。

```json
{
  "seed": 3030,
  "split_seed": 2026
}
```

以上為更換訓練 seed 的設定片段，其他設定照舊。保留同一 `paths.dataset_dir`（預設 `data/processed/reproduce_paper/seed2026/`，切分表為其中的 `games.parquet`）；各 run 的權重與 log 放各自的 `output_dir`。

Dataset manifest identity 格式更新為 version 2，以 `split_seed` 取代原本混用的 `seed`。舊版 manifest 不會自動覆寫／遷移，須保留原產物並在新版本資料目錄重新準備。`PaperDataset` 的 D4 seed 預設 2026，也可明確傳入訓練 seed；它不從切分 seed 推導訓練隨機性。

已建立 `config.py`、`configs/strength/se_seed2026.json`（專案根目錄下）、`prepare.py` 與 `dataset.py`。正式資料準備入口：

```bash
python -m go_aicup.tasks.strength.reproduce_paper.prepare
# 或指定獨立 config
python -m go_aicup.tasks.strength.reproduce_paper.prepare --config path/to/config.json
```

此命令會讀取既有 packed cache，預設輸出為 `data/processed/reproduce_paper/seed2026/`：

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

`PaperDataset` 支援批次 `__getitems__`、process-local memmap、LRU 及 spawn workers；增強由 seed／round／sample index 決定。第 4 步已加入 `GroupBatchSampler` 與 `collate_groups`（操作見下方）。識別資訊不進入模型通道；未指定 `label_dir` 時不提供 value label。`soft_value_target(p_black, player_color)` 提供有限值／範圍檢查及黑白視角轉換。

### 已查核的 state/action 契約

固定作者版本 `d9498964e2ba84b5798c40e3f5b8a91a8791d66c`：

- [`strength/misc/game_wrapper.cpp`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/misc/game_wrapper.cpp) 的 `calculateFeatures` 直接呼叫 loader。
- [`minizero/minizero/environment/base/base_env.h`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/environment/base/base_env.h) 的 `getFeatures(pos)` 只重播索引小於 pos 的著手，`getPolicy(pos)` 使用 pos 的 action。
- [`minizero/minizero/environment/go/go.cpp`](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/environment/go/go.cpp) 的 `getFeatures` 以最新歷史優先，每對通道為當手方／對手；不足八步補零，最後兩層為黑／白回合。

因此本版對選中的 ply 讀取 cache 的 `ply-1` 及更早盤面，白方交換黑白歷史通道；policy 使用所選 ply 的著手。`ply=0` 歷史全零；`ply >= 30` 不會清空前 30 手的歷史。盤面及 policy 共用相同 D4；本 repo 維持自身 row-major 座標，pass 類別為 361（官方訓練資料無 pass）。此為來源閱讀與合成棋局對照，未執行作者 C++ 二進位逐值比對。

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

## 合成測試與歷史驗收

執行資料與固定題組測試：

```bash
PYTHONPATH=src python -m pytest tests/test_reproduce_paper_data.py tests/test_reproduce_paper_questions.py -q
```

以下為 2026-11-01 記錄的先後結果，非本次文件整理重新執行：

| 階段 | 結果 | 驗證範圍 |
|---|---|---|
| 初版資料與樣本 | 7 passed，6.80s | 十級 split／玩家重疊、有效位置與不足拒絕、來源重用、提子／落子前歷史／白方通道／padding、全部 D4 座標與 pass、soft value 非法值、pickle／spawn |
| 加入固定題組 | 17 passed，6.38s | 前述項目加題組平衡、train／valid 隔離、同玩家同色、題內無重複、跨 seed 重用、獨立生成一致、題數衝突、損壞與 eligibility 不足拒絕 |
| 固定 split seed 整合 | 24 passed，7.17s | 前述項目加訓練 seed／模型／output_dir／loader 變更後產物雜湊與修改時間不變、跨 seed 初建一致、非法 split_seed 與索引損壞拒絕、cache → prepare → questions 整合 |

上述僅為 CPU 合成驗收，不代表官方全量資料或 GPU 驗收。Teacher 測試檔 `tests/test_reproduce_paper_value.py` 的執行情況見開頭帶日期的實作紀錄。

## 第三步：準備每個有效盤面的 v

先具備 `prepare.py` 產物；config 與 Slurm 預設均為 1 visit。

```bash
# 固定 training 前 100 局；同一小樣本比較不同 visits 時必須換 out。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out outputs/rank/reproduce_paper/value_pilot_v1 --split train \
  --visits 1 --limit-games 100

# 正式分片範例：每個 worker 各自使用 shard 0..7，指定自己的 GPU。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out data/processed/reproduce_paper/value_v1 --split train \
  --visits 1 --num-shards 8 --shard 0

# 全部分片成功後匯出一個 sample_index 對齊的 v 向量（此命令不啟動引擎）。
python -m go_aicup.tasks.strength.reproduce_paper.label_value \
  --out data/processed/reproduce_paper/value_v1 --split train --export
```

本次標籤生成範圍為 training。另需 valid 標籤時，用同一組 teacher 設定改 `--split valid` 分片執行與匯出。以 `CUDA_VISIBLE_DEVICES` 配置各 worker；不會自動提交 Slurm 或啟動多 GPU。`--engine-config` 可指定執行配置，變更後須使用新輸出版本。

每個 `<out>/<split>/shard_XXXXX/` 保存與該分片有效位置同序的 `value.npy`、`p_black.npy`、`score_black.npy`、`score_stdev.npy`、`visits.npy`、`status.npy` 及來源 manifest。分片按共用 game array 的 row modulo 分配，局內按原有效 ply 排序。狀態 0=未完成、1=成功、2=失敗；缺失值為 NaN。每局完成先 flush 數值再提交 status，`--retries` 預設 2 次；重跑相同命令只處理非成功位置。錯誤見 `failures.jsonl`／`katago_stderr.log`。硬中斷留下 `.writer.lock` 時先確認原程序已停止再移除鎖。

### 匯出、清理與恢復

`--export` 驗證所有位置成功、分片完整、來源與 teacher 身分一致後，原子發布 `<out>/<split>/value.npy`，再自動清除該 split 的分片、續跑狀態、log、暫存檔與 manifest。其他 split、原始 cache 與 prepared 索引不受影響。

- 輸出為一維 float32，長度等於有效位置數，與 `{split}_ply.npy`／`PaperDataset` 的 `sample_index` 同序；可經 game／offsets／ply 與 games.parquet 回溯 `(rank, game_id, ply)`。
- 匯出失敗保留續跑資料；清理中斷時，重跑相同 `--export` 接續清理。
- 完成後重跑 export 僅檢查既有 v 的長度、dtype 與數值範圍，不啟動引擎；標籤生成命令拒絕重新建立分片。Dataset 檢查 shape／dtype 與讀取樣本值。
- 生成及首次匯出期間，以 SHA-256 綁定來源、引擎、已登記 kata1 強網路、設定與程式，拒絕 HumanSL。最終只留 `value.npy`，不再保留來源或 teacher 驗證能力；`label_dir` 必須指向正確 split／資料版本。

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

成本以「有效位置數／實測 positions/s」估算；各分片每位置 21 bytes 數值陣列，匯出 v 額外 4 bytes（尚不含來源、log 與暫存匯出檔）。全量磁碟、RAM 與 GPU-hours 未實測。

### Nano4 Slurm 執行

腳本：`slurm/label_value.slurm`。預設 `8gpus` partition、單節點 8 張 H200、8 tasks（每 task 1 GPU／12 CPU）、總 RAM 128GB、時限 48 小時。這是資源申請上限，並非實測完成時間；未量測 GPU-hours。執行前須已有 prepare 產物，腳本不自動重新準備資料。

從 repo 根目錄提交，不帶參數即使用 1 visit；可用第一個參數覆寫：

```bash
# 預設 1 visit、只處理 training。
sbatch slurm/label_value.slurm

# 也可指定輸出位置與單一 split。
sbatch slurm/label_value.slurm 1 \
  data/processed/reproduce_paper/value_v1 train
```

預設只執行 train 的八個分片，全部成功才執行 `--export`；匯出與恢復行為見[匯出、清理與恢復](#匯出清理與恢復)。任一分片失敗即停止，修正後以相同 visits、task 數、設定與輸出目錄重提。遭 Slurm 強制終止時，先確認所有原作業程序停止，再移除殘留 `.writer.lock` 才能續跑。

預設使用 `$HOME/.conda/envs/go/bin/python`，可透過 `PYTHON_BIN`、`CONFIG` 環境變數覆寫；工作目錄沿用既有腳本固定為 `/work/jason1122/go_aicup_2026`；Slurm 負責每個 task 的 GPU 綁定，不手動覆寫 `CUDA_VISIBLE_DEVICES`。account 預設 `GOV115094`，需變更時用 `sbatch --account=你的account ...` 覆寫。KataGo 的 `LD_LIBRARY_PATH` 沿用 `slurm/humansl.slurm` 的 `data/external/cuda-libs-12.8/{lib,stdcxx}`。若改為單卡，可加 `--ntasks=1 --gpus-per-node=1 --mem=16G`，但既有分片數不得改變，須使用新輸出目錄。Slurm 自己的 stdout／stderr 合併至 `outputs/slurm-se-value-JOBID.out`（提交前確保 outputs 目錄存在），不混入 label 目錄。

腳本亦沿用既有環境變數寫法：`VISITS=1 RUN=data/processed/reproduce_paper/value_v1 SPLIT=train sbatch slurm/label_value.slurm`。位置參數仍可使用且優先於環境變數。`slurm/` 依既有 `.gitignore` 規則為本機腳本目錄。

Nano4 GPU partition 要求明確指定 `--gpus-per-node`；本腳本已加入 `#SBATCH --gpus-per-node=8`，與 8 tasks／每 task 1 GPU 一致。分析 srun 明確指定每節點 8 GPU，匯出 srun 指定 1 GPU，避免繼承完整八卡設定。


### 第四步：完整排序組與 teacher value batch（2026-11-02）

```python
from torch.utils.data import DataLoader
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.dataset import (
    PaperDataset, GroupBatchSampler, collate_groups,
)

config = read_config()
dataset = PaperDataset(
    config['paths']['dataset_dir'], split='train', augment=True,
    seed=config['seed'], max_open_shards=config['loader']['max_open_shards'],
    label_dir='data/processed/reproduce_paper/value_v1/train',
)
sampler = GroupBatchSampler(
    dataset,
    groups_per_update=config['training']['groups_per_update'],
    updates=config['training']['updates'],
    start_step=0,  # Resume with the number of updates actually completed.
    rank=0, world_size=1,  # Use the process rank/world size under DDP.
)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_groups,
                    num_workers=0)
try:
    batch = next(iter(loader))
    # planes: [local_groups, 10, 7, 18, 19, 19]
    # policy_target/value_target: [local_groups, 10, 7]
finally:
    dataset.close()
```

每組按 rank 0..9 排列，每級先均勻抽一盤 training 棋局，再不放回抽七個有效位置；跨組／step 可以重抽同盤。`value_target` 直接讀取既有指定玩家視角的 v，不再翻轉黑白符號。`collate_groups` 拒絕缺標籤、錯 rank 順序、跨盤／跨色或重複位置及無效 value 的 batch。

抽樣由 `(seed, step, global_group)` 決定，D4 沿用 `(seed, step, sample_index)`；不依賴 worker RNG。8 卡／128 組時，每卡保留 16 個完整組、1,120 個位置，沒有補齊組。續跑時傳入消費端已完成 updates 作為 `start_step`，不能使用 DataLoader 預取進度。使用 spawn workers 時，將執行入口放在 `if __name__ == '__main__':` 內，並設定 `multiprocessing_context='spawn'`；worker／prefetch 數不影響抽樣與增強。

資料管線、三頭模型與 loss 已完成；optimizer 更新／訓練器仍待第六步，以上範例不啟動訓練。


### 第五步：三頭模型與分組 loss（2026-11-02）

#### 固定作者版本查核與網路結構

本次逐檔查核 commit `d9498964e2ba84b5798c40e3f5b8a91a8791d66c` 的 [BTNetwork／ScoreNetwork](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/bt_network.py)、[AlphaZeroNetwork](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/network/py/alphazero_network.py)、[network_unit.py](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/network/py/network_unit.py)、[train.py](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/train.py) 與 [se_go.cfg](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg/se_go.cfg)。這些 MiniZero Python 檔直接存在該 commit 的 tree 中，並非以另一個未固定版本的安裝套件代替。

`model.py` 的 `StrengthEstimator(**config['model'])` 沿用作者的以下運算結構：

| 部位 | 實作及作者對齊項目 |
|---|---|
| Stem | `Conv3×3(18→256) → BN → ReLU`，保持 19×19 |
| 20 個 residual blocks | `Conv3×3 → BN → ReLU → Conv3×3 → BN`，與輸入相加後再做 `ReLU` |
| Policy head | `Conv1×1(256→2) → BN → ReLU → flatten(722) → Linear(722→362)`；輸出 raw logits，包含 pass |
| Value head | `Conv1×1(256→1) → BN → ReLU → flatten(361) → Linear(361→256) → ReLU → Linear(256→1) → tanh` |
| Strength head | 與 value head 的隱藏層結構相同，但最後沒有 tanh，輸出無界 scalar beta |
| Bias／BN／初始化 | 所有卷積有 bias；BN 使用 PyTorch 預設 `eps=1e-5`、`momentum=0.1`、affine 與 running stats；保留各層預設初始化，沒有套用 V1/V2 的自訂初始化 |

專用 residual block 是必要差異：既有 shared block 缺少相加後 ReLU、卷積為 `bias=False`，直接沿用會改變作者網路。Value 與 strength 各自擁有獨立 head 參數。

`forward` 接受浮點 `[B,18,19,19]` 或 `[G,10,7,18,19,19]`。分組輸入先展成 `G×70` 個位置做一次共享骨幹 forward，再恢復原有組／rank／位置軸；BN 對本次傳入的所有位置計算，不逐組執行網路。回傳字典：`score`／`value` 為 `[G,10,7]`，`policy_logits` 為 `[G,10,7,362]`；逐位置輸入則分別為 `[B]` 與 `[B,362]`。Dataset 的 uint8 planes 由呼叫端轉為浮點，模型不隱式轉換。

相對作者程式的介面／範圍差異：本版省略 `bt_use_weight=false` 下未參與 loss 的 learned weight head；不額外輸出 policy softmax，CE 直接使用 logits；scalar 輸出省略最後長度 1 軸並可保留組軸，名稱為 `policy_logits`。State-dict 名稱不同，未提供作者 checkpoint 匯入。資料仍是幽玄十級、soft teacher value 與已確認修改的訓練預算；這些不構成 Fox 十一級原始結果重現。

#### Loss 計算與數值尺度

`grouped_bt_loss(scores)` 嚴格接受 `[G,10,7]`，rank 0 最弱、rank 9 最強：

```text
b[g,i] = FP32_mean(scores[g,i,:])
L_BT = mean_g { sum(i=1..9) [logsumexp(b[g,0:i+1]) - b[g,i]] / 9 }
L_policy = mean_over_all_positions CE(policy_logits, policy_target)
L_value = mean_over_all_positions (predicted_value - teacher_value)^2
L_total = L_BT + L_policy + L_value
```

程式用 `torch.logcumsumexp(b, dim=1)` 一次建立弱到強的所有 prefix，排除只有最弱一級的零項，再對 `G×9` 個項目取平均。這等價於論文強到弱的 suffix 公式及作者除以 `n-1` 的尺度；沒有先把不同組的同級 scores 混在一起。每組先平均七個 strength scores，不平均 exp(score)，也不先將所有組同級平均。

三項 loss 都轉為 FP32 計算，包含級內平均、logsumexp、CE 與 MSE；不直接計算 `exp(beta)`，避免大 score 溢位。Policy target 使用整數 action ID（含 361 pass），與作者 one-hot CE 在本資料規格下等價。Value target 直接使用已轉為指定玩家視角的 `[-1,1]` soft value，不轉成硬勝負、不重做黑白正負號轉換。

`grouped_losses` 回傳可微的 `total`／`bt`／`policy`／`value`，三項權重固定為已確認的 1。輸入會檢查 shape、device、浮點／整數類型、有限值、action 範圍及 value 範圍。當前嚴格要求完整標籤；可傳入 Dataset 的 `value_weight`，但只接受與 target 同形且全為 1，尚未加入品質遮罩或缺標籤跳過政策。因無遮罩，policy／value 的分母都是 `G×70`。

#### 與第四步接線

以下承接第四步取得的 `batch`，只展示一次 forward／backward，並非正式訓練器：

```python
from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
from go_aicup.tasks.strength.reproduce_paper.losses import grouped_losses

model = StrengthEstimator(**config['model'])
output = model(batch['planes'].float())
losses = grouped_losses(
    output, batch['policy_target'], batch['value_target'], batch['value_weight'],
)
losses['total'].backward()
# losses['bt'], losses['policy'], losses['value'] remain separately reportable.
```

DDP 每張卡必須拿相同數量的完整組，這由第四步 sampler 保證；各卡本地 loss 平均後 backward，DDP 的參數梯度平均就是全域平均，不需再乘／除 world_size，也不需在 loss 內做可微 all-reduce。訓練器若要顯示全域 loss，仍須另行聚合 detached metrics。當前不支援不等組數或空 rank。

#### 驗證範圍與限制

`tests/test_reproduce_paper_model_loss.py` 涵蓋作者結構／bias／BN 設定、residual 相加後 activation、分組與逐位置輸出一致、strength 無界／value tanh、直接公式（絕對差 ≤1e-6）、正反排序及組邊界、極端 score 的有限 loss／梯度、BF16 輸入下 FP32 loss、one-hot CE／soft MSE 等價、所有 head 與共享骨幹反向傳播及無效輸入拒絕。資料測試另包含 sampler → Dataset → collate → model → 三項 loss → backward 的合成整合。

兩程序 CPU Gloo 測試在 FP32、固定 BN running stats（`model.eval()` 但保留 autograd）下，比較全組與分組的四項 loss 及全參數梯度向量。以梯度向量相對 L2 ≤1e-5 驗收，另報最大絕對誤差及近零分量（參考梯度絕對值 ≤1e-7）的最大絕對誤差。這項測試驗證分組／梯度尺度，不代表 training-mode 普通 BN 與 SyncBN 等效。

2026-11-02 完整 reproduce_paper CPU 合成回歸結果為 **67 passed（27.04s）**。Gloo 全參數梯度向量相對 L2＝**1.37e-6**、最大絕對誤差＝**3.69e-7**、近零分量最大絕對誤差＝**2.68e-8**；執行命令與完整紀錄見根目錄 RESULTS。

CUDA／NCCL、GPU AMP、SyncBN、microbatch、SGD schedule、checkpoint 與正式收斂仍待第六步。未載入作者權重，未做作者原程式／C++ 二進位逐值比對；沒有新增外部訓練資料。k=5／10／15／20 分數、增益與訓練成本仍未知。
