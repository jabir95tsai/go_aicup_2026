# SE 幽玄十級方法移植：實作與驗收計畫

更新日期：2026-11-02。

本文件將 [README.md](README.md) 的已確認方法拆成可逐步交付、驗收的工程工作；專案規則仍以根目錄 [CLAUDE.md](../../../../../CLAUDE.md) 為準。設定檔為根目錄 [configs/strength/se_seed2026.json](../../../../../configs/strength/se_seed2026.json)。新增決策或短測結果須同步回填設定與實驗帳本，避免本文件形成另一套規格。

## 1. 實作範圍

### 本次包含

- 使用官方幽玄十級資料移植基本 SE 方法，建立 strength／policy／value 三頭模型。
- 沿用唯一 parser、既有 packed board cache，以及 `data.cv` 的資料切分與題組產生介面。
- 按級別抽棋局、同盤指定玩家抽七個不同有效位置，分組計算 Bradley–Terry 排序 loss。
- 使用既有 KataGo 強引擎分析每個有效落子前盤面，產生 soft value 偽標籤並保存來源。
- 單卡與單節點 DDP 訓練、固定更新數的 SGD schedule、checkpoint 與可重現續跑。
- 全量 training centroid、nearest-centroid 解碼及固定 validation 題組的分 k 評估。
- 對應論文 §4.2 的評估流程；與競賽題組流程分開列報，明記資料與 protocol 差異。

### 本次排除

- SE-infinity、合成最低級及隨機合法著手生成器。
- SLsum／SLvote 及其對照訓練或增益 gate。
- SE-MCTS、chess、2-rank／3-rank 實驗與附錄 B 位置消融。
- Adam／AdamW／Muon 等 optimizer 比較，以及競賽主線融合與自動上傳。
- 宣稱完成 Fox 十一級原始結果或整篇論文重現；本版使用幽玄十級、修改後的訓練預算及 teacher 偽標籤。

### 假設與效果目標

方法假設為：多位置平均上的排序 supervision 能學到棋力順序。本版 k=5／10／15／20 的實際分數、相對競賽主線的增益及收斂速度均未知，不設定未經量測的增益承諾，也不另訓練 SL baseline。工程驗收、正式執行完成與模型效果分開判定；training loss 降低不足以證明泛化改善。

## 2. 固定設定與資料契約

| 項目 | 已確認設定 |
|---|---|
| Seed | Split seed 固定為 **2026**；模型初始化／訓練抽樣 seed 預設 2026，可獨立調整，不改變切分 |
| 級別 | `D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D`，編碼 0..9 |
| Split | 每級 100,000 盤，80,000 train／20,000 valid；首次以 `data.cv.make_strength_v1_split(seed=2026)` 產生，所有 reproduce_paper 訓練共用同一份切分 |
| 分離單位 | 按棋局切分、棋局零重疊；記錄玩家重疊，不宣稱未見玩家泛化 |
| 有效位置 | 指定玩家落子且零起算 `ply >= 30`；training、validation、centroid 使用相同規則 |
| 歷史 | 落子前八步盤面、最新優先，每對為當手方／對手；不足補零，另加黑白回合兩層 |
| Policy | 預測所選當手，row-major 0..360；361 為 pass，盤面與 target 同步 D4 |
| 取樣 | 每組十級，各級均勻抽一盤，再抽同盤指定玩家七個不同有效位置；跨組／step 可重複 |
| Global batch | 128 組 × 10 級 × 7 位置 = 8,960 個位置／update |
| 模型 | 256 channels、20 residual blocks、三頭；bias、post-add ReLU、BN／head 已依作者固定版本查核 |
| Loss | `L_BT + L_policy + L_value`，初始權重皆為 1 |
| Optimizer | SGD，momentum=0.9、weight_decay=0.0001 |
| Schedule | 35,000 optimizer updates；第 1–25,000 步 LR=0.04，第 25,001–32,500 步 LR=0.02，第 32,501–35,000 步 LR=0.002 |
| Teacher | 既有 KataGo 強網路、日本規則／6.5；visits=1（2026-11-01 使用者決定） |
| Value | 黑方 `2*p_black-1`，白方取相反數；不將 soft target 硬分類 |
| Centroid | 全部 800,000 盤 training 的全部有效位置，按位置等權；不使用 validation 擬合 |
| 題目解碼 | 題內所有有效位置的 scalar score 平均後取最近 rank centroid |

唯一棋局鍵為 `(rank, game_id)`，另以 `game_index` 對齊快取；不能只用 `game_id` join。ID 與檔名不得成為模型特徵。排除前 30 手只限制樣本位置，不裁切歷史，也不裁切 teacher 重播。

### 共用切分規則（2026-11-01）

此規則僅適用於 `reproduce_paper`，不調整 V1／V2、HumanSL 或玩家辨識的既有切分。

- 本目錄所有模型、初始化 seed、超參數設定與續跑，共用一份固定 training／validation 棋局名單；不按 run、模型或訓練 seed 各自重新切分。
- Split seed 永遠固定為 **2026**，維持每級 80,000／20,000 盤。首次由唯一 CV 模組產生並保存，後續直接讀取與驗證同一產物。
- 共用切分表使用 `data/processed/reproduce_paper/seed2026/games.parquet`，對應 manifest 記錄來源與切分身分；各 run 記錄相同切分表的雜湊。
- 模型初始化、分組抽樣與 D4 的訓練 seed 預設 2026，後續可以獨立改變，但不得變更 training／validation 成員或重新生成共用切分。
- Teacher 標籤、training centroid 與固定 validation 題組都依同一切分表判定角色；各 run 重用固定 validation 題組，不因訓練 seed 更換題目。
- 原始資料或切分規格真的變更時，須另建明確版本並記錄原因，不覆寫既有共用切分；不同切分版本的成績不可當成同一 holdout 比較。

驗收已完成：不同訓練 seed 的 run 具有完全相同的 train／valid 名單、切分雜湊與固定 validation 題組；重跑不改寫產物。`prepare.py` 使用獨立且固定為 2026 的 `split_seed`，資料 identity 不包含訓練 seed；初次建置與共用讀取皆已驗證。模型與分組 loss 已完成第五步；訓練器及實際 checkpoint 續跑仍屬後續階段。

### 128 組與四倍 LR 的最新決策

2026-11-01 使用者指定以 128 組／update、LR ×4 為實作計畫，取代 512 組／10,000 updates；2026-11-02 另加 LR=0.002 的 2,500 步尾段。以作者 32 組／130,000 updates 為參考：

- 128 組／update，35,000 updates，共 4,480,000 組；十級七手共 313,600,000 次位置使用。
- 初始 LR=0.04；完成第 25,000 次更新後降為 0.02，完成第 32,500 次後降為 0.002，再跑 2,500 步（2026-11-02 使用者決定）。前 32,500 步保留原 schedule，新增尾段後總位置使用量增加 7.69%。
- 8×H200 各 16 個完整組、1,120 個位置；三項 loss 維持平均，SGD momentum=0.9、weight_decay=0.0001 不變，未加入 warmup。
- 此為接近四個小步的線性縮放近似，並非已證明等效；BN、momentum、更新時點與精度影響須短測。分 k 增益與正式加速比未知。

## 3. 實作順序與驗收

### 第一步：設定、split 與有效位置索引

**狀態：設定／索引、獨立固定 split_seed 與跨 run 共用切分已實作並通過合成驗收；官方 prepared 索引及固定 validation 題組已存在（2026-11-02 查核）。**

實作範圍：`config.py`、根目錄 `configs/strength/se_seed2026.json`、`prepare.py`。沿用 packed cache metadata／實際顏色，產生 split、每盤有效位置及 manifest；不重新 parse 官方 CSV／SGF，不複製全量盤面。

交付：`games.parquet`、`{train,valid}_{game,offsets,ply}.npy`、`manifest.json`。

資料準備流程第五步「固定 validation 題組」已另由 `questions.py` 完成：讀取共用切分表，呼叫唯一 CV 模組產生題組與 eligibility，預設 10,000 題、固定 seed=2026，不受訓練 seed 影響。產物為 `validation_questions.parquet`、`validation_eligibility.parquet`、`questions_manifest.json`；題數須為 40 的正整數倍，四個 k／十級 cell 等量，題內同玩家同色且不重複，與 train 棋局零重疊。來源或題組損壞／設定衝突明確拒絕覆寫。操作入口見 README；官方固定題組產物已存在。

驗收目標：

- 正式資料每級精確 80,000／20,000 盤；train／valid 棋局交集為 0，玩家重疊有數字紀錄。
- 索引完整包含指定玩家且 `ply >= 30` 的位置，每盤至少七個不同位置；不足即報錯。
- seed、來源 cache、程式與產物雜湊、各級盤數／位置數均可追溯。
- 相同設定可驗證並重用；設定衝突、資料損壞或未完成目錄不得靜默覆寫。
- 所有訓練 run 讀取同一份 seed=2026 切分；改變模型初始化／抽樣 seed，train／valid 名單、切分雜湊及固定驗證題組不變。

### 第二步：樣本組裝與 state/action 對齊

**狀態：程式已完成，來源查核與合成資料驗收通過；未執行作者 C++ 二進位逐值比對。**

實作範圍：`dataset.py` 包裝既有 cache，提供落子前的 18-plane 輸入、policy target、rank 與追溯 metadata；支援批次讀取、LRU、spawn worker 及可重現 D4。提供 soft value 視角轉換介面，尚不生成／填補標籤。

驗收目標：

- 逐手重播對照落子前八步歷史，包括提子、`ply=0`、padding、白方通道與回合層。
- 輸入不包含要預測的當手；八種 D4 × 全部 361 個盤面座標均與 policy target 一致，pass 不變。
- 黑白視角的 value target 正負相反；非有限、越界機率與非法顏色被拒絕。
- 同 seed／round／樣本索引的增強一致，pickle 及兩個 spawn workers 的讀取結果與單程序一致。
- 沒有 teacher 標籤時不輸出虛構 value target。

現有第一、二步測試：`tests/test_reproduce_paper_data.py`，七項測試已通過。這僅代表合成資料工程驗收，不代表官方全量資料或 GPU 訓練已驗收。

### 第三步：逐有效盤面 teacher 與標籤準備

**狀態：使用者回報全部 training target 已取得；已確認 `value_v1/train/value.npy` 存在。合成 protocol／重試／匯出測試已通過；本次未重新分析 teacher 或逐值稽核官方標籤，品質與效能量測仍未知。**

2026-11-01 使用者明確改為「逐有效盤面分析，各自取得 v」，取代原末盤共用標籤。有效位置維持指定玩家且 `ply >= 30`，每個落子前盤面各自分析。Teacher 只收到所需歷史 prefix，不加入虛構 pass、不續弈、不改寫 RE。

交付：`label_value.py`、`configs/strength/se_value_analysis.cfg`、`PaperDataset(label_dir=...)`。逐位置保存 `p_black`、指定玩家視角 `value`、黑方 `score_black`、可用的 `score_stdev`、實際 visits 與 status；缺少 scoreStdev 保存 NaN，並不視為已知信心。依固定 game rows 分片，失敗重試與續跑保留成功位置，來源／teacher／分片設定不同拒絕混用。棋局來源表及索引雜湊綁定 `game_index`、`(rank, game_id)` 與 ply。

`--export` 只接受全量完整分片，輸出 `{train,valid}/value.npy`：一維 float32，與既有 `{split}_ply.npy`／dataset sample_index 精確同序，每個有效盤面恰有一個 v；完成完整性檢查及原子發布後，自動清除該 split 的分片、續跑狀態、log、暫存檔與 manifest，最終只留 `value.npy`（2026-11-01 使用者決定）。Dataset 讀取後提供 `value_target` 與 `value_weight=1`，D4 不改變 v。失敗不補 0.5／硬勝負，不刪 ranking 樣本；當前採嚴格完整標籤，尚未決定品質遮罩政策。

待執行驗收與正式執行 gate：

- 使用已登記強網路，記錄引擎／網路／config／程式 SHA-256、規則／貼目、visits、BLACK 視角；拒絕 HumanSL。
- 合成測試涵蓋落子前時點、完整歷史、黑白轉換、失敗與續跑、分片混用拒絕、匯出及 dataset 對齊。包含完成清理、清理中斷續作及重複匯出；2026-11-02 已執行合成測試通過，不代表官方標籤已逐值重新稽核。
- 使用者已指定 `teacher.visits=1`，Slurm 預設同為 1。後續固定 training 小樣本可用 `--limit-games` 量測穩定性、失敗率、positions/s 與品質；本次不執行。
- 全量成本須按全部有效位置數估算，不能沿用每局一次末盤的估時。短測後回報時間、GPU-hours、RAM、磁碟再啟動長作業。
- 2026-11-01 使用者決定本次只生成 training 標籤；Slurm 預設只處理 train。Validation 標籤不在本次執行範圍。

操作見 README。Training 標籤由使用者完成，本次未重跑標籤生成；validation 標籤仍不在範圍內。

### 第四步：分組 sampler 與 teacher 標籤讀取整合

**狀態：已實作，合成驗收通過（2026-11-02）；未啟動 GPU／DDP 訓練。**

交付：`dataset.py` 新增 `GroupBatchSampler` 與 `collate_groups`，沿用 `PaperDataset(label_dir=...)`。每組按 rank 0..9 排列、每級先均勻抽棋局再抽七個互異有效位置；value 已為指定玩家視角，不再次翻轉符號。Collate 保留 `[local_groups, 10, 7, ...]`，拒絕缺標籤、錯級別順序、跨盤／跨色、重複位置與非有限／越界 value。

抽樣 RNG 由 `(seed, step, global_group)` 決定；完整組依全域編號連續分配給各 DDP rank，128 組／8 卡時每卡 16 組、1,120 個位置。D4 沿用 `(seed, step, sample_index)`。`start_step` 是消費端已完成 optimizer updates，迭代／prefetch 不修改此游標；同設定可重建任意後續 step，不受 worker 數影響。操作範例見 README 第四步。

驗收目標：

- 每組十級各出現一次；每級七手來自同盤同一指定玩家，七手互異且都符合有效位置規則。
- 級內按棋局均勻抽樣，不因棋局較長而提高該盤被選中的機率。
- 同盤不同位置各自使用該落子前盤面的 teacher 機率，依視角轉換；標籤錯配明確報錯。
- 同 seed 與消費端 step 可重建抽樣與增強，不受 worker 數、prefetch 或續跑影響。
- DDP 依全域組編號分配完整組；8 卡時各 16 組（1,120 個位置），無額外補齊組；保留未來跨 step 重抽同盤的合法行為。

### 第五步：三頭模型與分組 loss

**狀態：已實作並通過 CPU／兩程序 Gloo 合成驗收（2026-11-02）；CUDA／SyncBN／正式訓練留待第六步。**

交付：`model.py` 的 `StrengthEstimator`／專用 residual block、`losses.py` 的 `grouped_bt_loss`／`grouped_losses`、`tests/test_reproduce_paper_model_loss.py`，以及第四步資料到三頭 backward 的整合測試。完整架構、來源對齊／偏離、公式與 API 見 [README 第五步](README.md#第五步三頭模型與分組-loss2026-11-02)。

- 固定作者 commit `d9498964e2ba84b5798c40e3f5b8a91a8791d66c` 的 BT／MiniZero network、train 與 SE config 已查核。專用骨幹保留卷積 bias、residual 相加後 ReLU、預設 BN eps=1e-5／momentum=0.1 及各層初始化；不改 V1／V2 shared model。
- Policy：1×1 conv→2 channels→flatten 722→362 logits。Value／strength：1×1 conv→1 channel→flatten 361→256→1，僅 value 接 tanh。支援逐位置與分組輸入；同次展平 forward 後恢復 `[G,10,7,...]`。
- FP32 級內七手平均，再對每組使用 `logcumsumexp` 計算弱到強 prefix 排序 loss，平均 9 項／各組。Policy CE、soft value MSE 與 BT 等權，回傳分項及 total；value 不再次翻號或硬分類，只接受完整標籤／全 1 value_weight。
- 作者 `bt_use_weight=false` 下未使用的 learned weight head 省略；本版只輸出 logits、scalar 去除最後長度 1 軸、state-dict 名稱不同，不匯入作者 checkpoint。幽玄十級、teacher 與訓練預算差異保留。

驗收結果：完整 reproduce_paper CPU 合成回歸 **67 passed（27.04s）**。直接公式 loss 絕對差通過 ≤1e-6，極端 score／BF16→FP32 及所有 head 梯度通過。兩程序 Gloo 固定 BN running stats 後，全參數梯度向量相對 L2 **1.37e-6**、最大絕對差 **3.69e-7**、近零分量最大絕對差 **2.68e-8**。這是等組數、FP32 小模型驗收；尚未執行完整 20×256 GPU forward 或 training-mode BN／SyncBN 對照。

驗收目標：

- 完整列出與作者網路的對齊項目及偏離，不僅以深度／寬度相同認定等價。
- 每組每級先平均七個 strength scores，再以強到弱順序計算 listwise loss、除以 `10-1`，最後平均各組；不可先平均全域各級再算一次 loss。
- 級別平均與 logsumexp 使用 FP32；與直接公式的 loss 絕對差 ≤ `1e-6`，極端輸入下保持有限。
- Policy CE、value MSE、BT 各自可報告，總 loss 權重皆為 1；有遮罩時明確記錄 value loss 的有效樣本與歸一化方式。
- 在可控 BN 行為的 FP32 基準中，完整組計算與兩程序分組計算的 loss 相對差與全參數梯度向量相對 L2 ≤ `1e-5`；接近零的分量另報最大絕對誤差。

### 第六步：訓練器、8 卡 DDP 短測與續跑

**狀態：8 卡／100 步短測入口、量測與 checkpoint 續跑已實作；CPU 合成驗收及 GPU 實測狀態見 README。依賴：第三步標籤、第四、五步。2026-11-02 使用者限定只完成 8 卡，不安排單卡 GPU 短測。**

實作範圍：新增 `train.py`；整合 SGD、step scheduler、BF16／FP32、單節點 8 卡 DDP、三項 loss 紀錄及原子 checkpoint。先交付 100 步量測，不啟動正式 35,000 步。沿用 shared runtime／checkpoint 能力，適配本版 sampler 與資料身分。

驗收目標：

- 一次 optimizer update 對應 128 個完整組；訓練停止條件為 35,000 updates，與 epoch／讀過多少不同棋局無關。
- LR 在完成第 25,000／32,500 次更新後分別切換為 0.02／0.002；短測須直接驗證兩個邊界與第 35,000 步停止，含邊界前後續跑。Config 以 `lr_drop_after_updates`／`lr_after_drop` 表示第一段降率，`final_lr_after_updates`／`final_lr` 表示尾段降率；checkpoint 保存模型、optimizer、scheduler、RNG、消費端 step、抽樣身分及 config／資料來源。
- 固定 seed 的中斷續跑與連續執行具有相同抽樣、增強、optimizer steps 及 LR；在固定 FP32／確定性短測條件下比較 loss 與參數差異。
- 普通 BN／SyncBN、BF16 與必要 microbatch 的行為另行查核、記錄；不可把 BN 差異誤判為分組 loss 錯誤或宣稱跨硬體逐值一致。
- 短測須檢查 LR=0.04 的三項 loss、梯度及參數更新是否有限且穩定；以相同 128 組資料比較四個 32 組／LR=0.01 小步與一個 128 組／LR=0.04 大步的參數更新方向及幅度，先控制 BN 再檢查正式設定。這是近似誤差診斷，不以單卡／DDP 正確性容差要求兩種優化軌跡相同。
- H200 dev 直接跑 8 卡 100 steps，global 128 組／每卡 16 組，預設 BF16＋SyncBN，每卡一次 forward/backward 處理完整 16 組；不安排單卡 GPU 短測。記錄各卡三項 loss、梯度／參數／更新 L2 norm、各級 score 平均、資料等待、forward/backward、optimizer、診斷與 checkpoint 秒數，及 CUDA allocated／reserved 峰值、主程序＋worker RSS、Linux I/O counters、checkpoint bytes。前 5 步為量測暖機，不改 LR；後 95 步以最慢 rank 的端到端耗時估算吞吐與 35,000 步成本。
- 依實測選定監控與保存頻率；不得假定 8 卡可線性加速八倍。

### 第七步：逐位置推論、centroid 與評估

**狀態：待實作。依賴：第一、二步及模型介面；可用小型模型先驗收，不等待正式訓練。**

實作範圍：新增 `infer.py`、`evaluate.py`；分片保存逐手 score、每盤 `score_sum`／`position_count`，以位置等權計算 training rank centroid 及 validation 題目分數，nearest-centroid 解碼，保存報表與重建圖表所需資料。

驗收目標：

- 小資料手算結果與串流／分片聚合一致；正確處理不同盤長，題目平均為 `sum(score_sum) / sum(position_count)`。
- 正式 centroid 使用全部 800,000 盤 training 的全部有效位置，不限每盤七手，不用 validation 擬合或挑切點。
- 固定 validation 題組來自唯一 `data.cv` 介面；每題同玩家同色、k 盤互異，沒有 training 棋局，題組與 seed 可重建。
- k=5／10／15／20 分別報 exact、within-one、官方 `exact + within-one-only/e` 分數及四桶等權平均，保留各級結果與混淆矩陣。
- Scalar nearest-centroid 不套用 V1 切點或 V2 機率 `score_rank` 解碼；centroid 並列時採固定規則並記錄。
- 論文式各級／各 N 重複 500 次評估，與競賽題組流程分開呈現；抽樣單位、資料限制及 95% CI 算法須先明記再產生正式報表。

原論文 §4.2 的參考數字為 N=15 exact >0.80、N=100 約 0.975、N=6 within-one >0.99；這些不是幽玄移植版的必過門檻。原始資料清單、抽樣 seed、CI 與重現容差尚有未知，不以達到某個數字宣告原版重現完成。

### 第八步：正式執行與結果交付

**狀態：待啟動。依賴：前七步工程驗收、所需全量標籤及長作業確認。**

實作範圍：完成官方資料準備與 training teacher 標籤，執行正式 35,000 updates，使用最終模型完成全量 training centroid 與 validation 評估。

驗收目標：

- 啟動前分開回報資料準備、teacher、訓練、training centroid 推論、validation 推論的估時與磁碟需求；總 GPU-hours 依實測估算，不套用論文 A5000 成本。
- 每項長作業依 CLAUDE.md §9 回報並確認啟動；開始與完成後紀錄實際硬體、版本、耗時與失敗狀態。
- 完成 35,000 次 optimizer updates；最終權重、有效 config、資料／teacher 身分、checkpoint 及訓練紀錄可以支援續跑與重建。
- 全量 training centroid 和固定 validation 報表可從保存產物重建；結果無論高低都寫入根目錄 `RESULTS.md`，來源及貢獻寫入 `EXTERNAL_RESOURCES.md`。
- 分數解讀明記 game-disjoint、玩家可能重疊，以及 teacher 偽標籤等偏離；不能直接和主線 player-disjoint 成績宣稱優劣。

## 4. 交付位置與目前進度

| 類型 | 路徑（相對專案根目錄） |
|---|---|
| 專用程式與本計畫 | `src/go_aicup/tasks/strength/reproduce_paper/` |
| 預設設定 | `configs/strength/se_seed2026.json` |
| 已有合成測試 | `tests/test_reproduce_paper_data.py`、`tests/test_reproduce_paper_questions.py`、`tests/test_reproduce_paper_value.py`、`tests/test_reproduce_paper_model_loss.py` |
| 預設 prepared 資料 | `data/processed/reproduce_paper/seed2026/` |
| 預設 run 輸出 | `outputs/rank/reproduce_paper/se_seed2026/` |
| 實驗與外部資源紀錄 | `RESULTS.md`、`EXTERNAL_RESOURCES.md` |

第一、二步、共用切分及固定 validation 題組已完成；使用者已取得全部 training target，檔案位於 `data/processed/reproduce_paper/value_v1/train/value.npy`。第四步 sampler／標籤整合與分組 batch 已完成，涵蓋按盤等權、每級同盤七手互異、teacher／D4 對齊、8 卡完整組分配、消費端 step 續跑及 spawn workers／prefetch 一致性。第五步三頭模型／分組 loss 亦已完成；第六至八步仍待完成。

2026-11-02 驗證：`tests/test_reproduce_paper_data.py`、`tests/test_reproduce_paper_questions.py`、`tests/test_reproduce_paper_value.py` 合計 **41 passed（21.50s）**，包含新第四步測試與既有資料／題組／teacher protocol 回歸；未重新生成 teacher、未逐值稽核官方 value.npy，未啟動 GPU 或正式訓練。Teacher visits=1；品質、精度、BN／SyncBN、microbatch、監控與 checkpoint 頻率及正式成本仍待後續短測。k=5／10／15／20 的模型成績與增益均未知。

第五步後完整回歸為 **67 passed（27.04s）**，包括第四步資料接線、三頭模型、公式與兩程序 Gloo loss／梯度驗證。架構／loss 實作方式及對齊／偏離詳見 README 第五步。GPU 訓練品質、分 k 成績與成本仍未知。

下一步為第六步：整合 SGD、固定 update LR schedule、8 卡 DDP 訓練器、原子 checkpoint 與消費端 step 續跑；先驗證 FP32／BN 行為，再依計畫執行 H200 短測、回報正式估時。第七步推論／centroid 可在小模型上另行驗收。

2026-11-02：新增 `train.py`、`configs/strength/se_smoke100.json`、`slurm/se_smoke100.slurm` 與 CPU 合成測試。此入口最多 500 步，預設只跑 100 步；CUDA 強制 8 processes，CPU 僅作工程驗證。第 50／100 步保存 latest-only checkpoint，config／資料標籤 hash／world size／程式不一致拒絕續跑；正式完整訓練、GPU 驗收與四個 32 組小步對照診斷仍待完成。


### 2026-11-02：大顯存效能實作補充

依使用者指定實作前兩個方向：

1. 第六步新增 cuDNN benchmark／channels-last 的 2×2、8 卡 100 步對照設定。固定 global 128 組、每卡 16 組及既定訓練參數；另建同版本基準，不覆寫 job 673013。比較吞吐／耗時／顯存與 loss／梯度／更新幅度，不自動採用候選；預期加速與分 k 增益未知。
2. 第七步先交付 `infer_positions` 與 8 卡 batch benchmark：同一 checkpoint、每卡同一 71,680 個位置，batch=1,120／2,240／4,480／8,960，eval／無 D4；量測吞吐／資源及 score 差異，OOM 停止更大候選。CPU 合成驗證涵蓋尾批、空分片、布局／batch 不變性及多程序接線。正式 centroid、全量輸出／每盤聚合、評估仍待完成。

只實作與 CPU 測試，未提交新的 GPU 作業；既有 job 673013 checkpoint 僅供推論讀取，不以新版 trainer 繞過程式 hash 檢查續跑。操作與限制見 README。


### 2026-11-02：選定 channels-last，關閉 cuDNN 搜尋

依使用者決定，reproduce_paper 訓練採 `channels_last=true`、`cudnn_benchmark=false`，同步 `configs/strength/se_seed2026.json` 與 `se_smoke100.json`。預設短測改用獨立 `outputs/rank/reproduce_paper/se_smoke100_default_8gpu/`，不改寫已完成的 baseline／對照 config 或輸出。模型、128 組、LR schedule、35,000 步預算不變；正式完整訓練入口尚未完成。

依據：job 673150（channels-last）5.133 steps/s、allocated 峰值 16.85 GiB；job 673193（兩者合用）5.152 steps/s、18.70 GiB，額外吞吐僅約 0.37%。此為效能設定決策，不以 100 步 loss 宣稱棋力改善；尚無同版本新 baseline 或重複量測。未提交新 GPU 作業。


## 5. 完整訓練入口實作提案（2026-11-02，待確認後寫程式）

本節依上述固定規格、[RESULT.md](RESULT.md) 及現有程式整理下一輪實作；屬規劃，不表示入口或驗收已完成。依根目錄 CLAUDE.md §9.1，先確認實作方案；正式長作業另依 §9.6 回報估時後確認啟動。

### 5.1 目標、範圍與現有缺口

目標是讓既有 `train.py` 從初始化完成 35,000 次更新，支援安全中斷、同版本續跑及最終模型交付。沿用 sampler、三頭模型、loss、SGD、scheduler 與 shared checkpoint，不另建 trainer，不增加依賴。

- 現有入口強制 `probe_updates <= 500`，CLI 預設 smoke config；正式 config 缺少 label_dir、device、precision 及 loader／checkpoint 執行設定，不能直接完整訓練。
- 現有 scheduler 已具備兩次降率；已有測試涵蓋倍率及部分 restore，仍需補 optimizer 實際更新與完整 trainer 在邊界前後續跑的驗收。
- 現有逐步診斷會複製全參數、收集各 rank 資源，並將所有詳細 rows 留在記憶體；這是短測設計，正式入口改用串流紀錄與累積摘要。
- 尚無 Slurm 預告中斷的安全停止流程；硬中斷可能殘留 `.writer.lock`，不能直接刪鎖後並行寫入。
- `infer.py` 現在提供逐位置推論及 batch benchmark，尚非全量 centroid／評估入口。完整方法仍須第七步；訓練完成不等於已取得棋力成績。

既定假設仍是平均位置上的排序 supervision 能學到棋力順序。k=5／10／15／20 的官方得分與相對增益均未知；本輪不增加模型效果承諾或 SL 對照訓練。

### 5.2 第一批：正式 trainer 與設定

1. 同一 `train()` 同時支援 smoke 與正式訓練：有 `probe_updates` 時保留 1..500 限制，未指定時以 `training.updates` 為終點。正式預算為 35,000；`--stop-after` 是本次停止的絕對 completed-update，不修改有效 config 或總預算。既有 smoke Slurm 均明確傳入原 smoke config。
2. CLI 預設切換到 `se_seed2026.json`；補齊 `paths.label_dir=data/processed/reproduce_paper/value_v1/train`、CUDA／BF16、每 rank 4 workers、SyncBN、每卡 16 組、TF32=false、deterministic=false、channels_last=true、cudnn_benchmark=false。CUDA 保留目前 8 processes 限定，CPU 僅供合成驗收。
3. 固定 global 128 組，每組十級各七手，每卡一次 forward/backward 共 1,120 位置；不更改抽樣、D4、loss 平均或加入 LR warmup。第 1–25,000 次更新用 0.04，第 25,001–32,500 次用 0.02，第 32,501–35,000 次用 0.002。
4. 每步保存 step、當步／下一步 LR、三項 loss／total 及 finite checks；每 100 步另保存八卡資源、梯度／參數／更新 norm、各級 score 等完整診斷。Smoke 維持逐步完整量測；診斷降頻不移除每步非有限梯度／loss／參數的檢查。詳細 rows 寫 JSONL，不常駐保留全部歷史；summary 用累積值，耗時分位數最多保留每步一個 scalar。
5. 每 500 updates、正常提前停止、收到停止要求及第 35,000 步保存 latest-only checkpoint。保留 model／optimizer／scheduler、每 rank RNG／BN buffers、消費端 step、config／資料／程式身分；抽樣與 D4 依已完成步數重建，prefetch 不前移游標。
6. 補入實際影響續跑的 shared runtime／training／checkpoint、資料組裝依賴的程式指紋。延續嚴格 config／data／world／code 檢查，不將既有 100 步 checkpoint 冒充正式 run 的起點，不放寬 hash 來跨版本續跑。
7. 新增 `slurm/se_train.slurm`，沿用既有 conda／torchrun 設定：單節點、8×H200、96 CPU、128 GiB RAM、`8gpus` partition，訓練階段先申請 4 小時。新 run 使用既定 `outputs/rank/reproduce_paper/se_seed2026/`；非空目錄仍須顯式 `--resume`。

### 5.3 第二批：中斷恢復與最終產物

- Slurm 在時限前預告（提案 120 秒），將訊號送達實際 trainer processes；handler 只設停止旗標。每個完整 optimizer update 邊界先跨 rank 合併停止要求，再由所有 ranks 一起寫 checkpoint 並退出，禁止 handler 直接做 collective／檔案寫入。腳本須驗證 srun／torchrun 的訊號傳遞，不能只加 SBATCH signal 指令就宣稱完成。
- SIGKILL、節點故障等無法保證最後保存；由最近已完成 checkpoint 重跑，至多損失 499 個已完成更新。若保存中斷，沿用 shared checkpoint 先發布再清除舊檔的順序，舊 last.json 仍可恢復。
- 保留寫入鎖；硬中斷殘留鎖的處理須先用 Slurm 狀態確認原作業已停止，再人工清理。不自動依 PID 判定跨節點鎖失效。
- 續跑截斷 checkpoint 之後的未提交訓練紀錄；能處理最後一行寫到一半的 JSONL。原始環境紀錄保留，續跑環境與 wall／GPU-hours 另記 session，避免覆蓋初始來源或只回報最後一段成本。
- 第 35,000 步先發布可續跑的最終 checkpoint，再原子發布 `model.pt`：最終 state dict、有效 config、完成步數、資料／程式／權重身分。重載驗證後才標記 training completed；提前停止不可發布成完整最終模型。不依 validation 選 best checkpoint，也不清除續跑 checkpoint。
- 正式輸出包含 `config.json`、初始及各 session 環境、`data_identity.json`、`train.jsonl`、session／training summary、`last.json`、latest checkpoint、`model.pt`。重跑已完成 run 應驗證並回報完成；若只缺最終模型發布，從最終 checkpoint 補發布，不再做第 35,001 次更新。

### 5.4 必要驗收與通過條件

| 驗收 | 方法與 gate |
|---|---|
| 入口與相容性 | 正式設定通過檢查；既有 smoke 限額、獨立輸出及 inference benchmark 測試保留；缺 teacher、非法 batch／精度／world size 明確拒絕 |
| LR 與停止 | CPU 小模型的 optimizer 實際走過 24,999／25,000／25,001、32,499／32,500／32,501 及 34,999／35,000 邊界；可用一致的合成 checkpoint 直接到邊界，無須跑完整 CNN 35,000 步；不多更新一次 |
| 續跑 | 固定 FP32／確定性條件下，比較連續與中斷兩段的抽樣、D4、loss、參數、momentum、scheduler／LR 與消費步數；CPU 參數逐值一致。補兩程序 Gloo 與 workers／prefetch 回歸 |
| 停止與發布 | 任一 rank 請求停止，所有 ranks 同步保存；partial JSONL、非空目錄、hash 不符、寫入失敗保留舊 checkpoint、最終模型發布重試、完成後不多訓練均有合成測試 |
| GPU 接線 | 新程式版本以同 config 分別完成連續 100 步及 50＋50 步續跑；八卡無 hang、各 loss／梯度／參數有限、有非零更新，sample／LR／step 一致。BF16＋非確定性不要求參數逐值相同，另報 loss／參數相對 L2 及最大絕對差 |
| 四小步／一大步 | 同初始權重、同 128 組資料，比較四次 32 組／LR=.01 與一次 128 組／LR=.04；先固定 BN、再用正式 SyncBN／BF16，報更新 cosine、norm ratio、relative L2 及分項 loss。此為近似誤差診斷，沒有理論等價門檻；非有限值直接失敗，不用 CPU 梯度正確性容差宣稱等價 |

只跑與改動相關的 `tests/test_reproduce_paper_*.py`，不另跑全 repo 訓練或重製 teacher。正式提交前須完成上述工程驗收並如實列出 GPU 差異；四小步診斷不自動改 batch／LR。

### 5.5 第七步銜接與效果驗收

完整訓練入口先交付訓練；第七步獨立交付全量推論／評估命令，通過各自驗收後才安排正式全量計算，避免把未知成本直接綁進訓練 Slurm 作業。

- 重用 `infer_positions`，eval、無 D4、無 teacher；按 sample_index 分片原子保存 float32 逐位置 scores，完整覆蓋且不補齊尾批。配合已完成分片續作，避免中斷後重算全量。
- Training 共 76,763,475 位置，validation 共 19,183,737 位置。由逐位置分數保存每盤 score_sum／position_count，以 float64 累計 training 十級 centroid。題目使用 `sum(score_sum) / sum(position_count)`，nearest-centroid；相同距離固定取較低 rank index，記錄規則。
- 重用固定 10,000 題，報 k=5／10／15／20 的 exact、within-one、官方得分與 Equal-k，各級結果及混淆矩陣；不套用 V1 切點或 V2 score_rank。論文式各 N／500 次評估另依第七步先定抽樣與 CI protocol。
- 工程通過條件為手算／分片結果一致、完整覆蓋、centroid 僅用 training、報表可重建。官方 0.4631 僅作競賽參考，不當作原論文重現 gate；此 split 有 7,855／7,855 位 validation 玩家同時出現在 training，不與主線 player-disjoint 成績直接比較。

### 5.6 成本、磁碟與執行順序

| 階段 | 估算依據與預算 |
|---|---|
| CPU 工程驗收 | 小模型／合成資料，0 GPU-hours；實際 wall time 未知 |
| 8 卡續跑驗收 | 連續 100＋分段合計 100，共 200 updates；依已測 session 約數分鐘內、暫預留 0.5 GPU-hours（不含排隊），正式腳本訊號驗收另外記帳 |
| 四小步診斷 | 有限固定 batch，成本未知；先實作 CPU 診斷，再回報 GPU 短作業成本，不與正式訓練混算 |
| 正式訓練 | RESULT.md 舊設定 0.2643 s/update → 2.57 h／20.56 GPU-hours；選定 channels-last 的 job 673150 summary 為 0.194814 s/update → 1.894 h／15.152 GPU-hours。以 2–3 h／16–24 GPU-hours 規劃，4 h job 配額為 32 GPU-hours 上限，非預期耗用 |
| Training／validation 推論 | 合計 95,947,212 位置；尚無完整推論吞吐，時間未知。須先用推論 benchmark 量測，以各 split 位置數／實測全節點 positions/s 估算，不套用訓練速度 |
| 新增磁碟 | 全量 float32 scores 為 383,788,848 bytes，約 366.0 MiB；latest checkpoint 約 185.4 MiB，原子替換需同時容納新舊檔，再加 model.pt、每盤統計及 logs。全流程先保留至少 2 GiB 可用額度，正式執行前重查配額；不複製既有 cache／teacher |

實作順序為第一批 trainer／config／Slurm → 第二批續跑／最終發布 → CPU 驗收 → 經確認執行 GPU 短驗收與診斷 → 回報更新後正式成本 → 經確認啟動 35,000 步。第七步可先用合成模型實作與驗收；全量推論須另回報時間與配額再啟動。

預計修改 `train.py`、`config.py`、`configs/strength/se_seed2026.json`，新增 `slurm/se_train.slurm`，補 `tests/test_reproduce_paper_train.py`；新增設定如影響推論／資料 fixtures，連同相關測試一併調整。第七步擴充 `infer.py`、新增 `evaluate.py` 及聚合／評估測試。同步本計畫、README、根目錄 RESULTS.md 的實際交付與驗收狀態；不覆寫 RESULT.md 的既有未提交修改，不更改 V1／V2 或 shared checkpoint 的公共契約。Slurm 目錄受 ignore 規則影響，交付時須確認腳本有納入版本控制。
