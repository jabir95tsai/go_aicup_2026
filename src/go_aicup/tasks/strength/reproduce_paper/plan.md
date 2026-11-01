# SE 幽玄十級方法移植：實作與驗收計畫

更新日期：2026-11-01。

本文件將 [README.md](README.md) 的已確認方法拆成可逐步交付、驗收的工程工作；專案規則仍以根目錄 [CLAUDE.md](../../../../../CLAUDE.md) 為準。設定檔為根目錄 [configs/strength/se_seed2026.json](../../../../../configs/strength/se_seed2026.json)。新增決策或短測結果須同步回填設定與實驗帳本，避免本文件形成另一套規格。

## 1. 實作範圍

### 本次包含

- 使用官方幽玄十級資料移植基本 SE 方法，建立 strength／policy／value 三頭模型。
- 沿用唯一 parser、既有 packed board cache，以及 `data.cv` 的資料切分與題組產生介面。
- 按級別抽棋局、同盤指定玩家抽七個不同有效位置，分組計算 Bradley–Terry 排序 loss。
- 使用既有 KataGo 強引擎分析完整棋譜的末盤，產生 soft value 偽標籤並保存來源。
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
| 預設 seed | **2026**，包含 split、抽樣與初始化；其他實驗的 seed 不受影響 |
| 級別 | `D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D`，編碼 0..9 |
| Split | 每級 100,000 盤，80,000 train／20,000 valid；呼叫 `data.cv.make_strength_v1_split` |
| 分離單位 | 按棋局切分、棋局零重疊；記錄玩家重疊，不宣稱未見玩家泛化 |
| 有效位置 | 指定玩家落子且零起算 `ply >= 30`；training、validation、centroid 使用相同規則 |
| 歷史 | 落子前八步盤面、最新優先，每對為當手方／對手；不足補零，另加黑白回合兩層 |
| Policy | 預測所選當手，row-major 0..360；361 為 pass，盤面與 target 同步 D4 |
| 取樣 | 每組十級，各級均勻抽一盤，再抽同盤指定玩家七個不同有效位置；跨組／step 可重複 |
| Global batch | 512 組 × 10 級 × 7 位置 = 35,840 個位置／update |
| 模型 | 256 channels、20 residual blocks、三頭；block／head 細節依作者固定版本查核 |
| Loss | `L_BT + L_policy + L_value`，初始權重皆為 1 |
| Optimizer | SGD，momentum=0.9、weight_decay=0.0001 |
| Schedule | 10,000 optimizer updates；完成第 7,692 次後，LR 由 0.01 降至 0.005 |
| Teacher | 既有 KataGo 強網路、日本規則／6.5；visits 由固定 training 小樣本短測決定 |
| Value | 黑方 `2*p_black-1`，白方取相反數；不將 soft target 硬分類 |
| Centroid | 全部 800,000 盤 training 的全部有效位置，按位置等權；不使用 validation 擬合 |
| 題目解碼 | 題內所有有效位置的 scalar score 平均後取最近 rank centroid |

唯一棋局鍵為 `(rank, game_id)`，另以 `game_index` 對齊快取；不能只用 `game_id` join。ID 與檔名不得成為模型特徵。排除前 30 手只限制樣本位置，不裁切歷史，也不裁切 teacher 重播。

## 3. 實作順序與驗收

### 第一步：設定、split 與有效位置索引

**狀態：程式已完成，合成資料驗收通過；官方全量資料尚未準備。**

實作範圍：`config.py`、根目錄 `configs/strength/se_seed2026.json`、`prepare.py`。沿用 packed cache metadata／實際顏色，產生 split、每盤有效位置及 manifest；不重新 parse 官方 CSV／SGF，不複製全量盤面。

交付：`games.parquet`、`{train,valid}_{game,offsets,ply}.npy`、`manifest.json`。

驗收目標：

- 正式資料每級精確 80,000／20,000 盤；train／valid 棋局交集為 0，玩家重疊有數字紀錄。
- 索引完整包含指定玩家且 `ply >= 30` 的位置，每盤至少七個不同位置；不足即報錯。
- seed、來源 cache、程式與產物雜湊、各級盤數／位置數均可追溯。
- 相同設定可驗證並重用；設定衝突、資料損壞或未完成目錄不得靜默覆寫。

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

### 第三步：末盤 teacher 入口、品質短測與標籤準備

**狀態：待實作。依賴：第一、二步。**

實作範圍：新增 `label_value.py`；沿用既有引擎協定，逐盤重播完整棋譜，只分析末盤。保存每盤 `p_black`、黑方 score estimate、實際 visits、可用的不確定性、狀態與來源 manifest；支援分片、重試、續跑，將標籤接入 dataset。

驗收目標：

- 每個成功輸出的棋局只有一筆有效末盤結果，以 `game_index` 與 `(rank, game_id)` 雙重對齊。
- 記錄引擎／網路 SHA-256、規則、貼目、分析設定與勝率視角；不使用 HumanSL 作為強 teacher。
- 不加虛構 pass、不自動續弈、不覆寫 SGF／RE；失敗不補成 0.5 或硬勝負。
- 在固定 training 小樣本比較 visits，量測勝率穩定性、失敗率與 games/s，凍結 visits／品質處理，數值門檻由短測結果明列，不能事後依 validation rank 調整。
- 續跑不重複已成功資料，來源不同的分片不能混合。
- 品質遮罩／權重只影響 value loss，不刪掉 ranking 樣本或破壞十級分組；正式訓練前每盤標籤都有明確處理狀態。
- 全量前回報 800,000 盤 training 的時間、GPU-hours、RAM 與磁碟需求，依 CLAUDE.md §9 確認長作業。Validation 若要報 value MSE，另用同一凍結設定生成 200,000 盤標籤。

標籤格式及小樣本流程完成後，可繼續開發第四至七步；正式訓練前須完成所需全量標籤。

### 第四步：分組 sampler 與 teacher 標籤讀取整合

**狀態：待實作。依賴：有效位置索引、樣本介面及第三步標籤格式。**

實作範圍：擴充 `dataset.py`，按 step 產生完整排序組，保留 `[groups, 10, 7, ...]` 結構；將已對齊的每盤 teacher target 依樣本視角接入。抽樣使用 training，不以全資料 epoch shuffle 取代分組抽樣。

驗收目標：

- 每組十級各出現一次；每級七手來自同盤同一指定玩家，七手互異且都符合有效位置規則。
- 級內按棋局均勻抽樣，不因棋局較長而提高該盤被選中的機率。
- 同盤不同位置共用同一末盤 teacher 機率，依視角轉換；標籤錯配明確報錯。
- 同 seed 與消費端 step 可重建抽樣與增強，不受 worker 數、prefetch 或續跑影響。
- DDP 依全域組編號分配完整組；8 卡時各 64 組，無額外補齊組；保留未來跨 step 重抽同盤的合法行為。

### 第五步：三頭模型與分組 loss

**狀態：待實作。依賴：第二、四步的資料契約。**

實作範圍：新增 `model.py`、`losses.py`；核對固定作者版本的 residual activation、BN、各 head 結構與 value 輸出範圍。Strength 為 scalar，policy 為 362 logits，value 接收 soft target。

驗收目標：

- 完整列出與作者網路的對齊項目及偏離，不僅以深度／寬度相同認定等價。
- 每組每級先平均七個 strength scores，再以強到弱順序計算 listwise loss、除以 `10-1`，最後平均各組；不可先平均全域各級再算一次 loss。
- 級別平均與 logsumexp 使用 FP32；與直接公式的 loss 絕對差 ≤ `1e-6`，極端輸入下保持有限。
- Policy CE、value MSE、BT 各自可報告，總 loss 權重皆為 1；有遮罩時明確記錄 value loss 的有效樣本與歸一化方式。
- 在可控 BN 行為的 FP32 基準中，完整組計算與兩程序分組計算的 loss／參數梯度相對差 ≤ `1e-5`；接近零的量另報絕對誤差。

### 第六步：訓練器、單卡／DDP 短測與續跑

**狀態：待實作。依賴：第三步小樣本標籤、第四、五步。**

實作範圍：新增 `train.py`；整合 SGD、step scheduler、AMP、單節點 DDP、三項 loss 紀錄及原子 checkpoint。沿用 shared runtime／checkpoint 能力，適配本版 sampler 與資料身分。

驗收目標：

- 一次 optimizer update 對應 512 個完整組；訓練停止條件為 10,000 updates，與 epoch／讀過多少不同棋局無關。
- LR 在完成第 7,692 次更新後切換；checkpoint 保存模型、optimizer、scheduler、RNG、消費端 step、抽樣身分及 config／資料來源。
- 固定 seed 的中斷續跑與連續執行具有相同抽樣、增強、optimizer steps 及 LR；在固定 FP32／確定性短測條件下比較 loss 與參數差異。
- 普通 BN／SyncBN、BF16 與必要 microbatch 的行為另行查核、記錄；不可把 BN 差異誤判為分組 loss 錯誤或宣稱跨硬體逐值一致。
- H200 dev 單卡 100–500 steps 短測通過後，再測所需 DDP 配置，記錄 steps/s、峰值顯存、RAM、GPU-hours、I/O 與 checkpoint 成本。
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

實作範圍：完成官方資料準備與 training teacher 標籤，執行正式 10,000 updates，使用最終模型完成全量 training centroid 與 validation 評估。

驗收目標：

- 啟動前分開回報資料準備、teacher、訓練、training centroid 推論、validation 推論的估時與磁碟需求；總 GPU-hours 依實測估算，不套用論文 A5000 成本。
- 每項長作業依 CLAUDE.md §9 回報並確認啟動；開始與完成後紀錄實際硬體、版本、耗時與失敗狀態。
- 完成 10,000 次 optimizer updates；最終權重、有效 config、資料／teacher 身分、checkpoint 及訓練紀錄可以支援續跑與重建。
- 全量 training centroid 和固定 validation 報表可從保存產物重建；結果無論高低都寫入根目錄 `RESULTS.md`，來源及貢獻寫入 `EXTERNAL_RESOURCES.md`。
- 分數解讀明記 game-disjoint、玩家可能重疊，以及 teacher 偽標籤等偏離；不能直接和主線 player-disjoint 成績宣稱優劣。

## 4. 交付位置與目前進度

| 類型 | 路徑（相對專案根目錄） |
|---|---|
| 專用程式與本計畫 | `src/go_aicup/tasks/strength/reproduce_paper/` |
| 預設設定 | `configs/strength/se_seed2026.json` |
| 已有合成測試 | `tests/test_reproduce_paper_data.py` |
| 預設 prepared 資料 | `data/processed/reproduce_paper/seed2026/` |
| 預設 run 輸出 | `outputs/rank/reproduce_paper/se_seed2026/` |
| 實驗與外部資源紀錄 | `RESULTS.md`、`EXTERNAL_RESOURCES.md` |

第一、二步已完成程式與七項合成測試；第三至八步尚未完成。Teacher visits／品質門檻、精度、BN／SyncBN、microbatch、監控與 checkpoint 頻率、正式成本，依後續查核與短測回填。既有官方全量資料準備、teacher 分析及正式訓練均尚未執行。

目前下一個交付為第三步的「末盤分析入口＋固定 training 小樣本短測」，先確立標籤契約與成本，再安排全量分析。
