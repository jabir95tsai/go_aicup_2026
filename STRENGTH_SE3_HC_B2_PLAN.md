# 棋力新實驗：凍結 SE3＋H＋C，重新訓練 B2

2026-11-06。狀態：使用者已核准實作；程式、設定、Slurm 與操作文件已建立，CPU 合成驗收與既有來源唯讀檢查已完成；尚未執行正式資料準備、GPU 推論、B2 正式訓練或提交作業。實作目錄依使用者指定為 `src/go_aicup/tasks/strength/SE_H_C_B2/`，操作見該目錄 README；正式長作業仍先回報實測估時。

## 1. 研究問題與實驗範圍

將現行 AH＋C＋B2 的 A（A-024 V2）換成已完訓 SE3 的十級 strength head 輸出，記為 **SE3＋H＋C＋B2**。SE3 的 backbone／strength head、HumanSL H、C 的級別條件向量與其 HumanSL backbone 全部凍結，只重新訓練 B2 的兩層分類頭及擬合其標準化器。

這裡的「預訓練模型」是既有特徵產生器的集合，沒有一個共同的 AH＋C 神經網路。SE3 需連同其 backbone 載入，不能只把 SE head 權重接到 V2 backbone。舊 A／舊 B2 的權重不移植到新 B2；替換後沒有 V2 的十維輸入，特徵仍為 10＋30＋20＝60 維。

假設：SE3 用十級 CE，另受 policy／value 輔助監督，其逐盤十級分布可能提供不同於 V2、H、C 的棋力訊號。B2 可重新學習各訊號與級別的對應和信心尺度。**實際增益及機制未知**；SE3 單獨較高或較低都不能推論融合增益。

成功目標預先定為同題 Equal-k 至少 **+0.005**，任一 k 退步不超過 **0.005**；這是值得繼續的門檻，並非預測一定會提升。依使用者更新，B2 改為 `Linear(60,32) → ReLU → Linear(32,10) → Softmax`，最後十維不使用 ReLU；k=5／10／15／20 仍各自一個分類器。暫不加入 HD、SE3 微調、HumanSL／C 重訓、attention、ensemble 或新 teacher。

## 2. 固定模型與資料來源

目前工作區的舊產物已搬到 `outputs/rank/humansl/`、`outputs/rank/v2/`；舊程式與模型 info 仍含搬移前路徑。實作必須使用存在的來源並記錄內容雜湊，不能直接照抄舊預設。

| 元件 | 固定來源 | 使用方式 |
|---|---|---|
| SE3 | `outputs/rank/reproduce_paper_3/se3_seed2026_15pos/model.pt` | best 第 30,000 步，training 在 45,000 步 early stop；18 blocks／256 channels，十級 `rank_logits` |
| H | `data/external/humansl-v0/` 的既有 HumanSL 網路及原 analysis 設定 | 15 個原 rank profiles，沿用原規則／貼目／位置選擇與數值精度 |
| C | `outputs/rank/humansl/humansl_rankvec/vectors.pt` | 十級 × 本人／對手 × 384 維的既有條件向量；不呼叫 C 的 train 入口 |
| 舊 A，配對對照 | `outputs/rank/v2/v2_strength/model.pt` 及既有逐盤分類產物 | 原十維逐盤平均 log-prob |
| 歷史提交模型 | `outputs/rank/humansl/humansl_rankvec_model/model.pkl` | 唯讀重現／對照，不覆寫 |
| SE 共用資料 | `data/processed/reproduce_paper/seed2026/` | 原始 800,000／200,000 game split、位置索引與 metadata，不重切 |
| 棋盤快取 | `data/processed/boards/strength_v1/` | 重用既有棋步／盤面與顏色，不重新 parse 百萬盤 |

已查核內容身分：

- SE3 SHA-256：`74c65097c53d85897c8e4239e910415eec1b8f3b474945b1d3357925a95a8cd0`（既有完成 marker／evaluation manifest 一致；執行前仍重驗）。
- C vectors SHA-256：`a9c5b4586f8409054d6919ce0ed659feac05b383d30cacedef96a99fadc85886`。
- 舊 A SHA-256：`5b9f4bade366b8d770db0507f13c96c6a94ab9abd83d4ca62216ad258619a3d9`。
- 舊 AH＋C B2 SHA-256：`130d1f4670ec0789f6f482394a9146faf7fb7f62f3f808464274695b569f2314`。

所有神經網路推論使用 `eval()`、`requires_grad_(False)`、`inference_mode()`；BN running statistics／counters 也不得改動。C 向量以一般 tensor 讀入，不進 optimizer。每輪前後核對 state／buffer 與來源檔雜湊。B2 改用 AdamW，只更新四個兩層分類頭的參數；四個 scaler 各自只從該 k 的擬合題計算統計量。上游 SE3／H／C 不進 optimizer，訓練 B2 時只讀已快取的六十維特徵。

## 3. 特徵與解碼契約

SE3 每盤特徵為指定玩家、零起算 ply≥30 的全部有效落子前盤面：

`S_g = mean_t(log_softmax(SE3.rank_logits(board_before_move_t)))`，共十維，級別順序固定 `D,C,B,A,1D,2D,3D,4D,5D,6D`。

沿用 SE3 的本人／對手八步歷史 18 planes。推論不做隨機 D4；不是訓練時只抽十五位置，也不是先將所有位置混成一題。`log_softmax` 用 FP32，累加沿用 FP64，保存逐盤均值及有效位置數。policy／value 輸出不進 B2；本次不修改 SE3 forward 來省掉其他 head。

H 每盤本人 15＋對手 15＝30 維；C 每盤本人十個級別假設＋對手十個假設＝20 維。H／C 原本的落子範圍與推論配方保持，**不因 SE3 的 ply≥30 同步截斷**，否則多改一個因素。

新特徵列固定 `[se3_0..9, H 原 30 欄, C 原 20 欄]`。每題先對 k 盤的六十維各自等權平均，再交給對應 k 的 B2。B2 的十類後驗以 `metrics.score_rank` 解碼；SE3 單模型既有 argmax 成績只作背景資料，不能直接當新 B2 對照。沒有額外溫度、centroid、人工 rank 比例校正或切點。

ID／rank label／HD／執色不作額外數值特徵；ID 只用於 join、分組與洩漏檢查，label 只作 B2 supervision。執色仍用於正確還原本人視角及同色抽題。

## 4. 已查到的洩漏與評估設計

SE3 training 有 8,007 位玩家；validation 的 7,855 位全部在 training 中。它與舊 V2 player-disjoint 的含義不同。

舊 HumanSL 資料池的實際對齊如下；`game_index` 與共用 board metadata 的 rank／game_id／player_id／player_color 已逐列核對一致，沒有僅按 `game_id` 合併。

| 舊資料角色 | 棋局 | 玩家 | 在 SE3 training | 在 SE3 validation |
|---|---:|---:|---:|---:|
| S-fit＋S-extra | 38,589 | 4,094 | 30,872 | 7,717 |
| S-select | 26,634 | 906 | 21,307 | 5,327 |
| 全部 Valid-S | 65,223 | 5,000 | 52,179 | 13,044 |
| 舊 U 題組的棋局 | 11,766 | 676 | 9,353 | 2,413 |

舊 U 共 1,000 題、每 k 250 題，**四個桶各自都是 0 題全部來自 SE3 validation**。因此不能把新模型在舊 U 的分數當成未見棋局／未見玩家的驗證，也不能直接拿它突破 0.6934 就 promote。

### 4.1 開發診斷：沿用原資料角色

第一階段以 S-fit＋S-extra 的 38,589 盤訓練兩個配對 B2，用原 S-select 的 1,000 題選擇各 k 的停止 epoch 並比較兩臂，僅供開發、信心尺度及是否值得進入完整評估的診斷。S-select 玩家與該階段 B2 擬合玩家互斥，亦不在 C 學習玩家中；但部分棋局被 SE3 訓練過，不能宣稱整個系統獨立驗證。

第二階段固定第一階段各臂／各 k 的最佳 epoch 數後，兩臂各以全部 Valid-S 的 65,223 盤從零重訓相同 epoch 數，再在新評估題組一次比較。原 U 的 1,000 題可同步報告，但欄名必須標示「SE3 training 棋局重疊，診斷」。兩階段不得用 U 或新評估題組標籤做 scaler、擬合或超參選擇。

B2 擬合集約八成棋局已被 SE3 用於訓練，其信心分布可能與新棋局不同，會使融合器高估 SE3 或產生校準偏差；本輪固定原擬合池，以新棋局 gate 檢驗結果，不以高擬合分數佐證成功。真正的上游 OOF 特徵需要重訓 SE3，超出本次凍結實驗範圍。

### 4.2 主要配對評估：新棋局題組 `U_newgames`

從**舊 U 題組的 676 位玩家**出發，依原 SE prepared metadata 選擇：

1. 只取 SE3 validation 棋局，與 SE3 training 棋局零交集。
2. 排除本 SE3 run `monitor/positions.npz` 的全部棋局，而非只排除十五個抽樣位置。
3. 排除 C 學習與監控的全部棋局，並驗證評估玩家不在 C 學習／監控玩家中。
4. 評估玩家與兩階段 B2 擬合玩家均零交集，評估棋局亦不參與 B2。

唯讀盤點後，符合上述條件的池有 **16,439 盤、673 位玩家、1,576,762 個 SE3 有效位置**。C 學習／監控玩家與原 U 玩家交集為 0。每級 k=20 可用的 player-color groups（D→6D）為 **15／29／17／13／29／29／24／30／21／15**；全部四十個 k／rank cell 均可抽題。

用固定 seed=2027，凍結 **1,000 題＝每個 k／rank cell 25 題**。先在符合 k 的玩家間等權抽樣，再選該玩家可用顏色，從同色棋局不放回抽 k 盤。跨題可重用玩家／棋局，題內不得重複；完全相同的題組組合去重，抽題前驗證各 cell 足夠，不足就回報，不換成 training 棋局或縮小 k。模型兩臂共用逐字相同的題組。

題組建立與 checksum 固定要在新模型結果產生前完成。輸出各 cell 的玩家數、棋局數、重用率及與各上游資料的交集斷言。這是「SE3 已見玩家的新棋局」評估；部分玩家其他棋局參與過 SE3 訓練／選模，且 SE3 全量 validation 已被分析過，因此**不稱全系統獨立 holdout，也不證明未見玩家泛化**。

凍結既有 SE3 的條件下，無法補出該官方 training 玩家集合內的真正 unseen-player 評估。若未來要回答那個問題，需另訓 player-disjoint SE3，超出本次範圍。正式測試的未見玩家效果仍未知。

## 5. B2 配方與必要對照

現有 `humansl.RankModel._fit_synthetic` 只按 `player_id` 分組，可能把同一玩家的黑白棋局混成訓練題。正式題目是同色，因此本實驗固定採 **player_id＋player_color** 抽樣，舊 A 對照也用相同修正，避免把抽樣修正歸因於 SE3。

原 S-fit＋S-extra 在 k=5／10／15／20 可用同色 groups 為 **2,193／1,067／651／418**；全 Valid-S 為 **3,868／2,183／1,327／835**，所有 k 都涵蓋十級。實作時另檢查每個 k 的十類訓練標籤確實齊全。

| 代號 | 特徵／B2 | 用途 |
|---|---|---|
| R0，歷史參考 | 既有 AH＋C＋舊 B2，原始抽樣 | 原 U 分數 0.6244／0.6977／0.7303／0.7213，Equal-k 0.6934；僅供重現，不作新題組的絕對門檻 |
| R1，配對對照 | 舊 A＋凍結 H＋凍結 C，重新訓練同色 B2 兩層頭 | 與 R2 相同資料、題組、scaler 配方、兩層架構與訓練／選模規則 |
| R2，實驗候選 | 凍結 SE3＋凍結 H＋凍結 C，重新訓練同色 B2 兩層頭 | 與 R1 唯一的特徵差異是十維 A→SE3 |

每階段先存相同的 synthetic training questions，R1／R2 共用。每個 k **20,000 題**、合計 80,000 題；玩家等權抽樣，再在可用顏色中選擇，題內不放回。符合各 k 的玩家數不同，另報告 rank／color／玩家分布，不把 80,000 題當獨立玩家樣本。

### 5.1 兩層分類頭

每個 k 的架構固定為：

`題內平均六十維 → StandardScaler_k → Linear(60,32) → ReLU → Linear(32,10) → Softmax`

只有 32 維隱藏層使用 ReLU，最後十維為可正可負的原始 logits；不加入 dropout、BN 或殘差。每頭 2,282 個參數，四頭合計 9,128 個；scaler 統計量另計。這是新的 B2 分類頭，與歷史 B2 的單層 logistic regression 區分記錄。

訓練直接把十維 logits 傳給 `CrossEntropyLoss`，不先做 Softmax。CE 內部計算穩定的 log_softmax 與負 log-likelihood，等價於對 Softmax 機率取交叉熵；避免先轉成機率再被 CE 重複轉換。推論才顯式 Softmax，將後驗交給 score_rank。這是同一個機率模型的訓練／推論用法，沒有省略隱藏層 ReLU。

### 5.2 訓練集與標準化

不把 SE3 原本 800,000 training 直接全部當 B2 訓練資料，也不拿正式測試題訓練。資料分兩階段：

| 階段 | 來源棋局 | 分類頭的訓練樣本 | 驗證／停止方式 |
|---|---|---|---|
| 開發選模 | S-fit＋S-extra：38,589 盤、4,094 位玩家 | 每 k 固定 20,000 道同玩家同色題，各盤六十維取平均 | S-select：1,000 題，每 k 250 題，按各 k 官方得分選最佳 epoch |
| 最終重訓 | 全部 Valid-S：65,223 盤、5,000 位玩家 | 重新生成每 k 固定 20,000 題，兩臂共用 | 沿用各臂／各 k 的開發最佳 epoch，不用 U_newgames 選模 |
| 主要評估 | U_newgames：從 16,439 盤、673 位玩家的池建立 | 固定 1,000 題，每 k 250 題 | 完全不參與 B2 scaler、loss、epoch 或超參選擇 |

每個 k 的 scaler 僅以該階段的 20,000 道訓練題六十維向量 fit，再原樣 transform 該 k 的驗證／測試題。最終重訓重擬合 scaler；不沿用開發 scaler 或舊 B2 係數。訓練題組固定，epoch 只改題目順序，不重新抽題，避免將題量與更新預算一起變動。類別分布延用上述玩家等權抽題，不另加 class weight 或 label smoothing。

### 5.3 訓練 schedule（待驗證的固定初始配方）

| 設定 | 配方 |
|---|---|
| 訓練裝置／數值 | 預設 CPU、FP32；分類頭不需要八卡 DDP |
| optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| 初始 learning rate | 0.001 |
| weight decay | 0.0001，套用分類頭全部參數 |
| loss | 十類 CrossEntropyLoss，逐題等權平均，輸入原始 logits |
| batch size | 每個 k 各 512 題；保留最後不足 512 的尾批 |
| epoch 上限 | 開發階段每個 k 最多 100 epochs |
| 每 epoch 更新數 | ceil(20,000／512)=40，最多 4,000 optimizer updates／頭 |
| LR scheduler | 每次 optimizer update 後走 cosine decay；固定 4,000 updates 從 0.001 降至 0.00001 |
| warmup | 不使用 |
| validation | 每個 epoch 結束，用該 k 的全部 250 題，以 Softmax＋score_rank 計官方得分 |
| best／early stopping | 該 k 官方得分嚴格提高才更新 best；並列保留較早 epoch；連續 10 epochs 沒有提高即停止 |
| 初始化／shuffle | seed=2027，按 k 派生獨立 RNG；兩臂的相同 k 使用相同初始化及題目 shuffle 順序 |
| checkpoint | 每 epoch 保存 latest（模型／AdamW／scheduler／RNG／游標／best／patience）及最佳模型，來源與題組身分綁定；中斷可續跑 |

各 k 是獨立模型，因此依各 k 分數選模；四個最佳模型組成候選後才彙整 Equal-k。S-select 是開發選模資料，不當最終成績。schedule 不依 U_newgames 或 LB 回調。

最終重訓：對每臂／每 k 記錄開發最佳 epoch E_k，重新初始化、以全部 Valid-S 的新擬合題訓練恰好 E_k epochs。不再 early stop，也不以已納入訓練的 S-select 回頭選權重。**cosine 總長仍固定 4,000 updates**，只執行前 40×E_k 步，不把整段 cosine 壓縮到 E_k，保持與開發階段相同的 LR 軌跡。兩臂的最佳 E_k 可以不同，選擇規則相同；逐頭公開 epoch／updates／LR 與分數。

兩階段 × 兩臂 × 四個 k＝16 次小型頭訓練；最多 64,000 個 optimizer updates，實際通常因最佳 epoch／early stopping 少於此上限，但耗時尚未量測。這些新 schedule 數值是提案，並非已證明最佳；不做學習率、隱藏寬度或 weight decay 搜尋。

R1 與 R0 的差異包含同色抽題、分類頭架構與訓練法；R2 與 R1 的差異才用於判斷 SE3 替換效果。不能將 R2−R0 全部歸因於 SE3，也不能將新 AdamW weight decay 當成與舊 LR 的 C=1 等價。

## 6. 快取、推論與成本

優先重用 SE3 `evaluation/validation_games.parquet` 的全量 200,000 盤逐盤 log-prob，不重跑既有 validation，也不載入 teacher value 標籤。

舊 S＋U 聯集有 **76,989 盤**，其中 **15,457 盤**已有 SE3 validation 特徵，另 **61,532 盤**需以凍結 SE3 推論，合計 **5,918,345 個有效位置**。只對指定棋局推論：分別使用既有 train／valid 索引查詢，不推論全部 800,000 training。

`U_newgames` 的 SE3 十維均可重用現有快取；舊 A、H、C 按實際所需棋局查 coverage，只補缺失。新題組共 12,500 個棋局引用，unique games 至多 12,500；不為方便而把 16,439 盤整池都跑 H／C。正式測試 Public／Private 各 5,000 盤，共 10,000 盤，按內容與模型身分重用 release 快取。

快取至少綁定 model／vectors SHA、盤面與索引身分、input encoding、min_ply、profile／規則／貼目、precision、欄位順序、game_index 及位置數。先檢查來源 metadata 後才接受 game_index join；原始資料唯一鍵仍是 `(rank, game_id)`。缺失／非有限特徵不得 inner join 靜默丟盤；評估缺任一棋局即失敗，補齊後重跑相同題組。

SE3 延用 BF16、channels-last、batch=2240、無 D4／無 TF32；H／C 各自維持原配方，不能因 SE3 關閉 TF32 而悄悄改 C 的既有設定。分開記錄各路數值設定，抽樣比對快取與重算結果，再決定可接受的數值誤差。不同模型／精度產物不得混用。

| 工作 | GPU 成本／估時方式 |
|---|---|
| 本次規劃、metadata 稽核 | 已完成，0 GPU-h |
| 抽題、join、16 次小型分類頭訓練、配對統計 | 預設 CPU／FP32，0 GPU-h；CPU 時間未知，首輪記錄實測 |
| SE3 缺失特徵 | 5,918,345／實測總 positions/s；8 卡 wall hours ×8 記 GPU-h |
| 新評估題組缺失 H／C | 按選中 unique games 的實際位置量，分別除以 H／C 實測吞吐；未知 |
| 正式測試缺失特徵 | 至多 10,000 盤，依 cache coverage 與各路吞吐估時；未知 |
| SE3／HumanSL／C 再訓練 | 不安排，0 GPU-h |

核准後先做獨立短推論：SE3 從缺失池按 rank／color 分層取 1,000 盤；H／C 從新評估缺失池各取固定 100 盤。使用既有 8 卡分片推論慣例，量測暖機外的 end-to-end throughput、峰值顯存、I/O、失敗率及完成片段成本。短測合計配置上限先抓 **8 卡 ×0.5 小時＝4 GPU-h**，這是資源上限而非耗時預測；若預估超出便先回報。

依短測回報完整 wall time／GPU-h／儲存需求及排隊假設，再確認是否啟動完整作業。不能由 SE3 訓練 2.8 steps/s 或歷史 35.5 GPU-h 推論本次純推論成本；H／C 缺失分析可能才是主要成本。

六十維 float32 約每十萬盤 24 MB；另計 metadata、FP64 原始 SE3 表、分片與原子暫存。只保留逐盤特徵、題組及預測，不保存全量逐位置十維輸出，不複製模型／棋盤／teacher 大檔。執行前查當下配額，不採用歷史剩餘空間當現況。

## 7. 報表、統計與判斷 gate

每個資料角色都同題列 R1／R2 的 k=5／10／15／20 官方得分、Equal-k、Δ、題數、玩家數；另列 exact／相鄰比例、per-rank 成績與混淆矩陣。R0 的 0.6934 和 SE3 單模型 0.623873 題組／解碼不同，不能直接排名。

對 `U_newgames` 的 R2−R1 做 **2,000 次配對玩家 bootstrap**，固定統計 seed；每次以玩家為單位重抽，保留該玩家全部題目，兩臂使用相同重抽索引，每次重新計算四桶與 Equal-k。按 rank 分層以維持評估的等級結構；空 k 的 replicate 不計並回報。相同玩家跨 k 的題目不可當獨立資料。

- 工程 gate：來源與 frozen state 檢查、完整特徵／題組、同色抽樣、所有 k 十類齊全、loss／梯度有限、schedule／選模正確、保存／載入預測一致，全部通過。
- 開發 gate：S-select 四桶與 Δ 如實記錄；若全桶退步且 ΔEqual-k≤0，不擴大新 H／C 推論，保留失敗結果。因 SE3 重疊而有的提升只供決定是否繼續，不能宣稱成功。
- 主要候選 gate：`U_newgames` 的 Equal-k **Δ≥+0.005**，且每個 k **Δ≥−0.005**。兩臂的絕對分數目前未知，因此不先填新題組的數字。
- 證據強度：bootstrap 95% 區間下界>0 才稱「本評估設計下有正向證據」；跨零只列不確定的探針候選。即使通過仍須保留「已見玩家的新棋局」限制。
- 正式主線採用：完成格式／推論彩排後，由使用者進行同版本 AH＋C 與 SE3＋H＋C 受控 LB 探針。Public 400 題的微小差異不足以保證 Private 提升；依既有提交紀律做最終選擇。本地 gate 通過不自動覆寫主線或提交。

本計畫只安排兩臂比較，未過 gate 不追加搜尋直到變好。若希望再研究 SE3 的增量來源，可另提 H＋C 消融，先不加入本輪必要算力。

## 8. 最小實作範圍與驗收

新版本放在 `src/go_aicup/tasks/strength/SE_H_C_B2/`，重用現有資料與模型，不修改 SE3 訓練契約。下列範圍已建立，執行與限制見本目錄 README：

| 範圍 | 必須完成的內容 |
|---|---|
| 獨立 config／入口 | 固定模型路徑、雜湊、run 路徑、兩階段資料角色、題組／sampling seed、兩層分類頭／AdamW／schedule／選模設定、推論設定；不增加可切換模型家族的通用框架 |
| 特徵準備 | 重用 `PaperDataset`／`reproduce_paper_3.evaluate.infer_games`；selected train／valid、舊快取讀取、缺失補算、欄位 manifest、凍結驗證及可續跑分片 |
| B2 訓練／評估 | 重用題內均值與 `metrics.score_rank`，沿用 per-k scaler 規則；新增本實驗兩層頭、同色共用題組、CPU 訓練器／AdamW／schedule／選模與專用 checkpoint 格式，不把新頭塞入舊 sklearn `RankModel`；舊流程保持 |
| 正式推論／輸出 | 讓新入口能讀官方 prepared test，使用 SE3 本人視角／ply 規則及 frozen H／C，按保存的欄位契約交給新 B2；不能繼續從 `category.npy` 讀 V2 當新 A |
| release 接線 | 新增獨立候選檔名；校對 `humansl_predict.aggregate` 的 V2 欄位假設與現有 fallback；不得覆寫現有主線產物或把新模型直接塞入舊 loader |
| tests／Slurm／文件 | 小型合成端到端、固定題組與故障驗證、8 卡短推論 launcher、README、執行後更新 RESULTS；launcher 放在新套件的 `slurm/`，避開 repo 根目錄 `/slurm/` 忽略規則 |

最低必要驗收：

1. 不同位置數的兩盤，SE3 盤內均值與題內盤等權結果正確，與逐手混池可區分；只取指定玩家且 ply≥30。
2. 在訓練器執行前後，SE3／H／C 的所有參數與 BN buffers 完全不變；沒有 teacher 或上游 backward。
3. 同色、同玩家、題內不同棋局；兩臂共用同一份抽樣。評估玩家／棋局交集與 SE3 monitor 排除全部通過；故意混入一局 training 必須拒絕。
4. 來源 game_index／metadata 不一致、缺特徵、NaN、欄位錯序、錯模型雜湊、某 k 缺級別均拒絕；不能縮題後沿用原 k。
5. scaler 只 fit 擬合題，四個 k 各有獨立模型；檢查 60→32→10 shape、只有中間一個 ReLU、logits 接 CE、推論 Softmax 與 score_rank；序列化重載的 posterior／rank 一致。另驗證 LR 首尾／尾批、最佳 epoch／patience、逐步中斷續跑的 AdamW／scheduler／RNG／shuffle 狀態及最終固定 epoch 重訓。
6. 小型假官方 CSV 彩排黑／白與 k=5／10／15／20，確認全盤推論、同色聚合、rank 字串與 sample_submission 格式一致；分片中斷不混用不同模型快取。
7. 正式資料某盤不能重播或缺 H／C 時，整題明確標記並使用既有 AH＋C 該題 posterior 作 fallback，再解碼；沒有可用舊 posterior 時直接失敗。不得靜默丟盤／縮 k，fallback 題數與原因獨立報告。

輸出根目錄固定為 `outputs/rank/se3_hc_b2/se3_30k_mlp32_seed2027/`，建議保留 `config.json`、`manifest.json`、`audit.json`、`questions/`、`features/`、`models/`、`reports/`。每階段兩臂模型分別保存，最終候選為全部 Valid-S 擬合的 R2；評估題不加入最終 B2 擬合集。

正式候選依官方 README 合併 Public／Private 為 `submissions/rank/submission_rank_se3_hc_b2.csv`，800 列、`question_id,pred_rank`、sample 順序，保留現有 AH＋C CSV 供對照／fallback。實際欄位與列序以本地官方 sample_submission 驗證，不從舊文件猜測。產檔後在 RESULTS 登記模型身分、四桶本地成績、重疊限制、fallback、預期 LB（未知）與假設，由使用者手動上傳。

## 9. 執行順序與完成定義

使用者要求一次提交後，已新增 `src/go_aicup/tasks/strength/SE_H_C_B2/slurm/train.slurm`，將以下 prepare、開發、gate、最終訓練與新棋局評估串成單一任務；開發 gate 不通過時保留報表並停止。預設 8 GPU、12 小時上限＝最多 96 GPU-h，包含 CPU 階段；實際耗時未測，仍以短測或現場記錄重估。各階段原子分片與 checkpoint 供重新提交時重用，操作見套件 README。

1. 確認計畫後，先做獨立設定、metadata／模型身分 audit 與固定 synthetic／`U_newgames` 題組；本輪唯讀數字在執行時重驗。
2. 實作最小特徵接線與合成驗收；既有產物與他人未提交變更保持原狀。
3. 進行短推論、回報全量估時，確認後只補缺失特徵。
4. 完成 S-fit→S-select 的 R1／R2；若符合繼續條件，再補新評估所需 H／C。
5. 固定各臂／各 k 的最佳 epoch，以全部 Valid-S 重擬合 scaler 並從零重訓分類頭，不再早停；一次評估固定 `U_newgames`，另附原 U 的受污染診斷及 bootstrap。
6. 通過候選 gate 才產正式兩份候選、完整彩排、登記結果；未過亦保留報表與模型，不取代現行主線。

實作完成後可依新目錄 README 逐步重建 frozen-feature B2、配對四桶結果與 CSV；訓練與評估也可由單一 Slurm 任務依開發 gate 自動接續，正式測試 CSV 仍使用獨立流程。CPU 合成驗收不代表正式實驗已執行完成；新模型各 k 分數、融合增益、LB、純推論吞吐與完整 GPU 成本目前全部未知。
