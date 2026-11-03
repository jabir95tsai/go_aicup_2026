# SE3：18-block ResNet＋十級 CE 分類實驗計畫

狀態：五項實作小任務與 CPU／Gloo 工程驗證完成；尚未啟動 GPU 短測或正式訓練。日期：2026-11-03（台灣時間）。

本文件整合本次使用者已確認的設定。沿用 SE／SE2 的盤面表示與工程流程；新實驗獨立保存，不覆寫原實驗。專案規範以根目錄 `CLAUDE.md` 為準。

## 1. 假設、比較與成功門檻

- 假設：以十級分類 CE 取代 scalar Bradley–Terry 排序，讓棋力監督直接對應十級分類；保留 policy／value 輔助任務。
- 每盤由七個增加為三十個有效位置，以增加單盤訓練訊號；依固定題目官方分數選模，避免僅依 training loss 選權重。
- 同時改架構、loss、抽樣位置數、optimizer 與訓練長度，結果不能單獨歸因於其中一項。
- 預期增益：未知，不宣稱 CE、AdamW 或較多位置必然改善分數。
- 建議成敗 gate（尚未獨立核准）：同題、同位置評估條件下，Equal-k 相對原 SE 至少 +0.005，且任一 k 桶退步不超過 0.01。此門檻不參與 early stopping。
- 歷史參考：原 SE 全量 10,000 題 Equal-k=0.5979；SE2 新增 50k／70k 步為 0.5746／0.5412。數字依 `RESULTS.md`；抽樣監控與全量結果不可直接混比。
- SE 系列使用 game-disjoint，validation 玩家可能在 training 出現；與 V2 的 player-disjoint Valid-U 不同，不能直接排名架構效果。

## 2. 模型與 loss

### 2.1 主幹與三個 head

- 從頭初始化，seed=2026，不載入 SE／SE2 的 20-block 權重。
- 沿用 `reproduce_paper/model.py` 的 stem／ResidualBlock 語意：卷積 bias、BN、residual 相加後 ReLU 與預設初始化。
- 輸入維持 `[B,18,19,19]`；channels=256，residual blocks=18。主幹輸出 `[B,256,19,19]`。
- 棋力分類 head：`Conv1x1(256→1) → BN → ReLU → Flatten(361) → Linear(361→256) → ReLU → Linear(256→10)`。
- 原先要求的 362 維已由使用者修正為 361；不將 policy logits 串接至棋力 head。
- policy head 保持 `Conv1x1(256→2) → BN → ReLU → Flatten(722) → Linear(722→362)`，包含 pass。
- value head 保持原 scalar head 與最後 tanh。
- 支援逐位置推論及每組十級、每盤三十位置的訓練輸入；不得沿用寫死 `(10,7)` 的 shape 驗證。

### 2.2 監督與 loss

- 棋力 CE 取代 BT；逐位置以該棋局的 rank_id（0..9）為標籤。
- policy CE、teacher soft value MSE 保留，使用既有逐位置 teacher value。
- 三項 loss 等權：`total = strength_ce + policy_ce + value_mse`。
- CE 直接接收原始 logits；不先 softmax 再傳入 CE。
- CE／MSE 與全域平均採 FP32，DDP 梯度與平均語意保持一致。
- 不新增 label smoothing、ordinal loss、centroid 或 learned weighting head。

## 3. 資料與抽樣

- 共用資料：`data/processed/reproduce_paper/seed2026/`，棋盤 cache 為 `data/processed/boards/strength_v1/`。
- teacher value：`data/processed/reproduce_paper/value_v1/train/value.npy`。
- split_seed=2026；每級 training 80,000 盤、validation 20,000 盤，共 800,000／200,000 盤；重用現有固定切分，不重抽或覆寫。
- 有效位置維持指定玩家、零起算 ply>=30 的落子前盤面；本人／對手八步歷史及顏色層沿用既有 18-plane 編碼。
- 一組為十級各一盤，每盤抽三十個不同有效位置，共 300 位置。
- 沿用 SE2：每級棋局分別 shuffle，每輪完整用完才重新 shuffle；同輪棋局不重複，不要求遍歷同盤全部位置。
- 保持 training D4 增強，棋盤與 policy target 同步變換；抽樣、D4 與 worker 預取皆由已消費的 step 可重建。
- 實作時先檢查所有相關棋局是否至少有三十個有效位置；不足時明確報錯，不靜默補抽、漏盤或改切分。若實際遇到不足，再提出處理方案。
- 既有 prepared manifest 將每盤七位置寫入 identity；新 sampler 的三十位置屬訓練設定。需區分「可用位置索引身分」與「本 run 抽樣數」，驗證後唯讀重用索引，不修改舊 manifest 或假裝舊設定與新設定完全相同。

## 4. Optimizer、batch 與步數

| 項目 | 設定 |
|---|---|
| Optimizer | AdamW |
| betas／eps | (0.9,0.999)／1e-8 |
| weight_decay | 0.0001，沿用 SE2 全參數同一衰減方式 |
| 第 1–70,000 次更新 | LR=0.01 |
| 第 70,001–100,000 次更新 | LR=0.005 |
| Warmup | 無 |
| Global batch | 128 組／update |
| GPU | 單節點 8×H200，每卡 16 組 |
| 位置數 | 每步 38,400，每卡 4,800 |
| 最大更新數 | 100,000 optimizer updates |
| 計算設定 | BF16、SyncBN、channels_last=true、cudnn_benchmark=false，其他沿用 SE2 |

- 完成第 70,000 次更新後切換 LR，避免 off-by-one；early stopping 可能在降 LR 前結束，不延後早停等待降 LR。
- 不按 batch 額外放大 LR，不增加第三段 LR。
- 每輪 625 步，最多 160 輪；最多使用 3,840,000,000 次位置樣本，不等於獨立盤面數。
- LR 數值依使用者指定採論文基準；AdamW 與原 SGD 不具有已證實的固定換算倍率，不宣稱相同 LR 代表相同有效更新。

## 5. 題目聚合與訓練中驗證

### 5.1 唯一正式聚合規則

1. 每個位置以 `log_softmax(rank_logits)` 取得十維 log-prob，使用穩定實作，不先 softmax 再取 log。
2. 盤內對位置的 log-prob 等權平均。
3. 題內各盤的十維平均 log-prob 再等權平均，不依盤長加權。
4. 對聚合向量取 argmax 作為級別；並列取較低 rank index。

不使用 centroid、機率算術平均、溫度校準或 score_rank 解碼。官方得分公式與解碼方法分開：以 argmax 預測後計官方 exact／相鄰級得分，不代表用 score_rank 解碼。

### 5.2 固定監控集合

- 每完成 5,000 updates 驗證一次，包含第 100,000 步，最多二十次。
- 從既有固定 validation 題庫分層選最多 5,000 題；目標 5,000 題，k=5/10/15/20 各 1,250 題，每個 k／rank cell 125 題。
- 固定 seed=2026、題目 ID、每盤三十個位置及 manifest；所有模型驗證重用同一集合，不每次重抽。
- 題內仍同玩家同色、棋局互異，且與 training 棋局零重疊；不同題目共用棋局時重用同一盤的推論結果。
- 每盤無放回固定抽三十個有效位置，eval 模式、不做 D4；只推論監控題目涉及的棋局。
- 取消原先「一百組建立 centroid」：分類流程不需要這個階段。
- 分別報 k=5/10/15/20 的官方分數：完全正確=1、差一級=1/e、其他=0；四桶等權平均 Equal-k 為唯一 early-stopping 分數。
- 選模使用未四捨五入的分數；保存每桶得分組成與題數，避免顯示值相同卻被誤認為精確同分。

## 6. Early stopping、checkpoint 與續跑

- 首次成功驗證建立 best，未改善次數=0。
- `score >= best_score`：接受本次權重為 best；同分保留較晚模型；未改善次數歸零。
- `score < best_score`：未改善次數加一；連續三次後停止。
- 驗證失敗、非有限值或缺少 k 桶不計成較差分數；保存可恢復狀態並報錯，不默默繼續選模。
- 最早可在第 20,000 步早停：5,000 步首次 best，後三次都較低。
- 每 1,000 步保存 latest 完整 checkpoint，驗證邊界與停止時也保存；另保存完整 best，不沿用 training loss 的 best_1／best_2 規則。
- checkpoint 包含模型／BN、AdamW、scheduler、每 rank RNG、已完成 step、資料與程式身分、best score／step、連續較低次數、固定監控集合身分及驗證進度。
- DDP 所有 rank 同步驗證結果與停止決定；評估後還原訓練模式及所需 RNG，不讓評估改變後續抽樣或訓練軌跡。
- 驗證完成記錄與 best／latest 發布須可重試；重啟不重複累計 patience，不跳過待完成驗證。
- 達 100,000 步也發布最佳驗證權重，不直接使用最後一步。
- best／latest 以原子寫入與重載驗證保護；確認最佳模型發布成功前不刪除恢復來源。
- Slurm／人工中斷屬暫停，保存 latest 供續跑；只有 patience 觸發或達預算上限才執行最終全量評估。

## 7. GPU 短測與正式作業

使用者已同意：8 卡、100 步，並執行一次固定最多 5,000 題的驗證。

- 獨立 smoke config／output，不混入正式 run、不以 smoke 權重接續正式訓練。
- 使用相同 18-block 模型、global 128 組、三十位置、AdamW LR=0.01、BF16／SyncBN／channels-last。
- smoke 在第 100 步額外執行監控驗證，供成本與接線驗收；正式頻率仍為每 5,000 步。
- 檢查三項 loss、梯度與更新有限且有效、顯存、吞吐、資料供應、checkpoint 重載及驗證各 k 分數。
- 短測不能證明 LR 長程穩定或模型已收斂；不以一百步成績判模型優劣。
- 如每卡 4,800 位置 OOM，先報告實測再調整；不靜默縮 global batch。梯度累積會改變 BN batch 語意，不宣稱與完整 batch 等價。
- 短測後回報每步與單次驗證耗時、峰值顯存、正式時間／GPU-hours／儲存需求估算，再依專案規範確認長作業；不自動提交正式訓練。

## 8. 最佳模型全量評估

早停或滿十萬步均執行，不能沿用舊 evaluator「步數必須等於預算」的限制而拒絕合法早停模型。

- 發布且重載驗證最佳模型為 `model.pt`，記錄 best_step、實際停止 step、停止原因與監控分數。
- 使用完整 validation 200,000 盤的全部有效位置推論；不限制每盤三十位置。
- 使用共用固定 10,000 題，以第 5.1 節同一 log-prob 聚合與 argmax 解碼評估。
- 最終評估的位置數與題數大於訓練中監控，分數須分開標示，不用最終分數回頭改選 checkpoint。
- 不做 training 全量推論或 centroid；本版分類評估不需要它們。
- 輸出四桶與 Equal-k 官方分數、各級結果、confusion matrices、逐盤／逐題結果及 loss／監控分數曲線。
- 保存足以重建報表的逐盤十維 log-prob 平均及位置數；不要求保存全量逐位置十維輸出。
- 全量評估失敗可單獨重試，不重訓、不改 best；推論分片須綁定模型與資料身分，禁止混用其他權重結果。

## 9. 實作範圍與預定產物

- 新套件：`src/go_aicup/tasks/strength/reproduce_paper_3/`，含專用 config、model、dataset／sampler、losses、train、evaluate 與 README；具體拆檔依既有可重用邏輯決定，不為預留用途增加模組。
- 重用 SE 的盤面輸入、cache、policy D4／teacher target；重用 SE2 的全輪棋局排列原則及 shared runtime／checkpoint 工具。
- 需改寫或適配的邊界：寫死七位置的 sampler／collate／model shape、三十位置與 prepared identity、分類輸出與 loss、定期驗證／早停、最佳模型發布、早停模型載入與全量分類報表。
- 原 SE／SE2 checkpoint 與現有訓練入口不得因新實驗失去可讀性；不修改原模型定義以破壞舊 code fingerprint。
- 預定正式設定：`configs/strength/se3_seed2026.json`。
- 預定短測設定：`configs/strength/se3_smoke100.json`。
- 預定 Slurm：`slurm/se3_train.slurm`、`slurm/se3_smoke100.slurm`；留意目前 repo 忽略 `/slurm/` 的規則，不默默更動 Git 追蹤政策。
- 預定正式輸出：`outputs/rank/reproduce_paper_3/se3_seed2026/`。
- 預定短測輸出：`outputs/rank/reproduce_paper_3/se3_smoke100_8gpu/`。
- 正式產物：config／資料與監控身分、training／validation logs、latest／best checkpoints、最佳 `model.pt` 與完成標記、最終 `evaluation/` 報表及 PNG。
- 更新 `CLAUDE.md`、`RESULTS.md` 的狀態與實際驗收數字；沒有計算的分數明記未知。

## 10. 驗收與成本限制

### 工程測試

- 三頭 shape、分類 logits、三十位置輸入及 CE／policy／value loss 的有限性與梯度。
- 八 rank 分組、每輪全棋局覆蓋、每盤三十位置不重複、worker 預取中斷續跑一致性。
- prepared 索引唯讀重用與不相容設定拒絕；不足三十位置明確失敗。
- 分層固定題組、固定位置與零 train 棋局重疊。
- 以合成數字驗證 log-prob 盤內平均／盤間等權／argmax，含不同盤長與並列情況。
- LR 第 70,000／70,001 步邊界。
- 同分重設 patience、三次嚴格較低停止、驗證失敗不累計、續跑與失敗重試不重複計數。
- CPU／Gloo 驗證全域平均與梯度；CPU 通過不替代八卡 GPU 驗收。
- 早停與預算完成皆發布 best 並執行全量評估；eval 重試不增加訓練步數。

### 成本與風險

- 先前依 SE2 文件估時、位置數與 block 數粗略外推：滿十萬步純訓練約二十小時／160 H200 GPU-h，尚未包含最多二十次監控與最終全量推論；此為未實測估計，正式以短測重估。
- 不將舊 smoke 的吞吐或顯存直接視為本版數字；每卡位置數增加至原先的 4.29 倍。
- AdamW LR=0.01 的穩定性、正式收斂、各 k 分數與增益未知；early stopping 不保證避免所有過擬合。
- 固定 validation 用於模型選擇，最終同切分報表不是獨立未見測試估計；不據此宣稱新玩家泛化或官方 LB 成績。
- 本輪已完成獨立程式、設定、Slurm 與 CPU／Gloo 合成工程測試；僅唯讀檢查官方位置索引，沒有重新準備官方資料、GPU 短測、正式訓練、官方全量評估或上傳。


## 11. 實作進度（2026-11-03）

依序完成五個小任務，程式及操作見 [README.md](README.md)：

| 次序 | 小任務 | 狀態 |
|---|---|---|
| 1 | 獨立 config、18-block 三頭分類模型、等權 FP32 CE／CE／MSE | 已完成；正式模型 21,754,485 parameters，361→256→10 |
| 2 | 三十位置 sampler／collate、舊索引唯讀適配、固定監控題組／位置 | 已完成；八 rank 全輪覆蓋與 spawn／prefetch 續跑驗證 |
| 3 | 穩定 log-prob 聚合、監控、全量 validation 分片／分類報表與 PNG | 已完成；CPU 合成全量流程及報表重建驗證 |
| 4 | AdamW／LR、定期驗證、early stopping、原子 latest／best、最佳模型發布 | 已完成；四處驗證失敗注入、最終發布失敗、半行 log、早停／續跑驗證 |
| 5 | CPU／兩程序 Gloo、短測／正式 config 與 Slurm、文件／結果更新 | 已完成程式與工程驗收；八卡 GPU 驗收尚待執行 |

SE3 最終合成測試 20 項通過（52.34s）；既有 SE／SE2 相關回歸 87 項通過（合併較早一輪 105 項／140.60s，當時 SE3 18 項；其後補入 smoke 與最終發布失敗檢查）。Gloo 使用固定 BN 檢查全域 loss／梯度，另測一般 BN 的兩程序訓練、續跑與評估；不能代替 CUDA／SyncBN。測試命令及限制見 README。

官方索引唯讀檢查：training 800,000 盤、validation 200,000 盤的最少有效位置均為 35，不足三十位置均為 0；未重切或改寫 prepared／teacher。監控與最終 eval 的實際官方 k=5／10／15／20 分數、顯存、吞吐與長程 LR 穩定性均未知，尚未提交 GPU 工作。既有 SE／SE2 定義及入口未修改。

接續作業是執行 `slurm/se3_smoke100.slurm`，回報一次監控與純訓練成本後，再確認正式長作業。`/slurm/` 仍被 Git 忽略，兩份 launcher 只存在工作區，未更動追蹤政策。
