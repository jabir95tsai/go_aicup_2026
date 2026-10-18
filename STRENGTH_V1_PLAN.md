# 第一版棋力辨識計畫：盤面 ResNet 與級別平均排序 loss

> **2026-10-18 已確認的新流程（取代下文舊 calibration／輸出規格）：**
> 三個 seed 2027／2028／2029 各自切分並訓練 1 epoch；每 100 optimizer steps 與最後一步，凍結 BN／關閉 D4，完整推論 training／validation，先求整份資料各 rank 的平均 score 再算 ranking loss。
> 切分表、validation 題目與 train／valid 逐手索引統一放 `data/processed/strength_v1/seed{seed}/`，取消獨立 cv 目錄、calibration 子集和 training 校準題。
> 最後完整推論保存所有指定玩家落子 score（float32 npy，與 processed 索引對齊）；9 個切點用所有 training 逐手 score／標籤等權擬合，validation 不參與。
> 成功驗收後 run 只留 `model.pt`、`losses.npy`、`train_scores.npy`、`valid_scores.npy`；權重內嵌 seed、設定、雜湊及切點。中途保留最新續跑 checkpoint；最終成功才清理。評估／繪圖讀保存分數，validation 分 k 評分。
> 操作及當前限制以 [README.md](README.md#棋力-v1資料準備訓練與評估) 為準；下文保留原設計背景。未執行正式訓練或 GPU 驗收。


更新日期：2026-10-14。狀態：已確認訓練方案；模型、DataLoader、切分／索引、loss、DDP 訓練與校準評估程式已撰寫，完整訓練與單／多卡驗收仍未執行。已精簡 SGF／pass 支援與重複檢查，訓練資料仍保留 `board_cache.py` 的 `*_colors.npy`。清理期間的部分測試結果見 `RESULTS.md` A-011；其後依使用者要求停止測試，最終整套驗證尚未完成。命令與實際程式路徑見 [README.md](README.md#棋力-v1資料準備訓練與評估)。

本文件記錄本次討論確認的第一版棋力辨識方案。第一版的架構、資料切分、輸入與訓練目標以本文件為準；舊文件中的 Transformer、10-way softmax 與 player-disjoint CV 路線不作為這一版的實作規格。

## 1. 已確認的設定

| 項目 | 第一版設定 |
|---|---|
| 級別 | D、C、B、A、1D、2D、3D、4D、5D、6D，共 10 級 |
| 棋局切分 | 每級 100,000 盤，80,000 training、20,000 validation |
| 切分方式 | 每級按棋局隨機切分，固定 seed 並保存切分表 |
| 訓練樣本 | 只取該盤指定玩家落子後的盤面 |
| 基礎盤面 | `T×2×19×19`，T 為雙方總手數，兩層為黑／白棋子 |
| 磁碟快取 | 位元打包為正式格式：`uint8[T,2,46]`，每手 92 bytes；按 batch 解包 |
| 模型輸入 | 讀取樣本時才組合 `18×19×19` |
| 網路 | 256 channels、20 個 residual blocks、單一實數輸出 |
| Batch | 全域 16,384 個盤面，打亂所有 training 樣本索引 |
| Loss | 每個 batch 中各級的輸出先平均，再用 10 個平均分數計算排序 loss |
| 級別區間 | 模型固定後，只用 training data 的題目分數分布擬合 9 個有序切點；validation 使用固定切點評估 |
| 算力 | Nano4 H200；暫定單節點 8 卡 DDP，直接 forward／backward |

第一版不使用為本機 8GB 顯存設計的兩遍 forward 或小批次重算方案。

## 2. 棋局切分與樣本索引

1. 每級的 100,000 盤以固定 seed 打亂，前 80,000 盤為 training，其餘 20,000 盤為 validation。起始 seed 暫定為 2026。
2. 先切棋局，再產生盤面樣本，保證同一盤的不同手不會跨 training／validation。
3. 棋局唯一鍵使用 `(rank, game_id)`；模型只接收盤面，ID 只用於索引、切分與稽核。
4. 樣本索引記錄棋局、零起算 ply 與 rank_id。僅保留指定玩家的 ply；依訓練集黑先、黑白交替的已知條件，可用 `ply % 2 == player_color` 選取（黑=0、白=1）。
5. 每個 epoch 全域打亂樣本索引，再按 batch 分配給各卡；不搬動或打亂實體盤面快取，不重複解析 SGF。

| 集合 | 每級棋局 | 10 級棋局合計 | 以平均每盤 200 手估計的指定玩家盤面 |
|---|---:|---:|---:|
| Training | 80,000 | 800,000 | 每級約 800 萬，合計約 8,000 萬 |
| Validation | 20,000 | 200,000 | 每級約 200 萬，合計約 2,000 萬 |

「每級約 1,600 萬訓練盤面」是包含黑白雙方落子的估算；確認只取指定玩家後，樣本數約減半。正式數量以實際棋譜統計為準，各級盤面數不保證完全相等。

本版按盤切分，不額外強制玩家互斥。必須記錄 training／validation 的玩家重疊率；這份驗證不能直接代表未見玩家的泛化能力。既有 player-disjoint 切分保留供後續獨立實驗使用，不取代本版 80,000／20,000 盤的切分。

## 3. 兩層盤面與動態 18 層輸入

### 3.1 現有轉換介面

沿用 [sgf_planes.py](src/go_aicup/features/sgf_planes.py) 的 `sgf_to_planes()`／`moves_to_planes()`：

- 回傳連續的 `numpy.float32` 陣列，形狀為 `T×2×19×19`，值為 0／1。
- 每手完成落子與提子後，channel 0 表示黑棋、channel 1 表示白棋。
- 轉換函式本身不寫磁碟；快取與樣本索引由後續資料準備模組負責。
- 訓練集已確認黑先、黑白交替且 pass=0（`RESULTS.md` A-002）。V1 不再要求 pass 相容支援；當手顏色可由零起算 `ply % 2` 得到。
- 訓練資料仍依 `board_cache.py` 保存各分片的 `shard_XXXXX_colors.npy`（`int8[T]`，黑=0、白=1），維持現有快取格式。

基礎盤面必須保留雙方的每一手。只在樣本索引過濾指定玩家的手，不能先刪掉對手盤面，否則歷史會變成每隔一手。

### 3.2 組合規則

令 `P[t]` 為第 t 手落子後的黑／白兩層盤面。Dataset／collate 在訓練或驗證讀取樣本 t 時組合：

| 零起算 channel | 內容 |
|---|---|
| 0、1 | `P[t]` |
| 2、3 | `P[t-1]` |
| … | … |
| 14、15 | `P[t-7]` |
| 16 | 這一手是黑棋則全 1，否則全 0 |
| 17 | 這一手是白棋則全 1，否則全 0 |

歷史不足補零，禁止跨棋局讀取；最後兩層表示**這一手的顏色**。歷史按雙方每一手往前取。盤面保持固定黑／白編碼，不改成我方／對方。

### 3.3 記憶體與資料載入

- 全量資料以位元打包保存兩層基礎盤面，不預先展開全量 18 層 tensor；按 batch 解包後才組合歷史與顏色層。
- 同 dtype 下，兩層的元素數為 18 層的 1/9。例如 T=200 時，float32 約由 4.96 MiB 降至 0.55 MiB，不含索引與中間暫存。
- 正式快取使用 `uint8[T,2,46]`：每層按 row-major 將 361 個 0／1 值打包為 46 bytes，固定 `bitorder="little"`，最後 7 個 padding bits 為零。解包僅取前 361 bits，還原成 `T×2×19×19`，須與現有 float32 輸出逐元素一致。
- 快取按棋局保存 offset／length，以分片、memmap 讀取；分片大小與解包效能依 Nano4 短測調整。uint8 未打包盤面只作小型一致性／效能對照，不另存全量未打包副本。
- 磁碟仍保存全量盤面，但每手只占 92 bytes；以 2 億手估算 payload 約 18.4 GB，不含 move store、索引、metadata 與暫存。RAM 逐盤轉檔、逐 batch 讀取／解包，禁止累積全量盤面後再 stack／concatenate。
- 預取、worker 數、pinned memory 與非同步傳輸依 Nano4 實測設定；不把全部資料轉成常駐的浮點 tensor。
- Training 暫定使用 D4 增強，同一樣本的 8 個歷史盤面套用同一變換。Validation 固定、不增強。

## 4. 模型架構

| 階段 | 操作 | 輸出 |
|---|---|---|
| 輸入 | 動態組合 18 層 | `B×18×19×19` |
| Stem | Conv3×3(18→256, padding=1)、BN、ReLU | `B×256×19×19` |
| 主幹 | 20 個 residual blocks | `B×256×19×19` |
| 壓縮 | Conv1×1(256→1)、BN、ReLU | `B×1×19×19` |
| 展平 | Flatten | `B×361` |
| 隱藏層 | Linear(361→256)、ReLU | `B×256` |
| 分數 | Linear(256→1) | `B×1` |

Residual block 為 `Conv3×3 → BN → ReLU → Conv3×3 → BN → 加回輸入`；按原圖，相加後不額外加 ReLU。所有 3×3 卷積 stride=1、padding=1。接 BN 的卷積不使用 bias，Linear 使用 bias。

網路約 2,370 萬個參數；末端不加 sigmoid／softmax。分數越高表示棋力越強。8 卡版本以 SyncBatchNorm 同步 BN 統計。

## 5. Batch 內級別平均與排序 loss

全域 batch size 固定為 16,384，指所有 GPU 合計的盤面數。8 卡時每卡為 2,048；不是每卡各 16,384。

對 rank r 的樣本計算：

\[
\mu_r=\frac{1}{N_r}\sum_{i:y_i=r} f_\theta(x_i)
\]

將 10 個平均依真實棋力由強至弱排列為 `mu_1=mu_6D, ..., mu_10=mu_D`，使用：

\[
L=-\frac{1}{9}\sum_{i=1}^{9}\left[\mu_i-\operatorname{logsumexp}(\mu_i,\ldots,\mu_{10})\right]
\]

- 10 為級別數，9 為 loss 項數。使用 logsumexp 計算，各項先等權。
- 每級以自己的實際樣本數平均，不再按 batch 中的級別數量加權。
- 全域隨機打亂不保證各級數量一致；級別均衡時每級平均約 1,638 個盤面。計算前檢查 10 級均有樣本，缺級時明確回報，不除以零或靜默省略。
- 每個完整全域 batch 一次 forward／backward 與 optimizer update，不改成逐盤面排序 loss，也不改成題目排序訓練。
- 每 epoch 遍歷該 epoch 的 training 樣本排列。尾批保留，以實際樣本數平均；多卡需要補齊時使用有效樣本 mask，不把補齊樣本計入 loss。若尾批缺級，需明確記錄並停止該步更新。
- 以約 8,000 萬 training 盤面估算，每 epoch 約 4,883 steps；正式 steps 依實際樣本數計算。

此目標約束各級平均分數的順序，不保證每個盤面都能正確分類。各級平均即使排序正確，個別分數仍可能大量重疊，因此驗證須同時看分布與題目級評分。

## 6. Nano4 多卡訓練

目標為 Nano4 H200 單節點，暫定 8 GPU、DDP／NCCL／torchrun，環境使用現有 Conda `go`。1／2 卡模式用於正確性短測；全域 batch 的定義保持不變。實際 GPU 配置與專案絕對路徑待 Slurm 作業確認。

每一步：全域打亂索引 → 各卡取不同樣本 → 動態組合 18 層 → BF16 mixed precision forward → 跨卡各級輸出 sum／count → 10 個全域平均 → 排序 loss → backward → optimizer update。

跨卡 sum 必須使用梯度正確的聚合方式，並驗證其與 DDP 梯度平均的尺度一致。不能先在各卡算局部級別平均 loss，再平均 loss，因為這不等於全域級別平均的訓練目標。Counts、級別 sum／mean 與 loss 使用 float32 累積／計算。

| 項目 | 起始設定；尚未實測調參 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | 前 5% optimizer steps warmup，之後 cosine decay |
| 梯度裁剪 | 最大 norm=1 |
| 訓練長度 | 暫定最多 10 個完整 epochs，先測一個 epoch 再決定續跑 |
| Checkpoint | 每 epoch、固定時間間隔，以及 validation 最佳；保存 last |

Checkpoint 保存模型、optimizer、scheduler、AMP 狀態（若使用）、epoch／batch 位置、資料排列與各 process 隨機狀態。續跑恢復相同切分與抽樣順序。記錄 seed、git commit、config、環境、GPU 數、吞吐量、GPU 小時與實際樣本數。

完整訓練時間與效能增益目前未知。先做短測估算 epoch 時間；不以本機顯存設定或先前的 2 GPU 小時試跑上限限制 Nano4 正式訓練。

## 7. Validation 與題目級辨識

Validation 固定樣本順序，使用 eval 模式、凍結 BN 統計，不更新權重。報告：

1. 固定 validation batches 的排序 loss；跨 batch 統計時記錄樣本數與尾批處理。
2. 整份 validation 各級平均、排序、標準差／分位數、分數分布與相鄰級重疊程度。整份資料的級別平均 loss 與 batch loss 分開列出，不混為同一數字。
3. 指定玩家每手分數先平均成每盤分數，再將同一玩家、同色的 k 盤等權平均成題目分數；不讓長局獲得較高的題內權重。
4. k=5／10／15／20 分別回報官方分數、Accuracy、MAE 與混淆矩陣；總分取四個 k 桶等權平均。

將標量分數轉成 10 級時，模型訓練完成後固定權重、切換 eval 模式並凍結 BN，只用 training 棋局重新推論各級分數分布，不使用隨機 D4 增強。Training 指定玩家每手分數先平均成盤，再從同一玩家、同色的 training 棋局合成 k=5／10／15／20 題目，依題目分數擬合一組 9 個有序切點。切點按四個 k 桶等權的官方分數選擇，與對應 checkpoint 一起保存；validation 不參與切點擬合，使用固定切點評估。

令切點為 `t1 <= ... <= t9`，由低至高將分數對應到 D、C、B、A、1D、2D、3D、4D、5D、6D；D 為 `(-∞, t1)`，6D 為 `[t9, +∞)`，中間各級使用左閉右開區間。相同分數使用固定邊界規則，不依真實標籤分配。

預期各級平均滿足 `mu_6D > mu_5D > ... > mu_D`，但 loss 不保證完整分布不重疊，因此最佳切點不能只用相鄰平均的中點決定。須同時保存各級手／盤／題分布與切點擬合分數；training 上的最佳區間仍可能有樣本內偏差，辨識能力由固定切點後的 validation 四桶分數判定。本版按盤切分的玩家重疊率仍需記錄，不新增校準／評估玩家互斥要求。這一項屬模型訓練後的解碼，不改變 batch 級別平均 loss。

## 8. 實作順序與驗收

| 階段 | 工作與通過條件 |
|---|---|
| 資料準備 | 各級 80,000／20,000 盤、棋局零重疊；位元打包／解包逐元素一致、padding bits 為零；保存玩家重疊率與實際盤面數 |
| 動態輸入 | 對照現有兩層輸出；驗證 8 個歷史時刻、補零、提子、顏色層與棋局邊界 |
| 模型／loss | 驗證 shape、級別平均、強弱方向、有限數值；所有分數相等時 loss 約 log(10!)/9=1.678 |
| 多卡正確性 | 小型相同 batch 的單卡／多卡 loss 與梯度對照，含各卡級別數量不一致、尾批 mask 與 SyncBN |
| 小樣本學習 | 關閉增強、固定小資料；確認 loss 能下降、各級平均形成正確順序 |
| Nano4 短測 | 測快取讀取、18 層組合、forward／backward、通訊與 checkpoint 續跑，估算完整 epoch 時間 |
| 正式實驗 | 先跑一個完整 epoch，保存權重與 validation 分布，再決定後續 epochs |
| 題目級評估 | 僅 training data 擬合切點，固定後評估 validation 四個 k 桶；不能只以平均排序成功判定棋力辨識成功 |

## 9. 檔案與輸出位置

既有 [sgf_planes.py](src/go_aicup/features/sgf_planes.py) 提供兩層盤面；[test_sgf_planes.py](tests/test_sgf_planes.py) 驗證此介面。以下程式已於 2026-10-12 撰寫，但未執行；新測試檔列為後續驗收工作，不代表已有測試結果：

| 路徑 | 用途 |
|---|---|
| `src/go_aicup/data/board_cache.py` | 兩層盤面位元打包／解包、分片快取、offset／length |
| `src/go_aicup/data/cv.py` | 新增本版按級別 80,000／20,000 盤切分，保留既有切分功能 |
| `src/go_aicup/models/strength_resnet.py` | 20-block／256-channel ResNet |
| `src/go_aicup/tasks/strength/dataset.py` | 指定玩家樣本索引、全域 shuffle、動態 18 層與 D4 |
| `src/go_aicup/tasks/strength/losses.py` | 全域級別平均與排序 loss |
| `src/go_aicup/tasks/strength/train_rank_v1.py` | Nano4 DDP 訓練、validation、checkpoint／續跑 |
| `src/go_aicup/tasks/strength/evaluate_rank_v1.py` | 手／盤／題聚合與有序切點校準 |
| `configs/strength/rank_cnn_v1.json` | 可重現設定 |
| `tests/test_strength_v1.py` | 動態輸入、loss 與分散式正確性 |

衍生資料預計放在 `data/processed/cv/`、`data/processed/boards/` 與 `data/processed/strength_v1/`。每次 run 的 config、log、權重、切點、環境及 validation 預測／分數放在 `outputs/rank_cnn_v1_<run-id>/`。

本機專案根目錄為 `D:\college\go_aicup_2026`；Nano4 的絕對路徑尚未指定，所有資料／輸出路徑走設定。文件更新不啟動轉檔或訓練，不產生權重與提交檔。
