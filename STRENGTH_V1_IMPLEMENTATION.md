# 棋力 V1：模型、訓練與 SGF 平行轉檔實作設計

日期：2026-10-11。依據：[STRENGTH_V1_PLAN.md](STRENGTH_V1_PLAN.md)。本文件是待實作的工程設計，未建立模型／訓練入口，未啟動全量轉檔或 GPU 訓練。

沿用 V1 已確認設定：各級 80,000／20,000 盤、固定黑／白編碼、指定玩家落子後取樣、8 個雙方歷史盤面、20-block／256-channel ResNet、全域 16,384 盤面及級別平均排序 loss。舊文件的玩家互斥訓練切分、我方／對方正規化與 softmax 不替換本版設定。

## 1. 建議的實作順序

```text
官方 CSV
  → 既有 preprocess：唯一 SGF parser → moves／colors／metadata
  → board_cache：跨棋局平行重播 → 兩層盤面分片
  → cv：按級別固定棋局切分 → 指定玩家 sample index
  → Dataset：全域 sample shuffle → 動態 18 層 → D4
  → ResNet：每盤面一個標量
  → 全域各級 sum／count → 排序 loss → DDP backward
  → validation：手 → 盤 → 同玩家同色 k 盤 → 有序切點 → 官方分數
```

先完成「一盤從原 SGF 到模型輸入」及「相同 batch 的單卡／多卡梯度一致」，再擴大轉檔與訓練。資料準備用 CPU 多程序；模型訓練用單節點 GPU DDP。兩者不要共用同一套 worker 或在每個訓練 process 各做一次全量轉檔。

假設是級別平均排序可學到具有棋力訊號的盤面標量，且手／盤／題平均可降低局面雜訊。對 k=5／10／15／20 的效能增益均為**未知**，不由 loss 下降推定。最終 gate 是獨立校準後的四桶官方分數及分布；0.4631 是官方門檻，不是這份按盤驗證必然可轉移的 LB 預測。既有 0.4117 線性探針採用不同切分，不能直接當受控增益比較。

## 2. 現有程式的接點與限制

| 現有檔案 | 可以沿用 | 本版要新增或注意 |
|---|---|---|
| `data/sgf.py` | 唯一 parser、實際 move colors、pass 編碼 | 不另寫 regex parser |
| `data/preprocess.py` | `ProcessPoolExecutor` 按批 parse、moves／colors npy、metadata parquet | 現在逐 rank 轉換，再合併；不是跨棋局盤面快取 |
| `data/store.py` | `GameStore.game(i)` 的 moves／colors memmap slice | 本版不用 `perspective()`，保留黑／白 |
| `features/sgf_planes.py` | Numba 重播、提子、pass、實際顏色、落子後盤面 | 目前只回傳一盤 float32 陣列，不寫快取 |
| `data/cv.py` | metadata、同玩家同色合成題、保存切分表的模式 | 新增具名 V1 按盤切分；保留舊 player-disjoint API |
| `metrics.py` | 四個 k 桶的官方 score、acc、MAE | 本版總分另明確取四桶等權平均；目前 `all` 是所有題直接平均，僅桶數相等時等價 |

目前 `data/processed/` 沒有 move store；原始資料實際在 `data/raw/training/`，但 `preprocess.py` 的 `TRAIN_DIR` 指向 `data/training/`。正式轉檔前需把來源／目的路徑接入 config 或 CLI，不能直接假設現有預設命令會找到 CSV，也不需要搬動原始資料。

既有 parser 偏容錯；board cache 啟動前要檢查 metadata 的 `n_errors`、19 路／無 setup 等資料稽核條件。`moves_to_planes()` 只看 moves／colors，無法自行發現 SGF 的 AB／AW 等原始設定。異常棋局需帶唯一鍵明確報錯，不能靜默略過而讓 80,000／20,000 盤數失真。

## 3. 快取格式：邏輯兩層，磁碟優先比較 uint8 與位元打包

float32 留在現有公開轉換介面；快取層處理壓縮，讀取後仍得到相同的 0／1 盤面。以全量 **2 億個雙方 ply** 為估算基準，以下使用十進位 GB，不含 metadata、索引、檔案 header 與暫存：

| 格式 | 每手 bytes | 全量估算 | 特性 |
|---|---:|---:|---|
| `float32[T,2,19,19]` | 2,888 | 577.6 GB | 介面直接，但磁碟與讀取代價高 |
| `uint8[T,2,19,19]` | 722 | 144.4 GB | 最容易驗證的磁碟參考版 |
| `uint8[T,2,46]` 位元打包 | 92 | 18.4 GB | 每層 361 bits，分別補至 46 bytes |

建議先支援 `uint8`／`packed` 兩種快取 codec，以同一批資料在 Nano4 做冷讀與熱讀短測後選正式格式。`packed` 是優先候選：每手小於 uint8 的 1/7，且整份快取較有機會留在節點的 OS page cache。配額、可用 RAM 和共享儲存吞吐量仍未知，不能保證常駐。

打包規格必須固定：按 `[channel,row,column]` 的 row-major 順序，兩層各自 `packbits(..., axis=-1, bitorder="little")`，每層最後 7 個 padding bits 為零；解包只取前 361 bits。manifest 記錄 shape、dtype、codec、bitorder、版本與來源 fingerprint。

### 本次只讀小型檢查

使用既有 parser／`moves_to_planes()`，每級前 100 盤，共 1,000 盤。這是連續前綴抽樣，**不是全量隨機代表樣本**：

| 項目 | 本機量測 |
|---|---:|
| 雙方總手數 | 222,231 |
| 指定玩家樣本數 | 111,355 |
| parse 時間 | 0.463 s |
| 第一次 replay（含 JIT 或 cache 初始化） | 2.653 s |
| 暖機後 replay＋打包＋逐元素還原驗證 | 0.776 s |
| 打包 payload | 20,445,252 bytes，恰為 92 bytes／手 |
| 還原與 float32 原輸出不一致 | 0 |

這只證明小樣本 codec 等價，未測磁碟寫入、多程序、Nano4 或訓練吞吐。不能把這個秒數線性乘成全量正式預算。

## 4. 如何平行把 SGF 轉成訓練資料

### 4.1 第一階段：SGF → move store

沿用 `preprocess.py` 的多程序 parse。CSV 必須由 CSV reader 正確處理引號，不按任意 byte offset 切行。現在 10 個 rank 依序執行，各 rank 內再使用 process pool；這已經可以平行化 SGF 解析。

第一版不必同時把「跨 rank」和「rank 內」都開滿 worker，否則形成 10×W 程序。若調整成全域工作佇列，則按 rank 的 CSV chunks 供給一個 process pool，任務帶來源行號，合併時保持固定排序與 `(rank, game_id)` 對應。SGF strings 可在 parse 階段按批傳送；解析後的大型盤面不可再由 process queue 傳回主程序。

現有單 rank 一次讀入 100,000 盤、`parse_many()` 集中收集結果的方式可先保留，但不能照搬到全量 float32 盤面：盤面結果遠大於 move tokens。

### 4.2 第二階段：move store → board shards

一盤的 t+1 依賴 t，所以**盤內循序重播、盤與盤平行**。不要把同一盤的手分給不同 process，也不要為每個訓練樣本重播棋譜。

| 項目 | 建議起點 |
|---|---|
| 任務單位 | 2,048–4,096 盤，保持棋局完整，不跨 rank 邊界 |
| worker 輸入 | move-store 路徑、game range、shard ID、codec、輸出目錄 |
| worker 讀取 | 自行唯讀 memmap moves／colors，讀取所屬棋局 |
| worker 計算 | 每盤 `moves_to_planes()` → 檢查 0／1 → uint8 或 packbits |
| worker 寫入 | 自己唯一的 shard 暫存檔；不與其他 worker 共寫 |
| worker 回傳 | shard ID、盤數、手數、bytes、耗時及檢查結果等小型摘要 |
| 主程序 | 排程、顯示進度、收集狀態、驗證 manifest；不收大型 tensor |

以每盤 200 手估算，4,096 盤的 packed shard 約 75.4 MB；全量約 250 shards（按 rank 邊界切，尾 shard 較小）。實際按總手數與檔案大小監控，不能固定假設每盤一樣長。

worker 根據該 shard 的 lengths 預先計算 local offsets／總 ply 數，用 `open_memmap` 配置 `.tmp.npy`，逐盤寫入 slice。一盤 T=200 的 float32 暫存約 0.55 MiB，因此不必先累積整個 shard 的 float32。重播後即可釋放該盤陣列。這樣峰值 RAM 不隨全量棋局數上升。

### 4.3 CPU 配額、初始化與續跑

- 單一 process pool，worker 數依 Slurm 實際可用 CPU／affinity，而非只看 `os.cpu_count()`；比較 W=4／8／16／32，沒有配額就不啟用該組。
- 每個 worker 的 Numba／BLAS／OpenMP thread 設為 1，避免程序數乘上執行緒數；重播函式本身是循序 JIT。
- worker initializer 先跑一個極小棋局暖機；共用 Numba cache 的初始化／寫入競爭需驗證，JIT 開銷與穩態吞吐分開記。
- Windows 用可 pickle 的頂層函式及 `__main__` guard；worker 在自身 process 開啟 memmap，不 pickle 開啟的 store 或大型陣列。
- 限制 pending tasks（例如 2×workers），避免提交整份 SGF／metadata 或把 RAM 用在佇列。
- 完成順序可不同，shard ID／game range 必須固定；manifest 最終按 game range 排序。
- shard 完整 flush、關閉 handle、shape／bytes／hash 檢查後，在同目錄原子 rename。最後才寫完成標記。續跑要核對來源 fingerprint、converter／codec 版本；不能只因 `.npy` 存在就跳過。
- worker 失敗只重做未完成 shard；主程序禁止把缺 shard 的快取發布為可訓練資料。錯誤包含 rank、game_id、source_row 與 ply。

在 Slurm 計算節點執行轉檔，CPU 數遵循實際 allocation；是否允許單獨 CPU job 目前未知，不能假設 H200 佇列提供 CPU-only 作業。轉檔不需要 torchrun，也不需要 GPU 計算。

### 4.4 分片與索引布局

```text
data/processed/boards/strength_v1_<cache-id>/
  manifest.json
  games.parquet                  # game_index → shard_id, local_offset, length
  shard_00000.npy                 # [total_ply,2,46] packed 或 [total_ply,2,19,19]
  shard_00000.done.json
  ...
data/processed/cv/
  strength_v1_games.parquet       # game_index, rank, game_id, split
  strength_v1_calibration_players.parquet
  strength_v1_evaluation_players.parquet
data/processed/strength_v1/<dataset-id>/
  train_game.npy                  # uint32[N_train_samples]
  train_ply.npy                   # uint16[N_train_samples]
  valid_game.npy
  valid_ply.npy
  manifest.json
```

兩層快取保存**所有雙方 ply**；train／valid sample index 才選 `colors[ply] == player_color`。pass 納入，不按奇偶猜顏色。rank／player_color 從 game metadata 查，無須在每個 sample 重複保存。game_index 是 store row 對應，不能把不唯一的 game_id 直接當 array index。

`game:uint32 + ply:uint16` 以兩個陣列保存共 6 bytes／樣本；約 8,000 萬 training 樣本是 480 MB。載入時把 batch 的索引轉成框架需要的整數型別。寫入前檢查 game／ply 上限；未來來源超過 dtype 上限須報錯或升版。

快取不依 train／valid 另複製；按盤切分先於樣本索引生成。manifest 綁定 cache ID、split hash 與 sample counts，避免續跑或評估接到不同資料版本。

## 5. Dataset 與讀取效能

### 5.1 動態組合歷史

給定 `(game_index, ply=t)`，讀 `P[t], P[t-1], ..., P[t-7]`。負數時刻直接補零；不得讀到同 shard 的上一盤。channel 16／17 使用 move store 的當手實際顏色。所有歷史使用同一個 D4 transform，顏色層不變。

packed 版一次讀取 `[8,2,46]` payload，解包得到 `[8,2,19,19]`，再 reshape 為 16 個棋子 channels，補兩層顏色成 `[18,19,19]`。CPU 優先保持 uint8，整批 pin memory、非同步 H2D，再在 GPU 轉成模型輸入 dtype。BF16 autocast 的輸出分數另轉 FP32 做 sum／loss。

首版先在 CPU vectorized 解包；若 profiling 證明解包限制速度，再比較搬 packed bytes 到 GPU 後解包。GPU 解包與自訂 kernel 尚未實作，也不預設一定更快。

### 5.2 全域 shuffle 與 batch I/O

每 epoch 建立唯一 `uint32[N_train_samples]` 隨機排列，seed 與 epoch 確定後落盤並保存 hash。8,000 萬項 payload 約 320 MB；避免 Python 的 8,000 萬項 list。各卡讀同一排列的不同區段，loader workers 不各自重新 shuffle 全部資料。

第 s 個完整 step 取排列 `[s×16384:(s+1)×16384]`，再分為 8 個互斥的 2,048 sample slices。這個 sampler 負責明確的全域 batch 邊界，不能以「各卡任意拿下一個 batch」推定全域樣本一定吻合。

Dataset 使用批量取樣介面，將當前 local batch 的樣本暫按 shard／offset 排序、成批 gather／去重歷史讀取，再恢復原 sample 順序。只改這一批的讀取次序；不改全域 shuffle 或 batch 成員。避免 2,048 次 Python 函式與大量 92-byte 零碎隨機 I/O。

每卡先測 `num_workers=2`、`prefetch_factor=2`，再測 4／8 workers；8 卡的總 loader workers 是 16／32／64，須與 Slurm CPU 配額一起看。使用 lazy memmap／有限 handle cache，防止每個 worker 開啟全部 shard 耗盡 fd。不要在 Dataset 保存全量 SGF strings 或 Python 物件列表。[PyTorch DataLoader 官方說明](https://docs.pytorch.org/docs/2.6/data.html)支持多程序載入、批量 Dataset 取樣與 pinned memory；這些起始參數的最終吞吐仍需本專案量測。

使用 `persistent_workers` 時，D4 由固定的 `(seed, epoch, sample_id)` 決定；epoch 隨 index 任務傳入。不要只修改主程序的 `dataset.epoch`，期待持續存活的 worker 副本自動同步。此方式也使 prefetch／worker 排程改變時，續跑仍可取到相同增強。

以 8,000 萬樣本估算，8 個歷史的 packed 邏輯讀量約 58.9 GB／epoch，uint8 約 462.1 GB，float32 約 1.85 TB；實際磁碟流量受 page 大小、cache miss、歷史重用及 prefetch 影響，可能與這些數字差很多。memmap 不等於資料已在 RAM；快取 cold／warm 兩種狀態都要測。

## 6. 模型實作

自寫 `StrengthResNet`，不直接套 torchvision ResNet：本版 residual 相加後沒有 ReLU，head 也使用展平與 MLP。

```text
Conv3x3(18,256,bias=False) → BN → ReLU
20 × [Conv3x3 → BN → ReLU → Conv3x3 → BN → add]
Conv1x1(256,1,bias=False) → BN → ReLU
Flatten(361) → Linear(361,256) → ReLU → Linear(256,1)
```

按以上結構、BN 有 affine、Linear 有 bias 計算，共 **23,748,611 個可訓練參數**，不含 BN running buffers。forward 返回 `[B]` 或 `[B,1]` 的實數，loss 層明確統一 shape。

先建模、轉 SyncBatchNorm、移至 local CUDA device，再包 DDP。正式起點是 BF16 autocast、AdamW 1e-3／weight decay 1e-4、5% steps warmup＋cosine、gradient norm clip=1；BN eps／momentum、初始化、BN／bias 是否 decay 均寫入 config。首版可全部參數採用同一 weight decay；日後排除 BN／bias 若作實驗，則單獨記錄，避免隱含預設改變。

顯存主要是 activation，不是權重。每卡 B=2,048 時，一個 BF16 `[B,256,19,19]` tensor 為 **0.353 GiB**；41 個同尺寸 tensor 就約 14.5 GiB，而 backward 還有其他 saved tensors、BN 暫存、gradients、cuDNN workspace 等，這不是峰值估計。H200 141GB 是否能以這組 kernel／layout 直接訓練，仍須實測。

先以本版完整網路、local batch 128／512／1,024／2,048 做短測，再測 8 卡全域 16,384。1／2 卡梯度對照用小型固定全域 batch（例如 80／160），只驗證數學正確性；不能要求單卡一定吃下正式全域 16,384。正式 V1 的 global batch 定義不變。

不預設兩遍 forward、梯度累積、activation checkpoint 重算或縮小正式 global batch。如果 2,048／卡實測 OOM，記錄峰值與 workspace，該配置判未通過；需修訂方案後才換訓練語義。`channels_last` 可作效能比較，`torch.compile` 留到 eager 正確後，編譯時間與穩態時間分開報。

## 7. 全域級別平均 loss：最重要的多卡正確性

各卡 d 先取得 FP32 的 `S[d,r] = sum(score_i)`、`C[d,r] = count(valid_i)`，全域為：

\[
S_r=\sum_d S_{d,r},\quad C_r=\sum_d C_{d,r},\quad \mu_r=S_r/C_r.
\]

`rank_id` 儲存弱→強 0..9；loss 先反轉為強→弱，再對每個位置的 suffix 做 logsumexp。可用反向 `logcumsumexp` 得到所有 suffix，最後一項恆為零，只平均前 9 項。

推薦基準路徑：sum 使用 `torch.distributed.nn.functional.all_reduce(..., SUM)` 的可微聚合；count 使用不需要梯度的 `dist.all_reduce(..., SUM)`。每張卡都對同一組全域 means 算相同 loss，再 backward。

**梯度尺度推導：**可微 SUM 的 backward 將 W 張卡對同一 loss 的梯度相加，產生 W 倍 local 貢獻；DDP 隨後平均 W 張卡的參數梯度，得到單卡全域 batch 的梯度。因此此路徑不再手動乘／除 world_size。這是依 [PyTorch 2.6 all-reduce 實作](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/distributed/nn/functional.py)及 [DDP 梯度平均說明](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/parallel/distributed.py)推導，必須由本專案梯度對照驗收。

不要直接對帶梯度 sum 使用原地普通 `dist.all_reduce`，也不要先在每卡算局部 means／排序 loss 再平均。若改採 detach＋本地梯度替身，需另推導 W 倍補償，不能沿用上面的尺度規則。

全部卡先 collective 檢查 counts／有限值，再作一致的更新或停止決定。一張卡自行 skip，其他卡進 backward，會造成 collective 卡住。缺級按原計畫明確記錄、不更新該步；tail 的 scheduler／optimizer step 也不前進。

### 必須通過的數值驗收

| 案例 | 通過條件 |
|---|---|
| 所有級別同分 | loss = `log(10!)/9 = 1.6782680637` |
| 強級分數增加 | loss 方向符合強→弱設定 |
| 全部分數加同一常數 | loss 不變 |
| 各卡級別數量不同，部分卡缺級但全域不缺 | 全域 loss／梯度與單卡參考相同 |
| FP32 無 BN 的極小模型 | 初始建議 atol 1e-6、rtol 1e-5；記最大絕對／相對誤差 |
| 完整模型＋SyncBN | 比較 outputs、loss、參數 grads、BN running stats 及一步更新；另訂合理浮點容差 |
| BF16 | 與 FP32 參考比較誤差及有限值，不能要求逐 bit 相同 |

排序 loss 對整體平移不敏感；末端 Linear bias 對這個 loss 的資料梯度應接近零。切點必須綁定 checkpoint，不能把另一 epoch 的切點直接搬過來。級別差距拉大會讓排序 loss 接近零，但個體分布仍可能重疊，所以要記標準差、分位數及四桶題目分數。

### 尾批與 SyncBatchNorm

原計畫要求保留尾批，補齊時 mask 排除補齊項。mask 只排除 loss 的 sum／count，**普通 SyncBatchNorm 仍把補齊盤面當有效輸入**，會影響有效樣本輸出與 running stats。

建議尾批按實際 N_tail 分配不等長 local batch（差最多 1），forward 前移除 placeholder，只讓真實盤面進網路；loss 的 count／sum 只含有效樣本。PyTorch 2.6 SyncBN 會按各卡實際元素 count 聚合統計，參見[官方 SyncBN 原始碼](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/_functions.py)。這個設計需要實測含不等長輸入的整條 DDP 路徑。

若某卡零樣本，所有卡仍須參與 collective；空 batch 的 Conv／BN／Flatten／Linear 及梯度連接需要額外驗收，Flatten 不能用含 `-1` 的空 tensor reshape。若全域尾批缺級則按計畫停止該步更新。不要用 duplicated inputs 加 loss mask 宣稱與未補齊的 SyncBN 完全等價，也不要直接用預設 DistributedSampler 的重複尾樣本而不記錄。

## 8. 訓練、估算與續跑

每 step：讀取預排全域 batch → local 資料載入／18 層／D4 → 單次 BF16 forward → FP32 全域級別 sum／count → loss → backward → clip → AdamW → scheduler。正式起點不混入題目級 loss。

約 8,000 萬 training samples 是 **4,883 steps／epoch**。僅 40 個主幹 3×3 卷積的 forward 約 0.279 PFLOP／全域 batch（乘加算 2 FLOPs）；backward 與其他操作另計。此算式是運算量，不是 H200 實测时间。

先在 Nano4 跑暖機後 100–200 steps，分別量 data wait、H2D／組合、forward、loss collective、backward、optimizer、checkpoint、GPU peak memory，記錄 p50／p95／吞吐。CUDA 非同步操作需用 events／同步或 profiler 測量，不能只用未同步的 Python 計時。再依實測 step 秒數估訓練時間：

| 假設實測 step 時間（非預測） | 單 epoch 純訓練 | 8 卡 GPU 小時 |
|---:|---:|---:|
| 2 s | 2.71 h | 21.70 |
| 10 s | 13.56 h | 108.51 |
| 30 s | 40.69 h | 325.53 |

上表不含全量轉檔、shuffle、validation、寫 checkpoint 與排隊。Validation 約 2,000 萬 samples，需另外量 eval throughput。10 epochs 只是上限，不在完成短測前預支全量預算；正式先跑一 epoch。

checkpoint 保存 model／BN buffers、optimizer、scheduler、global optimizer step、epoch、**最後完成 optimizer update 的 batch cursor**、per-rank RNG、permutation hash／path、split／cache hash、config、git／環境資訊。BF16 沒使用 GradScaler 時記錄 AMP 模式，不虛構 scaler state。

DataLoader 已預取的 index 不算已完成訓練；續跑從 update cursor 重建 loader，利用 stateless D4 重現增強。只存 seed 不夠：保留 permutation 或固定 RNG 演算法／版本並校驗 hash。精確續跑起點要求相同 world_size；換卡數視為另一 run／另驗 BN 與數值差異。

各卡保存自身 RNG，rank 0 保存主 checkpoint；都成功後才發布完成標記。定時寫 `last`（起始每 15 分鐘）、每 epoch／最佳 checkpoint，接近 Slurm walltime 時在完整 step 邊界停止，不等被殺才嘗試保存。

## 9. Validation 與有序切點

每盤指定玩家 scores 等權平均成 game score，再將同玩家同色的 k 盤 game scores 等權平均成 question score。不能直接把題內所有手平均，否則長局佔比較高。

20% validation 的玩家先按級別固定分成 calibration／evaluation 兩部分，再僅從對應部分的 **validation 棋局**合成題。不能把同一玩家的 training 棋局誤納入驗證題。兩部分玩家互斥；各自與 training 的玩家重疊率另外記錄。k=20 同色 validation 盤不足的玩家不以重複盤補足，列 eligibility／覆蓋率；某級某桶無合格玩家則明確報錯。

起始建議 calibration／evaluation 玩家各半、各合成 10,000 題，四個 k 桶等量，題內盤不重複。不同題可共用玩家／盤，必須按玩家做 bootstrap 或重抽比較；不能把 10,000 題宣稱為 10,000 個獨立玩家。

9 個切點可用動態規劃最佳化：對 calibration question scores 排序，相同分數不可拆開；每個分數區段配一個非遞減的 rank_id。reward 為 exact=1、差一級=1/e、其餘=0，四桶等權。利用 prefix reward 做最佳分段，固定平手規則，保存 boundaries、score 比較方向、模型 hash。空級／相同切點情況要定義合法的非遞減解與診斷，不能任意挪動平手樣本到不同級。

evaluation 題不參與切點搜尋。若反覆用這批 evaluation scores 選超參／epoch，這批數字便是模型選擇驗證，不再是完全獨立最終評估；需再留固定測試題／保留既有 player-disjoint holdout 作第二意見。

報告同時列固定 validation batch loss、整體級別平均 loss、各級分布、四桶 score／acc／MAE／混淆矩陣，以及玩家重疊／樣本 eligibility。不因全量 means 排序成功就判定模型成功。

## 10. 檔案分工與交付 gate

| 待實作檔案 | 責任 | 首次驗收 |
|---|---|---|
| `data/board_cache.py` | codec、shard builder、manifest、續跑 | 隨機盤／提子／pass 與現有介面逐元素一致；壞 shard 不可讀 |
| `data/cv.py` | 新增 V1 棋局切分、校準／評估玩家切分 | 每級 80,000／20,000、棋局零交集，保留舊 API |
| `tasks/strength/dataset.py` | sample index、global batch sampler、歷史、D4 | 不跨局、不看未來、只取指定玩家、全 epoch 不漏不重 |
| `models/strength_resnet.py` | 完整 ResNet 與 scalar head | shape、23,748,611 參數、block 相加後無 ReLU |
| `tasks/strength/losses.py` | 全域 means、排序 loss | 1.678268 基準、單／多卡梯度對照 |
| `tasks/strength/train_rank_v1.py` | DDP、短測、訓練／validation、續跑 | 尾批／SyncBN、一完整 step 的續跑對照 |
| `tasks/strength/evaluate_rank_v1.py` | 手→盤→題、切點、分桶評分 | 校準／評估玩家互斥，題內盤等權 |
| `configs/strength/rank_cnn_v1.json` | paths、codec、sampler、模型、optimizer、precision | 所有隱含預設明確化，保存 resolved config |

後續落地建議拆成四次可審查交付：① 快取與切分／索引；② 動態輸入＋模型／loss；③ DDP＋短測／續跑；④ 校準評估＋一 epoch 實驗。這是程式模組分工，不需要不同 agent 同時修改共用檔案。

本次實際新增輸出只有 **`D:\college\go_aicup_2026\STRENGTH_V1_IMPLEMENTATION.md`**，外部文件引用另登記於 `EXTERNAL_RESOURCES.md`。上面的 cache／indices／checkpoint 路徑是未來產物規劃，尚未生成。正式訓練 run 仍集中在 `outputs/rank_cnn_v1_<run-id>/`，不產生提交或代替使用者上傳。
