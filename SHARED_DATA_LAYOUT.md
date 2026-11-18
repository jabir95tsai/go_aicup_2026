# 共用資料位置與搬移紀錄

2026-11-16 依使用者要求盤點本機 `outputs/`，將可供多個實驗重用的資料移到 `data/processed/`。本文件說明儲存位置；競賽與資料使用規則仍以 [CLAUDE.md](CLAUDE.md) 為準。

## 現行位置

| 資料 | 正式位置 | 本次搬移檔案數 | 本次容量（GiB） |
|---|---|---:|---:|
| HumanSL likelihood、C／逐手特徵、引擎分析、盤面 pack、固定題組與切分表 | `data/processed/strength_features/humansl/` | 2,303 | 27.716606 |
| 玩家辨識的官方 store、盤面快取、各 encoder embedding、共用 draw／component 快取 | `data/processed/identity/artifacts/` | 529 | 4.048914 |
| 對手比對所需盤面、向量與特徵 | `data/processed/strength_features/opp_link/` | 37 | 0.524618 |
| 玩家合併與對手檢查的參考資料 | `data/processed/strength_features/{identity_merge,opponent_check}/` | 6 | 0.004250 |
| 11/06 release 的 store、盤面快取、HumanSL 特徵與推論索引 | `data/processed/release/rank_release_20261106/` | 48 | 0.237586 |
| 713770 測試流程的 store、盤面、HumanSL／SE3 特徵與推論索引 | `data/processed/release/rank_test_ahc_713770/` | 434 | 0.350685 |
| **合計** | **616 個搬移單位（檔案或快取目錄）** | **3,357** | **32.882659** |

容量以實體一般檔案的邏輯 bytes 計算，共 **35,307,486,117 bytes**，不把符號連結指向的內容重複計數。這是位置整理，沒有去重或減少總資料容量。

保留原有實驗／encoder 名稱、HD、貼目、分片、資料分區與版本層級，不把不同來源混成一份。副檔名相同不代表可互換；重用時仍須遵守各格式的來源身分、完成標記、玩家切分及題組驗證。快取目錄內的完成狀態與 KataGo 診斷 log 隨資料一起保存，供來源追溯；訓練及 Slurm log 保留原位。

原本已在 `data/processed/strength_train_hc/`（含 `standard/` 與 `legacy/`）、`strength_test_hc_713770/`、`boards/`、`cv/`、`identity/`、`reproduce_paper/` 的資料保留原位。例如 `outputs/rank/humansl/humansl_q1000` 等四個既有連結，仍指向 `strength_train_hc/legacy/`，未建立第二份資料。

## 常用路徑

| 用途 | 新位置 |
|---|---|
| 逐手 C 聚合特徵 | `data/processed/strength_features/humansl/permove/agg_{train,test_public,test_private}.npz` |
| 逐手 C 原始分片 | `data/processed/strength_features/humansl/permove/c_*/shard_*/` |
| 強引擎逐盤特徵 | `data/processed/strength_features/humansl/engine_full/features_{train,test_public,test_private}.parquet` |
| C4 共用盤面 pack | `data/processed/strength_features/humansl/c4x/{learn_A,learn_B,monitor}/pack_*/` |
| HD≤1 固定 E2 題組 | `data/processed/strength_features/humansl/hd_filter/questions_e2_hd01.parquet` |
| 官方兩賽落子 store | `data/processed/identity/artifacts/day0_1112/store/` |
| 新增玩家辨識測試集的固定 encoder embedding | `data/processed/identity/artifacts/pid2_bench/<encoder-run>/embeddings.npz` |
| 玩家辨識重複盤參考向量 | `data/processed/identity/artifacts/pid2_overlap/train.npy` |

完整逐項新舊路徑、檔案數與容量見 [搬移對照 TSV](SHARED_DATA_MIGRATION_20261116.tsv)。不能只把整個 `outputs/rank/humansl` 前綴換成新路徑：該目錄的權重、log、報表仍在 `outputs/`，僅對照表列出的項目已搬移。

## 相容方式與後續放置規則

每個搬走的舊路徑保留相對符號連結，指向 `data/processed/` 中的實體資料。既有程式、Slurm、保存設定及 manifest 仍可透過舊路徑存取；寫入既有連結指向的檔案／目錄也會落在新位置。未修改資料內容、設定雜湊或 checkpoint 身分，未批次改寫歷史實驗文件中的路徑。

新實驗產生的共用資料直接寫入上述對應 `data/processed/` 位置；模型、optimizer／checkpoint、訓練 log、分數、圖表與提交預測留在 `outputs/`。新版本須保留自身來源、encoder／權重、HD、貼目、split／question seed 與分片身分，不能覆蓋不同設定的既有資料。共用的模型權重仍按原本模型產物規則存放，不因被多個實驗使用就移入資料目錄。

搬移當下 `rank-policy-epoch`（Slurm job **787069**）仍在執行，`outputs/rank/humansl/policy_residual_epoch/` 內的 run 專屬棋局與題組、設定及訓練狀態保留原位。本次也保留最終 `predictions`、`scores.npy`／`candidates.npy`、驗證診斷與報表；可獨立重用的 embedding／component 特徵則已搬移。

資料與相容連結受 `/data/`、`/outputs/` 忽略規則保護，不進 Git。搬到其他機器時，須一併攜帶資料與保留相對連結的目錄結構；僅複製 Git 程式碼不會帶入這些本機檔案。不要刪除舊連結，既有程式與歷史設定仍可能引用它們；也不要用原子替換舊連結本身的方式更新資料，應更新其新位置，避免在 `outputs/` 重新產生一份實體副本。

## 搬移驗證

- 616 個搬移單位均採同檔案系統 rename，沒有大檔複製、覆蓋或重新計算。
- 3,357 個一般檔案從新、舊路徑檢查，device、inode、size、mtime_ns 全部與搬移前一致。
- 213 個原有符號連結的 target device／inode 全部一致；搬移前沒有失效連結。
- 搬移目錄內 208 個相對符號連結已重算相對路徑，仍可讀到原資料。
- 既有 `BoardCache` 讀取器對 4 組新舊路徑通過初始化驗證，首尾棋局盤面／顏色一致；184,698 列 × 24 欄引擎 Parquet 全表、逐手聚合 NPZ 與 encoder embedding NPZ 全陣列一致，145,118 × 256 的 NPY 首尾抽樣一致。沒有模型權重出現在搬移清冊中。
- 本機完整檔案清冊與驗證結果：`data/processed/migrations/outputs_to_data_20261116/{manifest.json,journal.jsonl,verification.json,reader_checks.json}`；manifest 包含各檔案的搬移前身分與新舊位置，journal 記錄完成的每次搬移。

未執行模型訓練、GPU 推論或全量分數重算；本次無模型或解碼修改，k=5／10／15／20 成績不新增、不宣稱改善。未全量重算檔案 SHA-256；以上是搬移身分與路徑完整性驗證。


## 2026-11-17：C4x 下載備份後移除 Nano4 副本

使用者回報已下載 `data/processed/strength_features/humansl/c4x/`，並刪除 Nano4 上的副本；本次唯讀檢查確認該目錄已不存在。下載目的地與備份完整性尚未驗證；IDE 顯示的 `data/processed/strength_features/humansl/c4x.sha256` 在本次 Nano4 檢查時亦不存在，因此未執行 SHA-256 核對。

刪除前最後一次盤點為 **25,782,120,448 bytes（約 24.01 GiB）**：`learn_A/` 11.61 GiB、`learn_B/` 11.73 GiB、`monitor/` 約 683.29 MiB，以及根目錄切分表與棋局清單約 3.20 MiB。這是刪除前容量，並非刪除後配額實測差值。上方 2026-11-16 搬移清冊與驗證數字保留為歷史紀錄，不代表該資料目前仍在 Nano4。

**移除範圍是整個 `c4x/`，包含 `halves.parquet`、`learn_*_hd*.parquet` 與監控資料，不只是 `learn_A/`、`learn_B/` 的 pack。** 既有 `outputs/rank/humansl/c4x/` 相容路徑若連到已移除資料，讀取也會失敗。`policy_residual/data.py` 的 prepare 仍引用該處的 `halves.parquet`；重做資料準備或重跑依賴 C4x 的實驗前，須從下載備份還原所需檔案與相容路徑。既有已準備完成的作業不能據此推定全部後續流程皆不受影響。

若重新生成盤面 pack，既有入口為 `python -m scripts.strength.rank_c4x tables` → `sbatch slurm/rank_c4x_pack.slurm`；先確認原始切分身分及路徑，再決定還原或重建。兩半 pack 歷史成本約 5.9 GPU-hours，非本次實測。本次僅記錄使用者已完成的下載／刪除，未再刪檔、還原、重建或提交作業。


### 2026-11-17：raw 與 strength_train_hc 僅保留 ZIP

依使用者要求，已將 `data/raw/` 與 `data/processed/strength_train_hc/` 分別封存為同層的 `data/raw.zip`（413,620,651 bytes）與 `data/processed/strength_train_hc.zip`（2,268,546,012 bytes），並刪除兩個原目錄。兩份封存均以逐檔 SHA-256 比對來源、核對完整項目清單及符號連結，確認來源未變後才刪除；合計一般檔案邏輯容量減少 1,437,097,326 bytes（約 1.34 GiB）。ZIP 內保留頂層目錄，還原分別執行 `unzip data/raw.zip -d data/` 與 `unzip data/processed/strength_train_hc.zip -d data/processed/`。讀取原始 CSV、重做前處理或讀取舊 H/C 特徵前須先還原；既有指向 strength_train_hc 的相容連結未移除，還原前無法讀取。未改動其他資料或模型。


### 2026-11-17：e2e_adam 僅保留 ZIP

依使用者要求，`data/processed/identity/e2e_adam/` 已完整打包為同層 `e2e_adam.zip`（4,224,570,410 bytes），再次逐檔 SHA-256 核對封存內容與來源一致、確認來源未變後刪除原目錄（原一般檔案共 5,376,719,380 bytes，約 5.01 GiB）。ZIP 包含 lr1e5／lr3e5 兩臂的完整快取；重做依賴它們的分析前，執行 `unzip data/processed/identity/e2e_adam.zip -d data/processed/identity/` 還原。其他實驗目錄與模型未刪除。


## 2026-11-17：strength_train_hc 下載備份後移除 Nano4 副本

使用者回報已下載備份並刪除 Nano4 上的 `data/processed/strength_train_hc.zip`；本次唯讀檢查確認該 ZIP 與原目錄 `data/processed/strength_train_hc/` 均不存在。上方「raw 與 strength_train_hc 僅保留 ZIP」為較早的封存紀錄，strength_train_hc 現況以本節為準；本次未變更 raw 或 e2e_adam 的封存狀態。

移除前 ZIP 為 **2,268,546,012 bytes（約 2.11 GiB）**，內含原目錄的一般檔案共 2,543,436,906 bytes（約 2.37 GiB）；封存當時已逐檔 SHA-256 比對來源。這不代表下載後的備份已驗證：下載目的地、備份完整性與校驗結果仍未知，本次無法核對外部備份。

原目錄包含 `legacy/`、`standard/`、`se3_hc_b2/` 及來源 manifest；既有指向該目錄的相容連結在還原前無法讀取。重做依賴這批舊 H/C 特徵的 prepare／實驗前，先從下載備份將 ZIP 傳回 `data/processed/strength_train_hc.zip`，再執行 `unzip data/processed/strength_train_hc.zip -d data/processed/`，還原原目錄與相對路徑。

本次僅更新文件，未再刪除資料、還原、重建或提交作業。


## 2026-11-18：C v2 向量搬入共用資料

依使用者要求，競賽二現行基底的 C 級別向量 `outputs/rank/humansl/cvec_v2/learn/vectors.pt`（33,688 bytes，SHA-256 `1640cc8c9493cc9f43a35246bb53c8dfaff636136e99b03ce90213224c07a355`）已移至 `data/processed/strength_features/humansl/cvec_v2/learn/vectors.pt`，與同目錄的 learn／ext／c 特徵放在一起；搬移前後 SHA-256 相同。原路徑保留相對連結，`rank_cvec_v2.py`、C4x 的 `INIT=` 等寫死舊路徑的程式不需修改。同目錄的 `train.json` 與 log 仍留在 `outputs/`。
