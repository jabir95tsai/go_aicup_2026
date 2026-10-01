# AI CUP 2026 圍棋競賽 — 專案 Context

更新：2026-09-23。競賽事實依下列官方頁面；規則異動以最新公告為準。

## 任務與評分

本專案從人類圍棋棋譜推論棋手身分與棋力，支援兩題：

| 題目／官方來源 | 輸入 → 預測 | 官方分數 |
| --- | --- | --- |
| [競賽一：棋手辨識](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688) | 查詢棋譜集 → 候選棋手的有序 Top-5 | 正解第 r 名得 exp(-(r-1))，r=1…5；未入榜得 0。官方公式對 400 題取平均 |
| [競賽二：棋力預測](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403) | 查詢棋譜集 → 一個棋力等級 | 完全正確得 1，相差一級得 1/e ≈ 0.368，其餘 0；對所有題目取平均 |

## 比賽時程

兩題相同，時間皆為台灣時間（UTC+8）。

| 階段 | 開始 | 結束／公布 |
| --- | --- | --- |
| 報名與組隊 | 2026-09-08 11:00 | 2026-11-03 23:59:59 |
| 訓練資料下載與模型訓練 | 2026-09-18 11:00 | 2026-11-24 23:59:59 |
| 測試資料下載與答案提交 | 2026-11-04 11:00 | 2026-11-24 23:59:59 |
| Private Leaderboard 公布 | — | 2026-11-25 16:00 |
| 得獎隊伍繳交報告與程式 | 2026-11-26 11:00 | 2026-12-06 23:59:59 |
| 最終名次公布 | — | 2027-01-04 11:00 |

每天最多提交 5 次，以最後一次上傳結果計分；Public 分數供參考，最終名次依 Private 分數及報告審查確認。

## 訓練集與格式

- **官方 repo**：[AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)；[Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) 提供 v1.0 完整資料，repo 本身提供說明與樣本。
- **下載檔**：`aicup2026_go_training.tar.xz`，官方標示 214 MB、解壓約 1.3 GB；用 `tar xf aicup2026_go_training.tar.xz` 解開。官方 SHA-256：`959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b`。
- **結構／規模**：`training/train_<rank>.csv`，10 檔、每檔 100,000 筆，共 1,000,000 盤匿名棋譜，執黑／執白各半。
- **棋力由弱到強**：`D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D`；程式可依此映射為 0–9，提交值仍依官方格式。

| CSV 欄位（有表頭） | 定義 |
| --- | --- |
| `player_id` | 匿名棋手 ID；同一棋手跨列一致，以字串保留 |
| `game_id` | 棋局編號，以字串保留 |
| `rank` | 該棋手的棋力，與檔名等級一致 |
| `color` | 該棋手在本局的執色：`B` 黑／`W` 白 |
| `sgf_content` | 單行 SGF，僅含棋盤大小與落子序列 |

- **樣本**：[sample_train_A.csv](https://github.com/AILAB-NDHU/aicup2026-go-dataset/blob/main/sample_train_A.csv)。已讀取的首筆為 `player_id=7e6c1a85646fdc7a, game_id=g_0000001, rank=A, color=B`；棋譜開頭為 `(;SZ[19];B[pd];W[pq];B[cd]...`（此處截短）。一列是指定棋手在一盤對局的紀錄；SGF 含雙方落子，須依 `color` 對齊目標棋手。
- **解析**：使用標準 CSV reader（如 `pandas.read_csv`），不要以 `split(',')` 切列；README 提醒 SGF 欄位可能含逗號。再解析 SGF；不可假設有勝負、時間、貼目或真實姓名等 metadata。`rank`／`player_id` 是標籤，不可直接作為預測特徵。
- **待核實**：完整資料中的 pass／異常棋譜、重複棋局與棋手分布，以及測試 query 組成、候選清單、sample submission。已核對 README 與公開樣本，尚未下載／驗證完整壓縮檔。
- **使用限制**：資料僅限 AI CUP 2026 競賽，不得用於識別真實使用者。

## 訓練環境與實作要求

- **目標環境**：本機單卡 NVIDIA CUDA、本機單卡 AMD ROCm，或國網中心「晶創 26」大規模訓練。晶創的 GPU、排程器、容器與儲存路徑尚未核實，部署時依實際環境設定。
- 優先採可跨 CUDA／ROCm 的 PyTorch 實作；相依版本按 GPU、作業系統與驅動確認。CUDA 專屬套件／kernel 必須可停用並有替代路徑。
- 同一訓練入口支援單卡，並預留 DDP／torchrun；資料路徑、batch size、梯度累積、workers、precision 與裝置皆可設定。提供 checkpoint 續訓與小樣本 smoke test。

## 開發與產物慣例

- 先完成資料解析 → baseline → 官方 metric 驗證 → 推論／提交檢查，再擴大訓練。切分須模擬官方任務；同局、重複棋譜及其衍生片段不可跨 train/validation。
- 每次實驗保存資料版本、split、seed、commit、完整 config、套件／硬體版本與分數；提交檔依官方範例驗證，不手動改答案。
- 原始資料放 `data/raw/`，快取放 `data/processed/`，實驗放 `outputs/<run-id>/`，提交放 `submissions/`；資料、快取及權重不進 Git。程式、設定與文件版控。
- 外部資料、模型、開源程式及生成式 AI 的來源與用途需留存，供最終報告揭露；遵守官方授權及不得跨隊私下分享程式／特徵等規則。
- 動手前檢查現有檔案與 Git 狀態，保留使用者變更。回覆用繁體中文；若產生或修改檔案，務必告知輸出的絕對路徑。
