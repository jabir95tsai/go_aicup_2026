# v1.1 讓子資料稽核與訓練提案

2026-11-06。來源：[官方 v1.1 release](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/tag/v1.1)。本次完成資料更新、擷取及描述性分析；下列訓練提案尚未實作或執行。

## 產物與重現

- `data/raw/training/train_{D,C,B,A,1D..6D}.csv`：官方 v1.1 十份 CSV，SGF 含 `HD[n]`。
- `data/raw/training/handicap.csv`：1,000,000 列，欄位 `rank,game_id,player_id,color,source_row,handicap`。`source_row` 是該 rank CSV 的零起算列號；唯一鍵為 `(rank, game_id)`。ID 與列號只供 join／稽核，不得當特徵。
- `data/raw/training/handicap_manifest.json`：官方壓縮檔雜湊、新舊 CSV 雜湊、各級及執色的 HD 計數。
- `outputs/rank_handicap_audit_v11/report.json`：與既有 prepared metadata 的對齊檢查、測試 HD 分布及既有 SE3 分數的分層診斷。
- `outputs/rank_handicap_audit_v11/se3_question_diagnostics.csv`：既有 10,000 題預測、HD=1 比例、是否含 HD≥2 與原始得分；沒有擬合或改預測。

資料與 outputs 均依現有規則不進 Git。重跑使用既有 `go` 環境，在 repo 根目錄執行：

```bash
curl -fL --retry 3 -o /tmp/aicup2026_go_training_v1.1.tar.xz https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/download/v1.1/aicup2026_go_training_v1.1.tar.xz
PYTHONPATH=src python scripts/extract_training_handicap.py /tmp/aicup2026_go_training_v1.1.tar.xz
```

腳本要求既有十份 training CSV，先驗官方 SHA-256，再解壓至暫存目錄，全量比對完成才逐檔原子替換。支援來源為 v1.0 或已更新的 v1.1；發布中斷可重跑，但十份 CSV 並非整組交易。成功更新後，舊版可由移除 root `HD[n]` 還原內容，不另存 1.3 GB 備份。官方 SHA-256：`e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`。

已全量驗證：除 root HD 外，列順序、player_id、game_id、rank、color 與 SGF 逐字相同；sidecar 與既有 SE prepared 的 1,000,000 個鍵、玩家、執色、source_row 全部一致。SGF 嚴格 parser 已接受可選 root HD，仍拒絕 setup 石、負數／非整數／重複 HD。HD 只保存在 `Game.props`，沒有新增盤面石頭、模型輸入或變更快取。

## 實際分布

訓練集 HD=0／1／≥2：**792,610／170,969／36,421**（79.261%／17.0969%／3.6421%），範圍 0–9。

| 棋力 | 盤數 | HD=1 | HD≥2 |
|---|---:|---:|---:|
| D | 100,000 | 32.431% | 2.878% |
| C | 100,000 | 25.663% | 1.672% |
| B | 100,000 | 21.423% | 1.320% |
| A | 100,000 | 15.958% | 1.761% |
| 1D | 100,000 | 14.339% | 2.787% |
| 2D | 100,000 | 14.336% | 2.772% |
| 3D | 100,000 | 8.753% | 4.578% |
| 4D | 100,000 | 9.144% | 3.582% |
| 5D | 100,000 | 12.659% | 6.754% |
| 6D | 100,000 | 16.263% | 8.317% |

棋力測試 Public／Private 各 400 題、5,000 盤；**全部有 HD，且 HD≥2 都是 0 盤**。HD=1 各為 1,155／1,125 盤（23.1%／22.5%）。以盤為單位的分 k 分布如下；這些是無標籤資料描述，不能解讀為棋力分數或直接反推測試 rank 先驗。

| k | Public HD=1／盤數 | Private HD=1／盤數 |
|---|---:|---:|
| 5 | 108／500（21.60%） | 99／500（19.80%） |
| 10 | 259／1,000（25.90%） | 266／1,000（26.60%） |
| 15 | 347／1,500（23.13%） | 316／1,500（21.07%） |
| 20 | 441／2,000（22.05%） | 444／2,000（22.20%） |

SE3 現有預測分層如下。這是同一模型的事後診斷，兩組玩家／rank／題數不同，不能把組間差當因果或訓練增益：

| k | 全部 HD≤1 題數 | 原始得分 | 含 HD≥2 題數 | 原始得分 |
|---|---:|---:|---:|---:|
| 5 | 2,233 | 0.579656 | 267 | 0.533020 |
| 10 | 2,161 | 0.623552 | 339 | 0.632477 |
| 15 | 2,169 | 0.630277 | 331 | 0.617067 |
| 20 | 2,074 | 0.663429 | 426 | 0.687466 |

含 HD≥2 的題目並非四桶一致較差，所以目前只能確認分布差異，不能宣稱排除後必定提升。

## 優先實驗，尚待確認

1. **先做凍結模型的 CPU 校正。** HD=1 比例由 D 的 32.431% 降至 3D 的 8.753%，之後回升，呈非單調訊號；不能把 HD 大小直接當棋力高低。以每題 HD=0／1 比例、執色及其交互項搭配既有 SE3 每盤／每題 log-prob，先測一個小型正則化分類器。按玩家切分擬合與選擇集，fit／標準化不可看到評估玩家；SE3 本身為 game-disjoint，這只能驗證校正器對 held-out 玩家的增益，不能宣稱整個 CNN 已驗證新玩家泛化。沿用 SE3 argmax，避免同時變更多個因素；官方 score_rank 的另一次消融分開做。
2. **再測訓練抽樣只保留 HD≤1。** 現有 800,000 training 有 29,168 盤 HD≥2；保留 **770,832 盤**，不重切分、不刪原始資料、仍按級等權抽樣。假設是降低 train/test 差異，預期增益未知。固定原 10,000 題作主 gate，另外報告全部 HD≤1 的題目子集。不同盤數不改「相同 optimizer updates」對照；這不保證提速。若確認有收益，再考慮獨立版本把 HD 與執色接入 rank head 或新增常數 plane。

提案 1 的成本為 **0 GPU-h**，只讀已有 validation log-prob 並做 CPU 擬合；實際 CPU 秒數未知。提案 2 需要新的 H200 訓練與完整評估，成本未知，須先短測估時；不在目前 run 上續訓或覆寫。

兩項都以 **k=5／10／15／20 的官方得分與 Equal-k** 判斷，先約定 gate：相同評估題的 Equal-k 至少 **+0.005**、任何 k 不低於對照 **0.005** 以上，並檢查玩家 bootstrap 95% 區間；區間跨零時只列候選。SE3 原固定 10,000 題對照為 **0.574675／0.624762／0.628528／0.667525，Equal-k 0.623873**；若另切 calibration 評估玩家，須在相同子集重算對照，不能拿它直接對比此全量數字。**兩項實際增益與所有新分 k 分數目前都是未知**。模型／超參選擇只看本地 held-out labels，不用測試題預測分布反推標籤。

實作範圍：sidecar→metadata 的驗證 join、fold-safe calibration 與測試聚合；若重訓，還須同步 sampler eligibility、config／run 身分、固定監控、最終評估及推論。既有 packed boards、split 與逐位置 teacher 的索引保留，不能因篩盤後重新編號而錯位。HD 尚未加入 `feature_policy.py` 白名單，須在核准特徵實驗時同步處理。

## 尚未確定的語意

官方只說 HD 是讓子資訊；沒有提供 `HD=1` 對應的貼目，也沒有提供 HD≥2 的 `AB` 起始座標。棋步本次完全不變，不代表已證實盤面還原正確；不要由 HD 自動補星位、套用 HA setup 或修改目前 teacher 的 6.5 貼目。若之後取得官方語意與起始盤面，才評估是否重建對應 boards／value labels。玩家辨識的候選庫與測試題未在目前 `data/raw/tests` 提供，本次未驗證其 HD，也未變更玩家辨識模型。

驗證：`PYTHONPATH=src python -m pytest -q tests/test_sgf_planes.py tests/test_foundation.py tests/test_release_pipeline.py tests/test_training_handicap.py tests/test_board_cache.py`，**84 項通過（15.01s）**。未重新重播官方全部盤面，也未啟動 GPU 工作。
