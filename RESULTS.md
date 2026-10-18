# 實驗帳本

> 只增不刪。失敗的實驗也要記——上屆最有價值的知識來自失敗的那幾條。
> 每次實驗前先寫：tag / 假設 / 預期增益 / gate。實驗後補：4 個 k 桶分數 + 標準差 / 結論。

---

## W1 — 稽核

### A-001 訓練集稽核（2026-09-24）

**工具：** `src/sgf.py`（parser）、`src/audit_train.py`（稽核）
**報告：** `notebooks/01_audit.md`（**全量 1,000,000 盤，已完成**）、`notebooks/01_audit_sample.md`（每檔 3,000 列）
**狀態：** 候選庫與測試檔尚未發布，只能稽核訓練集。

#### 決定性發現

| # | 發現 | 對計畫的影響 |
|---|------|------------|
| A1 | SGF 只有 `SZ[19]` + 落手，**沒有 `HA`/`KM`/`DT`/`AP`/`TM`，也沒有 AB/AW setup 石** | **L3（讓子貼目）作廢。** 沒有任何 metadata 捷徑，全部訊號必須從落手本身來 → L1/L5 的相對價值上升 |
| A2 | distinct `player_id` = **8,007**；**rank 是玩家級屬性**（0 位玩家跨越多個 rank） | 競賽二的有效樣本數是 **8,007 位玩家**，不是 100 萬盤。這是最大的過擬合風險來源，見 A5 |
| A3 | `game_id` **不唯一**：1,000,000 列只有 100,000 個 distinct `game_id`，唯一鍵是 **`(rank, game_id)`** | 前處理／join／去重若只用 `game_id` 會 10:1 靜默塌縮。`preprocess.py` 必須用複合鍵 |
| A4 | 每位玩家盤數極度偏斜：p25=23、中位數=51、p75=155、max=1800。**只有 1,281 位玩家有 ≥120 盤黑棋** | 競賽一的忠實合成驗證（400 人 × 100 盤 gallery + 20 盤 query）**最多只能用 1,281 位玩家** → 3 組互斥的 400 人池，或 5 組重疊重抽。且這些是「重度使用者」子集，有偏，要在報告中說明 |
| A5 | 各 rank 的玩家數差 3.6 倍：1D 只有 **433** 位（每人 230.9 盤）、A 有 **1,582** 位（每人 63.2 盤） | 1D/2D 等級的玩家太少，模型很容易學成「這是玩家 X，X 是 1D」而不是「這種棋力是 1D」。測試時玩家未見過 → 崩盤。**這是上屆 V15 玩家特徵不轉移的同型病** |
| A6 | 棋局長度 **100–445** 手（中位數 228、p99 323），無短局、無損毀、盤面全為 19 路 | **位置上限必須取 448**（全量 max=445，取樣版的 359 低估了尾巴）。p99 只有 323 → 用 **length bucketing** 而非固定 padding，否則 ~40% 算力浪費在尾端。不建議截斷：官子精度是棋力訊號 |
| A7 | 黑方第一手高度集中：`pd` 55.2%、`qd` 19.8%（兩點合計 75.0%），全量共 **293** 個不同點位 | 單看第一手辨識力弱，**L7 必須用前 10–20 手的 n-gram**，不能只用第一手 |
| A8 | **全量 1,000,000 盤內零筆完全重複的 `sgf_content`**；round-trip 還原 500/500 一致；解析錯誤 0；全為 19 路 | parser 可信，可進入 `preprocess.py`。注意：`game_id` 雖跨檔重複，但棋譜內容確實各不相同 |

#### 由 A2 + A5 衍生的架構修正

競賽一要**放大**玩家身分訊號，競賽二要**壓抑**它——同一個 encoder 承受相反的壓力。
因此「完全共用 encoder」是錯的。正確做法：**共用預訓 trunk → 任務各自微調**，競賽二額外加
player-adversarial 頭（gradient reversal）或「每批次每位玩家只取一盤」的取樣策略。

#### 新假設 L10（待測試檔發布後第一件事驗證）

**競賽二的測試玩家是否就在訓練集的 8,007 人之中？**
若是，競賽二可簡化為「用競賽一的技術辨識出玩家 → 查他的 rank」。這會是整個專案價值最高的單一發現。
驗證方法（便宜）：對每題跑玩家比對，看分數是否顯著尖峰於某位訓練集玩家（對照 null 分布）。
主辦方很可能刻意把測試玩家排除在外，但**檢查成本極低、回報極高**。

### A-001 勘誤（2026-09-24，全量 1M 盤轉檔後）

- **A6 修正：棋局長度上限是 445，不是 359。** 359 來自抽樣稽核；全量 1M 盤中 p50=228、p99=323、p99.9=348、
  **>360 手有 207 盤（0.02%）、>400 手 8 盤**，各 rank 都有（最長 C/D 445）。
  → 序列模型位置上限依 `EXECUTION_PLAN.md` D1 取 **448**（零截斷）+ length bucketing（p99 只有 323，固定 padding 到 448 會浪費大量算力）
- A-001 的 `notebooks/01_audit.md` 全量報告**沒有產出**（`logs/audit_full.log` 停在 5D 解析完），只有每檔 3,000 列的抽樣版。
  關鍵數字已由 A-002 的全量轉檔獨立重驗：parse 錯誤 0、pass 0、全數黑先且黑白交替、無重複 `sgf_content`

### A-002 `preprocess.py` 全量轉檔（W1 gate ✅）

- 1,000,000 盤 → `data/processed/train_{moves,colors}.npy` + `train_meta.parquet`，**221,894,008 手**，int16 共 444 MB
- 耗時 151 秒（18 workers；每 rank 分片 9–16 秒，分片可續跑）
- **Gate：隨機 1,000 盤從 store 反解回 SGF，與原始字串逐字比對 1000/1000 一致** ✅
- 解析錯誤 0；非交替 0；白先 0；pass 0；`sgf_hash` 重複 0（訓練集內無逐字重複棋譜）
- meta 保留 `sgf_hash`，測試檔發布後可直接 join 做「測試題是否逐字出現在訓練集／候選庫」稽核
- 已寫好 `release` 模式（候選庫 + 寬格式測試題 melt 成每盤一列），測試檔一放進 `data/raw/` 就能 `python src/preprocess.py release`

### A-003 `cv.py` 切分與合成題組（W1 gate ✅）

**競賽二：** 以玩家為單位分層（rank × log 盤數）→ 10% 固定 holdout（fold −1，801 人）+ 5 fold。
每 fold 每 rank 都有玩家；1D 最少（每 fold 77–79 人）。零重疊 assert 通過。

| | f0 | f1 | f2 | f3 | f4 | holdout |
|---|---|---|---|---|---|---|
| 盤數 | 178,907 | 183,141 | 179,048 | 177,535 | 181,675 | 99,694 |
| 合成題數 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 4,000 |
| 題中 distinct 玩家 | 1,322 | 1,318 | 1,339 | 1,331 | 1,341 | 719 |

- 題目 k ∈ {5,10,15,20} 各 25%、每 rank 各 10%（**rank 平衡是假設**，測試集是否平衡未知 → L4）、黑白約各半、同題同色同玩家
- ⚠️ 1D 每 fold 只有約 78 人卻要出 1,000 題 → 同一玩家平均出現約 13 次。**合成題之間不獨立**，
  報告標準差時要用「以玩家為單位的 bootstrap」，不能把 10,000 題當 10,000 個獨立樣本

**競賽一：** ≥120 盤黑棋的合格玩家 = **1,281**（與 A-001 A4 一致）。5 組 × 400 人，每人 100 盤 gallery + 1 題查詢（k 各 100 題，一對一）。
- split 0/1/2 玩家**完全互斥**；split 3/4 為重抽，與前三組各重疊 117–130 人 → **split 3/4 的分數不是獨立樣本**
- assert 通過：查詢盤 ∩ gallery = ∅；每題正解必在該 split 的 400 人中；全部黑棋
- ⚠️ 合成候選庫的 rank 組成偏斜：split 0 中 A 級只有 23 人、2D 有 54 人（A 級玩家盤數少，較少人達到 120 盤門檻）。
  真實候選庫的組成未知，這是合成驗證的已知偏差
- 規則：評估 split s 時，任何模型（含 L9 的棋力模型）都**不可在 split s 的 400 人上訓練** → 用 `cv.pid_train_players(meta, gallery, s)`

### A-004 算力實測（本機 RTX 3060 Ti 8GB，bf16）

| 模型 | 參數 | 訓練吞吐 | 1M 盤 / epoch | 顯存 |
|---|---|---|---|---|
| Transformer d=256, L=6, T=360 | 5.0M | 854 盤/秒 | **19.5 分** | 1.3 GB |
| Transformer d=384, L=8, T=360 | 14.6M | 374 盤/秒 | **44.5 分** | 2.5 GB |

（純前向＋反向的 synthetic benchmark，未含資料載入；真實數字預期再慢 10–20%）

### A-005 訊號 EDA（2026-09-25）

**工具：** `src/game_stats.py`（numba 盤面重播，1M 盤 / 2.2 億手 7.5 秒，每盤本人＋對手各 26 種統計）、`src/eda_report.py`
**報告：** `notebooks/02_eda.md`（全量，約 1.5 分鐘 CPU）；引擎單元測試 `tests/test_game_stats.py`
**性質：** 描述性分析 + 兩個量化訊號用的探針，**不是提交模型，不佔上傳額度**

| # | 發現 | 對計畫的影響 |
|---|------|------------|
| E1 | 單盤手工特徵棋力訊號弱：最強 `me_own_adj` rank η²=0.057；80–95% 變異是盤內雜訊 | k 盤聚合必需；天花板在單盤表徵品質（L1/L5） |
| E2 | 本人特徵的「同 rank 內玩家成分」是 rank 成分的 1.5–5 倍 | A5「記住玩家」風險的量化；k→∞ 也突破不了 → 競賽二必須有反記憶設計 |
| E3 | **對手特徵的棋力訊號 ≈ 本人**，距離／應手類的玩家成分只有本人一半（例 `d_opp` 0.063 vs 0.151）；提子類例外（「對手提子」= 本人被提，屬本人） | **L2 升級為主力**：rank 頭吃雙方的手 |
| E4 | 線性探針（題均值特徵、f0–f3 訓練、f4+holdout 評估）：本人+對手 acc 0.225/0.267/0.301/0.313（k=5/10/15/20）、MAE 1.74→1.23、QWK 0.69→0.83；只用本人 all acc 0.258、只用對手 0.249 | P0 GBDT 的參考下限；錯誤呈 ordinal 結構 |
| E5 | **開局指紋極強**：黑前 4 手逐手 naive Bayes（零訓練）在 3 組互斥合成候選庫：top-1 0.773/0.920/0.970/0.983、top-5 0.917/0.983/0.997/1.000、MAP@5 all 0.938。打亂標籤 → top-5 0.018（排除 bug）；α=1/5/20 穩定 | 競賽一 P0 = 這個計數模型；戰場在 k=5 桶與排序品質。8/16 手反而變差（獨立假設的缺陷），不代表後段無資訊 |
| E6 | D4 正規化代價：黑前 8 手 k=5 top-5 0.867→0.813、MAP@5 all 0.910→0.863；3 組間 sd 0.01–0.03 | 競賽一禁用 D4 有數字支持；棋力的空間訊號 D4 不變 → 競賽二可用 D4 |
| E7 | `game_id` 為打亂流水號（每檔 1–50000 = B、50001–100000 = W），無時間資訊；勝負代理（最後一手）η²=0.0003、終局子數 η²≈0 | 風格時間漂移**未知**、無法驗證；若官方依時間切候選庫/查詢，合成驗證會高估競賽一 |

其他：棋力越高 → 棋局越短、脫先越多、一線手越少、單子提（打劫）越多、大塊被提越少、照流行下法走得越久（開局唯一化深度 D 9.2 → 6D 11.2 手）。
兩色的棋力訊號強度相同，玩家黑／白風格均值相關 0.81–0.89 → 兩色可共用模型。

**已知偏差：** 合成候選庫 = ≥120 盤黑棋的重度玩家（習慣集中度：個人最常用第一手佔比中位數 0.95）；k=20 合成題只來自各 rank 活躍子集（A 44%～1D 86%）。

### A-006 官方指標回填 + 既有探針重算（2026-09-27）

**來源：** 官方競賽頁（使用者提供截圖與說明）。細節見 CLAUDE.md §0。
- 競賽二：exact 1 / ±1 級 1/e / 其他 0；baseline **0.4631**
- 競賽一：正解在第 r 名得 e^{-(r-1)}（1 / .368 / .135 / .050 / .018）；baseline **0.2845**
- 測試集 11/04 11:00 發布；上傳截止 11/24 23:59:59；報告 11/26–12/06

**程式：** `src/metrics.py` 新增官方 `score` 欄（兩賽）與 `score_rank`（競賽二 Bayes 最佳解碼）；單元測試通過。

**重算（不是新實驗，同一資料、同一模型，只換計分與解碼）：**

競賽二 — E4 線性探針（本人+對手 26 維題均值、LR；f0–f3 訓練、f4+holdout 14,000 題評估）：

| 解碼 | k=5 | k=10 | k=15 | k=20 | all |
|---|---|---|---|---|---|
| argmax | 0.341 | 0.392 | 0.434 | 0.443 | 0.403 |
| 期望值四捨五入 | 0.344 | 0.387 | 0.407 | 0.422 | 0.390 |
| **`score_rank`** | **0.350** | **0.398** | **0.440** | **0.459** | **0.412** |

- `score_rank` 在四個 k 桶都 ≥ argmax（+0.006～+0.016），**免費增益，之後所有棋力模型一律用它解碼**
- 期望值解碼提高 within-1，但犧牲 exact，淨分較低 → 驗證了「差兩級 = 0 分」讓期望值解碼不划算
- 線性探針 0.412 < 官方 baseline 0.4631 → P0 GBDT 至少要 +0.05 才過門檻；k=5 桶是最大缺口

競賽一 — E5 黑前 4 手 naive Bayes（零訓練，3 組互斥合成候選庫平均）：

| | k=5 | k=10 | k=15 | k=20 | all |
|---|---|---|---|---|---|
| **官方 score** | 0.809 | 0.939 | 0.977 | 0.989 | **0.929** |
| top-1 | 0.773 | 0.920 | 0.970 | 0.983 | 0.912 |

- 官方分數 ≈ top-1 + 0.02：名次 2–5 貢獻極小，**優化目標就是 top-1**
- ⚠️ **合成 0.929 vs 官方 baseline 0.2845，差 3 倍。** 機制未知，四個候選解釋見 CLAUDE.md §0。
  在 11/04 查清前，合成驗證的絕對分數不可信、只能用來比較方法間相對優劣。
  若真實情境下開局習慣失效，競賽一的主力要從 L7（開局指紋）轉向全盤風格 embedding（L8）

### L-001 文獻調查（2026-09-27）

**性質：** 文獻與既有專案調查，不是實驗，不佔上傳額度。結論與設計影響整理在 STRATEGY.md §1b；外部資源登記在 `EXTERNAL_RESOURCES.md`。
**方法：** 背景 agent 兩輪網路搜尋（第一輪碰到搜尋上限，第二輪補完）。以下「已驗證」= 讀過原文 PDF／原始碼／官方頁面。

| # | 發現 | 狀態 | 對計畫的影響 |
|---|---|---|---|
| R1 | Kuboki 2025 表 4：**同一玩家** k 盤 exact 58.1／61.0／63.2%（k=5／10／15），同段位隨機抽盤則是 70.4／82.8／89.4%；±1 同一玩家約 95–97% | 已驗證（PDF） | 競賽二預期要用同一玩家數字；盤數邊際效益小 → 重點在玩家不變性 |
| R2 | Chen 2025（ICLR）與 Kuboki 2025 主結果的查詢題都是「同段位」而非「同一玩家」，文中都沒說是否 player-disjoint | 已驗證 | 文獻中的高數字不能和我們的驗證直接類比 |
| R3 | KataGo HumanSL `b18c384nbt-humanv0`：以段位為條件的走子網路；段位尺度 = KGS；授權 = KataGo Neural Network License | 已驗證（原始碼 + 授權頁） | L5 成本下降，可考慮提前；gate = 先測吞吐量 |
| R4 | HumanSL 訓練資料來源：原始碼列舉 OGS/KGS/Fox/Tygem/GoGoD/Go4Go，官方未明示；不含幽玄 | 部分驗證（推論） | 只能當特徵，不能直接解碼段位 |
| R5 | Behavioral stylometry（NeurIPS 2021）：ArcFace 不收斂 → GE2E；只用第 15 手後訓練 0.39 vs 含開局 0.99；400 人池 P@1 0.953 | 已驗證（PDF）；程式碼未公開 | 競賽一 loss 並列比較 GE2E；開局必須入模 |
| R6 | 同一玩家換開局（e4 vs d4），兩組風格距離 ≈ 與他人距離；文獻中無時間切分的 stylometry 數字 | 已驗證（PDF） | 支持「開局指紋可能在真實測試失效」假設，未證實 |
| R7 | AI CUP 2023 圍棋賽題目不同（下一手預測 + 棋風分類）；前幾名方案未找到 | 已驗證（競賽頁） | 參考價值低 |
| R8 | 8 篇方法論文 arXiv ID 已核對（GE2E、Attention MIL、Set Transformer、CSLS、k-reciprocal、PT-MAP、CORAL、CORN） | 已驗證（arXiv API） | 實作細節未讀 |

**未找到：** AI CUP 2023 前幾名報告、Kaggle Finding Elo 得獎方法、時間切分下的 stylometry 數字、Kuboki 2025 程式碼、Hsueh et al. 2026 HumanSL 論文預印本。
**未驗證：** 「Fox 1d ≈ KGS 2k」、主辦方講者與 Kuboki 論文共同作者為同一人（推論）。

### A-007 合併 partner 骨架 + 改為套件結構（2026-10-03）

**性質：** 工程整理，不是實驗，不影響任何分數。
**起因：** 兩人各自開了 repo、接到同一個 remote，history 沒有共同祖先；W1 的程式從未推上 GitHub，partner 看不到。已用 `--allow-unrelated-histories` 合併。

- **路徑變更：** `src/*.py` → `src/go_aicup/{data,features}/`、`metrics.py` 留在套件根、`audit_train`/`eda_report` → `scripts/`；`data.py` 改名 `store.py`。
  **本檔 A-001～A-006 內的 `python src/xxx.py` 指令是當時的路徑，不改**；現行指令見 README
- repo 根目錄改由 `go_aicup/paths.py` 單一計算；`pyproject.toml` + `pip install -e .`；`requirements.txt` 補上原本漏鎖的 scipy／numba／matplotlib
- **重現性驗證（搬移前後）：** pytest 15/15；`cv` 9 張切分表與 `game_stats` 輸出 sha256 **逐位元相同**；`preprocess --verify-only` 結果相同；`eda_report` 重產 02_eda.md，全部數字相同、圖檔位元相同，只有時間戳與路徑文字變動
- 新 run 改用 `outputs/<run-id>/`（partner 的慣例）；`logs/` 留給 W1 舊 log
- partner 的 `reference.md` 補了 L-001 沒有的來源：AIIDE 2025 multi-policy 棋力估計（R02）、Moudřík GoStyle（R03/R04）、Animiral/go-strength-model（R05，KataGo trunk 特徵）、**主辦方 ICGA 2025 舊 AI CUP 回顧（R09，對應 L-001 R7 的「未找到」）**、SupCon。尚未讀原文，未驗證
- partner 的 `reference.md` §10 列的「同局黑白兩方跨 split 污染」風險，在本資料不適用：A-001/A-002 已證實 1M 盤零重複 `sgf_content`，每盤只出現一次
- **待確認：** partner 列的目標算力（國網晶創 26、AMD ROCm）是否真的可用 → 已加入 CLAUDE.md §0 待確認表
  - **更新（同日）：** 已確認晶創 26 與 ROCm 都可用 → CLAUDE.md §0 移到「算力 ✅」；EXECUTION_PLAN 算力預算待重估

---

## 待辦（測試檔發布後立刻做）

- [ ] 真實候選庫的「個人最常用第一手佔比」分布 vs 合成候選庫（0.95）→ 判斷 E5 是否高估

- [ ] **L10 驗證**：競賽二測試玩家是否在訓練集內
- [ ] train / candidates / test 三邊 property 存在率比對（確認 test 側也是純落手）
- [ ] 測試題 SGF 是否逐字出現在候選庫或訓練集
- [ ] `player_id` 空間在 train 與兩個候選庫之間是否重疊
- [ ] 400 題 vs 400 候選是否一對一（決定 L6 用 Hungarian 還是 Sinkhorn）
- [ ] 測試題 `num_games` 是否真的 5/10/15/20 各 100
- [x] 兩賽的評估指標與 baseline 分數（A-006）
- [ ] **競賽一 baseline 0.2845 vs 合成驗證 0.93 的落差來源**（A-006 警訊）：真實候選庫第一手集中度、查詢盤與候選盤的開局分布是否一致


### A-008 硬體支援範圍調整（2026-10-10）

**性質：** 專案需求調整，不是實驗，不影響模型與分數。

- 依使用者決定，必須支援的目標環境縮為 **RTX 3060 Ti 8GB 與國網中心「晶創 26」**，取消 AMD 單卡／ROCm 相容性要求；本紀錄取代 A-007 中的支援範圍，舊紀錄保留。
- 本機以 CUDA 為基準，可使用 CUDA 專屬 kernel，無須為 ROCm 提供替代實作；保留 DDP／torchrun 與資料路徑、batch、precision、裝置的 config 要求。
- 同步更新 `CLAUDE.md` §0、`AGENTS.md` 的指標說明及 `EXECUTION_PLAN.md` 算力預算待辦。晶創 26 的 GPU 型號、排程器、配額仍未知，算力預算仍待重估。
- 全專案搜尋確認目前沒有 ROCm／AMD 專用程式分支或設定需要移除；`reference.md` 的外部工具後端資訊保留。

### A-009 晶創 26 H200 環境查核（2026-10-10）

**性質：** 官方文件調查與需求回填，未登入 HPC、未提交作業、未訓練模型，不影響分數。

- 使用者確認帳號可使用 H200 `dev`、`8gpus`；官方規格、佇列限制與來源集中於 `CLAUDE.md` §0，文件引用登記於 `EXTERNAL_RESOURCES.md`。
- 已回填 GPU／節點規格、Slurm job 時限、CPU／RAM 限制、登入／傳輸與 Modules 操作。
- 同步更新 `EXECUTION_PLAN.md` 待辦與 README 入口；本紀錄補完 A-008 的 GPU 型號／排程器未知項，舊紀錄保留。帳號額度、軟體版本、實測吞吐量仍未知；算力預算仍待短測重估。

### A-010 Strength V1 程式交付（2026-10-12，未驗證）

**性質：** 依已確認的 `STRENGTH_V1_PLAN.md`／`STRENGTH_V1_IMPLEMENTATION.md` 撰寫程式。使用者指示「先寫程式不驗證」；本次沒有執行新程式、測試、正式轉檔或訓練。

- ResNet：256 channels、20 residual blocks、18 層輸入、單一實數輸出；相加後無 ReLU。23,748,611 是設計參數數量，未於本次執行計數驗證。
- 直接使用既有 `board_cache.py` 的 `uint8[total_ply,2,46]` 分片、實際 colors 與 game metadata；不變更快取格式、不另存全量 18 層。
- 新增 V1 按棋局 80,000／20,000 每級切分與同玩家同色四桶題組；指定玩家 ply 索引、全域 uint32 排列、stateless D4、尾批有效 mask、FP32 跨卡 sum/count 與排序 loss。
- 新增 DDP／SyncBN／BF16、AdamW、warmup／cosine、定時與 epoch checkpoint、各卡 RNG／資料指紋與續跑入口；預設先跑 1 epoch。未執行單／多卡梯度、SyncBN 尾批、續跑或顯存驗收。
- 校準只用固定 training 題組的唯一棋局，以凍結模型推論；手→盤→題等權聚合，用四桶等權官方 score 擬合九個有序切點，validation 固定切點評估。切點綁定 checkpoint SHA-256。

| 指標 | k=5 | k=10 | k=15 | k=20 |
|---|---|---|---|---|
| validation 官方 score／accuracy／MAE | 未知 | 未知 | 未知 | 未知 |
| 預期效能增益 | 未知 | 未知 | 未知 | 未知 |

本次訓練 updates=0、測試執行=0、新增模型權重=0；沒有提交檔、LB 預測或上傳。正式盤面數、玩家重疊率、GPU 記憶體與 GPU 小時仍未知。

**程式／設定輸出：** `src/go_aicup/models/strength_resnet.py`；`src/go_aicup/tasks/strength/{dataset,prepare_rank_v1,losses,runtime,train_rank_v1,evaluate_rank_v1}.py`；`src/go_aicup/data/cv.py`；`configs/strength/rank_cnn_v1.json`。操作文件更新於 `README.md`，外部資源登記於 `EXTERNAL_RESOURCES.md`。

**日後執行才產生：** `data/processed/cv/strength_v1_seed2026/`、`data/processed/strength_v1/seed2026/`、`outputs/rank_cnn_v1_seed2026/`；這些正式 V1 索引／權重／分數本次未產生。本機 run 輸出完整路徑為 `D:\college\go_aicup_2026\outputs\rank_cnn_v1_seed2026\`。

### A-011 資料流程與重複防呆清理（2026-10-14）

**性質：** 依使用者逐項清理指示整理程式與文件，不是模型實驗。訓練 updates=0，未執行正式全量轉檔、GPU 訓練或提交；k=5／10／15／20 分數與效能增益均未知。使用者其後指示「先不跑測試」，因此停止後續測試，最終整套驗證尚未完成。

- 唯一 SGF parser 改為競賽的線性 `SZ[19]`／0～360 落點格式；取消 pass、setup 座標清單、coverage 與額外 SGF 格式相容支援，刪除重複 skeleton 檢查及未使用的 `move_tokens()`。稽核仍可讀取 property 名稱，異常資料入口報錯。
- 一般盤面轉換只重播落子與提子；占點、自殺、立即劫改由 `validate_moves=True` 明確啟用，稽核只需兩個歷史盤面。EDA 刪除 pass 分支與第 362 個落點類別。
- 移除逐盤完整打包／解包比對；`board_cache verify` 保留全檔 hash／shape／padding／colors，並固定 seed 2026 抽每個 shard 一盤做合法性稽核重播與快取對照。shape 檢查移到首次開啟 shard，取消每次讀取的 padding／顏色值掃描。
- 保留 `shard_XXXXX_colors.npy`、實際 colors 取樣與 18 層黑／白編碼；指定玩家、歷史不跨局、索引範圍、尾批 mask、缺級處理、DDP loss、checkpoint／資料指紋與 training-only 校準檢查仍保留。
- CV 刪除玩家唯一性已涵蓋的 fold 集合逐對比對；GameStore 移除未知顏色 `-1` 回退，入口要求合法 B/W。刪除沒有使用端的 `FoldSafeTargetEncoder`；共用 artifact hash／原子 JSON 與 `assemble_inputs()`。
- 測試棋譜改為無 pass、黑先交替的資料，保留提子、歷史邊界、顏色、LRU／pickle、續跑與損壞檔案驗收。最後調整的多子／多群提子與自殺測試棋譜尚未重跑。

**停止測試前已完成的分組結果（非最終整套驗證）：**

| 部分 | 驗證 | 結果 |
|---|---|---|
| setup 清單 | `test_sgf_planes.py` | 39 passed |
| SGF／pass 格式 | SGF、foundation、board_cache | 65 passed、1 skipped |
| EDA／棋步稽核模式 | SGF、game_stats | 44 passed |
| 快取檢查集中化 | board_cache | 12 passed、1 skipped |
| CV／顏色入口 | foundation | 16 passed |
| 未使用功能清理 | 現有特徵與增強規則 | 3 passed、12 deselected |
| 共用工具／18 層組裝 | board_cache | 12 passed、1 skipped |

官方各級前 20 盤，共 200 盤（連續前綴抽樣，非全量代表樣本）的一般／棋步稽核模式盤面逐元素一致。略過項均為目前 `D:\anaconda\python.exe` 缺少 torch 的實際 DataLoader 多程序測試；未執行 DDP 驗收。依停止測試指示，之後只進行 Python AST 靜態語法檢查與 `git diff --check`，不匯入專案模組或啟動測試。

**檔案位置：** 新增共用工具 `D:\college\go_aicup_2026\src\go_aicup\artifacts.py`；修改位於 `src/go_aicup/{data,features,tasks/strength}/`、`scripts/{audit_train,eda_report}.py`、對應 `tests/`，並同步 `CLAUDE.md`、`README.md`、兩份 V1 計畫。本次沒有新增正式資料索引、模型權重、提交檔或 `outputs/` run 產物。


### A-012 Nano4 Strength V1 8-GPU short run（2026-10-17）

使用者要求執行短跑；基於 commit `8ed6d1f`，256 channels／20 blocks、BF16、SyncBN、global batch 65,536（每卡 8,192），限制 100 optimizer updates，跳過評估。只修改獨立短跑 config 的 output_dir、runtime 上限與 log 頻率，未修改模型或正式設定。

- `561669`：快取 verify 通過（1,000,000 games、221,894,008 plies、20,414,248,736 packed payload bytes、250 sampled games），正式索引產生成功；其後在 25a-hgpn062 發生 CUDA unknown error，updates=0，根因未知。
- `561693`：25a-hgpn145 CUDA 可用，但 NCCL socket 發生 No route to host，updates=0。
- `561694`：設定 `NCCL_SOCKET_IFNAME=lo`、`NCCL_IB_DISABLE=1` 後完成，Slurm COMPLETED／exit 0，elapsed 10m47s；8×H200、driver 580.65.06、torch 2.6.0+cu124。完成 100 updates、6,553,600 samples；loss 1.678584 → 0.796883，前／後 10 steps 平均 1.676225／0.832316，所有已記錄 loss 與梯度 norm 有限，未出現 OOM。
- 每步 p50 1.515966s、p95 1.517836s；排除前 10 steps 的吞吐量中位數 43,228 samples/s。rank 0 PyTorch peak allocated 120.677 GiB（不是全卡最高值，也不是 nvidia-smi reserved 顯存）。訓練程序 session 617.894s、1.373 GPU hours，未包含前兩次失敗等完整成本。
- session 估算純 training epoch 2,257.901s（37.63min），不含 shuffle、validation/calibration、checkpoint、初始化與排隊；10 epochs 純 training 約 6.27h，完整端到端時間未知。首輪全量 memmap shuffle 有明顯初始化耗時，尚未單獨精確計時。
- 唯一 checkpoint 約 286.46MB，last.json SHA-256 驗證通過；未執行 resume、完整 epoch、尾批或評估驗收。k=5／10／15／20 官方分數皆未知，loss 下降不代表驗證棋力分數提升。

資料索引：train 800,000 games／88,763,475 samples；valid 200,000／22,183,737；calibration 109,500／12,145,674。V1 game-disjoint 設計下 valid players 7,855 人全在 train 出現，overlap fraction=1.0，不是 player-disjoint 驗證。

產物：`outputs/rank_v1_short_100_20261017_135242/summary.json`、`socket.slurm`、`socket-561694.out`／`.err`、`run_socket/train.jsonl`、`run_socket/session_*.json`、`run_socket/last.json` 與 `run_socket/checkpoints/*.pt`；正式資料索引位於 `data/processed/strength_v1/seed2026/`，CV 題組位於 `data/processed/cv/strength_v1_seed2026/`。沒有提交完整 10-epoch 作業或競賽預測。

### A-013 Strength V1 compact per-seed pipeline（2026-10-18）

依使用者確認修改：seed2027／2028／2029 各自切分與訓練 1 epoch；每 100 optimizer steps 與最後一步，以 eval 模式／無 D4 完整推論 train、valid，使用全資料各 rank 平均 score 計 ranking loss。取消獨立 cv 目錄、calibration 子集與 training 校準題；processed 以 seed 為單位集中保存。切點使用全部 training 逐手 score 與標籤等權擬合。訓練成功後只留 model.pt、losses.npy、train_scores.npy、valid_scores.npy；原 seed2026 資料與 run 保留。新增 infer_rank_v1.py 與三份 seed config，預設 config 指向 seed2027。評估／繪圖讀保存分數，不重新推論；舊格式繪圖仍保留。

在 go Conda 環境使用合成資料執行 `PYTHONPATH=src python -m pytest -q`：**81 passed（34.92 秒）**。新增驗證包含：切點對照窮舉最佳解、完整 loss／逐手索引對齊、BN 推論不變、不中斷與中斷續跑的權重／loss／score 逐元素一致、評估及最終切點中斷恢復、成功後四檔清理、score hash 損壞拒絕、三張實際圖檔，以及兩個 CPU/Gloo process 對照 serial（401 samples，最後一個 process 無有效尾批樣本）。另通過 syntax compile 與 git diff whitespace 檢查。

此次未啟動官方資料準備或正式模型訓練，沒有新增正式 outputs run。新流程的 CUDA／SyncBN／8 卡與完整資料切點擬合成本尚未驗收；k=5／10／15／20 正式分數均未知。操作方式與預估成本見 README。
