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


### D-001 文獻對照計畫修訂（2026-10-05）

**性質：** 使用者要求修改計畫；僅文件修改，沒有新實驗、分數或訓練授權。同步更新 CLAUDE.md、STRATEGY.md、EXECUTION_PLAN.md。

- 主力由預設座標 Transformer 改為先做文獻對照，再比較盤面與序列；HumanSL 小樣本與 GE2E 盤面路線提前。
- Kuboki AIIDE 2025 正式版 Table 6 同一玩家 accuracy（5/10/15）為 56.9/60.0/61.6%，同段位抽樣為 68.1/81.3/88.7%。L-001 的舊版數字保留歷史，後續引用須註版本。來源：https://ojs.aaai.org/index.php/AIIDE/article/download/36812/38950/40889
- NeurIPS 正式版 Table 2，去前 15 手的主要 unseen 設定 P@1=0.860，reference/query 各 100 盤。L-001 中 0.39 與 ArcFace 不收斂的確切條件／版本本次未定位，暫不作設計依據。來源：https://papers.nips.cc/paper_files/paper/2021/file/ccf8111910291ba472b385e9c5f59099-Paper.pdf
- 不再把手級監督宣稱為獨立樣本增加 200 倍，也不以合成題數推論本地驗證必然更可信。修正 11/04 前上傳及每日收尾的過時排程。
- 各實驗數值 gate、抽樣規模與長訓練預算仍須先討論；此修訂不填造預期增益。

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


### A-014 Three-seed CPU visualizations（2026-10-18）

依使用者要求在登入節點 `25a-lgn05`，使用 go 環境、單執行緒 CPU，讀取 seed2027／2028／2029 已保存結果，各產生兩張 PNG：`outputs/rank_cnn_v1_seed{seed}/plots/loss_vs_step.png` 與 `score_distributions_cutpoints.png`，共六張。未執行模型推論或訓練。分布使用 validation 全部 200,000 盤、每級 20,000 盤，指定玩家逐手 scalar score 先取盤內平均，每盤等權；製圖程式已驗證 score／索引 hash、每盤覆蓋與手數一致。loss 圖為當步 training batch loss（train mode）與全 validation rank-mean loss（eval mode），不混接定義不同的 legacy_full_losses.json；2027／2028 各 12 個紀錄點（step 300–1355），2029 共 14 點（step 100–1355）。新增 `--skip-confusion` 選項並修正 loss 圖例，README 補重跑指令；三次製圖均 exit 0，六張 PNG 驗證通過，git diff --check 通過。本次未計算 k 桶分數。


A-014 更正（2026-10-18）：上段將橘線 valid_loss 誤記為全 validation，實際依 train_rank_v1.validation_sample_indices 為每級固定不放回抽樣 10,000 手，共 100,000 手，由各 seed 決定；每個評估 step 使用同一組樣本。已修正繪圖圖例、README 並重畫三個 seed 的 loss_vs_step.png。完整 validation 分數仍用於每盤分布圖，與 loss 抽樣範圍不同。


### A-015 Training game-mean cutpoint calibration（2026-10-18）

依使用者要求，切點由全部 training 的指定玩家逐手 scalar score 先取每盤平均，再以每盤等權官方 score 擬合有序區間；每 seed 800,000 盤，不使用 validation 標籤。登入節點單執行緒 CPU 使用既有 score 重擬合 seed2027／2028／2029，原子更新 model.pt 的切點，並逐元素確認模型權重不變，保存的 losses／scores 未改寫。訓練收尾流程亦改用每盤擬合；新增 --refit-cutpoints 供既有 run 使用。

| Seed | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 2027 | 0.312799 | 0.330387 | 0.338884 | 0.350654 | 0.333181 |
| 2028 | 0.300215 | 0.323040 | 0.352314 | 0.372288 | 0.336964 |
| 2029 | 0.301448 | 0.319964 | 0.324084 | 0.345367 | 0.322716 |

各 run 的 plots/ 保存 cutpoints_before_game_refit.json（舊逐手切點）、game_cutpoint_evaluation.json（新完整評估），並重畫 loss_vs_step.png 與 score_distributions_cutpoints.png。驗證：切點 DP 對照窮舉、完整收尾及續跑一致性共 2 passed、6 deselected（20.97s）；新增斷言確認 200 盤合成 training 的擬合筆數與官方每盤得分一致。語法解析與 git diff --check 通過。此變更不改 loss 定義、訓練權重或 split；本結果沿用 V1 game-disjoint split，非 player-disjoint 泛化分數。


### A-016 Strength V2 config and whole-game monitoring indices（2026-10-19）

依使用者要求完成 V2 實作第二步。新增 `configs/strength/rank_cnn_v2_seed2027.json`、`tasks/strength/v2/config.py` 與 `v2/prepare.py`；seed=2027、global batch=65,536、1 epoch，optimizer／scheduler／BN／SyncBN／precision／D4 沿用 V1。共用 prepare 沿用 `data.cv` 的同 seed 切分與固定 validation 題目產生器；V2 dataset 另存 `data/processed/strength_v2/seed2027/`，預定模型輸出 `outputs/v2_strength/`。

監控從 validation 每級固定無放回抽取 5,000 整盤，共 50,000 盤；保存 `monitor_games.npy` 唯一棋局清單，以及 `monitor_game.npy`／`monitor_ply.npy` 的指定玩家全部落子。manifest 保存監控設定、整盤數、總手數、每級手數及索引指紋，沿用 split 玩家重疊揭露。完成後重跑讀保存索引，不重新抽樣；改設定不覆寫既有 dataset。單盤 argmax 與多盤 score_rank 規格記於 config，評分程式仍未實作。

驗證：go Conda 環境以 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_strength_v2_prepare.py tests/test_rank_compact.py` 執行 **12 passed（26.44s）**，其中 4 項為新增測試。新增測試使用合成 metadata／顏色快取介面，涵蓋整盤平衡與固定抽樣、不同 seed、黑白全部指定玩家落子覆蓋、重跑不抽樣、設定變動及索引損壞拒絕，以及 V2／共用 V1 同 seed 切分、固定題目與完整 train／valid 索引逐位元組一致。既有 compact V1 訓練／評估／續跑回歸通過；V2 CLI help 與 git diff --check 通過。

未執行官方全量資料準備、V2 模型 forward 或正式訓練，未產生正式 V2 模型、loss 或分類結果。正式監控總手數、資料準備耗時與 GPU 小時未知；k=5／10／15／20 的 V2 分數均未知。模型／CE loss／訓練／評估／繪圖仍待後續步驟實作。


### A-017 Strength V2 ten-class model and global cross entropy（2026-10-19）

依使用者要求完成 V2 實作第三步，新增 `tasks/strength/v2/model.py` 與 `v2/losses.py`。模型直接沿用 shared backbone，末層 `Linear(256,10)` 回傳原始 B×10 logits，類別順序 D、C、B、A、1D..6D；預設參數 23,750,924。softmax 留在推論端；不載入 V1 訓練權重。CE 無類別權重、label smoothing=0、逐有效樣本等權，FP32 loss 總和除以 int64 全域有效樣本數，不要求每批涵蓋十級。mask 在 CE 前排除補齊樣本；各卡可有不同有效數或零有效數，全域零有效則拒絕。可微全域 SUM 的梯度倍數由 DDP 參數梯度平均抵消，不另加 world_size 縮放。函式回傳 differentiable loss、detached loss_sum 與 valid_count，供後續訓練／監控接線。

驗證：go Conda 環境執行 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_strength_v2_model_loss.py`：**6 passed（6.23s）**。涵蓋 B=1 輸出維度、softmax 列和、類別順序、meta device 正式模型參數計數、零 logits CE=log(10)、提高真實類別 logit 時 loss 降低、補齊樣本零梯度與非有限 padding 排除、BF16 logits 的 FP32 CE、非法標籤及全域零有效樣本拒絕；實際兩程序 CPU/Gloo DDP 線性模型驗證有效數 3:2、2:0（含實體空 batch）的全域 loss、sum、count、參數梯度與單程序對照一致，亦驗證已初始化 process group 內關閉同步的本地統計及跨卡非法資料／全空錯誤傳遞。

未啟動正式訓練或官方資料推論；完整 CNN 的 CUDA／SyncBN／多 GPU 訓練尚未驗收。V2 k=5／10／15／20 分數與正式 GPU 小時未知。第四步推論聚合及後續訓練、評估、圖表仍待實作。


### A-018 Strength V2 S4 per-game inference aggregation（2026-10-19）

依使用者 10/19 確認的 V2 下一步計畫（S4→S8；首輪改 player-disjoint，Valid-S／Valid-U，G 門檻 0.03／0.08，模型選擇與 baseline 一律看 Valid-U）完成 S4。新增 `tasks/strength/v2/aggregate.py`（純 numpy）與 `v2/infer.py`。

- `infer_games`：eval 模式、關閉 D4，每個指定玩家盤面在全部程序中恰好推論一次；softmax 與 log-softmax 以 FP32 由 logits 計算，CPU FP64 `np.add.at` 依 game_index 累加（一盤可跨 batch／跨程序），最後 all_reduce SUM，所有程序回傳相同結果；每盤手數須與索引檔一致。
- 每盤結果 `CATEGORY_DTYPE`：`game_index`、`rank_id`、`num_moves`、`confidence_sum[10]`、`log_probability_sum[10]`、`predicted_rank_id`（confidence 總和 argmax，同分取小）。相對 PLAN §7 新增 `log_probability_sum` 欄，供已授權的「log-prob 聚合」CPU 比較使用，且 validation CE 直接由它還原（逐手等權），不另做推論；confidence 改存 FP64。
- 題目聚合：`mean_probability`（正式：每盤平均機率、題內 1/k）與 `mean_log_probability`（診斷：每盤平均 log-prob 題內等權後 softmax，即正規化幾何平均）；解碼 `score_rank`（正式）／`argmax`（診斷）。

驗證：go 環境 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q tests/test_strength_v2_infer.py tests/test_strength_v2_model_loss.py tests/test_strength_v2_prepare.py`：**16 passed（18.16s）**，其中新增 6 項：同分取小、總和與平均 argmax 相同、官方計分；CE 由 log 總和逐手等權還原；長短局題內等權、`score_rank` 中段與 D／6D 邊界；log 聚合等於正規化幾何平均；合成棋局（每盤 2–4 手、eval batch 7 使盤跨 batch）逐盤 confidence／log 總和與逐手直接 forward 一致、BN／權重不變、手數與標籤正確；兩程序 Gloo（batch 4，第二程序空尾批）與單程序逐欄一致。git diff --check 通過。

未執行 GPU、官方資料推論或訓練；k=5／10／15／20 分數未知。S5 依賴的 player-disjoint 切分（Valid-S／Valid-U）尚未實作。


### A-019 Strength V2 S4b player-disjoint split with seen/unseen validation（2026-10-19）

依使用者 10/19 決定，V2 首輪不跑 game-disjoint 主 run，直接改 player-disjoint（CLAUDE.md §5），並以同一模型量「玩家記憶」落差 G = Equal-k(Valid-S) − Equal-k(Valid-U)。使用者確認：Valid-S 留出 10%；監控每級 Valid-U 2,500＋Valid-S 2,500 盤（共 50,000，成本同原設計）；G 門檻 ≤0.03／≤0.08／>0.08；模型選擇與 baseline 一律看 Valid-U，Valid-U 四桶等權 score_rank ≥0.4631 算過 baseline、≥0.47 才列提交候選。

- `data.cv.make_strength_v2_split(meta, seed, seen_holdout_fraction)`：以 `make_rank_folds(seed)` 分玩家；fold 0 → valid_unseen，fold −1 → excluded（完全不產生索引），fold 1–4 → train，但每個 (player, color) 隨機留出 floor(n×0.1) 盤為 valid_seen。assert：train∩unseen 玩家為空、excluded 不與其他重疊、seen 玩家 ⊂ train 玩家。不足 10 盤的玩家顏色組不貢獻 seen 盤，故 Valid-S 偏向重度玩家（記憶成分可能偏高估）。
- `make_strength_v1_questions` 改接受表中任何 split 名稱（V1 行為不變）。shared prepare 依 `split.policy` 分派；V1 game-disjoint 路徑輸出不變。V2 輸出 `strength_v2_games.parquet`、`strength_v2_valid_{unseen,seen}_questions.parquet`（各 10,000 題，seed+2／+3）、索引 `train`、`valid_unseen`、`valid_seen`、`monitor_unseen`、`monitor_seen`（seed／seed+1 抽樣）及 `monitor_*_games.npy`。V2 config：`split={"policy":"player_disjoint","seen_holdout_fraction":0.1}`、`monitoring.games_per_rank=2500`（每組每級）。

正式 metadata 記憶體內試算（seed 2027，未寫檔，9.3s）：

| 集合 | 玩家 | 盤數 | 指定玩家手數 | 每級盤數 min/max |
|---|---:|---:|---:|---|
| train | 5,767 | 650,823 | 72,298,918 | 63,590／67,072 |
| valid_unseen | 1,442 | 181,851 | 20,120,209 | 17,179／19,461 |
| valid_seen | 5,000 | 66,560 | 7,397,410 | 6,199／6,891 |
| monitor_unseen／seen | — | 25,000／25,000 | 2,762,965／2,783,567 | 2,500 |
| excluded（不使用） | 798 | — | — | — |

Valid-U 重疊率 0.0、Valid-S 1.0。題目各 10,000（每 k×級 250 題）；每 (k, 級) 至少涉及 66（U）／17（S）位不同玩家，Valid-S 的 k=20 題重複抽到少數重度玩家，噪音較大。train 由 88.76M（V1 game split）降為 72.30M 手 → global batch 65,536 下 1,104 steps，依 A-012 吞吐量純訓練約 27.9 min（推估）。

驗證：go 環境 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q`：**100 passed（57.12s）**（含 V1 回歸）。V2 prepare 測試改為合成 metadata（每級 10 重度＋15 輕度玩家）：切分玩家互斥／holdout 不碰／seen 留出數＝floor、與列順序無關、seed 敏感；五組索引來源與逐手覆蓋、monitor 每級平衡、題目只取自對應 split、完成後重跑不重抽、改設定拒絕；game policy 仍輸出 V1 佈局。git diff --check 通過。

未執行官方資料準備寫檔、GPU 或訓練；k=5／10／15／20 分數未知。


### A-020 Strength V2 S5/S6 trainer with seen/unseen monitors and final categories（2026-10-19）

完成 S5 訓練迴圈，原 S6（最終全量推論與 `model.pt` 收尾）併入同一入口 `python -m go_aicup.tasks.strength.v2.train --config configs/strength/rank_cnn_v2_seed2027.json`（torchrun 單節點 1–8 卡）。

- 把 V1 中與目標無關的 checkpoint 存取／latest-only 修剪／解析移到 `shared/checkpoint.py`（`code_sha256` 改由呼叫端傳入）；V1 以薄包裝沿用，行為與檔案格式不變，V1 fingerprint 清單加入新檔。V2 不從 `v1/` 匯入。
- `v2/train.py`：AdamW／warmup-cosine／BF16／SyncBN／尾批 mask／D4／定時 checkpoint／續跑沿用 V1；loss 改 `global_cross_entropy`，不再略過缺級 batch。每 100 steps（非末步）與最後一步，以凍結 BN、無 D4 完整推論 `monitor_unseen` 與 `monitor_seen` 全部指定玩家落子；`losses.npy` 每列 `[step, train_loss(最近 batch), unseen_loss, unseen_game_score, seen_loss, seen_game_score]`（CE 逐手等權、官方分單盤 argmax 按盤等權）。每次監控耗時、純訓練秒數、最終推論秒數記入 `model.pt["timings"]`，每 10 steps 的平均 step 秒數寫入 `train.jsonl`（供 S8b 判斷 15% 規則）。
- 訓練完成後完整推論 Valid-U／Valid-S 每盤 → `validation_unseen_category.npy`、`validation_seen_category.npy`（`CATEGORY_DTYPE`），再寫 `model.pt`（權重、config、類別順序、dataset／索引指紋、監控集合資訊、loss 欄名、玩家重疊、timings、環境、產物 SHA-256），reload 驗證雜湊後才清除 checkpoints／permutations／log。成功目錄只留 4 檔。

驗證：go 環境 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q`：**103 passed（55.72s）**，新增 `tests/test_strength_v2_train.py` 3 項（合成 4,000 盤、每級 10 位玩家、player-disjoint、留出 0.5）：端到端只留 4 檔、監控列為 4,8,…＋末步、`samples_seen` 等於 train 樣本數、`model.pt` reload 雜湊一致，由 `model.pt` 重推論的每盤分類與末列監控 loss／分數一致；第 5 步中斷（第 4 步監控後）續跑與不中斷的權重雜湊、`losses.npy` 與兩個分類檔逐位元組一致；兩程序 CPU/Gloo 訓練完整結束且分類檔可由 `model.pt` 重現。git diff --check 通過。

CPU 開發測試不含 SyncBN（需 CUDA）；CUDA／BF16／SyncBN／8 卡、實際吞吐與監控耗時留待 S8。未執行官方資料或 GPU；k=5／10／15／20 分數未知。


### A-021 Strength V2 S7 CPU evaluation and figures（2026-10-19）

新增 `python -m go_aicup.tasks.strength.v2.evaluate --run-dir outputs/v2_strength [--dataset-dir …] [--no-plots]`（實作位於 `tasks/strength/v2/evaluate.py`，取代 PLAN §8.1 預定的 `evaluate_rank_v2.py` 路徑）。只讀 `model.pt`、`losses.npy`、兩個分類檔與該 run 的 dataset 表／固定題目；不 forward、不改產物，報表印在終端，圖存 `<run-dir>/plot/`。

- 每個驗證集（Valid-U、Valid-S）：單盤（confidence 總和 argmax；診斷）整體 score／accuracy／差一級／差兩級以上、各級、黑白分組、10×10 混淆；題目 k=5／10／15／20 的 n、score、acc、within1 與四桶等權 Equal-k（不顯示 MAE），組合為 {mean_probability（正式）, mean_log_probability（診斷）} × {score_rank（正式）, argmax（診斷）}。任何 k 桶缺題即拒絕產出 Equal-k。
- 判定：G = Equal-k(S) − Equal-k(U)（正式規則），≤0.03／≤0.08／>0.08 三段文字；Valid-U ≥0.4631 標 passes_baseline，≥0.47 標 submission_candidate。不依 validation 選解碼或聚合。
- 10 張 PNG（200 dpi、Agg、不 show）：training_curves、game_confusion_{unseen,seen}、game_confidence_{unseen,seen}、game_by_rank、question_by_k（含 baseline 參考線）、question_confusion_{score_rank,argmax}（Valid-U，2×2 k）、question_aggregation（機率平均 vs log-prob）。配色 Valid-U 藍、Valid-S 橘、training 綠；診斷系列加斜線紋理，不只靠顏色區分。

驗證：go 環境全套 `pytest -q`：**104 passed（71.49s）**。新增測試以獨立迴圈重算正式規則（每盤平均機率 → 題內 1/k → `p[r]+(p[r−1]+p[r+1])/e` argmax → 官方分 → 四桶平均），與報表 Equal-k 差 ≤1e-12；單盤分數、混淆總數、G 與 baseline 判定一致；產生 10 張圖、評估不改 run 內任何檔案、重跑結果相同。已目視檢查合成資料的 3 張圖版面。git diff --check 通過。

未使用官方資料或 GPU；k=5／10／15／20 分數未知。


### A-022 Strength V2 S8 GPU short runs on Nano4 dev（2026-10-19，S8b 未完成）

使用者授權 S8 以 `dev` 佇列約 3 GPU-h。腳本 `slurm/v2_s8a_prepare.slurm`、`slurm/v2_s8_cuda_tests.slurm`、`slurm/v2_s8b.slurm`；設定 `configs/strength/rank_cnn_v2_seed2027_s8{a,b}.json`（只改 output_dir、batch、監控間隔）。原提案 8 卡做中斷／續跑比對（3 次啟動）估 4.3 GPU-h 超出授權，改為 2 卡合成 CUDA 測試驗續跑、8 卡只跑一次。

- 576385／576386：分配到 25a-hgpn001，0 秒以 signal 53 結束、無 stdout；探針作業 576392 在 25a-hgpn014 正常 → 腳本排除 25a-hgpn001（與既有 25a-hgpn069）。
- **S8c 576394（2×H200，driver 580.65.06，1m01s）通過：** `tests/test_strength_v2_cuda.py` 3 passed：FP32 SyncBN 1 卡 vs 2 卡（含非整除尾批）每列 loss／監控與兩個分類檔在 rtol 2e-3 內一致；BF16 2 卡分類檔可由 `model.pt` 單卡重推論重現；BF16 2 卡第 5 步中斷續跑與不中斷 **權重逐位元相同**。
- **S8a 576393（1×H200，16m47s）通過：** torch 2.6.0+cu124、CUDA 12.4、NCCL 2.21.5。V2 正式資料準備 26 s，`data/processed/strength_v2/seed2027/` 各集合筆數與 A-019 試算完全一致。1 卡 BF16、global batch 2,048、20 steps：loss 2.308→2.304、grad 有限；step 10／20 兩次完整監控（U 25,000 盤 2,762,965 手＋S 25,000 盤 2,783,567 手）各 **324 s**，約 17.1k moves/s／GPU。
- **S8b 576395（8×H200，25a-hgpn039）失敗，32m50s，updates=0：** rank 1–7 在第 9 個 NCCL broadcast 等 rank 0 達 30 分鐘逾時（SIGABRT）。rank 0 正在 `create_permutation` 以 memmap 原地 shuffle 72,298,918 筆 uint32；`/work` 為 WEKA 網路檔案系統，隨機寫入在被終止時仍未完成（.tmp 最後寫入 23:31）。S8a 同一步驟在 hgpn014 僅數分鐘，耗時受 I/O 狀況影響大。浪費約 4.4 GPU-h。
- **修正：** `shared.dataset.create_permutation` 改為記憶體內 `np.arange`＋同一 PCG64 shuffle 後一次寫出（約 289 MB RAM）；新增測試確認與舊 memmap 版逐元素相同（V1 排列不變）。全套 **105 passed、3 skipped（CUDA）**。

S8 GPU 用量約：S8a 0.28＋S8c 0.03＋S8b 4.38 ≈ **4.7 GPU-h**，已超過授權。8 卡吞吐、8 卡單次監控耗時與 15% 規則判定仍未知；V2 正式分數未知。`outputs/v2_s8b/` 為失敗殘留（無 checkpoint）。


### A-023 Strength V2 S8b 8-GPU timing; monitor interval 100 → 200（2026-10-20）

A-022 修正後重跑 **S8b 584688（8×H200，25a-hgpn030，driver 580.65.06）通過，5m02s（≈0.67 GPU-h）**：正式設定（global batch 65,536、每卡 8,192、BF16、SyncBN、D4），`--max-steps 100`，step 50／100 監控。排列建立已不再卡住（整次啟動 298 s）。

| 量測 | 數值 |
|---|---|
| 每步（第 11–100 步，每 10 步平均） | 1.505–1.506 s（第 1–10 步 3.21 s）→ 約 43.5k samples/s |
| 100 steps 純訓練（`timings.train_seconds`） | 151.05 s |
| 單次監控（U 2,762,965＋S 2,783,567 手） | 48.80 s／48.79 s → 約 113.7k moves/s |
| 監控／100 steps 訓練 | **32.4% > 15%** |

依使用者規則（>15% 改每 200 steps）把 `configs/strength/rank_cnn_v2_seed2027.json` 的 `evaluate_every_steps` 改為 200；此時約 48.8／301 = **16.2%，仍略高於 15%**，已回報。

**觀察（原因未知）：** step 50 → 100 的 training batch CE 2.269 → 2.193 下降，但 eval 模式監控 CE 由 2.299 升至 2.849／2.853（高於均勻 log 10 = 2.303），單盤分數 0.1368＝十級平衡下全部預測同一端點級（D 或 6D）的值，即 eval 預測塌縮。step 100 剛過 warmup 峰值（56 步）；可能是 BN running statistics 尚未穩定，也可能是真實問題，正式 run 的後續監控點會顯示。此為 100 步短跑，不代表最終分數。

**正式 1 epoch run 實測估時（8×H200，每 200 steps 監控）：** 1,104 steps × 1.506 s ≈ 27.7 min；監控 6 次（200…1000＋末步 1104）× 48.8 s ≈ 4.9 min；最終完整推論 Valid-U 20,120,209＋Valid-S 7,397,410 手 ÷ 113.7k/s ≈ 4.0 min；初始化／checkpoint／收尾約 2–3 min（推估）→ **約 39–40 min wall、約 5.3 GPU-h**。資料已準備完成，不需再跑 prepare。腳本 `slurm/v2_train.slurm`（`8gpus`、1h30m 時限、可續跑、完成後自動 CPU 評估與作圖），**尚未提交，等待使用者確認**。

S8 累計 GPU：A-022 約 4.7＋本次 0.67 ≈ **5.4 GPU-h**。V2 正式分數未知。


### A-024 Strength V2 first full run, player-disjoint seed2027（2026-10-20）

使用者確認後提交 **584751（8gpus，8×H200，25a-hgpn084）COMPLETED，37m22s**；1,104 steps、72,298,918 樣本、1 epoch；`model.pt` 記錄 gpu_seconds 17,571（≈4.9 GPU-h；含評估的作業總計 ≈5.0 GPU-h）。純訓練 1,649.5 s；6 次監控各 48.9–49.9 s；最終推論 Valid-U 150.5 s、Valid-S 58.0 s。程式基於 commit 534a06a 加未提交修改（git diff 存於 `model.pt["environment"]`）。產物：`outputs/v2_strength/{model.pt, losses.npy, validation_unseen_category.npy, validation_seen_category.npy, plot/}`。

**監控（每 200 steps；單盤 argmax 診斷分數）：**

| step | train batch CE | U CE | U 單盤分 | S CE | S 單盤分 |
|---:|---:|---:|---:|---:|---:|
| 200 | 2.1329 | 2.5388 | 0.1519 | 2.5277 | 0.1538 |
| 400 | 2.0674 | 2.0908 | 0.3276 | 2.0797 | 0.3411 |
| 600 | 2.0248 | 2.1559 | 0.2921 | 2.1373 | 0.3074 |
| 800 | 1.9969 | 2.0750 | 0.3320 | 2.0510 | 0.3517 |
| 1000 | 1.9832 | 2.0391 | 0.3557 | 2.0148 | 0.3741 |
| 1104 | 1.9725 | 2.0383 | 0.3539 | 2.0137 | 0.3726 |

A-023 step 100 的 eval 預測塌縮在 step 400 已消失；step 600 有一次回升，原因未知。

**題目分數（正式：每盤平均機率、題內 1/k、score_rank；各 2,500 題／桶）：**

| 集合 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| **Valid-U（未見玩家）** | 0.4643 | 0.4918 | 0.4950 | 0.5129 | **0.4910** |
| Valid-S（見過玩家） | 0.4737 | 0.5271 | 0.5322 | 0.5363 | 0.5173 |

- **G = 0.5173 − 0.4910 = 0.0263 ≤ 0.03** → 玩家記憶成分小，維持此切分、以 U 報告（Valid-S 偏重度玩家，G 可能偏高估，見 A-019）。
- **Valid-U 0.4910 ≥ 0.4631（過 baseline）且 ≥ 0.47 → 依使用者門檻列為提交候選。** 本地合成題目分數，不是 LB；測試玩家是否在訓練集（L10）、測試級別是否平衡仍未知。
- 單盤（診斷）：U 0.3502／acc 0.2372，S 0.3754。各級 U 單盤分：D 0.651、6D 0.683 最高；1D 0.215、2D 0.168 最低；預測偏向兩端（D 29,000、6D 34,218 盤 vs 2D 10,694）。黑／白 0.3497／0.3507。逐盤平均機率很平（每列最大約 0.25，見 `plot/game_confidence_unseen.png`）。

**已授權的 CPU 聚合比較（診斷，未依此改正式規則）：**

| Valid-U Equal-k | score_rank | argmax |
|---|---:|---:|
| 機率平均（正式） | **0.4910** | 0.4698 |
| log-prob 平均 | 0.5195 | 0.5502 |

Valid-S 同順序：0.5173／0.4969／0.5528／0.5941。log-prob 平均（每盤平均 log p 題內等權後 softmax）在 U 各桶都較高（score_rank：+0.015／+0.020／+0.038／+0.040，k 越大差越多），且搭配 argmax 再高 +0.031。機制推測（未驗證）：每手機率偏平，機率平均讓證據難以隨盤數累積；log 平均相當於乘積，k 越大越尖銳。score_rank 假設機率已校準，log 聚合後的機率尺度未經校準，score_rank 與 argmax 的高低因此不可解讀成「score_rank 不好」。是否改用 log 聚合或做溫度校準，需另立單變數實驗且不能以 Valid-U 挑選後仍用 Valid-U 報分（可用 Valid-S 或 training 題擬合）。

評估修正：混淆／機率熱圖的文字顏色門檻改依色階上限（比例圖 vmax=1），避免淺色格白字；只重畫圖，數值不變。`tests/test_strength_v2_train.py` 4 passed。


### A-025 V2 question aggregation / temperature experiment（2026-10-20，CPU）

使用者要求執行 A-024 提出的聚合實驗。預先登記設計後才執行：唯一變數＝題目聚合／校準規則，解碼固定 `score_rank`；只讀 A-024 已存每盤結果，無 GPU。Valid-S 依玩家（seed 2027）分兩半：S-fit（906 位玩家、5,000 題）只擬合溫度，S-select（906 位、5,000 題）只選規則；Valid-U（10,000 題）不參與擬合或選擇。溫度：每個 k 一個 T_k，p = softmax(T_k × 題目 logits)，在 [0.1, 100] 對數網格（601 點）上最小化題目 NLL。Gate：S-select Equal-k 最佳者須比 C0 高 ≥0.01，且 T_k 不在網格邊界。揭露：C1 的 U 分數已在 A-024 看過，不是盲測。

| 候選 | 規則 | T_5／T_10／T_15／T_20 | S-select Equal-k | **U k=5／10／15／20** | **U Equal-k** |
|---|---|---|---:|---|---:|
| C0 | 機率平均，T=1（現行正式） | 1／1／1／1 | 0.5151 | 0.4643／0.4918／0.4950／0.5129 | 0.4910 |
| C1 | log-prob 平均，T=1 | 1／1／1／1 | 0.5510 | 0.4790／0.5121／0.5334／0.5533 | 0.5195 |
| **C2** | **log-prob 平均，擬合 T_k** | **4.42／6.24／7.67／8.13** | **0.5999** | **0.5030／0.5501／0.5716／0.5965** | **0.5553** |
| C3 | 機率平均，擬合 T_k | 3.89／4.84／5.31／5.43 | 0.5359 | 0.4681／0.4913／0.4933／0.5061 | 0.4897 |

- **Gate 通過：** S-select 最佳為 C2，比 C0 高 +0.0848；T_k 皆不在網格邊界。C2 在 U 的 Equal-k 為 0.5553，比 C0 高 +0.0643，四個 k 桶全部提升（+0.039／+0.058／+0.077／+0.084）。
- **C3 對照：** 只校準機率平均時，S-select +0.021，但 U −0.001。所以增益主要來自 log 聚合（乘積式累積證據），不是單純把分佈變尖；C3 在 S 與 U 上不一致，原因未知。
- **機制解讀（推測，未驗證）：** T_k／k = 0.88／0.62／0.51／0.41。若各盤證據獨立，log 聚合的理想溫度應接近 k（各盤 log p 相加）。實測遠小於 k，且比例隨 k 下降，表示同一玩家各盤的模型誤差高度相關，多給盤數的邊際資訊遞減。
- **尚未採用為正式規則：** 依 CLAUDE.md，正式規則（機率平均＋score_rank）是使用者確認的。gate 由我預先登記，改用 C2 須使用者確認。若採用，測試推論只需每盤的 log-prob 總和與手數，加上這 4 個溫度（存於 `outputs/v2_agg_calibration/result.json`）。
- **限制：** 溫度擬合在 Valid-S（見過的玩家），U 證實可轉移（G 小，A-024）。但測試題的 k 分佈、級別分佈與玩家是否在訓練集內都未知；本地合成題目分數不是 LB。

程式：`aggregate.question_logits`／`tempered_probabilities`／`fit_temperature`（T=1 時與原本的機率平均相同）；`scripts/v2_aggregation_calibration.py`。新增測試涵蓋 T=1 恆等與已知溫度回收（真值 3.0，誤差 <8%）、網格邊界旗標。全套 **106 passed、3 skipped（CUDA）**；git diff --check 通過。


### A-026 V2 3-epoch run 中斷與續跑；重排進度（2026-10-20）

- **事故：** 使用者提交的 `slurm/v2_train.slurm`（seed2026、3 epochs、8×H200）作業 589164 於 step 400 存 checkpoint 時失敗：`FileNotFoundError: src/go_aicup/tasks/strength/shared/checkpoint.py`。原因是作業執行中，共用工作目錄由 `strength-v2-player-disjoint` 切回 `main`（`main` 尚無該檔）。Slurm 作業直接 import 該目錄的程式碼。損失約 0.15 h × 8 GPU。
- **處置：** 本地 `main` 快轉（`--ff-only`）到 `strength-v2-player-disjoint`（96342f8），未 push。由 step 200 的 checkpoint 續跑，作業 **590334**（25a-hgpn083）。續跑後 step 210–290 的 batch loss 與失敗作業逐位元相同（例：step 270 皆為 2.163703441619873）。此節點每步約 2.14 s（A-023 為 1.506 s，原因未知）；預估剩餘約 2h10m，接近腳本 2h30m 時限。若超時，可由每 900 s 的 checkpoint 再續跑。step 400 的監控在這個節點耗時 152.8 s（A-024 為 49 s），重估後會超過時限；`scontrol` 無法延長時限，因此另排 **590565**（`--dependency=afternotok:590334`，同一腳本自動續跑），只在 590334 失敗或超時時才啟動。
- step 400 監控（單盤 argmax 診斷）：U CE 2.2331／分數 0.2416，S 2.2175／0.2518。A-024 1-epoch run 同一步為 U 2.0908／0.3276；本次 seed（2026 vs 2027）與學習率設定（固定 0.001 vs warmup-cosine）都不同，兩者不可直接比較。
- **流程規則（寫入 EXECUTION_PLAN）：** 作業執行中不得在 `/work/jason1122/go_aicup_2026` 切換分支；需要其他分支時用 `git worktree`。
- `EXECUTION_PLAN.md` 頂端新增「2026-10-20 重排進度」（R1–R6），取代原 W5–W8 時間軸。砍掉：競賽二 GBDT、Transformer 預訓／scaling、KataGo、家族 B。


### A-027 競賽一開局模型與 L6 後處理（2026-10-20，CPU，合成候選庫）

**預先登記：** 所有超參（黑棋手數 D、α、β、溫度 τ）只依 split 0 的全體官方分數選擇；split 1、2（與 0 互斥）只用於報分。人口先驗只用該候選庫以外玩家的黑棋。合成候選庫由 `cv.make_pid_splits(seed 2026)` 依 board cache 的 `games.parquet` 重新產生（演算法同 A-003，但 meta 列順序不同，所以玩家組與 A-003 不同）。登入節點 CPU（4 執行緒），全程約 4 分鐘。

**模型**（`tasks/identity/opening.py`）：
- `slot`：逐手獨立 NB，即 E5 的方法。
- `pp`：前綴序列模型 P_c(m_t | m_<t) = (N_c(m_≤t) + β·P_pop(m_t | m_<t)) / (N_c(m_<t) + β)。N_c 是候選人 gallery 中前綴完全相同的盤數（uint64 前綴雜湊計數）；P_pop 是庫外人口的同前綴條件分布，再退回人口逐手分布。歷史包含白棋，只對黑棋計分。
- 第一版以「候選人逐手分布」作退回（split 0，D=16、β=1）只得 0.851，低於 slot。原因：離開候選人走過的路線後，深層手的雜訊被等權加總，與 E5「手數越多越差」同一機制。改成人口條件退回後，沒走過該路線的候選人在每一手的分數都相同，排序時互相抵消。
- `assign.py`（L6）：逐題 softmax(score/τ) → Sinkhorn（列、行交替正規化 50 次）→ top-5；或在 Sinkhorn 後做 Hungarian 一對一指派，指派結果放第 1 名，第 2–5 名依 Sinkhorn 排序。

**split 0 選出：** slot D=4、α=5（0.9255）；pp D=12、β=1（0.9441；D=8 同分，D≥8 後已無資訊）；pp+slot：pp D=8、β=0.3，加上 slot D=4（0.9684）；Sinkhorn τ=2；Hungarian τ=0.25（**位於網格邊界**，但 τ=0.25–2 的分數完全相同，對 τ 不敏感）。

**官方分數，split 1–2 平均（各 400 題，k 各 100）：**

| 方法 | k=5 | k=10 | k=15 | k=20 | 全體 | split 1／2 |
|---|---:|---:|---:|---:|---:|---|
| slot D=4（E5 重現） | 0.8096 | 0.9450 | 0.9793 | 0.9874 | 0.9303 | 0.9216／0.9390 |
| pp | 0.8603 | 0.9513 | 0.9855 | 0.9887 | 0.9465 | 0.9452／0.9478 |
| **pp+slot** | **0.8981** | 0.9740 | 0.9968 | 1.0000 | **0.9672** | 0.9624／0.9720 |
| pp+slot＋Sinkhorn | 0.9918 | 0.9937 | 1.0000 | 1.0000 | 0.9964 | 0.9968／0.9959 |
| pp+slot＋Hungarian | 0.9907 | 1.0000 | 1.0000 | 1.0000 | 0.9977 | 1.0000／0.9953 |

- pp+slot 相對 E5：兩個 split 分別 +0.041／+0.033，k=5 +0.089。增益主要在 k=5 桶。
- L6 再 +0.03。**前提是題目與候選人一對一**：合成庫依建構必然一對一，真實測試未確認（RESULTS 待辦）。若不是一對一，Hungarian 可能有害。11/04 後須以一次 LB 受控探針驗證。
- **合成驗證已飽和（≥0.99），之後無法再用它區分開局方法。** 官方 baseline 0.2845 與合成分數的落差仍無解釋。線索（未驗證）：E5 表中「只用第 1 手」的 top-5 為 0.30–0.43，與官方 baseline 同一量級，所以官方 baseline 可能只是很弱的方法，也可能真實測試確實比較難。R2 的非開局風格模型就是為後者準備的保險。

程式：`tasks/identity/{data,opening,assign}.py`、`scripts/identity_opening_eval.py`；產物 `outputs/pid_opening_v1/{results.csv,summary.json}`、`data/processed/identity/{train_prefix_60.npy,pid_gallery.parquet,pid_questions.parquet}`。測試 `tests/test_identity_opening.py` 4 passed（逐手 NB 對照直接計數、前綴雜湊順序／長度、人口退回在未匹配候選人上抵消、Sinkhorn 雙隨機與 Hungarian 拆解 hub）。尚未有測試集推論入口；短局（<D 手）與 pass 的處理留待 R3 提交管線。


### A-028 V2 3-epoch run（seed2026）完成結果（2026-10-21 補記）

- **作業：** 590334 於 2h30m 時限 TIMEOUT；預排的 590565（`afternotok`）自 checkpoint 續跑，14 分 27 秒後 COMPLETED（10/21 01:13）。產物在 `outputs/v2_strength_seed2026/`（`model.pt`、`losses.npy`、`validation_{seen,unseen}_category.npy`、`plot/`），報表在 `outputs/slurm-v2-train-590565.out`。
- **與 A-024 不可直接歸因：** seed 不同（2026 vs 2027），學習率不同（固定 0.001 vs warmup-cosine），epoch 數也不同（3 vs 1）。三者同時改變，所以下面的差異不能只歸因於 epoch 數。

**Valid-U 正式規則（機率平均 + `score_rank`）：**

| run | k=5 | k=10 | k=15 | k=20 | Equal-k | Valid-S Equal-k | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-024（1 epoch，seed2027） | 0.4643 | 0.4918 | 0.4950 | 0.5129 | **0.4910** | 0.5173 | 0.026 |
| A-028（3 epochs，seed2026） | 0.4717 | 0.4924 | 0.4980 | 0.4966 | **0.4897** | 0.5487 | **0.059** |

**Valid-U 聚合診斷（Equal-k）：**

| | 機率平均 score_rank | 機率平均 argmax | log-prob score_rank | log-prob argmax |
|---|---:|---:|---:|---:|
| A-024 | 0.4910 | 0.4698 | 0.5195 | 0.5502 |
| A-028 | 0.4897 | 0.4913 | 0.4816 | 0.5058 |

- 單盤診斷（Valid-U，178,907 盤）：score 0.3605、acc 0.2454。
- **判讀：** U 的正式分數沒有提升（−0.0013）；S 的分數 +0.031，G 從 0.026 升到 0.059，進入「注意」區間（≤0.08 時仍以 U 判斷）。這符合「多訓練時記住見過的玩家，但沒有學到更多棋力」的機制，但因為有上述三個混淆變數，這只是推測，尚未驗證。log-prob 聚合在 A-024 的大幅增益（A-025）在這次 run 縮小很多：score_rank 甚至低於機率平均。原因未知。可能的解釋是多 epoch 後每手機率變得較尖、較不校準，但沒有驗證。
- **處置：** 正式候選維持 A-024 的模型。A-028 不取代 A-024；它也過 baseline 與 0.47 門檻，可作為 ensemble 的元件候選。


### E1-001 手工特徵 + LightGBM（本機，2026-10-19）

**性質：** EXECUTION_PLAN E1，本機 CPU，約 3 分鐘；不佔上傳額度，不是提交候選。用來在 V2 結果出來前，先確認有沒有便宜的路線能過 baseline 0.4631。
**設定：** 與 A-006 線性探針相同：26 維 me_/op_ 統計（實際 47 欄 + length）、`rank_questions` 題組；評估集 f4 + holdout 共 14,000 題，與訓練玩家零重疊（程式內 assert）。一律用 `score_rank` 解碼。程式 `python -m go_aicup.tasks.strength.gbdt_e1`，輸出 `outputs/rank_gbdt_e1/`（report.csv、summary.json、各變體機率與模型）。

| 變體（相對前一列只改一個變數） | k=5 | k=10 | k=15 | k=20 | all |
|---|---|---|---|---|---|
| lr（f0–f3 題均值，重現 A-006） | 0.350 | 0.398 | 0.440 | 0.459 | **0.412** |
| gbdt（換成 LightGBM；f0–f2 訓練、f3 early stop） | 0.333 | 0.383 | 0.406 | 0.431 | 0.388 |
| gbdt_std（+ 題內 std 特徵） | 0.330 | 0.387 | 0.405 | 0.430 | 0.388 |
| gbdt_game（訓練單位改為單盤：fold 0–2 玩家的盤，題 = k 盤 log-prob 平均） | 0.351 | 0.389 | 0.400 | 0.427 | 0.392 |

- LR 逐位重現 0.4117 → 管線無誤。
- 三個 GBDT 變體在 all 都比 LR 低 0.02，k=15/20 低 0.03–0.04。只有 gbdt_game 在 k=5 打平 LR（0.351 vs 0.350）。
- 注意：gbdt 少用了 f3（拿去 early stop），訓練題數是 LR 的 75%，比較對 GBDT 略不利；但差距 0.024 不太可能全部來自這裡（未驗證）。
- 機制推測（未驗證）：題級訓練樣本只有約 4.3k 位玩家，非線性模型容易學到玩家特有的組合；線性模型較平滑。單盤訓練的樣本雖多，但同玩家的盤高度相關。
- **結論：** 在這組手工特徵上，換非線性模型沒有增益，E1 這條線過不了 0.4631。瓶頸在單盤表徵，不在分類器；保險線要改從表徵下手（例如 HumanSL 特徵），不再在 GBDT 超參上花時間。


### E0-001 競賽一合成驗證強健性（本機，2026-10-19）

**性質：** EXECUTION_PLAN E0，本機 CPU；評估實驗，不訓練、不佔上傳額度。基線為 A-005/A-006 的黑前 4 手 naive Bayes，在 3 組互斥合成候選庫（split 0–2，各 400 人 × 100 盤）上評估，分數取 3 組平均。程式 `python scripts/e0_pid_robustness.py`，輸出 `outputs/pid_e0_robustness/`。
**壓力測試的性質：** 只對查詢盤做隨機 D4 旋轉／鏡射，候選庫不動，seed 2026。這是假設情境，**不代表真實測試集有做方位打亂**（未知）。

| 變體 | k=5 | k=10 | k=15 | k=20 | all（組間 sd） |
|---|---|---|---|---|---|
| raw／不擾動（重現 A-006） | 0.809 | 0.939 | 0.977 | 0.989 | **0.929**（0.009） |
| raw／每題一個隨機方位 | 0.184 | 0.132 | 0.163 | 0.159 | 0.160（0.007） |
| raw／每盤各自隨機方位 | 0.096 | 0.092 | 0.087 | 0.088 | 0.091（0.004） |
| D4 正規化（兩側都做）／不擾動 | 0.757 | 0.917 | 0.968 | 0.988 | 0.907（0.013） |
| D4 正規化／每盤各自隨機方位 | 0.757 | 0.917 | 0.968 | 0.988 | 0.907（0.013） |

依真實玩家在候選庫中「最常用第一手佔比」分層（raw／不擾動，3 組合計 1,200 題）：

| 第一手集中度 | k=5 | k=10 | k=15 | k=20 | all | 題數 |
|---|---|---|---|---|---|---|
| < 0.60 | 0.771 | 0.868 | 0.947 | 0.988 | 0.897 | 196 |
| 0.60–0.80 | 0.743 | 0.945 | 0.954 | 0.988 | 0.902 | 192 |
| 0.80–0.95 | 0.836 | 0.901 | 1.000 | 0.975 | 0.928 | 224 |
| ≥ 0.95 | 0.839 | 0.971 | 0.984 | 0.996 | 0.948 | 588 |

- **方位是單一最大的脆弱點：** 查詢盤一旦被旋轉，raw NB 從 0.929 崩到 0.09–0.16，**低於官方 baseline 0.2845**。D4 正規化完全不受方位擾動影響（數學上必然），代價只有 all −0.022，主要集中在 k=5（−0.052）。
- **習慣集中度不是主要解釋：** 集中度最低一層仍有 0.897，與最高層只差 0.05。但合成庫全是 ≥120 盤黑棋的重度玩家，真實候選人的分布仍然未知。這只削弱了 CLAUDE.md §0 的假設 (c)，沒有排除它。
- 「每題一個方位」的 0.16 跟官方 baseline 0.28 在同一個量級，但這只是巧合程度的線索，**不能當作測試集有旋轉的證據**。
- **決定（2026-10-19 使用者確認）：** 在 11/04 稽核真實資料前，競賽一的預設採用 D4 正規化：用 −0.022 換掉可能高達 −0.8 的下行風險。11/04 時，先比較真實查詢盤與候選庫的第一手落點分布；若方位一致，再改回 raw。


### A-029 競賽一開局模型改 D4 正規化（2026-10-21，CPU，合成候選庫）

**目的：** E0-001 已決定競賽一預設採 D4 正規化，但 A-027 是用 raw 方位寫的。這次只改一個變數：方位處理（raw → D4 正規化）。其餘協定與 A-027 相同：超參網格相同，只在 split 0 選超參，split 1、2 只用來報分；合成庫與人口先驗也相同。登入節點 CPU，約 5 分鐘。

**正規化方式：** 每盤取 8 種對稱中，**整段 60 手前綴**字典序最小的那一個（`data.canonical_d4`）。候選庫、查詢與人口先驗都經過同一轉換。E0-001 的 `canonical_d4` 只用前 6 手當鍵，一旦前 6 手本身有對稱（例如都在對角線上），後面的手就不保證旋轉不變。但開局模型會用到第 16 手（黑 8 手）以後，所以這次改用整段前綴。這樣的正規化有兩個性質：(1) 對任何 D4 轉換嚴格不變；(2) 前 t 手相同的兩盤，正規化後前 t 手仍然相同，所以前綴比對照常成立。跑分前，先把 25,000 盤查詢盤各自隨機旋轉，再正規化，結果與原本完全相同（腳本內 assert）。測試新增 1 項，`tests/test_identity_opening.py` 共 5 passed。

**split 0 選出：** slot D=4、α=5；pp D=12、β=1；pp+slot：pp D=8、β=0.3，加上 slot D=4（三者都與 A-027 相同）。L6 的溫度 τ 都選到 **0.25（網格下界）**：Sinkhorn 在 τ=0.25 與 0.5 的分數同為 1.0；Hungarian 在 τ=0.25–32 全部是 1.0，對 τ 不敏感。

**官方分數，split 1–2 平均（括號內為與 A-027 raw 的差）：**

| 方法 | k=5 | k=10 | k=15 | k=20 | 全體 | split 1／2 |
|---|---:|---:|---:|---:|---:|---|
| slot D=4（E5） | 0.7645 | 0.9322 | 0.9730 | 0.9894 | 0.9148（−0.016） | 0.9146／0.9149 |
| pp | 0.7911 | 0.9308 | 0.9659 | 0.9855 | 0.9183（−0.028） | 0.9106／0.9260 |
| **pp+slot** | **0.8379** | 0.9710 | 0.9887 | 1.0000 | **0.9494**（−0.018） | 0.9458／0.9530 |
| pp+slot＋Sinkhorn | 0.9442 | 0.9894 | 0.9951 | 1.0000 | 0.9822（−0.014） | 0.9730／0.9913 |
| pp+slot＋Hungarian | 0.9505 | 0.9862 | 0.9951 | 1.0000 | 0.9829（−0.015） | 0.9746／0.9913 |

- **D4 的代價：** 全體約 −0.014～−0.028，集中在 k=5（pp+slot −0.060）；k=20 不受影響。slot 的代價（−0.016）與 E0-001（−0.022）在同一量級，兩者的 split 集合與正規化鍵不同，所以不完全一樣。機制推測（未驗證）：每位玩家「固定在哪個角落起手」的習慣也是指紋，D4 正規化把這部分抹掉了；盤數少的時候，這個訊號比較重要。
- **增益在 D4 下仍成立：** pp+slot 比 slot +0.035（raw 時為 +0.037）；L6 再 +0.033。所以 A-027 的方法增益不依賴方位。
- **處置：** 依 E0-001，pp+slot＋L6 的 D4 版成為競賽一的預設候選；raw 版保留，等 11/04 確認真實資料的方位分布後再比較。L6 依賴「題目與候選人一對一」的前提，這點仍未確認（同 A-027）。合成驗證仍接近飽和（≥0.98），無法再用來分辨更細的差異。

程式：`data.{d4_tables,canonical_d4,opening_prefix_d4}`；`scripts/identity_opening_eval.py --orient d4`（預設 raw，與 A-027 相同）。產物：`outputs/pid_opening_v1_d4/{results.csv,summary.json}`、`outputs/pid_opening_v1_d4.log`、`data/processed/identity/train_prefix_60_d4.npy`。


### D-002 計畫統整與決策（2026-10-22，使用者確認）

**性質：** 計畫修訂，不是新實驗，沒有新的模型分數。EXECUTION_PLAN.md 改寫為統整版（T1–T6），取代 10/05 的 E0–E6（D-001）與 10/20 的 R1–R6（A-026）。同步更新 CLAUDE.md（§0 baseline 表、V2 正式解碼段落、§7 路線圖、§11 第 5–6 項）與 STRATEGY.md §1。

**使用者確認的決策（全部採建議選項）：**
- **D1：** 競賽二正式解碼改用 C2（log-prob 平均＋每 k 溫度＋`score_rank`），取代 10/19 的機率平均規則。A-024 模型的 Valid-U Equal-k：0.4910 → 0.5553。測試題的 k 分布未知，T4 用一次 LB 探針確認。
- **D2：** 競賽一保險模型沿用 V2 盤面 ResNet 當逐手 encoder → 盤內平均 → GE2E。D-001 E2 的序列 Transformer 版留到 11/04 後再決定。
- **D3：** HumanSL／KataGo 特徵延到管線完成後，只當 ensemble 元件。
- **D4：** 之後的 V2 run 固定切分 seed 2027，只改初始化 seed。理由（本次新查到）：`make_strength_v2_split` → `make_rank_folds(seed)`，連 fold −1 的 holdout 也依 seed 重抽。所以 A-028（seed2026）的訓練集可能含 A-024 的 Valid-U 與 holdout 玩家，A-028 不能當 ensemble 元件；單獨評估 A-024 不受影響。
- **D5：** 依子目錄分工：一人負責競賽一保險模型，一人負責兩賽提交管線與 Day-0 稽核。具體人選待與 jb 確認。
- **T2 gate：** 競賽一保險模型在「去掉前 20 手」的合成題上，split 1–2 平均 ≥0.2845，且與開局模型合併後，不去開局的分數不下降。競賽二 ensemble 在共同 Valid-U 上比單模型高 ≥0.005，且以玩家為群組的 bootstrap 95% 區間下界 >0。

**A-025 產物補建：** A-025 記載的 `outputs/v2_agg_calibration/result.json` 不在工作目錄中（原因未知）。用同一腳本 `scripts/v2_aggregation_calibration.py --run-dir outputs/v2_strength` 在登入節點 CPU 重跑，結果與 A-025 逐位一致：T_k＝4.4157／6.2373／7.6736／8.1283；C2 的 U＝0.5030／0.5501／0.5716／0.5965，Equal-k 0.5553。log 在 `outputs/v2_agg_calibration.log`。


### A-030 T1 提交管線與假測試檔彩排（2026-10-22，CPU；競賽二 GPU 步驟待跑）

**性質：** EXECUTION_PLAN T1／T1b，依 D-002 已確認的 gate 執行；不訓練、不佔上傳額度。

**新增程式：**
- `sgf.parse_lenient`：只給官方 release 檔用。逐一讀 `;B[..]`／`;W[..]`，pass（`[]`、`[tt]`）丟棄並計數，setup 石／讓子記為 error 但仍回傳落子；訓練集仍走嚴格 `parse`。`preprocess release` 新增 `--raw-dir`／`--out-dir`／`--overwrite`，meta 多記 `n_passes`、`props`（SGF property 名稱）。
- `board_cache.write_store_cache`：GameStore → BoardCache（與 `_write_shard` 同一套重播與打包），讓 V2 推論直接讀 release 盤。
- `tasks/strength/v2/predict.py`：prepare（CPU）→ infer（bf16，與訓練相同）→ aggregate（D-002 的 C2＋`score_rank`）。`slurm/strength_predict.slurm` 為 1 GPU `dev` 入口。
- `tasks/identity/predict.py`：單一池＋其題目 → D4 pp+slot（超參讀 A-029 split 0 選出的 `summary.json`）→ Sinkhorn＋Hungarian → top-5；另存 `scores.npy` 供之後與保險模型融合。人口先驗＝訓練集中「不在該池」的黑棋盤（A-029 協定）。
- `scripts/make_fake_release.py`（假官方檔）、`scripts/rehearsal_check.py`（gate）、`scripts/day0_audit.py`（§5 的 1–7、9）、`scripts/l10_check.py`（§5 的 8）。`tests/test_release_pipeline.py` 新增 4 項；全套 115 passed、3 skipped。

**假測試檔：** 競賽二取 A-024 的 Valid-U 題目（seed 2027 切分），public／private 各 400 題、每個 k 100 題、互不重複；競賽一取合成候選庫 split 1 → public、split 2 → private（各 400 人 × 100 盤，加上 400 題）。SGF 字串逐字複製自訓練 CSV，題目順序打亂、題號重編。

**gate 結果：**

| 項目 | 結果 |
|---|---|
| (1) 每盤落子／顏色與訓練快取相同 | 六個 store 全部相同（池 40,000＋40,000、題目 5,000 × 4） |
| (2) 競賽一 top-5 與 A-029 路徑相同 | public 400/400、private 400/400 → **PASS** |
| (2) 競賽二題級類別相同、機率差 ≤1e-4 | 類別 public 400/400、private 400/400 相同；機率最大差 public 7.4e-6（**PASS**）、private **8.1e-4（FAIL）** |

**競賽二 gate 細節（2026-10-23，Slurm 602222，1 張 H200 bf16，每個檔約 45 秒）：** Slurm 控制器 10/22 23:00–10/23 07:45 無法連線，恢復後才跑。逐盤比較「平均 log-prob」與 A-024 存的 Valid-U 結果：public 5,000 盤中 4,999 盤**逐位相同**，1 盤差 1.5e-3（game 3134，在中段批次）；private 有 72 盤差 >1e-4（最大 5.3e-3），全落在最後一個不滿的批次（7,719 筆；game 4928 之後）。但 public 的最後批次（6,914 筆）卻完全相同，所以不是單純的「尾批」問題。推測（未驗證）：bf16 卷積結果會隨批次大小／組成而變（cuDNN 依批次大小選 kernel），參考值是訓練時 8 卡、不同批次組合算出來的。落子對應已由 (1) 與逐位相同的多數盤證實正確，這是數值差，不是管線錯誤；但 gate 是事先登記的，照實記 FAIL，**不事後放寬**。

**處置（2026-10-23 使用者決定選項 a）：** 接受為 bf16 批次相依的數值差，不改推論程式；競賽二管線以「類別 800/800 相同、落子逐盤相同」視為可用。gate 紀錄維持 FAIL，事後接受的決定另記於此，不改 gate 定義。

- 競賽一彩排分數（Hungarian）：public 0.911／0.9874／1.0／1.0，全體 **0.9746**；private 0.99／0.985／0.9902／1.0，全體 **0.9913**。與 A-029 的 split 1／2（0.9746／0.9913）逐位一致。每個池約 40 秒（登入節點 CPU）。
- 競賽二的推論路徑先用 3 盤做 CPU fp32 抽查：每盤手數與 Valid-U 已存結果相同，平均 log-prob 最大差 0.007–0.011（fp32 對 bf16 的正常差距；對錯盤會差到約 1）。這只證明盤與手的對應正確，**不是** 1e-4 gate。溫度 T_k 約 4–8 會放大 logit 差，GPU bf16 實跑能否壓到 1e-4 未知；若過不了，照實記錄，不事後放寬。
- **Day-0 稽核自檢**（`--train-self-check`）：長度 100／228／323／445（min／中位數／p99／max）、白先 0、全數黑白交替、重複落子序列 0、8,007 位玩家、跨級玩家 0，與 A-001／A-002 一致。對假 release 也跑通；跨檔重複數在假資料裡不代表任何事（都來自訓練集）。

**L10 檢查（`scripts/l10_check.py`，§5 第 8 項）：** 訓練集 8,007 人全當候選人，用開局模型（D4 pp+slot，A-029 超參；白方題用白方的手）打分，每題統計量＝第 1 名減第 2 名的 log-likelihood 差。參考題＝Valid-U 每個 k 100 題，先把它們的查詢盤從所有 gallery 移除；「seen」保留真實玩家其他盤，「unseen」把真實玩家那一欄拿掉當 null，門檻取 null 的第 95 百分位。彩排結果（登入節點 CPU 14.5 分鐘）：

| 題組 | 超過門檻（gap 17.97）的比例 | 中位 gap |
|---|---:|---:|
| 參考 unseen（null，依定義） | 0.050 | 2.29 |
| 參考 seen（玩家在訓練集、盤不同） | 0.2775 | 4.70 |
| 假 rank_public | 0.2775 | 4.70 |
| 假 rank_private（盤本身就在訓練集） | 0.975 | 102.8 |

- 假 rank_public 與參考題完全相同：兩者都用 `default_rng(2027)` 對 Valid-U 每個 k 取前 100 個排列，抽到同一批 400 題（巧合，不是錯誤）。所以它剛好就是「玩家在訓練集、盤不在 gallery」的情境，數字逐位相同。
- 判讀規則（11/04 用）：約 5% → 測試玩家是新玩家；約 28% → 是訓練集玩家（盤不同）；接近 1 → 測試盤本身在訓練集（第 2 項重複稽核也會抓到）。400 題下 5% 與 28% 的二項標準差約 1–2%，區分得開。檢定力只有 28%／5%，因為開局只是弱指紋；這只回答「是否在訓練集」，不回答是哪些人。

**已知限制：** 提交 CSV 是暫定格式（11/04 才知道）；開局模型要求每盤 ≥16 手（L10 的白方題 ≥17 手）且前段黑白交替，不符合的盤不計入開局分數。L10 正式跑請放 CPU 作業，不在登入節點。


### A-031 V2 切分 seed 與初始化 seed 拆開（2026-10-22，D-002 決策 D4 實作）

- config 新增選填 `split_seed`（`shared/runtime.dataset_seed`）：有設時 split、驗證題目與監控盤都用它，`seed` 只管初始化、資料順序與 D4 增強；沒設時與舊行為相同（V1 與既有 V2 config 不受影響）。dataset manifest 的 `seed` 欄位比對改用 `dataset_seed`。
- 新增 `configs/strength/rank_cnn_v2_seed{2028,2029}.json`：`seed` 2028／2029、`split_seed` 2027、`dataset_dir` 沿用 `data/processed/strength_v2/seed2027`，其餘與 A-024 的 seed2027 config 相同（1 epoch）。三個 config 都通過 `load_dataset_manifest`，不必重新 prepare。
- `slurm/v2_train.slurm` 可用環境變數 `CONFIG`／`OUTPUT_DIR` 覆寫。**尚未提交訓練**：每個 run 依 A-023／A-024 估約 40 分鐘 × 8 H200 ≈ 5.3 GPU-h，兩個共約 11 GPU-h（T2 預算上限），須使用者確認後才提交。


### A-032 T2 第一次 seed 2028／2029 誤用固定 LR（2026-10-23，8 H200 × 2，作廢為 ensemble 元件）

**經過：** Slurm 恢復後 07:50 同時提交（602228、602231），各 37 分鐘訓練＋約 4 分鐘最終推論與評估，08:27 完成，合計約 11 GPU-h。結果遠低於 A-024，比對 `model.pt` 內存的 config 才發現：A-024 用 **5% warmup＋cosine 衰減到 0**（`warmup_fraction` 0.05、`minimum_lr_ratio` 0.0），commit 96342f8（10/20，為 A-028 的 3 epochs）把排程改成固定 LR 0.001，並從 `rank_cnn_v2_seed2027.json` 移除這兩個欄位。A-031 的新 config 是從這個已改過的檔複製的，所以這兩個 run 不是「A-024 只換初始化 seed」。錯在我沒有先拿 A-024 的存檔 config 逐欄比對。

| run | LR | Valid-U Equal-k（舊規則：機率平均＋score_rank） | 同上，log-prob 平均（診斷） | 最終 U monitor loss | G（舊規則） |
|---|---|---:|---:|---:|---:|
| A-024 seed2027 | warmup＋cosine | **0.4910** | — | 2.038 | — |
| seed2028 | 固定 0.001 | 0.4358 | 0.4746 | 2.072 | 0.011 |
| seed2029 | 固定 0.001 | 0.3934 | 0.3851 | 2.107 | 0.034 |

- 同切分、同步數下，固定 LR 讓 Equal-k 低 0.055–0.098；兩個固定 LR seed 彼此也差 0.042，初始化 seed 的變異不小。這是兩個點，不是受控的排程實驗（A-024 只有一個 seed）。
- C2 溫度是為 A-024 擬合的，沒有對這兩個 run 套用；不做 ensemble gate。產物保留在 `outputs/v2_strength_seed{2028,2029}/`，不當 ensemble 元件。
- **處置（使用者 10/23 確認選項 a）：** `v2/train.py` 恢復可選排程：config 的 training 有 `warmup_fraction` 時用 `shared/training.scheduler_for`（總步數＝每 epoch 批次數 × epochs），沒有時維持固定 LR（A-028 的 config 不受影響）。seed2028／2029 config 加回這兩個欄位，輸出改到 `outputs/v2_strength_seed{2028,2029}_cosine/`；除 seed、split_seed、路徑外與 A-024 存檔 config 逐欄相同（已用程式比對）。排程抽查：step 0 → 1.8e-5、step 55 → 1e-3、step 552 → 5.4e-4、step 1103 → 0。重跑 602407、602408（另約 11 GPU-h，超出 T2 原預算，已獲確認）。
- Slurm 事故：恢復時有兩次 `sbatch` 回報 timeout 卻已建立作業，重試迴圈因此各多送一個（預測作業 602249、訓練作業 602267），都已取消；602267 在啟動時被「run directory is nonempty」擋下，沒有寫入任何檔案。


### A-033 T2 競賽二 3-seed ensemble gate（2026-10-23，CPU；元件 602407、602408）

**元件：** A-024（seed2027）＋ seed2028／2029 重跑（`outputs/v2_strength_seed{2028,2029}_cosine/`），三者同一切分（split 2027，Valid-U 1,319 位玩家、10,000 題）、同一 recipe（5% warmup＋cosine，1 epoch），只差初始化 seed。重跑各 2,230／2,224 秒訓練，09:30 前後完成；T2 實際共用 4 個 run ≈ 22 GPU-h（含 A-032 作廢的 11 GPU-h）。

**規則（元件跑完前寫進 `scripts/v2_ensemble_gate.py`）：** 每盤取三個模型「平均 log-prob」的平均 → 題內各盤等權 → 每個 k 的溫度只在 A-025 的 S-fit 擬合 → `score_rank`。對照＝A-024 單模型同一 C2 規則。自檢：只放 A-024 時逐位重現 A-025（0.5553、T_k 相同）。

| Valid-U，C2＋score_rank | k=5 | k=10 | k=15 | k=20 | Equal-k | T_k（5／10／15／20） |
|---|---:|---:|---:|---:|---:|---|
| A-024 seed2027（對照） | 0.5030 | 0.5501 | 0.5716 | 0.5965 | **0.5553** | 4.42／6.24／7.67／8.13 |
| seed2028（診斷） | 0.5069 | 0.5528 | 0.5652 | 0.5921 | 0.5543 | 4.37／6.46／8.04／8.51 |
| seed2029（診斷） | 0.5013 | 0.5474 | 0.5629 | 0.5883 | 0.5500 | 4.42／6.24／7.41／8.13 |
| **3-seed ensemble** | 0.5085 | 0.5500 | 0.5692 | 0.5948 | **0.5556** | 4.62／6.53／7.94／8.41 |

- 溫度都不在網格邊界。舊規則（機率平均）下三個單模型 Equal-k 為 0.4910／0.4916／0.4848；G（舊規則）0.029／0.026，屬「small」。
- **gate：** 增益 +0.0003（門檻 0.005），以玩家為群組的 bootstrap（2,000 次）95% 區間 [−0.0039, +0.0045]，下界不 >0 → **FAIL**。只有 k=5 小幅上升（+0.0055），k=15／20 反而略降；單桶差異都在雜訊範圍內。
- 三個 seed 的單模型 Equal-k 只差 0.005，A-024 不是特別幸運的 seed；固定 LR（A-032）才是先前落差的原因。
- 為什麼 ensemble 沒有增益：**未知**。可能（未驗證）三個模型資料、切分、recipe 完全相同，錯誤高度相關；而題級聚合已經對 k 盤、上千手平均，模型間的獨立雜訊在聚合後所剩不多。
- **處置：** 依事先登記的 gate，競賽二提交維持 **A-024 單模型＋C2**。seed2028／2029 cosine 權重保留（`outputs/` 下），不進提交管線；不再追加同 recipe 的 seed。要有 ensemble 增益，元件得有實質差異（例如 D3 的 HumanSL 特徵，T5 再議）。


### A-034 競賽一保險模型 GE2E 短跑（2026-10-23，Slurm 602579，8 H200，D-002 D2）

**設定（使用者 10/23 確認）：** `tasks/identity/ge2e.py`、`train_ge2e.py`、`configs/identity/pid_ge2e_v1_short.json`。A-024 V2 ResNet 主幹（stem＋20 blocks，載入 A-024 權重；丟掉 1-channel 棋力 head）→ 盤面平均＋最大池化（512）→ 2 層 MLP → 256 維逐手向量；盤向量＝**ply ≥20** 黑方手的平均（模型看不到開局）；GE2E softmax（w、b 可學，初值 10／−5）。每卡 64 人 × 8 盤 × 16 手＝8,192 盤面，8 卡各自一組 GE2E。訓練玩家排除合成庫 split 0–2 的 1,200 人，剩 5,438 位（黑棋 ≥8 盤，共 216,867 盤）。AdamW lr 3e-4、5% warmup＋cosine、300 steps、bf16、無 D4。

**吞吐（實測）：** 1.52 秒／step（資料載入跟得上），300 steps 訓練 7.6 分鐘；split 0 全部 45,000 盤、ply ≥20 的每一手推論 44 秒。整個作業 543 秒 ≈ **1.2 GPU-h**。GE2E loss 4.16（ln 64）→ 2.17，自身 centroid 命中率 1.6%（隨機）→ 41.8%；step 200 後 LR 已衰減，loss 仍在下降。

**split 0（開發用，λ 也在這裡選；不是報告分數）：**

| split 0 | k=5 | k=10 | k=15 | k=20 | 全體 |
|---|---:|---:|---:|---:|---:|
| 保險模型單獨（只看 ply ≥20） | 0.7974 | 0.9521 | 0.9805 | 1.0000 | **0.9325** |
| 開局單獨（A-029 pp+slot） | 0.8560 | 0.9787 | 0.9937 | 0.9937 | 0.9555 |
| 開局＋8 × 保險 | 0.9637 | 1.0000 | 1.0000 | 1.0000 | **0.9909** |
| 開局＋Hungarian／融合＋Hungarian | 1.0 | 1.0 | 1.0 | 1.0 | 1.0／1.0 |

- λ 網格 0–16，選到 8（不在邊界；16 為 0.9863）。Hungarian 在 split 0 兩者都飽和，無法分辨。
- **洩漏檢查：** A-024 主幹用棋力標籤訓練過部分合成庫玩家的棋。依玩家在 V2 split 2027 的身分分組：V2 train 玩家 295 題 0.930；A-024 從沒訓練過的玩家（excluded 37 題 0.926、valid_unseen 68 題 0.947）沒有比較低。樣本小，但沒有「見過的玩家分數虛高」的跡象。
- 只有 split 0，**T2 gate 尚未判定**（gate 看 split 1–2）。不去開局的保險模型單獨在 split 0 已遠高於 0.2845；合成庫的開局習慣性仍可能比真實資料強，這點與 A-029 相同。
- 機制推測（未驗證）：中盤落子仍帶強烈的個人習慣（定式後續、偏好的應手），A-024 主幹的盤面特徵足以描述；合成庫是 ≥120 盤的重度玩家，候選人 100 盤 gallery 很充分。


### A-035 競賽一保險模型 GE2E 長跑 3,000 steps 與 T2 gate（2026-10-23，Slurm 602606、602850）

**長跑：** 與 A-034 只差步數（300 → 3,000，cosine 依步數拉長）與 checkpoint 間隔，`configs/identity/pid_ge2e_v1.json`、輸出 `outputs/pid_ge2e_v1/`。訓練 4,739 秒（79 分鐘，1.52 秒／step）＋三個 split 推論各 44 秒，約 **10.5 GPU-h**。訓練玩家上的自身 centroid 命中率 42%（step 300）→ 78%（1,000）→ 92%（1,500）→ **98.9%**（3,000），loss 0.175。

**held-out 反而變差（只看 ply ≥20 的保險模型單獨，plain top-5）：**

| | k=5 | k=10 | k=15 | k=20 | 全體 |
|---|---:|---:|---:|---:|---:|
| split 0，300 steps（A-034） | 0.7974 | 0.9521 | 0.9805 | 1.0000 | **0.9325** |
| split 0，3,000 steps | 0.6574 | 0.8102 | 0.9050 | 0.9523 | **0.8312** |
| split 1–2 平均，3,000 steps | 0.6679 | 0.9000 | 0.9310 | 0.9547 | 0.8634 |

- 訓練命中率上升、held-out 下降 → 長跑在背 5,438 位訓練玩家，而不是學到可轉移的風格（推測；只有 300 與 3,000 兩個點，中間的 held-out 曲線未量，峰值位置未知）。
- **依 split 0 選模型 → 採 300 steps（A-034）。** 這個選擇只用了 split 0。

**選定模型（300 steps）在 split 1–2 的報告分數（`embed-only` 補推 split 0–2，1 張 H200 每 split 258 秒；與 8 卡版 split 0 向量 cosine ≥0.99984）：**

| split 1–2 平均 | k=5 | k=10 | k=15 | k=20 | 全體 |
|---|---:|---:|---:|---:|---:|
| 保險模型單獨（ply ≥20） | 0.8095 | 0.9666 | 0.9728 | 0.9968 | **0.9364** |
| 開局單獨 | 0.8380 | 0.9710 | 0.9887 | 1.0000 | 0.9494 |
| 開局＋8 × 保險（λ 在 split 0 選） | 0.9418 | 0.9968 | 1.0000 | 1.0000 | **0.9846** |
| 開局＋Hungarian | 0.9505 | 0.9862 | 0.9951 | 1.0000 | 0.9830 |
| 融合＋Hungarian | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |

split 1／2 分開：保險單獨 0.9385／0.9343，融合 0.9865／0.9828，融合＋Hungarian 1.0／1.0。

- **T2 競賽一 gate：PASS。** (1) 不去開局、只看 ply ≥20 的保險模型單獨 0.9364 ≥ 0.2845；(2) 與開局合併後不下降：plain top-5 +0.035、Hungarian +0.017（到 1.0）。長跑模型也通過（0.8634；+0.029／+0.017），但被 split 0 淘汰。
- 合成驗證在融合＋Hungarian 已完全飽和（1.0），無法再分辨更細的改進；真實資料的表現仍未知（CLAUDE.md §0 的警訊照舊）。
- **處置：** 保險模型採 `outputs/pid_ge2e_v1_short/model.pt`，融合權重 λ=8（尺度：GE2E w × cosine 加到開局 log-likelihood）。接進 `tasks/identity/predict.py` 是下一步（尚未做）。若要再訓練，應加 held-out（split 0）定期監控，並改善訓練玩家數不足的過擬合，例如 D4 增強或更少步數；未經提案不做。


### A-036 競賽二彩排 gate FAIL 的批次大小探針（2026-10-23，1×H200 dev，job 605076）

承 A-030：fake rank_private 的 V2 推論機率與 A-024 存檔結果最大差 8.1e-4（>1e-4 gate）。假設是 bf16 卷積結果隨批次大小而變。探針：同一份快取、同一模型，以 `--batch` 8192／4096／2048 各跑一次，比較每盤平均 log-prob（5,000 盤）：

| 比較 | 最大差 | 差 >1e-4 的盤數 | 預測類別改變 |
|---|---:|---:|---:|
| 8192 vs 先前 8192 run（602222） | 0 | 0 | 0 |
| 4096 vs 8192 | 5.0e-3 | 4,975 | 0 |
| 2048 vs 8192 | 5.3e-3 | 58 | 0 |
| 4096 vs 2048 | 3.1e-3 | 4,975 | — |

- 同批次大小重跑位元相同（決定性）；改批次大小，值最多變動約 5e-3。**假設成立：差異來自批次組成，不是管線錯誤。** 參考值是訓練時 8 卡批次算出，所以任何不同批次的推論都不可能保證 ≤1e-4。
- A-030 的 gate 仍記為 **FAIL，不事後放寬**。這個 gate 對 bf16 設計過嚴；是否改成「類別完全相同＋機率差 ≤2e-3」或推論改 fp32，由使用者決定（未決）。
- 此探針只測 log-prob，未測改變批次後 T_k 放大的題級分數差；題級預測類別在 400/400 題上的穩定性已由 A-030 的 gate 項目確認。


### A-037 彩排 gate 改定；競賽二 14-epoch 長跑事先登記（2026-10-24 00:10，使用者授權由 Claude 決策）

**彩排 gate 改定（使用者 10/24 選項 a）：** A-036 證實 bf16 推論值隨批次組成變動（約 5e-3），1e-4 不可能達到。T1 競賽二 gate 改成「題級類別完全相同，且機率最大差 ≤2e-3」（`scripts/rehearsal_check.py` 的 `PROB_TOL`、EXECUTION_PLAN T1 同步）。重跑：public 7.37e-6、private 8.13e-4，類別 400/400 → 兩檔 **PASS**。原 1e-4 下的 FAIL 紀錄（A-030）保留不改。

**長跑：為什麼現在做。** EXECUTION_PLAN 把「加長訓練」延後，依據是 A-028（3 epochs 未提升）。但 A-032 顯示 A-028 用的是固定 LR；固定 LR 下 1 epoch 只有 0.39–0.44（舊規則），A-028 的 3 epochs 到 0.4897，所以「cosine 下加長訓練有沒有用」其實沒測過。三個 cosine seed 的 Valid-U 只差 0.005（A-033），單一 run 的比較有意義。

**設定（單一變數＝訓練長度）：** `configs/strength/rank_cnn_v2_e14.json` 由 A-024 `model.pt` 內存 config 複製，逐欄比對只差 `epochs` 1→14、`evaluate_every_steps` 200→552（每 epoch 中段一次監控，省時間；不影響訓練）、`epoch_snapshots`（新功能：每個 epoch 結束存權重到 `snapshots/`，供事後找最佳 epoch；不影響訓練）、`output_dir`＝`outputs/v2_strength_e14`。初始化 seed、切分 2027、5% warmup＋cosine 到 0、LR 1e-3、batch 65,536、D4 增強都相同；warmup 與 cosine 依總步數 15,456 等比例拉長。

**估時：** 15,456 steps × 1.49 s ≈ 6.4 h，加監控 15 × 50 s、最終推論、評估與 gate，約 **6.8 h × 8 H200 ≈ 55 GPU-h**。`--time` 10 h；另排 `afternotok` 續跑作業。

**假設：** 1 epoch 只看每手一次；多看幾輪（D4 增強提供變化）能學到更多棋力特徵。風險是記住訓練玩家：A-028 的 G 從 0.026 升到 0.059（固定 LR 下）。

**事先登記的 gate（`scripts/v2_candidate_gate.py`，跑完自動執行）：** 候選與 A-024 各自用 C2（每 k 溫度只在 S-fit 擬合），Valid-U Equal-k 增益 ≥0.005、玩家群組 bootstrap 95% 下界 >0，且候選的 G（C2，S-select − U）≤0.08 → 取代 A-024 成為競賽二提交候選。任一不符 → 維持 A-024。A-024 的 C2 G＝0.0446（自檢：A-024 重現 0.5553／T_k 與 A-025 相同；seed2028 cosine 重現 A-033 的 0.5543）。
- 若終點未過但監控曲線顯示中途較好，**不在 Valid-U 上挑 epoch**（會用到 U 做選擇）；只能用 Valid-S 選 epoch，並把它記為新的事後候選，要另一次驗證才可採用。

**預期：** 增益未知。做得到 +0.01 以上才算值得；k=5 桶最弱（0.503），最想看它是否提升。


### A-038 競賽二 14-epoch 長跑結果（2026-10-24，Slurm 605588，8 H200）

**作業：** 605588 COMPLETED，6h52m（訓練 24,720 s；≈55 GPU-h）；續跑作業 605591 因依賴不成立自動取消。15,456 steps，14 個 epoch 快照在 `outputs/v2_strength_e14/snapshots/`。設定與 gate 見 A-037。

**Gate（`outputs/v2_strength_e14_gate.json`）：FAIL。**

| Valid-U，C2＋score_rank | k=5 | k=10 | k=15 | k=20 | Equal-k | G（S-select − U） | T_k |
|---|---:|---:|---:|---:|---:|---:|---|
| A-024（1 epoch） | 0.5030 | 0.5501 | 0.5716 | 0.5965 | **0.5553** | 0.0446 | 4.42／6.24／7.67／8.13 |
| 14 epochs | 0.4310 | 0.4801 | 0.4867 | 0.5010 | 0.4747 | **0.0847** | 0.68／1.12／1.57／1.93 |

- 增益 −0.0806，bootstrap 95% [−0.094, −0.067]；G 0.0847 > 0.08。三項條件都不符，四個 k 桶都下降。舊規則（機率平均）U 0.4720／S 0.5706。
- 溫度從 4–8 降到 0.7–1.9：每手機率變得過度自信，需要壓平而不是放大。

**監控曲線（每 epoch 中段；單盤 argmax 診斷分）：**

| step（epoch） | train CE | U CE | U 單盤分 | S CE | S 單盤分 |
|---:|---:|---:|---:|---:|---:|
| 552（0.5） | 2.049 | 2.095 | 0.328 | 2.080 | 0.345 |
| 1,656（1.5） | 1.906 | **2.065** | **0.350** | **2.027** | **0.383** |
| 2,760（2.5） | 1.780 | 2.181 | 0.347 | 2.116 | 0.374 |
| 4,968（4.5） | 1.425 | 2.522 | 0.330 | 2.426 | 0.363 |
| 8,280（7.5） | 1.011 | 3.276 | 0.319 | 3.145 | 0.346 |
| 15,456（14，終點） | 0.667 | 4.858 | 0.310 | 4.650 | 0.341 |

（A-024 終點：U CE 2.038、U 單盤 0.354。）

- **判讀：** 第 2 個 epoch 起就過擬合，train CE 一路降到 0.67，U 與 S 的 CE 同步上升。S 是**訓練玩家沒見過的盤**，它的 CE 也一起惡化，所以模型記住的主要是**訓練盤本身**（盤面 → 該盤標籤），不只是玩家習慣。機制屬推測：同一盤約 230 手共用一個標籤，模型看過幾次就能認出盤；1M 盤對 2,400 萬參數的 ResNet 不夠多。
- **快照：** 依 A-037 規則只能用 Valid-S 挑 epoch。S 監控在 epoch 1.5 最好（2.027），但那時 LR 仍接近峰值；同一時點的 U CE 2.065 也比 A-024 終點 2.038 差，看不出有快照能超過 A-024。**不另花 GPU 評估快照**；權重保留。
- **處置：** 競賽二提交候選維持 **A-024 單模型＋C2**。同 recipe 加長訓練砍掉（EXECUTION_PLAN 同步更新）。若要回收更多資料價值，方向應是抑制記盤，例如每批每盤只取少數手、更強的正則化或更小的模型，而不是多跑幾輪。這些都未驗證，須另提計畫。


### A-039 KataGo HumanSL 試算：吞吐量與單盤訊號（2026-10-24，使用者確認提前 D3，≤1 GPU-h）

**目的：** A-038 後判斷競賽二的瓶頸是「每盤一個標籤」的監督量；HumanSL 用大量人類棋譜、以段位為條件預測每一手，可能提供 A-024 沒有的訊號。使用者 10/24 同意把 D3 提前做小規模試算（原定 T5）。

**外部資源（需在報告申報）：** KataGo v1.18.2 `katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip`（2026-08-30，sha256 `2cb197fc…6c02`）、HumanSL 網路 `b18c384nbt-humanv0.bin.gz`（KataGo v1.15.0 release，sha256 `637746e4…4ab5`，KataGo Neural Network License，見 R3）。CUDA 12.8／cuDNN 9.8 共享函式庫取自 NVIDIA 的 PyPI wheel（cublas 12.8.4.1、cudart 12.8.90、nvrtc 12.8.93、cudnn 9.8.0.87），libstdc++ 取自 `go` 環境；全部在 `data/external/`（不進 git）。

**做法（`src/go_aicup/features/humansl.py`）：** HumanSL 當主模型，analysis engine 每盤每個 profile 一個 query、所有手數、1 visit、`includePolicy`、`ignorePreRootHistory=false`、規則 japanese、貼目 6.5（SGF 沒有貼目，A-001；幽玄一般為日本規則 6.5，未驗證）。讀出實際落子在該 profile 下的機率；落子轉換（row 0＝GTP 第 19 線）若錯，KataGo 會因非法著手報錯，並且程式要求實際落子的機率 >0。15 個 profile：rank_20k／15k／10k／7k／5k／3k／1k／1d–7d（不含 8d）／9d。

**吞吐量（1×H200，dev 佇列每卡上限 12 CPU）：** 單一引擎 256 threads 時每秒 3,136 個盤面（平均批次 170）；768 threads＋批次 512 時 3,306（批次滿 509，代表卡在單一 NN 服務執行緒）；加上 3 個 NN 服務執行緒時 **4,519**（平均批次 219，代表改由 CPU 餵資料限制）。多引擎並行會碰到使用者執行緒上限而崩潰。Python 解析 JSON 約每秒 3.1 萬行，不是瓶頸。小樣本檢查：各 profile 的機率明顯不同；最弱的 D 級盤最符合 5k–10k profile（10 盤，只是合理性檢查）。

**試算設計（`scripts/humansl_pilot.py`，事先登記）：** 樣本為 Valid-U 每級 200 盤（評估），以及 A-025 S-fit 那一半 Valid-S 玩家每級 200 盤（擬合），seed 2027，共 4,000 盤、887,445 個盤面 × 15 profiles。每盤特徵＝每個 profile 對指定玩家落子的平均 log-likelihood（另算對手落子的平均）。三個 multinomial logistic regression 只在 S-fit 擬合，用 `score_rank` 解碼，在 Valid-U 逐盤計官方分數：A（A-024 每盤平均 log-prob，10 維）、H（HumanSL 特徵）、AH（兩者合併）。
- **Gate：** AH − A 的單盤分數 ≥ +0.01，且以玩家為群組的 bootstrap 95% 下界 >0 → 進到題目層級（分 k 桶）評估與擴大計算。未過 → HumanSL 回到 T5，或放棄。
- 單盤增益不保證能轉成題目層級的增益；這只是是否值得繼續的第一關。
- 估算：887,445 × 15 ÷ 每秒約 4,800 ≈ 46 分鐘，約 0.8 GPU-h。

**結果（2026-10-24，Slurm 606915，1×H200 dev）：** 抽取 4,000/4,000 盤完成，13,311,675 次評估，1,946 s（≈0.54 GPU-h），穩定速度約每秒 6,840 個盤面（`numAnalysisThreads=768`、批次 256、3 個 NN 服務執行緒）。分析輸出 `outputs/humansl_pilot/result.json`。

| Valid-U 單盤官方分數（2,000 盤，733 位玩家） | 全體 | D | C | B | A | 1D | 2D | 3D | 4D | 5D | 6D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A：A-024 特徵 | 0.366 | 0.390 | 0.498 | 0.330 | 0.267 | 0.238 | 0.195 | 0.342 | 0.345 | 0.499 | 0.555 |
| H_own：HumanSL，只用本人落子 | 0.438 | 0.523 | 0.598 | 0.448 | 0.342 | 0.332 | 0.278 | 0.305 | 0.367 | 0.646 | 0.540 |
| H：HumanSL，本人＋對手 | 0.481 | 0.605 | 0.578 | 0.463 | 0.452 | 0.408 | 0.319 | 0.365 | 0.415 | 0.583 | 0.620 |
| AH：兩者合併 | **0.484** | 0.611 | 0.593 | 0.476 | 0.416 | 0.436 | 0.287 | 0.394 | 0.426 | 0.549 | 0.652 |

- **Gate：PASS。** AH − A＝+0.118，玩家群組 bootstrap 95% [+0.097, +0.139]。A 的 0.366 與 A-024 自己的單盤診斷分（0.350，A-024）同一水準，對照組合理。
- 單用 HumanSL（H）就比 A-024 高 +0.115；A-024 加進去只多 +0.003。對手落子讓 HumanSL 從 0.438 升到 0.481（+0.043），與 A-005 E3「對手走法也有棋力訊號」一致。
- 機制（推測）：HumanSL 每一手都有 361 選 1 的監督，學到「各段位的人在這個盤面會下哪裡」；這正是 A-038 指出 A-024 缺的東西。
- **限制：** (1) 這是單盤，題目層級（分 k 桶）還沒測；HumanSL 與 A-024 在 k 盤聚合後的差距可能縮小或放大，未知。(2) 擬合只用 2,000 盤、30 個特徵的 logistic regression，未調參。(3) HumanSL 的訓練資料未明示是否含幽玄棋譜（R4 推論不含）；若含，測試盤也來自同一來源，影響方向未知，無法驗證。(4) 貼目 6.5 為假設。


### A-040 HumanSL 題目層級 gate（2026-10-24，事先登記；使用者選 1,000 題）

**樣本（`scripts/humansl_questions.py select`，seed 2027）：** Valid-U 每個 k 250 題，共 1,000 題（評估）；A-025 S-fit 那一半 Valid-S 玩家每個 k 250 題，共 1,000 題（擬合）。兩者的盤取聯集：U 11,766 盤、S-fit 9,397 盤，4,715,928 個盤面 × 15 profiles ≈ 7,070 萬次評估；拆 8 個 1-GPU 作業，約 2.9 GPU-h。

**方法：** 每盤用 multinomial logistic regression（標準化、C=1）在 S-fit 盤上擬合，輸出每盤的 log-posterior；題目＝題內各盤 log-posterior 等權平均 → 乘上每個 k 的溫度（在 S-fit 題目上擬合，使用兩折、以玩家分折的 out-of-fold 值）→ softmax → `score_rank`。給 U 的模型用全部 S-fit 盤重新擬合。
- 候選：**AH**（A-024 每盤平均 log-prob ＋ HumanSL 本人／對手，40 維，主要候選）、H（只用 HumanSL）、A_LR（A-024 特徵走同一流程，檢查流程本身的影響）。
- 對照：A-024＋C2，完全照 A-025（T_k 用全部 S-fit 題目擬合），在同樣 1,000 題 U 上評分。
- **Gate：** AH 的 Equal-k 比對照高 ≥0.005，且玩家群組 bootstrap 95% 下界 >0。四個 k 桶分開列。
- 註：S-fit 題目裡的盤也是 logistic regression 的擬合資料，所以溫度一定要用 out-of-fold 值擬合；U 完全不參與擬合。

**結果（2026-10-24）：** 抽取共 8 個 1-GPU 作業。第一次送出因每節點每使用者 2,048 執行緒上限全數在啟動時失敗（每個引擎約用 analysis threads × 2 個執行緒；6 個作業擠在同一節點）；改為 shard 0–5 用 140、shard 6–7 用 200 個 analysis threads 重跑，21,163 盤全部完成，每卡每秒 3,500–4,900 個盤面。實際用量約 4.8 GPU-h（估計 2.9–3.5；執行緒變少拖慢速度），另有失敗啟動約 0.05 GPU-h。輸出 `outputs/humansl_q1000/result.json`。

| Valid-U 1,000 題（676 位玩家），`score_rank` | k=5 | k=10 | k=15 | k=20 | Equal-k | T_k（5／10／15／20） |
|---|---:|---:|---:|---:|---:|---|
| A-024＋C2（對照，A-025 協定） | 0.4930 | 0.5621 | 0.5576 | 0.6002 | 0.5532 | 4.42／6.24／7.67／8.13 |
| A_LR（A-024 特徵走 LR 流程，控制組） | 0.4467 | 0.5016 | 0.4549 | 0.5168 | 0.4800 | 1.88／2.34／2.48／2.75 |
| H（只用 HumanSL） | 0.5896 | 0.6558 | 0.6193 | 0.6126 | 0.6193 | 1.43／2.02／2.48／1.58 |
| **AH（A-024＋HumanSL）** | 0.5775 | 0.6415 | 0.6393 | 0.6347 | **0.6232** | 1.40／1.82／2.34／1.68 |

- **Gate：PASS。** AH − 對照＝+0.0700，玩家群組 bootstrap 95% [+0.044, +0.097]；H − 對照＝+0.0661 [+0.037, +0.096]。四個 k 桶 AH 都高於對照（+0.085／+0.079／+0.082／+0.035）。溫度都不在網格邊界。對照在這 1,000 題上是 0.5532，與 A-024 全部 10,000 題的 0.5553 接近，抽樣沒有偏。
- **流程本身會損失分數：** 同樣是 A-024 特徵，走 LR 流程（A_LR）只有 0.480，比 C2 的 0.553 低 0.073。可能原因（推測，未驗證）：題目 logit＝T_k ×（W·平均特徵＋b），截距 b（級別先驗）也被 T_k 放大；且 W 是為單盤判別擬合的。所以 AH 的 0.623 可能低估了 HumanSL 的價值；但改進聚合方式是新實驗，只能用 S-select 選，不能看 U。
- k=20 的 H（0.613）低於 k=10（0.656），每桶 250 題，標準誤約 0.03，可能是雜訊；AH 在 k=15／20 較平穩。
- **處置：** HumanSL 正式成為競賽二的主要候選訊號；是否改寫 D3／T2–T5 的時程與提交管線，待使用者決定。


### A-041 HumanSL：貼目、聚合方式、擬合資料量（2026-10-24，事先登記；使用者：三件都做）

**貼目檢查：** A-039 試算 S-fit 盤中每級前 20 盤（200 盤、43,919 個盤面）改用貼目 7.5 與 0 重跑，與 6.5 的結果比較。雜訊下限：同一盤在不同批次設定下，每盤平均特徵最大差 6.9e-3（試算與 q1000 重疊的 789 盤）。
- 判定：若改貼目後每盤特徵的最大差 <0.02（約雜訊下限 3 倍）→ 貼目不影響，維持 6.5。否則採用讓實際落子平均 log-likelihood 最高的貼目，並重抽。

**聚合方式（`scripts/humansl_aggregation.py`；方法實作在 `src/go_aicup/tasks/strength/humansl.py` 的 `RankModel`，提交管線用同一份程式）：** B0＝A-040 的 AH；B1＝截距（級別先驗）不乘溫度；B2＝每個 k 一個 LR，在 S-fit 玩家的合成題目（同一玩家抽 k 盤、特徵取平均，每個 k 20,000 題）上擬合，不用溫度。擬合都只用 S-fit。S-select＝另外 1,000 題 Valid-S（`select-sselect`，S-select 那一半玩家，每個 k 250 題，9,375 盤，與 q1000 的盤和玩家都不重疊）。
- 規則：S-select Equal-k 最高者若比 B0 高 ≥0.005 就取代 B0，否則維持 B0。選定後才在 Valid-U 1,000 題評分一次，並與 A-024＋C2 做玩家 bootstrap 比較。B0 在 U 上的分數 A-040 已報告，只當重現檢查。
- **學習曲線：** 選定方法只用一半 S-fit 玩家擬合 vs 全部，在 S-select 上比較；差 ≥0.01 → 值得再抽更多擬合資料。

**彩排容差（提交管線，第三項）：** release 路徑與訓練快取路徑對同一批盤的每盤 HumanSL 特徵最大差 ≤1e-2（雜訊下限 6.9e-3）；題目類別若有不同，須都發生在前兩名期望分數差 <0.01 的近乎平手題。

**結果（2026-10-24）：** 抽取 S-select 8 個 shard（9,375 盤，每卡每秒 3,900–5,500，約 2.0 GPU-h）與貼目檢查 2 個作業（約 0.08 GPU-h）。輸出 `outputs/humansl_aggregation.json`。

**貼目：**

| 貼目 | 實際落子平均 log-p（15 profiles、200 盤） | 每盤特徵與 6.5 的最大差／平均差 | 逐盤配對：每手 Δlog-p（95%） | 較高的盤比例 |
|---|---:|---|---|---:|
| 6.5 | −1.69077 | — | — | — |
| 7.5 | −1.69041 | 0.036／0.0022 | +0.00037 [+0.00014, +0.00060] | 0.57 |
| 0 | −1.68711 | 0.428／0.0119 | +0.00366 [+0.00224, +0.00502] | 0.63 |

- 兩者最大差都 ≥0.02，依規則應改用 likelihood 最高的貼目 0 並重抽。差異主要在開局前三分之一（貼目 0：+0.008／+0.002／+0.001）。效果很小（相鄰 profile 間的特徵差約 0.1 量級）但不是雜訊。
- **未執行重抽：** q1000＋S-select 重抽約 7 GPU-h，超出這輪估算（4.6 GPU-h）。規則沒有考慮成本，也沒有直接測下游分數；處置待使用者決定，這一節的其餘結果都用貼目 6.5。

**聚合方式（S-select 1,000 題選擇；擬合只用 S-fit）：**

| | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| B0 S-select | 0.5919 | 0.6844 | 0.6712 | 0.6789 | 0.6566 |
| B1 S-select | 0.5865 | 0.6552 | 0.6758 | 0.6907 | 0.6520 |
| **B2 S-select** | 0.6432 | 0.6832 | 0.6838 | 0.7133 | **0.6808** |
| **B2 Valid-U（選定後評分一次）** | 0.6055 | 0.6827 | 0.6682 | 0.6459 | **0.6506** |
| A-024＋C2 Valid-U（同 1,000 題） | 0.4930 | 0.5621 | 0.5576 | 0.6002 | 0.5532 |

- **選定 B2**（比 B0 高 +0.024 ≥0.005）。Valid-U：B2 − A-024＋C2＝**+0.0973**，玩家 bootstrap 95% [+0.069, +0.125]；比 B0（0.6232，與 A-040 逐位相同）高 +0.027。B1 沒有比 B0 好，所以 A-040 推測的「截距被溫度放大」不是主因；B2 直接在題目層級擬合（特徵先對 k 盤平均）才是關鍵。原因推測：每盤 LR 是為單盤雜訊尺度校準的，k 盤平均後特徵雜訊縮小、決策邊界應該不同。
- B2 在 U 的 k=20（0.646）低於 k=10、15，S-select 上沒有這個現象（0.713）；每桶 250 題，標準誤約 0.03，原因未知。
- **學習曲線：** 只用一半 S-fit 玩家 0.6285 vs 全部 0.6808（+0.052 ≥0.01）→ **擬合資料是瓶頸，值得再抽**。正式模型改用 S-fit＋S-select 擬合（玩家數加倍）。


### A-042 HumanSL 提交管線彩排（2026-10-24，Slurm 607492–607499，8×1 H200）

**管線：** release GameStore → V2 `predict`（prepare／infer／aggregate，提供 A-024 每盤特徵與備援預測）→ `features/humansl.py --cache <work>/cache --allow-failures`（每檔 4 個 1-GPU 作業）→ `tasks/strength/humansl_predict aggregate`（B2，`outputs/humansl_model/model.pkl`，以 S-fit＋S-select 共 18,772 盤／2,000 題擬合）。操作見 README「競賽二 HumanSL 路徑」。抽取 10,000 盤全部完成、0 個 query 被拒，約 2.5 GPU-h。

**Gate（A-041 事先登記）：**

| 項目 | rank_public | rank_private |
|---|---|---|
| (a) 每盤 HumanSL 特徵 max\|diff\| ≤1e-2 | 2.21e-2 → **FAIL** | 2.04e-2 → **FAIL** |
| (a) A-024 每盤特徵 max\|diff\|（參考） | 1.46e-3 | 5.27e-3 |
| (b) 題目類別相同；不同者須為近乎平手（期望分數差 <0.01） | 399/400，差異題 0.0003 → **PASS** | 399/400，差異題 0.0014 → **PASS** |
| 題目分數（對答案，k=5／10／15／20；全體） | 0.530／0.665／0.704／0.695；0.648 | 0.654／0.687／0.672／0.729；0.685 |

- (a) 照登記記為 FAIL，不事後放寬。差異分布與「同一路徑、不同批次設定」相同：release vs 訓練路徑 p50 3.6e-4、p99 7.5e-3（兩檔一樣）；訓練路徑兩次抽取之間（試算 vs q1000，789 盤）p50 4.6e-4、p99 7.3e-3、max 1.4e-2。所以是 fp16 批次雜訊的長尾，不是管線錯誤。**容差訂錯了：** 1e-2 是用「全部手數的平均」量出的 6.9e-3 推來的，但特徵是本人／對手分開的平均（各約一半手數，雜訊較大）。是否改成分布比較（例如 p99 ≤1e-2）由使用者決定。
- 這 800 題全部屬於 A-041 評估用的 1,000 題 U（`make_fake_release` 與 `select` 用同一個 rng 序列），所以題目分數不是新的驗證證據，只是確認管線端到端可以重現。


### A-043 貼目下游測試（2026-10-24，事先登記；使用者選「先做下游測試」）

**做法：** 用貼目 0 重抽 A-041 的 S-fit 盤（9,397 盤，q1000）與 S-select 盤（9,375 盤，qsel），其他設定不變。B2 分別在兩種貼目的 S-fit 上擬合、在同貼目的 S-select 1,000 題上評分（`scripts/humansl_komi_downstream.py`）；A-024 特徵兩邊相同。約 4.1 GPU-h。
- **規則：** 貼目 0 的 S-select Equal-k ≥ 貼目 6.5（0.6808）+0.005 → 改用貼目 0，之後的抽取（含 Valid-U 重抽與補擬合資料）都用 0；否則維持 6.5。另報玩家群組 bootstrap 區間供參考，不作為規則。Valid-U 不參與這個決定。

**結果（2026-10-24，Slurm 607782–607791，10×1 H200，約 3.6 GPU-h）：** 18,772 盤以貼目 0 重抽完成。輸出 `outputs/humansl_komi0/result.json`。

| B2，S-select 1,000 題 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| 貼目 6.5 | 0.6432 | 0.6832 | 0.6838 | 0.7133 | 0.6808 |
| 貼目 0 | 0.6240 | 0.6907 | 0.7070 | 0.7209 | 0.6856 |

- 貼目 0 − 6.5＝+0.0048，未達 +0.005 → 依規則**維持貼目 6.5**。玩家群組 bootstrap 95% [−0.006, +0.016]，包含 0。k=5 下降 0.019，其他三桶上升 0.008–0.023，方向不一致。
- 結論：貼目對下游分數的影響在 1,000 題的雜訊範圍內。A-041 的 likelihood 差異是真的，但不影響棋力判斷。之後的抽取（補擬合資料、測試集）都用 6.5，已抽的資料不必重抽。
- 這輪 HumanSL 線累計約 15 GPU-h（A-039 0.54、A-040 4.8、A-041 2.1、A-042 2.5、A-043 3.6、測試與失敗啟動約 0.3）。


### A-044 補 HumanSL 擬合資料：其餘 Valid-S（2026-10-24，事先登記；使用者同意，貼目 6.5）

**資料（`scripts/humansl_more_data.py select`）：** 尚未抽取的 Valid-S 盤 46,451 盤（10,317,378 個盤面）：S-fit 半 16,365 盤／894 人、S-select 半 17,259 盤／892 人、S-extra（不在任何 Valid-S 題目裡、留出盤少於 5 盤的玩家）12,827 盤／3,188 人。拆 10 個 1-GPU 作業，估約 9.5 GPU-h。
- **(i)** B2 以「舊 S-fit＋新 S-fit＋S-extra」擬合（程式檢查與 S-select 題目玩家不重疊），對照只用舊 S-fit，兩者都在 S-select 1,000 題評分。
- **(ii)** B2 以全部已抽 Valid-S 盤擬合，在 Valid-U 1,000 題評分一次，對照 A-041 的 B2（只用 S-fit，0.6506）與 A-024＋C2，玩家群組 bootstrap。
- **規則：** (i) 增益 >0 → 全部 Valid-S 擬合的模型成為正式模型；(ii) 不論結果都報告。

**彩排容差改定（使用者 10/24 同意）：** A-042 (a) 改為分布比較：每盤 HumanSL 特徵差的 p99 ≤1e-2（原為 max ≤1e-2；A-042 顯示 max 隨盤數增加，且量測雜訊下限時用錯了統計量）。A-042 原 FAIL 紀錄保留。


### A-045 競賽二主線比較（2026-10-24，CPU，診斷；無 gate）

`scripts/humansl_mainline_compare.py`，輸出 `outputs/humansl_mainline_compare.json`。B2 只用 S-fit 擬合（A-041 設定），S-select 對所有候選的擬合步驟都是樣本外。B2_H 的 Valid-U 數字是額外查看 U，只當診斷，不作為選擇依據。

| 候選 | 集合 | k=5 | k=10 | k=15 | k=20 | Equal-k | 完全對／差一級／差兩級以上 | 黑／白 |
|---|---|---:|---:|---:|---:|---:|---|---|
| A-024＋C2 | S-select | 0.507 | 0.635 | 0.645 | 0.609 | 0.5993 | — | 0.607／0.592 |
| A-024＋C2 | Valid-U | 0.493 | 0.562 | 0.558 | 0.600 | 0.5532 | — | 0.529／0.579 |
| B2_H（只用 HumanSL） | S-select | 0.632 | 0.668 | 0.701 | 0.711 | 0.6780 | 0.532／0.397／0.071 | 0.681／0.676 |
| B2_H | Valid-U | 0.609 | 0.685 | 0.670 | 0.666 | 0.6572 | 0.502／0.422／0.076 | 0.646／0.669 |
| B2_AH（A-041 選定） | S-select | 0.643 | 0.683 | 0.684 | 0.713 | 0.6808 | 0.537／0.391／0.072 | 0.699／0.664 |
| B2_AH | Valid-U | 0.606 | 0.683 | 0.668 | 0.646 | 0.6506 | 0.499／0.412／0.089 | 0.649／0.652 |

| Valid-U 各級 | D | C | B | A | 1D | 2D | 3D | 4D | 5D | 6D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-024＋C2 | 0.694 | 0.653 | 0.540 | 0.504 | 0.517 | 0.337 | 0.421 | 0.497 | 0.691 | 0.722 |
| B2_H | 0.831 | 0.717 | 0.775 | 0.486 | 0.576 | 0.544 | 0.528 | 0.560 | 0.768 | 0.834 |
| B2_AH | 0.843 | 0.688 | 0.760 | 0.433 | 0.647 | 0.536 | 0.492 | 0.572 | 0.753 | 0.839 |

- **G（S-select − U）：** A-024＋C2 0.046、B2_AH 0.030、**B2_H 0.021**。S-select 玩家是 A-024 的訓練玩家，所以 A-024 特徵在 S-select 上偏樂觀；只用 HumanSL 時差距最小，對「測試玩家是否在訓練集」（L10，11/04 才知道）最不敏感。
- B2_H 與 B2_AH：S-select 上 AH 高 0.003、U 上 H 高 0.007，都在雜訊內。A-024 特徵在未見玩家上沒有額外貢獻。
- HumanSL 在 10 個級別中 9 個高於 A-024；A 級是例外（0.486／0.433 vs 0.504）。2D、3D 這兩個 A-024 最弱的級別提升 0.07–0.21。
- 差兩級以上（0 分）的比例約 7–9%；完全答對約 50%，差一級約 40%。


### A-046 HumanSL 每盤特徵組（2026-10-24，事先登記；CPU，不重抽）

從已存的每手 log-likelihood 算四組特徵（`features/humansl.shard_features(feature_set=…)`，只用 HumanSL、不含 A-024）：F0 `mean`（本人／對手每 profile 平均，30 維，即 A-045 的 B2_H）；F1 `phase`（再加開局 <60 手、中盤 60–149、收官 ≥150 手各自的平均；某段沒有手時用全盤平均，120 維）；F2 `vote`（再加每手在 15 個 profile 之間做 softmax 後的平均，60 維）；F3 `full`（F1＋F2＋每 profile 標準差，180 維）。B2 在 S-fit 擬合、S-select 1,000 題評分（`scripts/humansl_features.py`）；A-044 資料完成後用 `--with-rest` 以更大的擬合集重跑一次。
- **規則：** S-select Equal-k 最高者比 F0 高 ≥0.005 → 採用；否則維持 F0。Valid-U 不使用。
- 註：`mean` 改寫後與 A-039～A-045 的特徵差 ≤1.9e-6（float32 加總順序不同）。

**結果（S-fit 擬合，9,397 盤）：** 輸出 `outputs/humansl_features.json`。

| 特徵組 | 維度 | k=5 | k=10 | k=15 | k=20 | S-select Equal-k |
|---|---:|---:|---:|---:|---:|---:|
| **F0 mean** | 30 | 0.6297 | 0.6659 | 0.7009 | 0.7122 | **0.6772** |
| F1 phase | 120 | 0.6246 | 0.6461 | 0.6758 | 0.6697 | 0.6540 |
| F2 vote | 60 | 0.6240 | 0.6604 | 0.6990 | 0.7173 | 0.6752 |
| F3 full | 180 | 0.6236 | 0.6851 | 0.7141 | 0.6853 | 0.6770 |

- **維持 F0。** 沒有一組比 F0 高；分段（F1）低 0.023。推測（未驗證）：擬合玩家約 1,300 人，特徵加到 120–180 維時 B2 的 LR 過擬合。資料量變大後結論可能改變，所以 A-044 完成後用 `--with-rest` 重跑。
- F0 這次 0.6772，與 A-045 B2_H 的 0.6780 差 0.0008（約 2 題），來自 `mean` 改寫後 ≤1.9e-6 的特徵差。

**結果（2026-10-24，Slurm 607987–607996，10×1 H200）：** 46,451/46,451 盤完成、0 個 query 被拒；每卡每秒 3,500–5,450（6 個作業擠在 hgpn031 的較慢），約 9.0 GPU-h。輸出 `outputs/humansl_vs_rest/result.json`。

| B2（A-024＋HumanSL 特徵） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| (i) S-select，只用舊 S-fit（9,397 盤） | 0.6406 | 0.6832 | 0.6863 | 0.7158 | 0.6815 |
| (i) S-select，＋新 S-fit＋S-extra（38,589 盤） | 0.6312 | 0.7194 | 0.7402 | 0.7495 | **0.7101** |
| (ii) Valid-U，全部 Valid-S（65,223 盤、5,000 人） | 0.6168 | 0.6792 | 0.7223 | 0.7147 | **0.6833** |
| (ii) Valid-U，只用 S-fit | 0.6055 | 0.6752 | 0.6707 | 0.6433 | 0.6487 |
| (ii) Valid-U，A-024＋C2 | 0.4930 | 0.5621 | 0.5576 | 0.6002 | 0.5532 |

- **(i) 增益 +0.0286 >0 → 正式模型改為全部 Valid-S 擬合**（`outputs/humansl_model/model.pkl` 已重新擬合：65,223 盤；`humansl_predict fit` 改讀三個 run，role 限 S-fit／S-select／S-extra，Valid-U 從不擬合）。
- **(ii) Valid-U：** 全部 Valid-S vs 只用 S-fit＋0.0346 [+0.014, +0.056]；vs A-024＋C2 **+0.1300 [+0.104, +0.156]**。k=5 改善最少（+0.011），k=15／20 改善最多（+0.05／+0.07）：B2 的合成題目需要每位玩家 ≥k 盤，新玩家多數只有幾盤，主要補到小 k；大 k 的改善可能來自 S-select 半那 892 位玩家（每人盤數多）。這是推測，未驗證。
- 「只用 S-fit」在這裡是 0.6487，A-041 為 0.6506，差 0.002（約 2 題），來自 `mean` 改寫後 ≤1.9e-6 的特徵差。
- 資料仍是瓶頸的跡象：(i) 擬合盤數 ×4 後 S-select +0.029。下一步可用的更多玩家只剩 train 切分（只能用不含 A-024 的特徵，A-045）與 excluded（需使用者同意）。

**`--with-rest` 重跑（擬合＝q1000 S-fit＋vs_rest S-fit／S-extra，38,589 盤、4,094 人；輸出 `outputs/humansl_features_rest.json`）：**

| 特徵組 | 維度 | k=5 | k=10 | k=15 | k=20 | S-select Equal-k |
|---|---:|---:|---:|---:|---:|---:|
| **F0 mean** | 30 | 0.6200 | 0.6970 | 0.7143 | 0.7788 | **0.7025** |
| F1 phase | 120 | 0.6145 | 0.6720 | 0.7118 | 0.7147 | 0.6783 |
| F2 vote | 60 | 0.6558 | 0.6829 | 0.7240 | 0.7586 | 0.7053 |
| F3 full | 180 | 0.6333 | 0.6636 | 0.6966 | 0.7133 | 0.6767 |

- F2 最高，但只比 F0 高 +0.0028（<0.005）→ **維持 F0**。擬合資料 ×4 後，F1、F3 仍比 F0 低約 0.025，所以「特徵多、資料少」不是唯一原因；分段與標準差特徵本身可能沒有額外訊號，或線性 LR 用不好。F2 在 k=5 高 +0.036、k=20 低 −0.020，方向不一致。
- 同一擬合集下，只用 HumanSL 的 F0（0.7025）比 A-044 (i) 的 AH（0.7101）低 0.0076。S-select 玩家是 A-024 的訓練玩家，A-024 特徵在這裡偏樂觀（A-045 的 G），所以這個差距不一定能轉移到未見玩家。


### A-047 HumanSL 擬合資料加入 train 切分玩家（2026-10-25，事先登記；使用者同意，約 23 GPU-h）

**資料（`scripts/humansl_train_players.py select`）：** train 切分 5,767 位玩家，每人隨機最多 20 盤（seed 2027），106,057 盤、23,581,944 個盤面。只用 HumanSL 特徵（F0 mean）：A-024 在這些盤上訓練過，A-024 特徵會偏（A-045）。拆 16 個 1-GPU 作業（dev 10＋8gpus 6）。
- **洩漏處理：** Valid-S 玩家本身就是訓練玩家。S-select 那一半的 906 位玩家在 train 裡也有盤，評估 S-select 時一律排除；正式模型與 Valid-U 評估不受影響（Valid-U 與 train 玩家不重疊）。
- **比較（S-select 1,000 題，B2）：** (a) H，擬合＝Valid-S 的 S-fit＋S-extra（A-046 F0，0.7025）；(b) H，(a)＋S-select 半以外玩家的 train 盤。參考：同 (a) 擬合盤的 AH＝0.7101（A-044 (i)）。
- **規則：** 資料增益＝(b) − (a)，照實報告。若 (b) ≥ AH＋0.005 → 正式模型改為 H、以全部 Valid-S＋全部 train 盤擬合，並在 Valid-U 評分一次，與 AH 全部 Valid-S（A-044，0.6833）做玩家 bootstrap 比較；否則維持 AH，不動 Valid-U。
- 偏差提醒：S-select 是 A-024 的訓練玩家，AH 在 S-select 上偏樂觀，這個規則對 H 不利。

**結果（2026-10-25，Slurm 609260–609276 中屬本作業的 16 個，16×1 H200）：** 106,057/106,057 盤完成、0 個 query 被拒，每卡每秒 4,180–5,580，每個作業 1.1–1.5 小時，約 20 GPU-h。輸出 `outputs/humansl_train20/result.json`。

| B2，H（只用 HumanSL），S-select 1,000 題 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| (a) 擬合＝Valid-S S-fit＋S-extra（38,589 盤、4,094 人） | 0.6200 | 0.6970 | 0.7143 | 0.7788 | 0.7025 |
| (b) (a)＋train 盤（另 87,937 盤，排除 S-select 半 906 人） | 0.6329 | 0.6945 | 0.7236 | 0.8005 | **0.7129** |
| 參考：AH，(a) 的擬合盤（A-044 (i)） | 0.6312 | 0.7194 | 0.7402 | 0.7495 | 0.7101 |

- **資料增益：** (b) − (a)＝+0.0104。擬合盤數 ×3.3、玩家從 4,094 增加到約 8,950 人；增益比 A-044（盤數 ×4 時 +0.029）小，資料的邊際效益在下降。
- **主線規則：** (b) − AH＝+0.0028 <0.005 → **維持 AH（全部 Valid-S 擬合），不切換、不動 Valid-U。** 照 A-047 的偏差提醒，S-select 對 AH 偏樂觀，所以 H＋train 在未見玩家上可能其實不輸，但規則依登記執行。
- k 桶分布不同：H＋train 在 k=20 最強（0.801），AH 在 k=10、15 較強。兩者的錯誤可能不完全相關。


### A-048 強引擎著手品質特徵試算（2026-10-25，事先登記；使用者同意「先試大幅加分的策略」）

**外部資源（需在報告申報）：** KataGo 網路 `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`（katagotraining.org，b18 系列最新一版，sha256 `9d7a6afe…a51f1d`，KataGo Neural Network License：MIT 式授權，需保留版權與授權聲明）。放 `data/external/kata1-b18c384nbt-s9996604416/`。

**做法（`src/go_aicup/features/katago_engine.py`）：** 每盤一個 query，位置 0..n 全部分析，1 visit，日本規則、貼目 6.5。每個位置取黑方視角的目數領先（`scoreLead`）與勝率；第 t 手的損失＝落子方視角下位置 t 到 t+1 的目數下降。每方 7 個特徵：平均損失（截在 0–20 目）、平均勝率損失、損失 >1／>3／>7 目的比例、實際著手在強網路 policy 的平均 log-prob、與 policy 第一選擇相同的比例。10 盤測試的數值合理（D 級每手平均損失 1–4 目，第一選擇命中 20–30%）。
- **試算資料：** q1000 S-fit（9,397 盤）＋ qsel S-select（9,375 盤），4,180,000 個盤面左右，4 個 1-GPU 作業，估約 0.3 GPU-h。
- **比較（B2，只用 q1000 S-fit 擬合，S-select 1,000 題）：** AH（A-024＋HumanSL）vs AH＋E（＋引擎 14 維）；另報 H＋E、E 單獨作診斷。
- **Gate：** AH＋E − AH ≥ +0.01 → 引擎特徵擴大到全部擬合資料與 Valid-U；否則停在這裡。


### A-049 級別平衡解碼模擬（2026-10-25，CPU，只用 S-select）

`scripts/balanced_decoding.py`，輸出 `outputs/balanced_decoding.json`。機率來自 B2（A-024＋HumanSL，A-044 (i) 的擬合盤）在 S-select 1,000 題；每次抽 400 題（每個 k 100 題），300 次。Sinkhorn＝把機率調成每級總和 Q/10 後再 `score_rank`；Hungarian＝硬性每級剛好 40 題、最大化期望官方分數。

| 情境（每 k 的級別比例） | score_rank 平均 | Sinkhorn 增益（p05, p95） | Hungarian 增益（p05, p95） |
|---|---:|---|---|
| 完全平衡（每級 10 題／k） | 0.7057 | +0.0067（−0.006, +0.020） | +0.0058（−0.008, +0.019） |
| Dirichlet α=50（輕微不平衡） | 0.7057 | −0.0067（−0.047, +0.020） | −0.0093（−0.032, +0.013） |
| Dirichlet α=10 | 0.7074 | −0.0375（−0.107, +0.008） | −0.0296（−0.068, +0.001） |
| Dirichlet α=3 | 0.7052 | −0.1049（−0.255, −0.006） | −0.0695（−0.127, −0.015） |

- **結論：不是大幅加分的方向。** 就算測試集完全平衡，期望增益也只有約 +0.006，而且 5% 分位是負的；只要不平衡就扣分，最多扣到 −0.10。推測：B2 在每級等量的合成題目上擬合，機率本身已接近平衡、校準得不錯，平衡分配能修正的空間很小。
- **處置：** 擱置。11/04 後若官方或討論區明確說明測試集每級題數相同，才值得用一次 LB 探針測。平衡時兩者增益相近；不平衡時 Hungarian 扣得比 Sinkhorn 少，所以若要測，優先測 Hungarian。

**結果（2026-10-25，Slurm 610632–610635，4×1 H200，約 0.25 GPU-h）：** 18,772/18,772 盤完成、每卡每秒 4,760–5,450（每個位置只算一次，每盤 3.5 分鐘內完成）。輸出 `outputs/engine_pilot/result.json`。

| B2，S-select 1,000 題（擬合＝q1000 S-fit） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| AH（參考） | 0.6406 | 0.6832 | 0.6863 | 0.7158 | 0.6815 |
| AH＋E | 0.6246 | 0.6737 | 0.7004 | 0.7175 | 0.6791 |
| H＋E | 0.6141 | 0.6695 | 0.6813 | 0.6943 | 0.6648 |
| E 單獨 | 0.4522 | 0.4478 | 0.4743 | 0.4379 | 0.4530 |

- **Gate：FAIL**（AH＋E − AH＝−0.0024 <+0.01）。不擴大計算。
- 強引擎著手品質單獨只有 0.453，比 A-024＋C2 在 S-select 的 0.599 還低；加到 AH 上沒有增益。推測（未驗證）：(1) 1 visit 的價值估計對單手損失很雜訊；(2) 在業餘段位的範圍裡，「像哪個段位的人」（HumanSL）比「下得多好」更能區分相鄰級別，著手品質的訊號已大致被 HumanSL likelihood 涵蓋。更多 visits 能否改變結論未知，但 E 單獨這麼弱，值得花的機率低。


### D-003 本地 gate 改為「排優先順序」，不據以刪除（2026-10-25，使用者決定）

**決定：** 11/04 才有正式分數，真實測試的切法與玩家組成可能讓本地結論翻盤（CLAUDE.md §0 競賽一警訊）。之後本地 gate 只決定「先做什麼、首選提交什麼」，沒過的方向保留程式、權重與已抽特徵，T5 再用 LB 探針判斷。EXECUTION_PLAN §7 已改為「已砍、延後或保留待 LB」。

**LB 解析度提醒：** Public LB 400 題，單一提交分數的標準誤約 0.02；本地差 ≤0.01 的候選在 LB 上通常分不出來，探針應優先給「本地差距大」或「機制上可能在真實資料表現不同」的候選（例如 H＋train 與 AH 的差別在於是否依賴 A-024 對訓練玩家的記憶）。

**候選與產生提交檔所需的工作（競賽二，測試集約 1 萬盤）：**

| 候選 | 本地 | 產生提交檔還需要 |
|---|---|---|
| AH＋B2（全部 Valid-S） | Valid-U 0.6833 | 測試集 HumanSL 抽取（約 2.5 GPU-h）；管線已彩排 |
| A-024＋C2 | 0.5532 | 已就緒（1 GPU、<1 小時） |
| H＋B2（Valid-S＋train） | S-select 0.7129 | 擬合正式模型（CPU）；測試特徵與 AH 共用 |
| AH＋E（強引擎） | S-select −0.002 | 測試集引擎抽取（約 0.15 GPU-h）；用 18,772 盤試算資料擬合 |
| 貼目 0 | S-select +0.005 | 測試集以貼目 0 重抽 HumanSL（約 2.5 GPU-h）；擬合資料只有 18,772 盤 |
| 平衡解碼 | 平衡時 +0.006 | CPU，可套在任何候選的機率上 |
| V2 14 epochs／其他 seed | 本地差 0.05–0.08 | `strength_predict` 換 run-dir 與 T_k（小改） |


### A-050 HumanSL PyTorch 路徑第 0 階段：重現引擎 policy（2026-10-25，事先登記；使用者同意）

**外部資源：** KataGo v1.18.2 原始碼（`v1.18.2.tar.gz`，sha256 `fefd554b…f8bf9`，MIT；使用 `python/katago` 的 Board、Features、SGFMetadata、model_pytorch、load_model）與 `b18c384nbt-humanv0.ckpt`（v1.15.0 release，sha256 `84b23476…c8b27`，KataGo Neural Network License）。放 `data/external/`。

**做法（`scripts/humansl_torch_check.py`）：** 用 KataGo 自己的 Python 程式載入 ckpt、重播棋局、算輸入特徵；每個 profile 的 metadata 照 `cpp/neuralnet/sgfmetadata.cpp` 的 `makeBasicRankProfile`（雙方同段位、人類、計分未知、讀秒 1200 秒＋30 秒×5、日期 2020-03-01、來源 KGS）；policy 取主 head、非法著手遮掉後做 softmax（pass 保留）。與 A-039 引擎的每手 log p 比較（試算 shard 0 的前 100 盤 × 15 profiles）。
- **Gate：** 每手 |Δlog-p| 平均 ≤3e-3 且 p99 ≤2e-2，且每盤特徵（本人／對手平均）最大差的 p99 ≤1e-2。引擎自身雜訊（A-041）：平均 1.1e-3、p99 5.9e-3。
- **2 盤小測：** 一般權重平均 1.19e-3、p99 6.5e-3、最大 1.2e-2；SWA 權重平均 5.5e-2、p99 0.34 → 引擎的 `.bin.gz` 對應的是一般權重，之後只用一般權重。

**結果（2026-10-25，1×H200 dev，100 盤 × 15 profiles＝325,500 筆，約 4 分鐘）：** 輸出 `outputs/humansl_torch_check.json`。

| | 每手 \|Δlog-p\| 平均 | p99 | 最大 | 每盤特徵最大差 p99 | 每盤特徵最大差 max |
|---|---:|---:|---:|---:|---:|
| 全部手數 | 1.37e-3 | 6.7e-3 | 1.25 | **1.21e-2** | 1.38e-2 |
| 診斷：只看第 10 手以後 | 1.26e-3 | 6.6e-3 | 0.71 | 1.09e-3 | 4.9e-3 |

- **Gate：照登記 FAIL**（每盤特徵 p99 0.0121 >0.01；前兩項通過）。
- **原因已找到，是引擎端的行為，不是 PyTorch 程式錯誤：** 差 >0.05 的 244 筆幾乎都在第 3–4 手，且不同盤的相同局面引擎值完全一樣。`cpp/neuralnet/nninputs.cpp` 的 `NNInputs::getHash` 註解寫明 KataGo 為了效能「容忍在不同歷史的換序局面之間重用快取評估」（快取鍵含 profile，`nneval.cpp:1131`，但不含手順）。開局四角常以不同順序到達同一局面，所以引擎回傳的可能是別盤歷史算出的機率；PyTorch 沒有快取，用的是每盤真正的歷史。排除前 10 手後每盤特徵差 p99 降到 1.1e-3，支持這個解釋。這也解釋了 A-041／A-042 兩次引擎抽取之間最大差 1.25 的來源。
- **影響：** 已抽的引擎特徵在開局有少量「換序污染」（每盤特徵約 0.01 量級），對 A-040～A-047 的分數影響推測很小（未驗證）。PyTorch 路徑沒有這個問題。
- **權重：** 引擎 `.bin.gz` 對應 ckpt 的一般權重，不是 SWA。
- **吞吐量問題：** 單一 Python 程序 100 盤約 240 秒（含特徵計算、合法性檢查與 15 個 profile 的前向），約 0.4 盤／秒。瓶頸在 KataGo 的純 Python 棋盤與特徵（含征子偵測）；100 萬盤需要大量平行化或改寫，第 1 階段前要先處理。


### A-051 HumanSL 主幹表示＋B2（C1，第 1 階段；2026-10-25，事先登記；使用者同意）

**表示（`src/go_aicup/features/humansl_torch.py`）：** PyTorch 版 HumanSL（A-050，一般權重、無 NN 快取）。每手取落子前局面的 `trunkfinal`（384 通道）兩種向量：全盤平均、實際落子點的向量；本人／對手分別對每盤平均。兩種 metadata 條件：`rank_1d`（照 `makeBasicRankProfile`）與「段位未知」（同 profile、雙方 rank 標為 unknown）。每盤 2×2×2×384＝3,072 維（float16）。40 盤測試：每 GPU 作業（11 個 CPU 程序算特徵）每秒約 650 個盤面，瓶頸在 KataGo 的純 Python 特徵。
- **資料：** Valid-S 擬合盤（q1000 S-fit＋vs_rest S-fit／S-extra，38,589 盤）＋ S-select（9,375 盤），共 47,964 盤、約 1,080 萬個盤面；10 個 1-GPU 作業，估約 4.3 GPU-h。
- **比較（B2，擬合＝上述 38,589 盤，S-select 1,000 題）：** 表示先用 PCA（只在擬合盤上擬合）降到 128 維。AH（參考，A-044 (i) 0.7101）、AH＋Emb、H＋Emb、Emb 單獨。
- **Gate：** AH＋Emb − AH ≥ +0.01 → 進一步（更多資料、非線性 head、或 C2 微調）；否則依 D-003 保留。

**結果（2026-10-25，Slurm 611261–611270，10×1 H200）：** 47,964/47,964 盤完成，每個作業每秒約 1,300 個盤面、約 13.5 分鐘，約 2.3 GPU-h。PCA 128 維。輸出 `outputs/emb_stage1/result.json`。

| B2，S-select 1,000 題 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| AH（參考） | 0.6312 | 0.7194 | 0.7402 | 0.7495 | 0.7101 |
| AH＋Emb | 0.6170 | 0.6690 | 0.6792 | 0.6509 | 0.6540 |
| H＋Emb | 0.6286 | 0.6541 | 0.6752 | 0.6593 | 0.6543 |
| Emb 單獨 | 0.6183 | 0.6448 | 0.6392 | 0.6202 | 0.6306 |

- **Gate：FAIL**（AH＋Emb − AH＝−0.0561）。依 D-003 保留表示與程式。
- **但表示本身有訊號：** 只用主幹表示 0.6306，高於 A-024＋C2 在 S-select 的 0.5993（A-045）。加進 AH 後反而下降，主要在大 k（k=20：0.750 → 0.651）。推測（未驗證）：B2 每個 k 的 LR 固定 C=1；k=20 的合成題目只能從每人 ≥20 盤的少數玩家抽，維度從 40 增加到 168 後記住了玩家。這是正則化問題，不是表示無用的證據；下一步 A-052 用擬合集內的交叉驗證選正則化，再評一次。


### A-052 C1 後續：在擬合集內選 B2 的正則化（2026-10-25，事先登記；CPU）

`scripts/emb_regularization.py`。擬合盤同 A-051（38,589 盤）。在其中以玩家分兩折交叉驗證：一折擬合 B2，另一折玩家的合成題目評分（每個 k 1,000 題：隨機一位盤數足夠的玩家，取其 k 盤）；PCA 只在訓練折上擬合。網格：C ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1}；含表示的組合另選 PCA 維度 ∈ {16, 32, 64, 128}。**S-select 不參與選擇。** AH 也用同一程序選 C（公平比較）。各自選定後，以全部擬合盤擬合，在 S-select 1,000 題評分一次。
- **Gate：** AH＋Emb（選定設定）− AH（選定設定）≥ +0.01。

**結果（2026-10-25，Slurm 611409，dev，CPU 12 核，1.5 分鐘）：** 輸出 `outputs/emb_regularization.json`。

| | 選定設定 | 擬合集內 CV Equal-k | S-select Equal-k（k=5／10／15／20） |
|---|---|---:|---|
| AH | C=0.3 | 0.7072 | **0.7114**（0.638／0.727／0.728／0.753） |
| AH＋Emb | PCA 16 維、C=1.0 | 0.6822 | 0.6764（0.631／0.702／0.705／0.667） |

- **Gate：FAIL**（−0.035）。CV 是在沒見過的擬合玩家上評分，加表示一樣變差，所以不是 C 沒調好，而是泛化失敗；PCA 維度越少越好（16 > 32 > 64 > 128），代表表示保留越多越傷。
- 推測（未驗證）：主幹表示強烈編碼了玩家的個人風格（同一人的盤彼此很像），B2 的合成題目重複使用同一批玩家，LR 學到的是「認人」的方向，換到新玩家就失效。只用表示仍有 0.63（A-051），所以訊號存在，只是線性 head 加在 AH 上用不起來。
- AH 選到 C=0.3 時 S-select 0.7114，比 C=1 的 0.7101 高 0.0013（雜訊內）；正式模型維持 C=1。
- **處置：** 依 D-003 保留表示、程式與結果。C1 這條線先停在這裡；C2（用幽玄級別微調 policy）仍是未試的選項。


### A-053 競賽一 GE2E 保險模型接進提交管線（2026-10-25，事先登記；T2 待辦）

**程式：** `tasks/identity/insurance.py`：release GameStore → `write_store_cache`（與競賽二同一套重播）→ A-035 編碼器（`outputs/pid_ge2e_v1_short`）對每盤 ply ≥20 的黑方手取平均（依存檔顏色選手，不靠奇偶）→ `ge2e.npz`。`identity/predict.py --insurance-dir`：開局分數＋λ×w×cos（λ＝8、w 取自 `eval_summary.json`／`model.pt`；與 `scripts/identity_ge2e_eval.py` 相同：每盤向量先正規化、再平均、再正規化），之後照常 Sinkhorn＋Hungarian（tau 用開局模型的值，與 A-035 相同）；沒有可用開局盤的題目只靠保險分數排序。
- **彩排 gate（假 release：public＝合成 split 1、private＝split 2）：** (a) release 路徑每盤向量與 A-035 訓練路徑向量（`embeddings_split{1,2}.npz`）的 cosine 全部 ≥0.999；(b) 融合＋Hungarian 的 top-5 與 A-035 驗證路徑（同一套開局分數＋訓練路徑向量）相同 ≥398/400，不同者須為近乎平手；(c) 分數與 A-035 的融合＋Hungarian（split 1／2 皆 1.0）相差 ≤0.005。bf16 與批次組成會造成微小數值差（A-036），所以 (a)(b) 不要求逐位相同。

**結果（2026-10-25，Slurm 611993，1×H200 dev，9.6 分鐘，約 0.16 GPU-h）：** 4 個 store 共 90,000 盤全部嵌入（pool 各 40,000 盤約 404 萬手；題目各 5,000 盤約 50 萬手）。輸出 `outputs/rehearsal/identity_insurance/rehearsal.json`。

| | (a) cosine 最小值（45,000 盤） | (b) top-5 相同 | (c) 分數（k=5／10／15／20；全體） |
|---|---:|---:|---|
| public（split 1） | 0.999996 | 400/400 | 1.0／1.0／1.0／1.0；1.0 |
| private（split 2） | 1.000000 | 400/400 | 1.0／1.0／1.0／1.0；1.0 |

- **Gate：PASS**（三項都通過）。λ＝8、w＝10.026；沒有題目或候選人缺保險向量。T2 競賽一的待辦完成。
- 合成驗證仍飽和（1.0），只證明管線正確；真實資料表現未知（CLAUDE.md §0 警訊照舊）。
