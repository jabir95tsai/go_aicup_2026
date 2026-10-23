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


### D-002 競賽一最糟情況規劃（2026-10-23）

**性質：** 使用者確認的計畫修訂；僅文件修改，沒有新實驗或分數。內容寫入 `EXECUTION_PLAN.md`「競賽一最糟情況規劃」，R2 改為先建壓力測試台。

- 理由：A-027 合成驗證已飽和（≥0.99），且與官方 baseline 0.2845 的落差未解釋；改以最糟情況（W1–W6）下的保底分數規劃。
- R2 先做 CPU 壓力測試台 S1–S4（開局漂移、去開局、非一對一；方位沿用 E0），再做不靠開局的風格模型。
- Gate G1–G4：S2 下風格模型全體 ≥ 0.30；S0 合併退步 ≤ 0.005；S1（p=100）合併 ≥ 只用風格；Hungarian 預設關閉，11/04 確認一對一後才開。
- 11/04 先交保底檔（D4 正規化 NB＋Sinkhorn），再做 Day-0 稽核與開局／風格／合併三份受控探針。
- 風格模型在圍棋去開局後的可達分數未知；GPU 估時未知，須先短跑回報。
