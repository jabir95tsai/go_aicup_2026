# AI CUP 2026 圍棋競賽：研究與實作參考

整理／查核日期：2026-09-23。用途：從匿名人類棋譜集預測 **棋手身分 Top-5** 與 **棋力等級**。

本文件優先收錄論文原文、作者網站、官方 GitHub 與工具文件。文中的「本專案建議」是依來源提出的實驗假設，尚未在本次資料集驗證；所有文獻成績都不是本次競賽的預期分數。本次完成線上資料查核，未下載完整訓練集、模型權重或執行外部 repo。

## 1. 優先閱讀與採用順序

| 優先 | 資料／方法 | 對本專案最有價值的部分 | 建議用途 |
| --- | --- | --- | --- |
| P0 | [ICLR 2025 Strength Estimator](https://arxiv.org/abs/2502.17109) | 從盤面與實際落子學出連續棋力分數，再跨手數聚合 | 棋力預測的主要研究基礎 |
| P0 | [AIIDE 2025 多棋力 policy](https://ojs.aaai.org/index.php/AIIDE/article/download/36812/38950/40889) | 合併 strength score、人類 policy likelihood、強引擎評估；另測同一棋手的多盤資料 | 棋力特徵與融合模型；特別注意評估設定 |
| P0 | [NeurIPS 2021 Behavioral Stylometry](https://papers.nips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html) | 落子 → 單局向量 → 棋手向量，對候選棋手做 few-shot matching | 棋手辨識最貼近的可遷移架構，原領域是西洋棋 |
| P0 | [KataGo HumanSL 文件](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide) | 取得不同棋力人類會選擇某一步的機率 | 小樣本特徵實驗、教師模型 |
| P0 | [sgfmill](https://github.com/mattheww/sgfmill) 與 [SGF 規格](https://www.red-bean.com/sgf/go.html) | SGF 解析、盤面還原、提子與 pass | 所有模型共用的資料管線 |
| P1 | [GoStyle／CIG 2015](https://arxiv.org/abs/1512.08969) | 局部棋形與逐手統計聚合 | 低成本、可解釋的 baseline |
| P1 | [TPNC 2016 棋力分類](https://link.springer.com/chapter/10.1007/978-3-319-49001-4_15) | 盤面 CNN，加總單盤預測 | 不依賴外部引擎的 baseline |
| P1 | [KataGo 特徵棋力模型](https://github.com/Animiral/go-strength-model) | 使用落子位置的內部特徵，避免儲存整個 trunk | 特徵快取與凍結 encoder 實驗 |
| P1 | [Deep Sets](https://arxiv.org/abs/1703.06114)、[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | 處理數量不同且無自然順序的棋譜集合 | 多盤聚合 |
| P2 | 舊 AI CUP 棋風研究、落子模仿研究 | 特徵與網路設計參考 | 補充閱讀，不套用舊競賽分數 |

**建議主線：** 先用官方資料訓練小型盤面／落子 encoder，兩題共用解析與快取；身分題做 metric learning，棋力題做分類／序位學習。再用少量 KataGo HumanSL 特徵測試是否值得擴大引擎分析。

## 2. 官方來源與任務邊界

| 來源 | 本次確認／用途 |
| --- | --- |
| [競賽一：棋手辨識](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688) | 正確棋手排第 r 名得 `exp(-(r-1))`，只計前五名；未入榜為 0 |
| [競賽二：棋力預測](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403) | 完全正確得 1，相差一級得 `1/e`，其餘 0 |
| [官方訓練資料 repo](https://github.com/AILAB-NDHU/aicup2026-go-dataset) | 100 萬盤匿名棋譜；10 級；CSV 含 `player_id, game_id, rank, color, sgf_content` |
| [官方資料 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) | 下載與版本追蹤；完整資料仍須另行下載、驗證雜湊與分析分布 |
| [AI CUP 最新公告](https://www.aicup.tw/news) | 後續追蹤競賽課程、baseline 與公告 |

本專案棋力順序沿用 `AGENTS.md`：`D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D`。不要自行把 D/C/B/A 換算成其他平台的 kyu 等級。`color` 用於指定待評估棋手，盤面仍須按雙方完整落子還原；`rank`、`player_id` 不可作為預測輸入。

**尚待官方測試格式確認：** 每個 query 的盤數與組成、候選棋手如何提供、候選是否包含訓練未見棋手、query 是否固定同一棋手、sample submission 與缺失值規則。模型介面應保留彈性，不把某篇論文的 reference/query 設定當成本次規則。

官方規則要求揭露外部資料、開源程式及生成式 AI 的來源與用途，且使用資源須有合法授權；公開可下載不等同已確認適合本次使用。此處只建立資源清單，實際採用時再逐項記錄版本、授權與用途。[官方規則](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)

## 3. 直接相關：棋力預測

### R01 — Strength Estimation and Human-Like Strength Adjustment in Games

Chun-Jung Chen、Chung-Chin Shih、Ti-Rong Wu，ICLR 2025。來源：[論文](https://arxiv.org/abs/2502.17109)、[PDF](https://arxiv.org/pdf/2502.17109)、[正式 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/17a1a1439421f1837e10cd612bf92861-Abstract-Conference.html)、[作者程式](https://github.com/rlglab/strength-estimator)。查核：全文方法與 repo README。

- **方法：** 對 `(盤面, 實際落子)` 預測 strength score `β`，以不同棋力群組的平均分數建立排序損失。`SE∞` 額外使用隨機合法落子作弱棋力參照。另有調整對弈強度的 SE-MCTS。
- **結果：** 摘要報告觀察 15 盤時，圍棋 rank prediction accuracy 超過 80%。需連同原文的群組取樣設定理解，不能解讀成任意同一棋手 15 盤都有此準確率。
- **本專案建議：** 優先重現 score 與聚合部分；不必為了預測標籤先重現 SE-MCTS。將輸出重新校準至官方 10 級。
- **成本／限制：** repo 建立於 MiniZero，README 要求 Linux、容器及 NVIDIA GPU；不是開箱即用的 Windows／ROCm 專案。GitHub 標示 GPL-3.0。

### R02 — Policies of Multiple Skill Levels for Better Strength Estimation in Games

Kyota Kuboki、Tatsuyoshi Ogawa、Chu-Hsuan Hsueh、Shi-Jim Yen、Kokolo Ikeda，AIIDE 2025。來源：[正式版 PDF](https://ojs.aaai.org/index.php/AIIDE/article/download/36812/38950/40889)、[arXiv](https://arxiv.org/abs/2505.00279)、[作者程式](https://github.com/kyotakuboki/go-chess-strength-estimation)。查核：正式版方法、Table 6 與 repo README。

- **方法：** 合併 s-score 統計、多棋力 policy 對實際落子的幾何平均機率、強引擎 advantage loss 統計，以 LightGBM 回歸棋力。
- **關鍵證據：** 正式版 Table 6 的另一組測試中，15 個 player instances 的 rank-group 取樣準確率為 **88.7%**，同一棋手取樣則為 **61.6%**；一個 instance 是一名棋手在一盤內的落子。兩種設定不可混用。結論的 87.0% 屬另一主實驗；arXiv 摘要亦有不同盤數的成績。
- **本專案建議：** 融合 score 與多棋力 likelihood；以符合 query 組成的方式聚合，先比較只用 HumanSL 與加上 s-score 的效益。
- **實作入口：** repo 的 `go/sgf_analyzer.py`、`go/go_engines.py`，以及 `testing_by_player/` 流程。引擎／權重另取；本次未確認整個 repo 的授權條款。

### R03 — Determining Player Skill in the Game of Go with Deep Neural Networks

Josef Moudřík、Roman Neruda，TPNC 2016，pp. 188–195。來源：[出版頁／摘要](https://link.springer.com/chapter/10.1007/978-3-319-49001-4_15)、[作者資源頁](https://moudrik.cz/archive/2016-08-02/)。查核：出版摘要與資源索引，未完整重現原模型。

- **方法與結果：** 以 188,700 盤訓練 CNN，預測三種棋力群組；單盤面準確率 71.5%，聚合一盤後 77.9%。
- **本專案建議：** 作為「直接分類＋多手聚合」的低成本基準，先改為官方 10 類。用 `state-only` 和 `state+action` 比較模型究竟靠局面還是棋手的決策。
- **限制：** 三分類遠比本次 10 級粗；作者頁雖列出模型、權重與資料連結，本次未逐一驗證下載和相依版本。

### R04 — Evaluating Go Game Records for Prediction of Player Attributes

Josef Moudřík、Petr Baudiš、Roman Neruda，IEEE CIG 2015，pp. 162–168。來源：[論文](https://arxiv.org/abs/1512.08969)、[PDF](https://arxiv.org/pdf/1512.08969)、[GoStyle 專案](https://gostyle.j2m.cz/)。查核：論文與專案說明。

- **方法：** 萃取每手的局部棋形、先手／後手等統計，再跨棋譜聚合，用機器學習預測棋力與棋風屬性。
- **本專案建議：** 先做便宜的特徵 baseline：落子距邊、前一手距離、局部黑白分布、提子數、氣數、分階段落子分布；這是本專案的候選特徵，不代表原文逐項都有。
- **限制：** 「攻擊性／實地性」不是匿名棋手 ID；將這些特徵用於辨識只是可測試的延伸。

配套閱讀：Moudřík、Neruda 的 [Evolving Non-linear Stacking Ensembles for Prediction of Go Player Attributes](https://arxiv.org/abs/1512.09254)，SSCI 2015。以演化搜尋組合 stacking 模型；適合參考特徵融合思路。對本專案而言，先完成嚴格的 out-of-fold stacking，比直接引入複雜搜尋更重要。

### R05 — KataGo 內部表示作為棋力特徵

來源：[Animiral/go-strength-model](https://github.com/Animiral/go-strength-model)、Peter Neubauer（2024）[Strength Estimation in the Game of Go](https://www.ac.tuwien.ac.at/files/pub/neubauer_24.pdf)。查核：repo README；論文作延伸閱讀。

- **方法：** 使用 KataGo 的 head、整個 trunk 或實際落子位置的 `pick` 特徵，再以 PyTorch 模型估計棋力。
- **工程價值：** repo 說明可續跑的特徵預計算與一致性檢查；整個 trunk 的時間／空間成本很大，`pick` 是值得比較的折衷。
- **限制：** 修改版 KataGo 明示只支援 CUDA；原流程還依賴棋手姓名、時間等 metadata。本次只能以匿名 `player_id` 做分組，且不可假設缺少的 metadata 存在。repo 有 MIT／GPL 等分檔授權，採用前讀對應 LICENSE。
- **本專案建議：** 先凍結 encoder、快取少量特徵測下游模型；若使用上游 KataGo 的多後端，不要假設此 fork 自動具有相同支援。

## 4. 直接相關：棋手辨識與棋風

### R06 — Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess

Reid McIlroy-Young 等，NeurIPS 2021。來源：[正式論文頁](https://papers.nips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html)、[PDF](https://papers.nips.cc/paper_files/paper/2021/file/ccf8111910291ba472b385e9c5f59099-Paper.pdf)、[作者 repo](https://github.com/CSSLab/behavioral-stylometry)。查核：全文模型／評估與 README。

- **方法：** Residual CNN 編碼落子前後盤面，Transformer 聚合手順成單局向量，再平均多局成棋手向量；用 GE2E loss 拉近同棋手、拉遠不同棋手，以 cosine similarity 比對候選。
- **結果的條件：** 作者摘要報告在數千候選中達 98% 識別率；正文評估使用每人 **100 盤 reference 與另 100 盤 query**，不是一盤就達 98%。
- **可用性：** GitHub 明示程式碼無法直接下載，須向作者申請。本次未申請；可讀論文自行實作。
- **本專案建議：** 改為 19×19、依 `color` 編碼目標棋手，保留落子 action；比較 mean pooling 與小型 Transformer。候選棋手可先計算 prototype，query 與所有候選比對後輸出 Top-5。

### R07 — Learning Models of Individual Behavior in Chess

McIlroy-Young 等，KDD 2022；預印本始於 2020。來源：[作者預印本](https://arxiv.org/abs/2008.10086)、[作者實驗室的出版資訊](https://csslab.cs.toronto.edu/research/human-ai/)。查核：摘要與作者出版資訊。

- **方法：** 從群體人類落子模型出發，針對個別棋手微調；個人化模型也能以落子相容性辨認棋手。
- **本專案建議：** 作為第二階段 reranker 的靈感：先用 embedding 找少量候選，再比較 query 在候選條件 policy 下的平均 log-likelihood。
- **限制：** 每位棋手各訓練完整模型成本高、長尾棋手樣本不足；共享 backbone 搭小型條件模組是本專案的延伸假設。

### R08 — Identifying fighting, balanced, and territorial go player styles with deep learning

Entertainment Computing，Vol. 56，2026，101036；DOI `10.1016/j.entcom.2025.101036`。來源：[出版頁](https://www.sciencedirect.com/science/article/pii/S1875952125001168)。查核範圍：可檢索的出版摘要與方法概述；直接全文存取未成功。

- **方法／結果：** CNN、residual connections 與 attention，分類 Fighting／Balanced／Territorial。摘要報告 70,000 盤原始棋譜、483,712 個增強樣本，以及另 10,000 盤測試上的 82.6% accuracy。
- **本專案建議：** 參考三種風格的局部與全局特徵，但不把三分類當成棋手 ID。尚未核實公開作者程式、標註細節及是否按棋手隔離測試，故不列為首要復現目標。

### R09 — 舊 AI CUP 競賽報告

Shi-Jim Yen、Xian-Dong Chiu、Lisye，**Go Strength Imitation and Playstyle Recognition Competition in AICUP**，ICGA Journal 47(1)，48–52，2025。來源：[出版頁](https://journals.sagepub.com/doi/10.1177/13896911251317623)、[PDF](https://journals.sagepub.com/doi/pdf/10.1177/13896911251317623)。查核：出版摘要與報告可讀內容。

這是 2023–2024 競賽的主辦方回顧，可了解 CNN／ResNet 等方法在舊賽事的使用。**舊任務的落子模仿、棋風分類，與 2026 的棋手身分檢索、10 級棋力預測不同。** 本次未核實可直接用於 2026 的歷屆冠軍開源方案，因此不把一般參賽 repo 當成得獎解法。

補充來源：[AI CUP 舊資料集入口](https://www.aicup.tw/%E5%89%AF%E6%9C%AC-%E8%B3%87%E6%96%99%E9%9B%86%E4%B8%8B%E8%BC%89)。舊資料用途限制及本次允許範圍須分開確認。

## 5. 可直接遷移的通用方法

以下來源不是在 AI CUP 2026 上驗證過的方案；價值在於其學習目標或聚合結構。

| 編號／來源 | 原方法 | 本專案建議 |
| --- | --- | --- |
| R10 [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)，Wan 等，ICASSP 2018 | 在同一 batch 內比較樣本與身分中心 | 把「說話者／語音」換成「棋手／棋局」；同棋手中心採 leave-one-out，避免自身相似度造成簡單解 |
| R11 [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)，Snell 等，NeurIPS 2017 | 以支援樣本形成類別 prototype | 每位候選由 reference games 產生 prototype，支援新增候選而不擴充分類 head |
| R12 [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)，Khosla 等，NeurIPS 2020；[作者 PyTorch 實作](https://github.com/HobbitLong/SupContrast) | 同標籤多正樣本的對比學習 | 同棋手不同棋局為正樣本；同棋力不同棋手作 hard negatives，測試是否學到超越棋力的個人特徵 |
| R13 [Deep Sets](https://arxiv.org/abs/1703.06114)，Zaheer 等，NeurIPS 2017 | 置換不變的集合函數 | 單局 embedding 做 mean/sum pooling，再接 MLP；跨棋局不憑空加入時序 |
| R14 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)，Lee 等，ICML 2019；[官方 PyTorch repo](https://github.com/juho-lee/set_transformer) | attention 處理集合及元素互動 | mean pooling 不足時再試；變動盤數需要 mask；局內手順仍須保留 |
| R15 [CORAL](https://arxiv.org/abs/1901.07884)，Cao 等，預印本 2019／Pattern Recognition Letters 2020 | 一致的序位門檻分類 | 把 10 級轉為 9 個「是否高於門檻」任務，和普通 10 類 cross-entropy 比較 |
| R16 [CORN](https://arxiv.org/abs/2111.08851)，Shi 等，2021 預印本 | 以條件機率處理序位一致性 | 保留完整 rank 機率，方便依官方報酬決策；不要只回歸再四捨五入 |

## 6. 圍棋表示與工具庫

### R17 — KataGo 與 HumanSL：最值得優先測試的現成教師

來源：[KataGo repo](https://github.com/lightvector/KataGo)、[Analysis Engine API](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)、[HumanSL 首次發布與模型入口](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)、David J. Wu 的 [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565)。查核：README、API 與 release 說明。

API 提供批次分析、`policy`、`humanPolicy`、局面評估與 ownership。HumanSL 由 `humanSLProfile` 指定棋力 profile；文件示例將其放在 `overrideSettings`，並用 `includePolicy: true` 取得機率。若要保留人類手順資訊，文件建議 `ignorePreRootHistory: false`。`humanPolicy` 是網路機率，不是搜尋 visit 比例。[API 文件](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)

目前 README 列出 CUDA、TensorRT、ROCm、OpenCL、ONNX、Eigen 等後端；是否適合本機仍需按 GPU／OS／驅動測試。這是引擎推論支援，不能推論所有周邊研究程式都能在 ROCm 執行。[KataGo README](https://github.com/lightvector/KataGo)

**本專案建議：**

1. 從各棋力、黑白、棋局階段抽取小批盤面，先測每秒處理量和特徵效益。
2. 對所有待測 profile 都計算 likelihood，再用官方標籤訓練映射；不要用真實 `rank` 選唯一 profile 後拿去當推論特徵。
3. 規則與貼目在本次 SGF 未必存在。採固定假設時寫入 config，並測試敏感性；score/value 類特徵尤其可能受影響。
4. 不先把 100 萬盤的每一手全送入搜尋。先測 raw policy／低成本特徵，再決定增加 visits 是否值得。
5. 快取記錄引擎版本、模型雜湊、profile、盤面歷史、規則、貼目、對稱處理與分析參數，避免混用不同定義的特徵。

### R18 — SGF 與盤面重建

來源：[SGF FF[4] Go 規格](https://www.red-bean.com/sgf/go.html)、[sgfmill repo](https://github.com/mattheww/sgfmill)、[sgfmill boards 文件](https://mjw.woodcraft.me.uk/sgfmill/doc/1.1.1/boards.html)。

- SGF 的 pass 可寫成空座標；規格亦說明舊式 `tt` 表示法在適用棋盤大小下的相容處理。座標轉換須依規格與函式庫定義測試。
- `sgfmill.boards.Board.play()` 可處理提子（含自殺提子），**不強制執行 ko 規則，也不維護完整歷史**。因此它不能單獨作為完整合法性驗證器。
- 本專案要把「解析 SGF」「還原盤面」「依指定規則驗證合法性」分清楚。不能只把歷史落子塗到平面上，否則提子後局面會錯。

### R19 — 落子模仿預訓練的背景

| 來源 | 值得借鏡 | 適用界線 |
| --- | --- | --- |
| [Move Evaluation in Go Using Deep Convolutional Neural Networks](https://arxiv.org/abs/1412.6564)，Maddison 等，2014 預印本 | 以專家落子監督盤面 CNN | 預測下一手是表示學習輔助目標，並非直接身分／棋力答案 |
| [Training Deep Convolutional Neural Networks to Play Go](https://proceedings.mlr.press/v37/clark15.pdf)，Clark、Storkey，ICML 2015 | 棋盤對稱與 CNN 的人類落子學習 | 可作不依賴外部權重的預訓練方向 |
| [Better Computer Go Player with Neural Network and Long-term Prediction](https://arxiv.org/abs/1511.06410)，Tian、Zhu，ICLR 2016；[DarkforestGo](https://github.com/facebookresearch/darkforestGo) | 多步預測、人類棋譜特徵與訓練管線 | 舊 Torch／Lua 生態，較適合讀方法，不是本專案首選底座 |
| [lightvector/GoNN](https://github.com/lightvector/GoNN) | 作者記錄 rank-conditioned policy、歷史資訊、global pooling 等實驗 | 研究 sandbox，可參考消融設計，勿直接視為現代訓練框架 |

## 7. GitHub 採用速查

「確認」指已讀到 README／頁面內容，**不代表已安裝、測試或確認所有相依授權**。

| Repo | 現成資產 | 本專案採用判斷 |
| --- | --- | --- |
| [rlglab/strength-estimator](https://github.com/rlglab/strength-estimator) | SE 訓練、配置、rank estimation | 優先讀；NVIDIA／Linux 流程需適配；GPL-3.0 |
| [kyotakuboki/go-chess-strength-estimation](https://github.com/kyotakuboki/go-chess-strength-estimation) | SGF 引擎分析、多來源特徵、LightGBM、按棋手測試 | 優先參考 Go 部分；引擎和權重另取；授權待確認 |
| [lightvector/KataGo](https://github.com/lightvector/KataGo) | 引擎、HumanSL、分析 API、訓練相關程式 | 優先用 API 做小型教師特徵實驗；程式與模型授權分別查核 |
| [Animiral/go-strength-model](https://github.com/Animiral/go-strength-model) | 特徵抽取、PyTorch 棋力模型、快取檢查 | 有價值但 fork 綁 CUDA；分檔授權 |
| [CSSLab/behavioral-stylometry](https://github.com/CSSLab/behavioral-stylometry) | 論文入口與聯絡說明 | 程式未提供直接下載，不能列為可 clone 復現的實作 |
| [mattheww/sgfmill](https://github.com/mattheww/sgfmill) | Python SGF parser、board utilities | 優先資料工具；完整合法性檢查另補 |
| [juho-lee/set_transformer](https://github.com/juho-lee/set_transformer) | 官方 PyTorch 集合模型 | 可讀核心 modules；GitHub 標示 MIT |
| [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast) | PyTorch supervised contrastive loss | 參考 loss 與 batch 結構；影像資料 loader 需重寫 |

## 8. 外部資料與其他資源

| 資源 | 內容／價值 | 使用前須解決的問題 |
| --- | --- | --- |
| [featurecat/go-dataset](https://github.com/featurecat/go-dataset) | repo 自述 2,110 萬盤，涵蓋多種棋力；R02 使用的 Fox 資料來源 | 平台段級位差異、與官方資料重複、資料授權；repo 標示 GPL-3.0 不應被解讀成已釐清所有原始棋譜權利 |
| [yenw/computer-go-dataset](https://github.com/yenw/computer-go-dataset) | 收集多種電腦圍棋相關棋譜，包含不同來源和 AI 對局 | 不可把 AI 自我對弈當人類棋手風格標籤；逐子資料集核對來源、內容與授權 |
| [GoStyle 專案頁](https://gostyle.j2m.cz/) | 早期棋風量化、專家風格問卷與研究入口 | 風格標籤具主觀性，與匿名 ID 不同；網站存在不代表服務仍可運作 |
| [TPNC 2016 作者資源頁](https://moudrik.cz/archive/2016-08-02/) | 模型、訓練程式、權重與資料索引 | 舊相依環境、下載有效性與授權尚待逐項驗證 |

**建議先用官方 100 萬盤完成可信驗證。** 外部資料最合理的初期用途是預訓練或 domain-shift 對照；不要以外部具名棋譜去回推本次匿名棋手的真實身分。

## 9. 本專案可實作的方法草圖

以下均為本文件提出的設計建議，並非已執行實驗。

### 9.1 共同輸入表示

以目標棋手一手落子之前的盤面 `s_t`、該手 action `a_t` 為基本單位。候選 channel 包括目標方／對手棋子、最近落子、實際 action mask、氣數、歷史盤面與手數；pass 用獨立表示。盤面資訊可含對手落子，但估計目標棋手決策品質時只聚合目標方的 action。

先比較 `state-only`、`state+action`、`before+after`。若 state-only 已很強，須檢查是否靠對手強度、特定局面或資料來源間接分類，而不是目標棋手的選擇。

棋盤 D4 旋轉／鏡射要同步套用盤面、歷史與 action；同一段手順使用一致的變換。棋力題通常適合測對稱增強；身分題可能含方向偏好，需比較「無增強／訓練增強／推論平均」而不是預設越多越好。

### 9.2 棋力：multi-policy likelihood

受 R02／HumanSL 啟發，對每種 profile `r` 計算：

```text
L_r(G) = mean over target-player moves t of log(max(pi_r(a_t | s_t), epsilon))
features(G) = [L_r1(G), ..., L_rK(G), optional strength/quality statistics]
```

這是幾何平均 likelihood 的 log 形式，可避免直接連乘下溢；每手平均也避免長棋局只因乘數更多而被扣分。跨盤先測每盤等權，再比較按有效手數加權。不要把不同 profile 的 likelihood 未經校準就當成官方 10 級的 posterior。

### 9.3 棋手：多盤向量檢索

```text
move embedding = CNN(state, action)
game embedding = masked pooling or sequence encoder(move embeddings)
player prototype = normalized mean(reference game embeddings)
query embedding = normalized mean(query game embeddings)
candidate score = cosine(query embedding, player prototype)
```

用 GE2E／SupCon／prototype 訓練，先驗證純 embedding，再測 ID classification 輔助 loss。同一個正樣本組應含不同棋局；只把一盤棋的旋轉版本當同棋手正樣本，容易學成棋局比對。

### 9.4 依官方報酬決策，而非只看 accuracy

棋力題若模型輸出校準後的 `p_k = P(rank=k | query)`，依官方得分自行推導，預測級別 `k` 的期望分數為：

```text
utility(k) = p_k + exp(-1) * (p_(k-1) + p_(k+1))
prediction = argmax_k utility(k)
```

邊界不存在的級別機率為 0。這個規則有時與 `argmax p_k` 不同；應在獨立 validation 上比較，不能宣稱一定改善。此推導依據 [官方棋力評分](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)。

身分題若每個候選有正確 posterior，依機率遞減排序可最大化期望遞減報酬；實際 embedding score 未必校準，仍須以官方指數分數挑模型，另報 Top-1、Recall@5、MRR。只提高第五名召回，不一定有明顯官方分數收益。[官方身分評分](https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688)

## 10. 驗證設計：比換更大的模型更優先

1. **先去重再切分。** 用 `game_id` 與正規化 SGF hash 查重；必要時用 D4 正規化序列輔助找同局變體。同一盤的黑白記錄、不同片段、增強版本不可跨 train/validation。
2. **棋力題至少測棋手隔離。** 若同一棋手跨兩側，模型可能記住身分後猜等級。正式主評估依官方 query 組成建立，另保留 unseen-player 壓力測試。
3. **身分題區分 seen 與 unseen。** seen-player：訓練可見棋手，但 reference/query 棋局隔離；unseen-player：encoder 未見過棋手，再以其 reference games 建立候選。不要混在一起報一個數字。
4. **跨棋手切分也要防同局污染。** 一盤若兩方分屬不同 split，不能讓該盤分別進入兩側；可排除這些衝突棋局並記錄保留比例。單靠 player group split 不足以保證 game isolation。
5. **聚合在切分之後。** 先分離原始棋局，再產生多盤集合。不能先做大量重疊集合後隨機 80/20 分，否則底層棋局會共享。
6. **query 盤數是主要變因。** 尚未確定官方設定前，測 1／5／10／20 盤等配置，並報有效手數；不拿百盤結果代表少盤效果。
7. **特徵與融合也要隔離。** 標準化、特徵選擇、校準及 stacking 都只在相應 training fold fit；監督式教師的 validation 特徵需使用未見該 fold 的模型。
8. **分層報分。** 棋力、黑白、棋局長度、開局／中盤／終盤分開看；置信區間以棋手或 query 集合為單位，不把高度相關的逐手樣本當獨立資料。

這些是針對本資料形態提出的驗證要求。R02 的 rank-group／player-specific 落差，以及 R06 的 reference/query 設計，是需要重視聚合與切分的直接研究依據。

## 11. 建議實驗順序與成本控制

| 階段 | 實驗 | 要回答的問題 | 成本概況 |
| --- | --- | --- | --- |
| E0 | CSV／SGF audit、去重、split、官方 metric 單元驗證 | 資料與分數是否可信？ | 以 CPU 為主 |
| E1 | 分階段統計特徵＋線性模型／樹模型；棋手特徵距離 baseline | 便宜的訊號能做到什麼程度？ | 低 |
| E2 | 小型 ResNet＋masked mean pooling，10 類 CE | 不依賴外部引擎的棋力基準 | 可先單 GPU 小樣本 |
| E3 | 同 encoder 加 GE2E／SupCon 的棋手向量 | 能否分辨同棋力的不同棋手？ | 依 batch 中棋手與棋局數控制 |
| E4 | 小批 HumanSL likelihood；與 E1／E2 比較及融合 | 外部教師收益是否足以支付預計算成本？ | 先 benchmark 再擴大 |
| E5 | SE／CORAL／CORN、不同多盤聚合 | 序位結構與集合資訊有多少增益？ | 中，逐項消融 |
| E6 | policy 預訓練、多任務共享、教師蒸餾、out-of-fold ensemble | 是否進一步提高官方 metric？ | 高，前面成立後再做 |

所有成本是實作方向的定性評估，非本機測量。建議每階段保留 seed、split hash、data version、commit、完整 config、硬體／套件版本、官方分數及處理速度。實驗放 `outputs/<run-id>/`，資料與特徵放 `data/raw/`、`data/processed/`，提交放 `submissions/`。

可先做的三組消融：`state-only vs state+action`、`mean pooling vs attention`、`只用官方資料 vs 加 HumanSL 特徵`。先回答這三組，再決定是否值得擴大訓練與多模型融合。

## 12. 尚未完成的查核與資源登錄

- 完整官方壓縮檔的 SHA-256、重複棋局、每棋手盤數、跨 rank 分布與異常 SGF 尚未實際分析。
- 正式 query/reference/candidate 規格與 sample submission 尚未在本次整理中核實。
- 外部 repo 未安裝；README 中的指令、效能與可用性不代表已在本機測試成功。
- ScienceDirect 棋風論文僅取得摘要／概述；部分歷史資源與模型下載未逐一驗證。
- GitHub 預設分支會變動。採用任何程式、模型或資料時，保存 commit／release／SHA-256，不能只保存會變動的網址。

實際採用時可沿用以下登錄欄位，供最終報告回溯：

| 資源 | 作者／來源 URL | 固定版本／雜湊 | 授權文件 | 用途 | 是否含外部訓練資料 | 影響的實驗 |
| --- | --- | --- | --- | --- | --- | --- |
| 待採用時填寫 | — | — | — | 解析／預訓練／特徵／模型／文件協助 | — | — |

本文件由 Codex 協助搜尋與整理公開資料；研究方法的原始貢獻歸各論文與專案作者。本次只建立參考文件，沒有訓練或產生競賽答案。
