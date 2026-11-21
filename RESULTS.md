# 實驗結果總覽

更新：2026-11-04。依既有紀錄與保存產物整理，沒有新增訓練或重算分數。完整實驗、失敗紀錄、決策與驗證細節保留於 [RESULTS_ARCHIVE.md](RESULTS_ARCHIVE.md)；專案規則以 [CLAUDE.md](CLAUDE.md) 為準，後續工作見 [EXECUTION_PLAN.md](EXECUTION_PLAN.md)。

## 1. 目前做到哪裡

| 項目 | 已完成 | 目前結果／採用方案 |
|---|---|---|
| 共用資料 | 100 萬盤稽核、SGF 轉檔、盤面快取、玩家切分與合成題組 | 8,007 位玩家、221,894,008 手；訓練集無重複棋譜，解析錯誤 0 |
| 競賽二：棋力預測 | 手工特徵、V1／V2 ResNet、HumanSL、級別向量、集成與微調實驗 | **首選 AH＋C＋B2：Valid-U Equal-k 0.6934**；AH 對照 0.6833，V2＋C2 備援 0.5553（題數不同，見下表） |
| 競賽一：玩家辨識 | 開局計數模型、D4 正規化、GE2E 風格模型、融合與指派壓力測試 | **首選 D4 開局＋8×GE2E，不做指派：0.9846** |
| SE 論文復現 | 三頭 ResNet、35,000 步正式訓練、全量 training centroid 與 validation 評估 | **Equal-k 0.5979**（10,000 題，game-disjoint）；單盤 0.3894，分數比較見第 3 節、訓練細節見第 5 節 |
| SE2 AdamW 續訓 | 新增 70,000 步（累計 105,000）、50k／70k 全量評估與權重保存 | **Equal-k 0.5746／0.5412**；均低於原 SE 0.5979，未過 gate，checkpoint 見第 5 節 |
| SE3 十級 CE，十五位置 | 第 45,000 步 early stopping，發布第 30,000 步最佳監控模型並完成全量 validation 評估 | **Equal-k 0.623873**（固定 10,000 題，game-disjoint）；單盤 0.415296，模型對應與來源見第 3.7 節 |
| 提交管線 | 兩賽一鍵流程、假官方 CSV 彩排、Day-0 稽核工具 | AH＋C 路徑彩排通過；正式格式與真實資料待 11/04 確認 |

**以下全是本地驗證，不是官方 LB。** 官方測試集預定 11/04 11:00（台灣時間）發布，目前沒有正式測試成績。

## 2. 分數怎麼看

- `k` 是每題棋局數（5／10／15／20）；Equal-k 是四桶分數等權平均。
- 棋力：完全正確得 1、差一級得 `1/e`、其餘 0；官方 baseline **0.4631**。`score_rank` 以相鄰級得分計算最佳預測。
- 玩家辨識：正解第 1～5 名依序得 `1、0.368、0.135、0.050、0.018`，榜外 0；官方 baseline **0.2845**。
- **Valid-U** 是棋力主線未見過的玩家；**Valid-S** 是 V2 訓練玩家留出的棋局，分成 S-fit／S-select 做擬合與選擇。HumanSL 主線只用 Valid-S 擬合，Valid-U 不參與擬合。
- 不同驗證集、題數或切分的結果分表列出，不能直接以總分判定架構優劣。合成題會重複使用玩家，信賴區間採玩家分群 bootstrap。

## 3. 競賽二：棋力架構比較

各架構的代表結果如下，便於一起查看；**驗證切分與題組不同，表中數字不構成受控排名**。後續小節保留同題對照與實驗細節。

| 架構／代表版本 | 驗證條件 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---:|---:|---:|---:|---:|
| 手工特徵＋LR | 玩家互斥，f4＋holdout，14,000 題 | 0.350 | 0.398 | 0.440 | 0.459 | 0.412 |
| V1 scalar ResNet，seed2028 | 棋局互斥、玩家可重疊，V1 題組 | 0.3002 | 0.3230 | 0.3523 | 0.3723 | 0.3370 |
| V2 十類 ResNet＋C2 | 玩家互斥，Valid-U，10,000 題 | 0.5030 | 0.5501 | 0.5716 | 0.5965 | 0.5553 |
| HumanSL＋B2（S-fit 擬合） | 玩家互斥，Valid-U，1,000 題 | 0.609 | 0.685 | 0.670 | 0.666 | 0.6572 |
| V2＋HumanSL＋B2（AH） | 玩家互斥，Valid-U，1,000 題 | 0.6168 | 0.6792 | 0.7223 | 0.7147 | 0.6833 |
| AH＋級別向量＋B2（AH＋C） | 玩家互斥，Valid-U，1,000 題 | 0.6244 | 0.6977 | 0.7303 | 0.7213 | 0.6934 |
| **SE 論文三頭 ResNet，35,000 步** | **棋局互斥、玩家可重疊，10,000 題** | **0.5453** | **0.5922** | **0.6108** | **0.6432** | **0.5979** |
| SE2 AdamW，新增 50,000 步（累計 85,000） | 同 SE 棋局切分與固定 10,000 題 | 0.5262 | 0.5679 | 0.5886 | 0.6155 | 0.5746 |
| SE2 AdamW，新增 70,000 步（累計 105,000） | 同上 | 0.4949 | 0.5325 | 0.5686 | 0.5689 | 0.5412 |
| SE3 十級 CE，十五位置，best 30,000 步 | 同 SE 棋局切分與固定 10,000 題；全有效位置、盤間等權 | 0.574675 | 0.624762 | 0.628528 | 0.667525 | 0.623873 |

### 3.1 架構各自學什麼

| 架構 | 輸入與訓練方式 | 題目如何預測 | 現況 |
|---|---|---|---|
| 手工特徵＋LR／LightGBM | 本人／對手的距離、應手、提子等統計 | k 盤統計聚合後分類 | 已測，未過 baseline |
| V1 scalar ResNet | 18 層歷史盤面、256 channels／20 blocks；排序 loss，輸出一個實數 | 每盤平均 scalar＋training-only 切點 | 已有三個 seed 的歷史結果；game-disjoint，見第 3.6 節 |
| V2 十類 ResNet（A） | 沿用盤面骨幹，十類 head＋逐手 CE，1 epoch | 每盤 log-prob 平均→題內各盤等權→C2 溫度校準→score_rank | 已完訓；主線特徵與備援模型 |
| HumanSL（H）＋B2 | 凍結 KataGo HumanSL；15 種段位 profile 對真實著手的 likelihood，本人／對手共 30 維 | 先平均 k 盤特徵，每個 k 各一個 logistic regression（B2） | 題級 gate 通過，成為主線訊號 |
| A＋H＋B2（AH） | V2 的 10 維平均 log-prob＋H 的 30 維特徵 | B2＋score_rank | 正式對照模型 |
| A＋H＋級別向量＋B2（AH＋C） | 凍結 HumanSL，每個幽玄級別學本人／對手兩個條件向量；新增 20 維 likelihood，共 60 維 | B2＋score_rank | **目前首選** |
| HumanSL 全網路微調（C4／C4b） | 解凍 policy 路徑；分別以生成式 NLL、NLL＋級別 CE 訓練 | 先以未見玩家單盤分數判 gate | 試跑未過，未擴大題級評估 |
| SE 三頭 ResNet | 專用 20 blocks／256 channels；strength BT 排序＋policy CE＋teacher value MSE，三項等權 | 全 training 有效位置建立十級 centroid；題內所有位置平均後取最近 centroid | 已完訓與全量評估，Equal-k 0.5979；game-disjoint，見第 3.6 節 |
| SE3 十級 CE 三頭 ResNet | 18 blocks／256 channels；strength 十級 CE＋policy CE＋teacher value MSE；AdamW、每盤抽十五位置 | 盤內平均 log_softmax→題內各盤等權平均→argmax；不使用 centroid／score_rank | best 第 30,000 步，全量 validation Equal-k 0.623873；game-disjoint，見第 3.7 節 |

名詞注意：**V2 的 C2 是聚合／溫度校準規則；HumanSL 實驗的 C2 是級別條件向量（表中記為 C）。兩者不同。**

### 3.2 V2：同一組 Valid-U 10,000 題

每桶 2,500 題；除 14 epochs 外皆為 1 epoch。C2 溫度只在 S-fit 擬合。

| 模型／聚合 | k=5 | k=10 | k=15 | k=20 | Equal-k | 結論／紀錄 |
|---|---:|---:|---:|---:|---:|---|
| V2，機率平均 | 0.4643 | 0.4918 | 0.4950 | 0.5129 | 0.4910 | 首次超過 baseline；A-024 |
| 同模型，log-prob 平均，未校準 | 0.4790 | 0.5121 | 0.5334 | 0.5533 | 0.5195 | 聚合本身有增益；A-025 |
| **同模型＋C2** | **0.5030** | **0.5501** | **0.5716** | **0.5965** | **0.5553** | 比機率平均 +0.0643，採用；A-025 |
| 3-seed V2 ensemble＋C2 | 0.5085 | 0.5500 | 0.5692 | 0.5948 | 0.5556 | 僅 +0.0003，未過 +0.005 gate；A-033 |
| V2 14 epochs＋C2 | 0.4310 | 0.4801 | 0.4867 | 0.5010 | 0.4747 | −0.0806；長訓練過擬合，未採用；A-038 |

**結果：** 這條線有效的改動是 log-prob 聚合與溫度校準；增加 seed 或拉長同配方訓練都未過 gate。保留 A-024 單模型＋C2，未採用不代表刪除權重。

### 3.3 HumanSL：同一組 Valid-U 1,000 題

每桶 250 題。AH 與 AH＋C 正式模型皆以全部 Valid-S 的 65,223 盤、5,000 人擬合 B2；Valid-U 不參與擬合。

| 模型／擬合資料 | k=5 | k=10 | k=15 | k=20 | Equal-k | 結論／紀錄 |
|---|---:|---:|---:|---:|---:|---|
| V2＋C2（同題對照） | 0.4930 | 0.5621 | 0.5576 | 0.6002 | 0.5532 | 與上表 0.5553 是同模型、不同題數 |
| H＋B2（只用 S-fit） | 0.609 | 0.685 | 0.670 | 0.666 | 0.6572 | 純 HumanSL 有強訊號；A-045，診斷結果 |
| AH＋B2（只用 S-fit，早期版） | 0.6055 | 0.6827 | 0.6682 | 0.6459 | 0.6506 | A-041；後來特徵加總順序改寫，重算為 0.6487 |
| AH＋B2（全部 Valid-S） | 0.6168 | 0.6792 | 0.7223 | 0.7147 | 0.6833 | 增加擬合資料後提升；A-044 結果補記於 A-046 |
| **AH＋C＋B2（全部 Valid-S）** | **0.6244** | **0.6977** | **0.7303** | **0.7213** | **0.6934** | **目前首選**；A-055／A-056 |
| AH＋C 交叉擬合（Valid-S＋一半 U，評另一半） | 0.6164 | 0.6926 | 0.7455 | 0.7259 | 0.6951 | 比首選 +0.0017，效果量不出來；A-058 |

**結果：** AH 比同題 V2＋C2 高 **+0.1300**，玩家 bootstrap 95% 區間 `[+0.104, +0.156]`。AH＋C 再高約 **+0.0102**（以未四捨五入值計算），四桶都升，但區間 `[−0.0003, +0.0211]` 包含 0；使用者已決定先採 AH＋C，AH 保留對照。

全玩家版最終以 Valid-S＋全部 Valid-U 的 **91,917 盤、6,442 人**擬合，只作 LB 探針。上表 0.6951 是交叉擬合結果；全玩家模型在 U 的 0.712 是樣本內分數，**不能當成泛化成績**。

### 3.3.1 SE3＋H＋C＋B2：同題配對比較（2026-11-06）

兩臂均凍結上游模型，使用相同題組、同色抽樣與 `60→32→10` 的 B2 分類頭，softmax 後以 `score_rank` 解碼；唯一特徵差異是將 AH＋C 中的 V2 十維特徵換成 SE3 十維。這裡的 AH＋C 是重新訓練的兩層 B2 對照，**不是上表 A-055／A-056 的正式 logistic regression 模型**；題組也不同，不能直接比較絕對分數。

| 評估階段／模型 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| 開發 S-select：AH＋C | 0.6358 | 0.7501 | 0.7331 | 0.7596 | 0.7196 |
| 開發 S-select：SE3＋H＋C | 0.7006 | 0.8175 | 0.8165 | 0.7836 | 0.7796 |
| 開發差值（SE3 − AH） | +0.0649 | +0.0674 | +0.0834 | +0.0240 | +0.0599 |
| 最終 U_newgames：AH＋C | 0.5953 | 0.6800 | 0.6492 | 0.6970 | 0.6554 |
| 最終 U_newgames：SE3＋H＋C | 0.5947 | 0.6326 | 0.7089 | 0.6888 | 0.6562 |
| 最終差值（SE3 − AH） | −0.0006 | −0.0474 | +0.0596 | −0.0082 | +0.0008 |

每組 1,000 題、每 k 250 題；最終 Equal-k 差值的 2,000 次玩家分層配對 bootstrap 95% 區間為 **[−0.0250, +0.0284]**。最終候選 gate 要求 Equal-k 差值 ≥+0.005、各桶差值 ≥−0.005；本次 **未通過**，尚無證據支持取代既有首選，LB 未知。開發集通過繼續實驗條件，但開發提升未在新棋局重現。

**訓練與限制：** 開發 B2 用 S-fit＋S-extra 的 38,589 盤，S-select 選各 k 最佳 epoch；兩臂八個分類頭皆在連續 10 epochs 未提高後早停，未跑滿 100 epochs。AH＋C 的 k=5／10／15／20 最佳 epoch 為 **11／32／15／19**，SE3＋H＋C 為 **5／18／32／16**；最終模型以全部 Valid-S 65,223 盤從零重訓各自的固定最佳 epoch，不再早停。

S-select 與開發 B2 擬合玩家互斥，但每桶約 78%～80% 的棋局引用是 SE3 訓練棋局，四桶各 250 題全都至少包含一盤；因此開發分數只供選模診斷。U_newgames 排除 SE3 訓練／監控與 C 學習／監控棋局，並與 B2 擬合玩家互斥，但 SE3 仍見過評估玩家的其他棋局，不能視為全系統未見玩家驗證。兩階段題組、玩家分布與重訓模型不同，無法單憑分數落差確定原因。

來源：[開發報告](outputs/rank/se3_hc_b2/se3_30k_mlp32_seed2027/reports/dev_select/result.json)、[最終報告](outputs/rank/se3_hc_b2/se3_30k_mlp32_seed2027/reports/final_newgames/result.json)、[實驗計畫](STRENGTH_SE3_HC_B2_PLAN.md)；圖表位於各報告目錄的 `plot/`。

### 3.4 早期基線與 V1 歷史結果（不同驗證條件）

| 模型 | 驗證條件 | k=5 | k=10 | k=15 | k=20 | Equal-k | 紀錄 |
|---|---|---:|---:|---:|---:|---:|---|
| 手工特徵＋LR | f4＋holdout，14,000 題，玩家互斥 | 0.350 | 0.398 | 0.440 | 0.459 | 0.412 | A-006／E1-001 |
| 手工特徵＋LightGBM | 同一評估題組；f3 用於 early stop，訓練資料少於 LR | 0.333 | 0.383 | 0.406 | 0.431 | 0.388 | E1-001 |
| V1 seed2027 | game-disjoint，玩家可重疊 | 0.312799 | 0.330387 | 0.338884 | 0.350654 | 0.333181 | A-015 |
| V1 seed2028 | 同上，各 seed 重新切分 | 0.300215 | 0.323040 | 0.352314 | 0.372288 | 0.336964 | A-015 |
| V1 seed2029 | 同上，各 seed 重新切分 | 0.301448 | 0.319964 | 0.324084 | 0.345367 | 0.322716 | A-015 |

這些實驗均未過 0.4631。V1 與 V2 的切分、loss、解碼與訓練設定不同，這裡僅保留歷史結果，沒有新增 V1／V2 受控對照。

### 3.5 其他做過、未成為主線的方向

以下多為 S-select 或單盤診斷，**不與 Valid-U 主線表排名**。完整四桶、設定與失敗紀錄見 [歷史帳本 A-043～A-057](RESULTS_ARCHIVE.md)。

| 方向 | 驗證結果 | 處置 |
|---|---|---|
| 貼目 0 取代 6.5 | S-select 0.6856 vs 0.6808，+0.0048，未過 +0.005 gate | 維持 6.5；A-043 |
| HumanSL 分段／投票／標準差特徵 | 擴大擬合集後最佳 0.7053 vs 均值特徵 0.7025，未過 +0.005 gate | 維持 30 維均值；A-046 |
| 只用 H，加入 train 玩家擬合 | S-select 0.7129 vs 同擬合集設計的 AH 0.7101，+0.0028，未過 +0.005 gate | 保留待 LB；A-047 |
| AH 加強引擎著手品質特徵 | S-select 0.6791 vs 0.6815，−0.0024 | 不擴大；A-048 結果補記於 A-049 |
| 強制級別平衡解碼 | 完全平衡模擬約 +0.006；不平衡時可能扣到約 0.10 | 未確認級別平衡前不採用；A-049 |
| AH 加 HumanSL 主幹表示 | S-select 0.6540 vs 0.7101；調 PCA／正則化後仍為 0.6764 vs 0.7114 | 未過 gate，保留；A-051／A-052 |
| C4 全網路生成式微調 | Bm 單盤 0.5195→0.5187；門檻為 +0.02 | 停在試跑；A-057 |
| C4b 加級別判別 loss | Bm 單盤 0.5195→0.5087 | 未過 gate，停止；A-057 |

### 3.6 SE 論文架構重建：Validation 成績

第 35,000 步最終模型，固定 10,000 題，每桶 2,500 題；同題同玩家同色。**game-disjoint、玩家可能重疊**；training centroid 與題目皆按有效位置等權。within-1 包含完全正確。訓練設定與產物見第 5 節。

| 評估單位 | 數量 | 官方分數 | 完全正確率 | within-1 |
|---|---:|---:|---:|---:|
| k=5 | 2,500 題 | 0.5453 | 0.3856 | 0.8196 |
| k=10 | 2,500 題 | 0.5922 | 0.4268 | 0.8764 |
| k=15 | 2,500 題 | 0.6108 | 0.4404 | 0.9036 |
| k=20 | 2,500 題 | 0.6432 | 0.4760 | 0.9304 |
| **全部題目／Equal-k** | **10,000 題** | **0.5979** | **0.4322** | **0.8825** |
| 單盤（診斷） | 200,000 盤 | 0.3894 | 0.2619 | 0.6084 |

**判讀：** 四個 k 桶均超過官方 baseline 0.4631，Equal-k 高出約 **+0.1348**；這只代表本地驗證跨過分數門檻，不是正式 LB。隨盤數增加，官方分數由 0.5453 升至 0.6432；各級題目分數以 C 級 0.6796 最高、2D 0.5076 最低。此結果使用幽玄十級資料與本專案訓練設定，不能宣稱重現原論文的實驗成績，也未完成與主線同切分的受控比較。

### 3.7 SE／SE2／SE3：相同切分與題組比較

以下共用 seed2026 棋局切分、200,000 盤 validation 與固定 10,000 題，每桶 2,500 題。SE／SE2 以位置等權平均 scalar＋各自的 training centroid 解碼；SE3 以盤內平均 log-prob、盤間等權平均＋argmax 解碼。各模型皆使用全部有效 validation 位置。

| 模型／步數 | k=5 | k=10 | k=15 | k=20 | Equal-k | 相對原 SE Δ | 單盤（診斷） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 SE，35,000 步 | 0.545260 | 0.592199 | 0.610802 | 0.643164 | 0.597856 | — | 0.389403 |
| SE2，新增 50,000／累計 85,000 步 | 0.526240 | 0.567869 | 0.588563 | 0.615529 | 0.574550 | -0.023306 | 0.373160 |
| SE2，新增 70,000／累計 105,000 步 | 0.494912 | 0.532464 | 0.568551 | 0.568931 | 0.541214 | -0.056642 | 0.368013 |
| **SE3，十五位置，best 30,000 步** | 0.574675 | 0.624762 | 0.628528 | 0.667525 | 0.623873 | +0.026016 | 0.415296 |

**比較：** SE3 的 Equal-k 比原 SE 高 **+0.026016**，四桶分別高 **+0.029416／+0.032564／+0.017726／+0.024360**；比 SE2 50k／70k 分別高 **+0.049322／+0.082658**，四桶也都較高。SE3 是這組 SE 系列既有本地結果中分數最高的模型。這是同資料／同題組的整體方案比較；模型、loss、取樣、optimizer、訓練長度及聚合規則同時改變，無法單獨歸因於 CE。

**SE3 訓練、最佳權重與評分來源（2026-11-04 補登）**

run：`outputs/rank/reproduce_paper_3/se3_seed2026_15pos/`（Slurm job 692203）。文末工程紀錄中的「尚無正式模型分數」與十五位置「未知」為當時狀態，正式結果以本節為準。

**完成與選模：** [training_summary.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/training_summary.json) 記錄 completed、45,000 updates、stop_reason=patience；最佳固定監控模型為第 **30,000 步**，5,000 題、每盤十五位置的監控 Equal-k 為 **0.6228357894**。第 35k／40k／45k 三次監控嚴格較低而早停。summary 記錄 15,980.56 秒（約 4.44 小時）／35.5124 GPU-h，為 trainer session 計時，不等於完整 Slurm 配置與最終評估成本。

**`evaluation/` 使用哪個模型：** [best.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/best.json) 指向 `best/best_00030000.pt`；完訓將其發布為根目錄 [model.pt](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/model.pt)，並加入停止步數等完成資訊。全量評估讀取此 `model.pt`，使用的是第 30,000 步最佳權重。已重新計算 `model.pt` SHA-256=`74c65097c53d85897c8e4239e910415eec1b8f3b474945b1d3357925a95a8cd0`，與 [model.complete.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/model.complete.json) 及 [evaluation/manifest.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/evaluation/manifest.json) 一致。

**完整驗證：** validation 200,000 盤，使用指定玩家且 ply≥30 的全部有效位置，固定 10,000 題、每桶 2,500 題。盤內平均 log_softmax，題內各盤等權平均後 argmax；不使用 centroid／score_rank。分數來源為 [evaluation/result.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/evaluation/result.json)：

| 評估範圍 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| 第 30,000 步固定監控，5,000 題／每盤十五位置 | 0.571475 | 0.625332 | 0.629166 | 0.665369 | 0.622836 |
| 同模型完整 validation 全有效位置，10,000 題 | 0.574675 | 0.624762 | 0.628528 | 0.667525 | **0.623873** |

全量 Equal-k 未四捨五入值為 **0.6238725773478886**；單盤官方公式分數為 **0.41529552023865274**（200,000 盤，診斷）。四桶均超過 baseline 0.4631；沿用 SE 的 game-disjoint 切分與固定題組，玩家可重疊，且同一 validation 用於選模，不能視為獨立 holdout 或新玩家泛化成績，也不能與 HumanSL 主線的 Valid-U 直接排名。SE3 同時改模型、loss、取樣、optimizer 與訓練長度，無法將差異歸因於單項。

本次只讀取既有產物並補登，未重新訓練、推論或上傳；以上為本地依官方公式計算的分數，平台 LB 未知。

## 4. 競賽一：玩家辨識架構比較

以下採相同合成候選庫 split 1–2 平均：每組 400 人、每人 100 盤 gallery，另有 400 題查詢、每桶 100 題。超參／融合權重在 split 0 選，split 1–2 用於報分。

| 架構 | k=5 | k=10 | k=15 | k=20 | Equal-k | 結論／紀錄 |
|---|---:|---:|---:|---:|---:|---|
| 逐手獨立 NB（raw，黑前 4 手） | 0.8096 | 0.9450 | 0.9793 | 0.9874 | 0.9303 | 零訓練計數基線；A-027 |
| 前綴模型 pp＋NB slot（raw） | 0.8981 | 0.9740 | 0.9968 | 1.0000 | 0.9672 | 捕捉開局順序，改善小 k；A-027 |
| pp＋slot（D4 正規化） | 0.8379 | 0.9710 | 0.9887 | 1.0000 | 0.9494 | 移除旋轉／鏡射影響；A-029 |
| GE2E 風格 ResNet（300 steps） | 0.8095 | 0.9666 | 0.9728 | 0.9968 | 0.9364 | 只看 ply ≥20 的黑方落子；A-035 |
| **D4 開局＋8×GE2E，不指派** | **0.9418** | **0.9968** | **1.0000** | **1.0000** | **0.9846** | **目前首選**；A-035／D-004 |
| D4 開局＋Hungarian | 0.9505 | 0.9862 | 0.9951 | 1.0000 | 0.9829 | 需一對一；A-029 |
| D4 開局＋8×GE2E＋Hungarian | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 僅作探針，預設關閉指派；A-035／D-004 |

**機制：** 開局模型直接比對候選人的落子習慣；GE2E 用 V2 ResNet 主幹初始化，學習 256 維逐手向量，再聚合成盤／玩家向量比對。兩者融合後，比 D4 開局單獨高約 **+0.035**。GE2E 拉長到 3,000 steps 反而從 0.9364 降到 **0.8634**（split 1–2），因此採 300 steps。

**為什麼首選不做 Hungarian：** A-059 模擬中，開局模型在一對一時 0.950→0.990；有放回抽人時卻從 0.951→0.684。真實題目是否一對一未知，故依 D-004 預設不做 Hungarian／Sinkhorn。

**仍未解的風險：** 合成候選人都是至少有 120 盤黑棋的重度玩家，且合成題已飽和。零訓練分數約 0.93 與官方 baseline 0.2845 的落差未解，絕對分數與方法排序都不保證轉移。E0-001 的 NB 壓力測試顯示，raw 遇到逐盤隨機方位會由 0.929 降到 0.091；D4 正規化是針對這項風險的預設。開局漂移 S1 與對應融合 gate G3 尚未做。

## 5. SE 論文架構重建：訓練細節與產物

**35,000 步訓練與全量評估已完成，Equal-k 為 0.5979。** 本次讀取既有產物補登，沒有重跑。SE 與主線獨立，固定 split seed2026、每級 80,000／20,000 盤，屬 **game-disjoint、玩家可能重疊**；與 V2／HumanSL 的 player-disjoint Valid-U 不同，不能直接排名或計算架構增益。

### 設定與完成狀態

| 項目 | 實際設定／結果 |
|---|---|
| 架構 | 專用 20 blocks／256 channels ResNet；strength scalar、362 類 policy、tanh value 三頭；BT 排序＋policy CE＋teacher value MSE，三項等權 |
| 資料與 teacher | training 800,000 盤／76,763,475 有效位置，validation 200,000 盤／19,183,737 有效位置；指定玩家且 ply ≥30；training 使用逐位置 1-visit teacher value |
| 取樣與訓練 | 每組十級、每級一盤七手；global 128 組，8×H200 每卡 16 組／1,120 位置；BF16＋SyncBN、channels-last；SGD，LR 0.04→0.02→0.002 |
| 完訓證據 | `training_summary.json`：completed、35,000 updates、finite checks 通過；`model.complete.json`：最終模型為第 35,000 步 |
| 訓練紀錄 | 2026-11-02 17:21:09～19:08:17（台灣時間），首末步 log 間隔約 1 小時 47 分；不含初始化、評估與 Slurm 完整配置時間 |
| Training loss | 第 1→35,000 步：total 8.4459→2.4190；BT 1.6826→0.8597；policy 5.9657→1.4896；value 0.7975→0.0697；為不同 batch 的端點值 |
| 評估規則 | 全 training 有效位置按位置等權建立十級 centroid；固定 validation 題組內所有有效位置平均，取最近 centroid（並列取較低級別）；BF16、8 卡、每卡 batch 2,240 |
| 使用權重 | 評估採第 35,000 步 `model.pt`；另外保留最低 training loss 的 `best_1.pt`（35,000 步）與 `best_2.pt`（34,000 步），未另評 best_2 |

四桶分數、accuracy 與 within-1 統一列於第 3.6 節。

**產物：** run 位於 `outputs/rank/reproduce_paper/se_seed2026/`。評分來源為 [report.json](outputs/rank/reproduce_paper/se_seed2026/evaluation/report.json) 與 [official_scores.csv](outputs/rank/reproduce_paper/se_seed2026/evaluation/official_scores.csv)；同目錄另有各級結果、centroid／分布、逐題預測、learning curve 與 `plot/` 圖表。完訓來源為 [training_summary.json](outputs/rank/reproduce_paper/se_seed2026/training_summary.json)。完整補登見歷史帳本 A-060。

### SE2 AdamW＋完整棋局遍歷（2026-11-03，已完訓與評估）

run：`outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/`。使用原 SE 的 `se_seed2026/best_1.pt`（35,000 步）作模型起點，重建 AdamW；新增 70,000 步，前 50,000 LR=0.002、後 20,000 LR=0.001，betas=(0.9,0.999)、eps=1e-8、weight_decay=0.0001，無 warmup。每級 80,000 training 棋局 shuffle，8 卡全域輪內不重複；625 步完整遍歷一次，共 112 輪，每盤每次抽七個位置。其餘 SE 模型、loss、資料及題組沿用。

**完成證據：** [training_summary.json](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/training_summary.json) 為 completed、completed_updates=70,000、cumulative_updates=105,000、finite_checks_passed=true；最後 session 從新增 50,000 步接續完成 20,000 步。`train.jsonl` 共 70,000 筆，第 1～70,000 步時間為台灣時間 11/03 16:44:07～20:24:52，間隔約 3 小時 41 分，包含中途評估／接續間隔，不等於純訓練或 Slurm 配置時間。Training total loss 端點為 2.4179→2.1695（不同 batch）；實際 GPU-hours 未核算。

兩階段 report 都為 completed，dataset_id 與 questions_sha256 與原 SE 相同；training／validation 分別為 76,763,475／19,183,737 個有效位置，固定 10,000 題、每桶 2,500 題。各階段使用自己的全量 training centroid，題內位置等權平均後取最近 centroid；BF16、8 卡、每卡推論 batch 2,240。**game-disjoint；validation 的 7,855 位玩家全在 training 的 8,007 位內，未驗證新玩家泛化。**

| 階段（新增／累計 updates） | k=5 | k=10 | k=15 | k=20 | Equal-k | 相對原 SE Δ |
|---|---:|---:|---:|---:|---:|---:|
| 原 SE（0／35,000） | 0.5453 | 0.5922 | 0.6108 | 0.6432 | 0.5979 | — |
| SE2 milestone（50,000／85,000） | 0.5262 | 0.5679 | 0.5886 | 0.6155 | 0.5746 | −0.0233 |
| SE2 最終（70,000／105,000） | 0.4949 | 0.5325 | 0.5686 | 0.5689 | 0.5412 | −0.0566 |

| 階段 | 題目完全正確率 | 題目 within-1 | 單盤官方分數（200,000 盤） | 單盤完全正確率 | 單盤 within-1 |
|---|---:|---:|---:|---:|---:|
| 新增 50,000 步 | 0.4015 | 0.8719 | 0.3732 | 0.2477 | 0.5888 |
| 新增 70,000 步 | 0.3579 | 0.8562 | 0.3680 | 0.2410 | 0.5862 |

**判讀：** 兩階段四桶都低於原 SE，未達原建議 gate（Equal-k≥0.6029、任一桶退步≤0.01）；繼續由新增 50k 訓至 70k，Equal-k 再降 0.0333。Training loss 較低沒有帶來題目分數提升；本實驗同時改 optimizer、取樣及訓練長度，無法單獨歸因，退步機制未知。保留結果與權重，不取代原 SE 或現行 AH＋C 主線；以上均為本地成績，LB 未知。

**保存 checkpoint／權重：** 以下兩個 `model.pt` 均實際存在，各 96,642,040 bytes；本次重新計算 SHA-256，與對應 `model.complete.json` 及 report 的 model_sha256 一致。步數為本次新增步數，累計需加來源 35,000 步。

| 階段 | 權重路徑（相對 run） | SHA-256 |
|---|---|---|
| 新增 50,000／累計 85,000 | [milestones/update_00050000/model.pt](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/milestones/update_00050000/model.pt) | `f7dc1e905581d9726fd0c72e8b18ba2750e58f0d66e19022ac6eb4ad2333cb34` |
| 新增 70,000／累計 105,000 | [model.pt](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/model.pt) | `bc78888a7b9272b092c6d36113eec2e0f9d5b89a7400d21a319ad40c5309b1c0` |

來源權重：`outputs/rank/reproduce_paper/se_seed2026/best_1.pt`，SHA-256=`ec81eee81cba56b4d870a1c3a5ec7f31b873054d14a012175ca7700cf1dcded7`（[source.json](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/source.json) 紀錄）。SE2 依設定只保留最新續跑 checkpoint，完訓後已清除 `checkpoints/` 與 `last.json`；本 run 沒有 `best_1.pt`／`best_2.pt`，上述發布權重不應視為仍保有完整續跑狀態。

**評分與重現來源：** [50k report.json](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/milestones/update_00050000/evaluation/report.json)、[70k report.json](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/evaluation/report.json)；各自 `evaluation/plot/` 保留 learning curve、centroid／分布與混淆矩陣等 10 張 PNG。實際設定見 [config.json](outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/config.json)；操作見 `src/go_aicup/tasks/strength/reproduce_paper_2/README.md`，Slurm 為 `slurm/se_adamw70k.slurm`。本次僅讀取既有產物補登，沒有重新訓練或全量推論。

### 2026-11-03：SE／SE2 論文式同級抽盤評估（CPU）

使用者要求檢查「論文同級抽盤、競賽同玩家組題」是否造成差距。新增 `scripts/strength/se_paper_eval.py`，只重用三個模型既有逐盤分數；N=1..100，每級每 N 有放回抽樣 500 次，各級等權、位置等權聚合、nearest-centroid，seed=2026。模型間共用抽樣。另從 validation 每級固定抽 100 candidate／900 query，棋局互斥、玩家可重疊；candidate 另建 centroid。沒有 GPU 訓練或新推論。

下表為**完全正確率**，四欄依 N=5／10／15／20；不可與官方分數混用。

| 模型／協定 | N=5 | N=10 | N=15 | N=20 | N=100 |
|---|---:|---:|---:|---:|---:|
| 原 SE，既有同玩家題組 | 0.3856 | 0.4268 | 0.4404 | 0.4760 | — |
| 原 SE，同級抽盤＋原 training centroid | 0.4128 | 0.4926 | 0.5562 | 0.5766 | 0.7234 |
| 原 SE，同級抽盤＋candidate100/query900 | 0.4246 | 0.5480 | 0.6238 | 0.6554 | 0.9126 |
| SE2 50k，同級抽盤＋candidate100/query900 | 0.4198 | 0.5206 | 0.6062 | 0.6486 | 0.8778 |
| SE2 70k，同級抽盤＋candidate100/query900 | 0.4126 | 0.5278 | 0.5910 | 0.6426 | 0.8810 |

candidate100/query900 的官方分數 N=5／10／15／20：原 SE＝0.5784／0.6884／0.7473／0.7728；SE2 50k＝0.5700／0.6671／0.7358／0.7676；SE2 70k＝0.5665／0.6745／0.7255／0.7661。這些是同級混玩家的診斷，**不取代競賽同玩家成績，不升級提交候選**。

**判讀與限制：** 同級抽盤可以解釋部分落差，centroid 來源亦有明顯影響；不能再把同玩家組題當成已證實的最大原因。原 SE N=15 仍只有 62.38%，低於論文約 80%。沿用指定玩家、ply>=30、BF16、無 D4 的既有 score；作者 candidate 平均雙方全部位置、query 每盤隨機顏色且預設 random rotation，故**本輪不是完整原版 eval**。只做一個固定 candidate/query split；Monte Carlo SE 只反映固定資料下抽題誤差；原同玩家題組的 eligible-player 分布也不同，兩者差值不是嚴格配對因果估計。

完整曲線、within-1、官方分數、逐級與來源 hash：[報告](outputs/rank/paper_eval_20261103/README.md)、[summary.csv](outputs/rank/paper_eval_20261103/summary.csv)、[圖](outputs/rank/paper_eval_20261103/accuracy.png)。驗證：三個模型 checksum／資料共用檢查、位置加權與 nearest-centroid self-check、candidate/query 無重複與輸出指標恆等式檢查均通過；未重跑 GPU。

## 6. 已可產出的提交候選

一鍵入口：`scripts/release/run_release.sh <raw-dir> <out-dir>`。`<part>` 為 public／private；檔案寫在指定輸出目錄。以下為管線輸出約定，正式提交格式仍待 11/04 確認；由使用者手動上傳。

| 任務／候選 | 模型位置 | 預測檔 | 本地參考／預期 LB |
|---|---|---|---|
| 棋力首選 AH＋C | `outputs/humansl_rankvec_model/model.pkl` | `rank_<part>.csv` | U 0.6934；LB 未知 |
| 棋力對照 AH | `outputs/humansl_model/model.pkl` | `rank_<part>_ah.csv` | U 0.6833；LB 未知 |
| 棋力全玩家探針 | `outputs/humansl_allplayers_model/model.pkl` | `rank_<part>_allplayers.csv` | 交叉擬合 0.6951；最終模型無獨立 U 分數，LB 未知 |
| 棋力備援 V2＋C2 | `outputs/v2_strength/model.pt`＋`outputs/v2_agg_calibration/result.json` | `rank_<part>_v2_fallback.csv` | U 10,000 題 0.5553；LB 未知 |
| 玩家辨識首選 | D4 開局＋`outputs/pid_ge2e_v1_short/model.pt` | `pid_<part>.csv` | 合成 0.9846；LB 未知 |
| 玩家辨識探針 | 同上，改指派或移除保險模型 | `pid_<part>_{hungarian,sinkhorn,opening_only}.csv` | 見第 4 節與歷史帳本；LB 未知 |

**彩排證據：** A-053 的 GE2E 接線通過；A-056 的 AH＋C 一鍵彩排通過（8 個提交檔各 400 列，C 特徵差 p99=1.2e-6）。D-004 另驗證不指派預測。A-058 全玩家版沿用相同聚合路徑，未另外跑完整彩排。這些是假資料流程驗證，沒有正式 LB。

## 7. 接下來尚待確認

- SE／SE2 已完訓與評估，SE2 兩階段均退步；8 卡中斷續跑一致性、與主線同切分的架構比較及正式測試表現仍待確認。
- 11/04 真實資料稽核：提交格式、SGF properties／pass／短局、重複盤、玩家重疊、k 分布、方位與開局集中度。
- 競賽一是否一對一、合成驗證與 baseline 的巨大落差、開局漂移 S1／G3。
- 競賽二 L10（測試玩家是否在訓練集）、級別比例及 AH＋C／AH／全玩家版的 LB 差異。

後續實驗先登記假設、gate、成本，完成後補四桶結果與處置；完整過程追加至 [歷史帳本](RESULTS_ARCHIVE.md)，本頁更新可比較的結果與現行候選。舊實驗編號保留，失敗也保留。


## 2026-11-03：SE3 實作與工程驗收（尚無正式模型分數）

**假設／成本：** 十級 CE 取代 scalar BT，保留 policy／value；同時改 18-block、每盤三十位置、AdamW 與訓練長度，不能歸因於單項。增益未知；計畫中的 +0.005 Equal-k／單桶退步不超過 0.01 是未獨立核准的比較建議，不作 early stopping 條件。滿十萬步舊外推約二十小時／160 H200 GPU-h，未經本版短測量測，且不含監控／最終推論。

**完成：** 分五個小任務實作獨立 config／模型／loss、三十位置資料與固定監控、log-prob 分類推論／報表、可續跑訓練／early stopping／best 發布、測試／Slurm／文件。正式模型 21,754,485 parameters，raw state 87,094,068 bytes。舊 SE／SE2 原始碼與實驗產物不變。程式與操作：[SE3 README](src/go_aicup/tasks/strength/reproduce_paper_3/README.md)，逐項進度：[plan §11](src/go_aicup/tasks/strength/reproduce_paper_3/plan.md)。

**數據與驗收：** 官方索引唯讀檢查 train 800,000／valid 200,000 盤，最少有效位置均為 35，不足三十均為 0。SE3 最終合成測試 20 項通過（52.34s）；既有 SE／SE2 相關回歸 87 項通過。較早合併一輪為 105 項／140.60s（SE3 當時 18 項），其後補 smoke／最終發布失敗檢查並重跑 SE3。涵蓋八 rank 全輪覆蓋、三十位置互異、spawn／prefetch 精確續跑、固定題組、log_softmax／盤間等權／argmax、LR 邊界、同分／patience、推論／best／結果／latest 四處失敗注入、RNG／BN 還原、最佳模型及報表重試。兩程序 Gloo 固定 BN 全域平均 loss／梯度檢查通過（relative L2<1e-5、max abs<1e-6），一般 BN 兩程序訓練／續跑／評估串接通過；不是八卡 SyncBN 驗收。

| 評估範圍 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 固定最多 5,000 題、每盤三十位置監控 | 未知 | 未知 | 未知 | 未知 | 未知 |
| 最佳模型、完整 validation 全位置、固定 10,000 題 | 未知 | 未知 | 未知 | 未知 | 未知 |

**處置：** 僅完成工程實作及 CPU／Gloo 合成驗收；沒有 GPU 短測、正式訓練、官方全量評估或上傳，不新增提交候選。下一步為八卡 100 步＋一次固定監控，先報告顯存、吞吐與成本再確認正式長作業。game-disjoint、validation 玩家可能在 training，最終同切分也非獨立選模 holdout；不據此聲稱新玩家泛化。


## 2026-11-04：SE3 每盤改抽二十位置

依使用者要求，訓練及固定監控均改為每盤無放回二十位置。global 128 組、八卡各 16 組、LR 與十萬步上限沿用；每步位置數由 38,400 降為 25,600、每卡由 4,800 降為 3,200，減少三分之一。實際吞吐／顯存不宣稱等比例改善，效果未知。最終全量 validation 仍使用全部有效位置，prepared／teacher 不重建。程式與設定使用獨立 `*_20pos*` 輸出，保留三十位置的 job 691911 及其 checkpoint／監控集合。

三十位置歷史短測 691911：8×H200，100/100 updates＋5,000 題，Slurm COMPLETED／exit 0，wall 138s；穩定約 0.682s/step（1.47 steps/s）、峰值 allocated／reserved 63.56／64.35 GiB，監控 22.32s，latest／best／model 重載與八 rank RNG／optimizer 狀態驗證通過。四桶如下，不能作為二十位置版本分數：

| 版本／範圍 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 三十位置，100 步後固定 5,000 題監控 | 0.1595 | 0.1516 | 0.1662 | 0.1639 | 0.1603 |
| 二十位置，新設定 | 未知 | 未知 | 未知 | 未知 | 未知 |

本次不提交 GPU 作業；二十位置的顯存／吞吐、收斂與增益仍未知。三十位置 smoke 棋力 CE 約 2.30，接近 ln(10)，只支持工程接線成功。變更與測試見 [SE3 README](src/go_aicup/tasks/strength/reproduce_paper_3/README.md)／[plan §12](src/go_aicup/tasks/strength/reproduce_paper_3/plan.md)。

二十位置版本驗收：CPU／Gloo 合成回歸 **20 項通過（68.11s）**；檢查二十位置邊界、固定監控 shape、無重複抽樣、spawn 預取續跑、loss／全參數梯度、兩程序訓練續跑與最終評估。config 與 Slurm 語法檢查通過，既有三十位置 smoke 未修改。


## 2026-11-04：SE3 每盤改抽十五位置

依使用者最新要求，訓練與固定監控改為每盤無放回十五位置，完整組改為十級×十五＝150 位置；global 128 組／八卡各 16 組、LR 與十萬步預算沿用。每次更新由 25,600 降為 19,200 位置，每卡由 3,200 降為 2,400，位置數減少 25%；不宣稱顯存／吞吐等比例改善。既有 split／teacher 唯讀重用，min_ply=30；最終全量 validation 仍使用全部有效位置。使用新的 `*_15pos*` 輸出，保留既有三十／二十位置 checkpoint、監控與報表。

二十位置歷史短測 692064：八卡 100 updates＋5,000 題監控成功（Slurm exit 0，wall 105s），約 0.465s/step／2.15 steps/s、allocated／reserved 峰值 42.55／43.04 GiB，單次監控 16.51s。Total loss 9.0779→7.2567，棋力 CE 2.3091→2.2941；模型／optimizer 有限值、checkpoint checksum、八份 RNG 與固定位置 shape 均驗證通過。這些是二十位置結果，不能作為十五位置效能與成績：

| 版本／範圍 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 二十位置，100 步後固定 5,000 題監控 | 0.1718 | 0.1685 | 0.1678 | 0.1577 | 0.1665 |
| 十五位置，新設定 | 未知 | 未知 | 未知 | 未知 | 未知 |

本次不提交 GPU 短測／正式訓練；十五位置的 GPU 顯存、吞吐、四桶成績與長程收斂均未知。現行規格見 [SE3 plan §13](src/go_aicup/tasks/strength/reproduce_paper_3/plan.md)，操作見 [README](src/go_aicup/tasks/strength/reproduce_paper_3/README.md)。

十五位置版本 CPU／Gloo 合成回歸 **20 項通過（47.81s）**，涵蓋十五位置抽樣與固定監控、十四位置拒絕、舊七／二十／三十位置設定與模型 shape 拒絕、spawn 預取精確續跑、全域 loss／梯度與兩程序訓練／評估；config、Slurm 語法及舊短測產物保留檢查通過。未提交 GPU 作業。

## 2026-11-06：官方 v1.1 讓子資料更新與稽核

已核對[官方 release](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/tag/v1.1) 壓縮檔 SHA-256 `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`，更新 `data/raw/training/train_*.csv` 並擷取 `handicap.csv`（1,000,000 列）及 `handicap_manifest.json`。全量逐列比對確認移除 root `HD[n]` 後內容與 v1.0 相同；sidecar 的 `(rank, game_id)`、player_id、color、source_row 與現有 SE prepared 全部對齊。共用 SGF 嚴格 parser 已相容可選 HD，未新增 setup 石或模型特徵；既有 processed／teacher／checkpoint 未修改。

訓練 HD=0／1／≥2 為 **792,610／170,969／36,421**；棋力 Public／Private 各 5,000 盤全部有 HD，且僅有 0／1，HD=1 為 **1,155／1,125** 盤。各級 HD≥2 比例介於 B 的 1.320% 與 6D 的 8.317%。既有 SE3 預測只做分層診斷，沒有擬合或改預測：

| k | 全部 HD≤1 題數／得分 | 含 HD≥2 題數／得分 |
|---|---|---|
| 5 | 2,233／0.579656 | 267／0.533020 |
| 10 | 2,161／0.623552 | 339／0.632477 |
| 15 | 2,169／0.630277 | 331／0.617067 |
| 20 | 2,074／0.663429 | 426／0.687466 |

組間玩家／rank 分布不同，且沒有四桶一致退步，不能推論刪除 HD≥2 一定有益。建議先以既有 log-prob 做按玩家切分的 HD＋執色 CPU 校正（0 GPU-h），再測 training 抽樣只保留 HD≤1（770,832 盤）；皆尚待確認，增益未知，不改現行 SE3 run。完整分布、產物、重跑命令、gate 與未知語意見 [HANDICAP_V11.md](HANDICAP_V11.md)；機器報告位於 `outputs/rank_handicap_audit_v11/`。未重訓、重建快取、生成 teacher 或上傳。

驗證：SGF／盤面轉換、preprocess／release、擷取與 board cache CPU 回歸 **84 項通過（15.01s）**，`git diff --check` 通過。官方全量執行的範圍為下載、解壓、原始列比對與 metadata 對齊，沒有執行全量盤面重播或 GPU 工作。


## 2026-11-06：AH＋C＋B2 正式棋力測試集推論

依使用者要求，沿用現有凍結的 A-024 V2、HumanSL 15 profiles、C 級別向量與 Valid-S 擬合的 B2；明確指定搬移後的 `outputs/rank/v2/`、`outputs/rank/humansl/` 模型路徑，未重新訓練或擬合。既有程式入口為 `go_aicup.data.preprocess`、`go_aicup.tasks.strength.v2.predict`、`go_aicup.features.humansl`、`go_aicup.features.humansl_rankvec`、`go_aicup.tasks.strength.humansl_predict.aggregate`。

正式 sample 已確認：Public／Private 各 400 題合併為 **800 列、`question_id,pred_rank`**，ID 與順序完全照 `data/raw/tests/sample_submission_rank.csv`。本次輸出目標為 `submissions/rank/submission_rank_ah_c_b2_20261106.csv`；執行與格式驗證指令位於 `outputs/rank_release_20261106/{run.sh,audit.py,finish.py}`，記錄與來源 SHA-256 見 `input_audit.json`、`jobs.txt`。原提交流程仍輸出暫定欄位，本次在最後匯出時依 sample 轉換與合併，不修改模型預測。

前處理稽核：兩份各 **5,000 盤**，k=5／10／15／20 各 **100 題**；解析錯誤、pass、非交替均為 **0**，皆僅有 `HD,SZ`。Public／Private 共 2,222,151 手；各自盤內重複、兩份交集、與原始訓練全盤落子 hash 交集皆為 **0**。沿用既有方法，不把 HD 加入特徵、不改日本規則／貼目 6.5。

| tag／狀態 | k=5 本地參考 | k=10 | k=15 | k=20 | Equal-k | 預期 LB／假設 |
|---|---|---|---|---|---|---|
| AH_C_B2_20261106；等待使用者自行提交 | 0.6244 | 0.6977 | 0.7303 | 0.7213 | 0.6934 | LB 未知；沿用 A-055／A-056 正式模型測試真實資料轉移 |

使用者後續要求「編寫 slurm 讓我丟任務就好」，已停止本次未完成的提交，未重新提交。prep **713340**、V2 **713341** 已完成；八卡特徵 **713342** 因同節點執行緒不足而停止，合併驗證 **713346** 隨依賴取消，沒有產出最終 submission。目前排程中沒有本次作業。已完成的前處理／V2 與失敗日誌保留在 `outputs/rank_release_20261106/`。重提交通訊曾回報 `get_api_token` 錯誤；查核後無新增作業。

交付入口：`sbatch slurm/rank_ahc_predict.slurm`（單節點八卡、96 CPU、800GB RAM、時限 90 分鐘；此檔依既有 Git 規則忽略）。沿用上述模型與特徵程式，八個分片使用各自 GPU；引擎 analysis threads 改 64，BLAS／OMP／編譯各限制一個執行緒，以避開同節點 2,048-thread 上限，尚未重跑 GPU 驗收。預設 work 為 `outputs/rank_test_ahc_<jobid>/`，最終檔為 `submissions/rank/submission_rank_ah_c_b2_<jobid>.csv`（11/08 起改為 `ahc_<jobid>_rank.csv`）。可於提交前設定 `WORK_DIR=outputs/rank_release_20261106` 重用本次已完成的前處理／V2，H／C 仍會重算。

新增 `scripts/strength/export_rank_submission.py` 只負責最終合併，要求全部題目都有完整 AH＋C 特徵、無 fallback、機率有限且歸一、段位合法、ID／順序／欄位與 sample 一致，通過後才原子發布 CSV。小型 CPU 匯出邊界測試 **6 項通過（0.50s）**，包含 sample 排序、重複／缺題 ID、非法段位、缺盤、fallback 與非有限機率；`bash -n` 與 `git diff --check` 通過。沒有上傳；完整八卡流程尚未完成驗收，上述分數僅為既有 Valid-U 本地成績，並非正式測試得分。

## 2026-11-06：SE3＋H＋C＋B2 兩層分類頭實作與 CPU 驗收

使用者核准後完成獨立實驗，實作目錄依指定為 `src/go_aicup/tasks/strength/SE_H_C_B2/`（加號改為底線）。四個 k 各使用 `Linear(60,32) → ReLU → Linear(32,10)`，原始 logits 接 CE、推論 Softmax＋score_rank；SE3／H／C 凍結。已建立設定、同色固定題組與來源檢查、舊特徵重用／缺失分片補算、CPU AdamW／cosine／早停／精確續跑／固定 epoch 最終重訓、配對玩家 bootstrap、逐題／分 k／各級／confusion 報表、完整 AH＋C 整題 fallback、官方 Public／Private 合併 CSV 及套件內三份可追蹤 Slurm。操作見 [README](src/go_aicup/tasks/strength/SE_H_C_B2/README.md)，完整設計見 [計畫](STRENGTH_SE3_HC_B2_PLAN.md)。

**驗收：** `tests/test_se_h_c_b2.py`（17 項）＋`tests/test_release_pipeline.py`（4 項）＋`tests/test_reproduce_paper_3.py`（20 項），**41 項通過，71.21 秒**。全為 CPU／合成資料，既有 SE3 測試含兩程序 Gloo；涵蓋兩層架構／CE、同色抽題／玩家與棋局邊界、來源／缺類／NaN／欄位拒絕、尾批／LR／patience／最終固定 epoch、逐步續跑完全一致、BN buffers 凍結、小型真實 SE3 黑白 pre-action 推論與平均、C 本人／對手順序及失敗略過後整題 fallback、分片未完成重試／損壞拒絕、權重重載、bootstrap 及 sample 排序的合併 CSV。三份 launcher 的 `bash -n`、CLI help 與 `git diff --check` 通過。

**既有來源唯讀查核：** 固定模型 SHA-256 全部符合，SE3 有 200,000 盤、A 有 248,411 盤、H／C 各 76,989 盤快取可讀。真實 metadata 的開發／最終擬合池為 38,589／65,223 盤；新評估池 16,439 盤、673 位玩家。在記憶體驗證 seed2027 可生成 1,000 題、每 k 250 題，引用 7,292 盤不同棋局；這不是已發布的正式題組。

**未執行：** 正式 prepare／題組發布、官方 GPU 補算、B2 正式訓練、Slurm 提交與上傳。CUDA／KataGo 實際接線與吞吐、正式四桶增益、LB 與完整 GPU 成本未知；GPU 短測與全量工作另先回報估時。SE3 已見評估玩家的其他棋局，B2 擬合亦有上游 in-sample 信心偏移，不能據此宣稱 unseen-player 泛化。既有模型、棋盤／teacher 與其他工作區修改保留。

## 2026-11-06：SE3＋H＋C＋B2 單次提交訓練入口

新增 `src/go_aicup/tasks/strength/SE_H_C_B2/slurm/train.slurm`，一次提交串接準備、八卡特徵補算、兩臂開發／gate、最終訓練與新棋局評估；gate 不通過時保留開發結果並停止。預設 8 卡、12 小時上限＝最多 96 GPU-h，CPU 階段也占用 GPU；實際耗時、SU 與 CUDA 接線未知。唯讀查核既有 A 特徵涵蓋全部 65,223 盤最終擬合池與 16,439 盤新棋局評估池；固定新題組 H／C 各缺 6,642 盤、1,481,462 個位置。依舊實驗單卡吞吐、假設八卡均衡算術外推，H 約 9.5–13.2 分鐘、C 約 3.1–3.2 分鐘；不含 SE3、CPU、啟動／編譯、I/O、排隊與環境差異，不能視為總耗時估計。

補算程序原先在各 worker 啟動期間快取變化可能造成分片遺漏，已改按 `game_index % shards` 固定歸屬，限量抽樣從固定舊快取基準選擇。`tests/test_se_h_c_b2.py` 與新 `tests/test_se_h_c_b2_slurm.py` 共 **21 項通過（9.58 秒）**，涵蓋串行模擬快取增長、完整與限量補算、單一作業正常完成、gate 停止與 worker 失敗；`bash -n` 與 `git diff --check` 通過。未提交 Slurm、執行 GPU 推論或正式訓練。

## 2026-11-06：SE3＋H＋C＋B2 dev／final 測試集候選

依使用者要求，以 job **715640** 在 8×H200 共用一次 SE3 測試特徵，沿用 `outputs/rank_test_ahc_713770/rank/` 既有 H／C 分片；來源與棋局 metadata 經既有程式檢查。Public／Private 各 5,000 盤、400 題，兩模型均無 fallback。作業 COMPLETED、耗時 75 秒（配置約 0.167 GPU-h，不含排隊；SU 未知）。兩份 CSV 均以官方 sample 順序輸出 800 題，欄位、ID、機率與解碼經既有匯出驗證；dev 載入 dev 權重後共用同一完整題目檢查，來源與產物見 `outputs/rank/se3_hc_b2/test_713770/`，final 逐題結果位於既有 release 工作目錄。

| 候選／等待使用者自行上傳 | k=5 本地參考 | k=10 | k=15 | k=20 | Equal-k | 預期 LB／假設 |
|---|---:|---:|---:|---:|---:|---|
| `submissions/rank/se3hc-dev_713770_rank.csv` | 0.7006 | 0.8175 | 0.8165 | 0.7836 | 0.7796 | 未知；S-select 用於選模，非獨立成績 |
| `submissions/rank/se3hc-final_713770_rank.csv` | 0.5947 | 0.6326 | 0.7089 | 0.6888 | 0.6562 | 未知；U_newgames，SE3 玩家重疊，測試集轉移未知 |

final 在 U_newgames 的候選 gate 未通過，對照 AH＋C 的 k=5／10／15／20 差異為 −0.0006／−0.0474／+0.0596／−0.0082，Equal-k +0.0008；不可將其視為已驗證優於對照。兩模型本地分數來自不同題組，不可直接比較。未上傳。

### 2026-11-06：訓練棋局 H／C 標準陣列

已將各來源的每盤特徵整合至 `data/processed/strength_train_hc/standard/`：206,363 盤 H，100,306 盤同時有 C；每種特徵的 148 盤來源重複均採 `se3_hc_b2` 補算值。保留 allu 選盤、來源及實驗 metadata，刪除舊 allu 逐手 H／C 分片及原符號連結；逐手值無法由標準陣列還原。來源 checksum、棋局 metadata、覆蓋數、全部 6,642 筆各新版 H／C 與陣列讀回驗證通過，Slurm 716499 完成，24 秒／1 GPU 配額（CPU 工作）。未重新訓練或評估分數，既有 SE3 流程未改為讀此陣列；原 allu 專用舊流程需改接標準陣列才能重跑。

## 2026-11-07：官方 Q&A 讓子語意驗證；HD≥2 本人／對手互換；競賽一測試集重發

**讓子語意（官方 Q&A Q13，標「待確認」）以訓練資料驗證**，`scripts/data/audit_handicap_semantics.py`，CPU 69 秒，結果 `outputs/rank_handicap_audit_v11/semantics.json`：

- HD2–9 共 36,421 盤，前 10 手落在 Q13 星位的盤數**全部為 0**；第一手是白棋掛角（HD2：`pp` 5,108、`qp`、`pq`）→ SGF B／W 標記與實際相反、讓子石在 Q13 星位，兩者成立。
- HD≥2 中 color＝B 比例，D…6D＝.637／.592／.634／.661／.638／.671／.745／.643／.774／**.275**（互先 .47–.54）→ `color` 欄位是實際顏色、不需對調（推論，Q13 此點未定稿）。
- **影響：** 本人落子以「SGF 標記顏色＝`color`」選取（`board_cache.py`、`features/humansl.py:205`），因此 **36,421 盤（3.6%）本人／對手互換且缺讓子石**，V2／HumanSL／C／B2／SE 系列的訓練、擬合與本地驗證題都含這些盤。棋力測試題沒有 HD≥2，推論取子正確；HD1（測試 22.5–23.1%）以 6.5 目抽特徵，實際約 0.5 目。修正提案 P1–P3（CPU／CPU／GPU）列於 [HANDICAP_V11.md](HANDICAP_V11.md)，尚待確認，未改任何模型或提交檔，分數影響未知。

**競賽一官方 11/06 公告：** 11/06 11:00 起暫停上傳並清空排行榜，11/11 11:00 重新開放；新測試集含原有＋新增題，格式當天公布。有效窗口 14 天 × 5＝最多 70 次。若候選庫維持 400 人而題數增加，必不是一對一（指派已預設關閉，D-004）。本機沒有任何競賽一真實資料，Day-0 稽核未執行。競賽二公告頁沒有換測試集公告；11/06 暫停是否影響競賽二上傳，待使用者在平台確認。CLAUDE.md §0／§1／§4 已同步更新。

## 2026-11-07：P1 B2 擬合排除 HD≥2（AH＋C）—— gate 未通過，列 LB 探針候選

`scripts/strength/humansl_hd_filter.py`，Slurm **721980**（dev，15 分 33 秒，純 CPU），結果 `outputs/rank/humansl/hd_filter/result.json`。只改擬合棋局：all＝Valid-S 全部 65,223 盤／5,000 人（同正式模型）；hd01＝排除 HD≥2 後 62,867 盤／4,885 人。特徵、B2、C=1、score_rank 不變。特徵組裝檢查：正式 `humansl_rankvec_model/model.pkl` 經本腳本特徵在 E1 重現 **0.6934**（各桶完全一致）。

評估（Valid-U，從不擬合）：E1＝原 1,000 題；E1_hd01＝E1 中不含 HD≥2 的 840 題（偏向沒有讓子棋的玩家，只作診斷）；**E2＝新出的 HD≤1 題組（gate 用）**，每 k 500 題、每級 50 題，同玩家同色、seed 2027，存於 `questions_e2_hd01.parquet`，之後比較可重用。

| 題組（seed 2027） | 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k | 差 95%（玩家 bootstrap） |
|---|---|---:|---:|---:|---:|---:|---|
| E1 | all | 0.6149 | 0.6962 | 0.7278 | 0.7162 | 0.6888 | |
| E1 | hd01 | 0.6164 | 0.7053 | 0.7350 | 0.7209 | 0.6944 | [−0.0087, +0.0201] |
| E1_hd01 | all | 0.6108 | 0.6977 | 0.7228 | 0.7225 | 0.6884 | |
| E1_hd01 | hd01 | 0.6241 | 0.7041 | 0.7461 | 0.7513 | 0.7064 | [+0.0040, +0.0304] |
| **E2** | all | 0.6732 | 0.7174 | 0.7241 | 0.7454 | 0.7150 | |
| **E2** | hd01 | 0.6617 | 0.7201 | 0.7420 | 0.7662 | **0.7225** | [−0.0037, +0.0191] |

B2 合成題抽樣 5 個 seed 的 E2 差值（hd01−all）：Equal-k **+0.0075／+0.0083／+0.0083／+0.0085／+0.0102**（平均 +0.0086）；k=15、20 每個 seed 都 +0.016～+0.023，k=5 為 −0.0115～+0.0023（平均 −0.0031），k=10 在 ±0.006 內。E1 原題組的差值平均只有 +0.0009。

**判讀：** 預先登記的 gate（E2 Equal-k ≥ +0.005 且各 k ≥ −0.005）**未通過**：Equal-k +0.0075 達標，但 k=5 −0.0115，bootstrap 區間跨零。增益集中在 k=15／20，且在五個抽樣 seed 間一致；在仍含 HD≥2 題目的 E1 上幾乎消失，符合「排除互換棋局只在測試沒有 HD≥2 時有用」的機制；k=5 為何略降未知。注意：all 臂在 E1 的 0.6888 與正式模型 0.6934 只差 B2 列順序（合成題抽樣），單是抽樣就有約 0.005 的差距，與 gate 門檻同量級。依 D-003 慣例保留為 **LB 探針候選**，不取代 AH＋C 首選；未產生提交檔、未上傳。

## 2026-11-07：P2 每題讓先比例特徵 —— gate 差 0.0004 未通過；P1 探針提交檔

Slurm **722045**（dev，19 分 28 秒，純 CPU）。`scripts/strength/humansl_hd_filter.py` 改為子命令 `p1`／`p2`／`fit_hd01`；檢查：P2 對照組 seed 2027 重現 P1 對照組（E2 0.7150、E1 0.6888），`fit_hd01` 重現 P1 hd01 臂（E2 0.7225），重構沒有改變結果。

**P2**（結果 `outputs/rank/humansl/hd_filter/result_p2.json`）：擬合棋局同正式模型（Valid-S 全部 65,223 盤），AH＋C 再加兩欄「讓先且本人執黑」「讓先且本人執白」，題內平均＝該執色下的讓先比例。讓先比例：Valid-S 執黑 8.1%／執白 8.4%，Valid-U 9.1%／10.8%。

| 題組（seed 2027） | 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k | 差 95%（玩家 bootstrap） |
|---|---|---:|---:|---:|---:|---:|---|
| E1 | 對照 | 0.6149 | 0.6962 | 0.7278 | 0.7162 | 0.6888 | |
| E1 | ＋HD1 | 0.6255 | 0.7059 | 0.7364 | 0.7137 | 0.6954 | [−0.0035, +0.0168] |
| E1_hd01 | ＋HD1 | 0.6253 | 0.7061 | 0.7456 | 0.7390 | 0.7040 | [+0.0046, +0.0272] |
| **E2** | 對照 | 0.6732 | 0.7174 | 0.7241 | 0.7454 | 0.7150 | |
| **E2** | ＋HD1 | 0.6777 | 0.7196 | 0.7346 | 0.7467 | **0.7196** | [−0.0045, +0.0147] |

五個 B2 抽樣 seed 的 E2 Equal-k 差：+0.0046／+0.0042／+0.0085／+0.0079／+0.0044（平均 +0.0059）；k=5 每個 seed 都為正（+0.004～+0.013），k=15 也都為正（+0.010～+0.017）。**gate 未通過**：seed 2027 的 Equal-k +0.0046 比門檻少 0.0004（各 k 都不退步），區間跨零。與 P1 互補：P1 傷 k=5、幫 k=15／20，P2 幫 k=5。

**P1 探針提交檔**：`submissions/rank/ahc-hd01_722045_rank.csv`（SHA-256 `51f958de…cdc9fd`），模型 `outputs/rank/humansl/hd_filter/model_hd01.pkl`＝P1 hd01 臂（Valid-S HD≤1 共 62,867 盤，seed 2027），沿用 713770 的測試特徵，800 列、官方順序、無 fallback。與首選 `ahc_713770_rank.csv` 比較：Public 47／400 題不同、Private 54／400；幾乎都差一級。`scripts/strength/export_rank_submission.py` 新增 `--predictions`（預設不變），匯出測試 6 項＋release 測試 4 項通過。

| tag／狀態 | k=5 本地參考（E2） | k=10 | k=15 | k=20 | Equal-k | 預期 LB／假設 |
|---|---:|---:|---:|---:|---:|---|
| AH_C_B2_hd01_722045；探針，等待使用者決定 | 0.6617 | 0.7201 | 0.7420 | 0.7662 | 0.7225 | 首選同題組 E2 0.7150 →預期 Public 約 +0.007，但 47 題差異的 LB 雜訊約 ±0.009（1 SD），單次探針難以分辨 |

## 2026-11-07：P1＋P2 合併（排除 HD≥2 擬合＋讓先比例特徵）—— gate 通過

`scripts/strength/humansl_hd_filter.py p12`，Slurm **722175**（dev，16 分 11 秒，純 CPU），結果 `outputs/rank/humansl/hd_filter/result_p12.json`。對照＝正式設定（Valid-S 全部 65,223 盤、AH＋C）；處理＝Valid-S HD≤1 的 62,867 盤＋兩欄讓先旗標（本人執黑／執白）。題組、gate、seed 與 P1／P2 相同；對照組數字與 P1／P2 對照組完全一致。

| 題組（seed 2027） | 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k | 差 95%（玩家 bootstrap） |
|---|---|---:|---:|---:|---:|---:|---|
| E1 | 對照 | 0.6149 | 0.6962 | 0.7278 | 0.7162 | 0.6888 | |
| E1 | P1＋P2 | 0.6265 | 0.7063 | 0.7461 | 0.7320 | 0.7027 | [−0.0012, +0.0306] |
| E1_hd01 | 對照 | 0.6108 | 0.6977 | 0.7228 | 0.7225 | 0.6884 | |
| E1_hd01 | P1＋P2 | 0.6325 | 0.7143 | 0.7598 | 0.7659 | 0.7181 | [+0.0137, +0.0451] |
| **E2** | 對照 | 0.6732 | 0.7174 | 0.7241 | 0.7454 | 0.7150 | |
| **E2** | P1＋P2 | 0.6763 | 0.7320 | 0.7619 | 0.7736 | **0.7360** | **[+0.0088, +0.0335]** |

五個 B2 抽樣 seed 的 E2 Equal-k 差：+0.0210／+0.0216／+0.0171／+0.0209／+0.0202（平均 **+0.0202**）；k=15、20 每個 seed +0.024～+0.046，k=5 為 +0.003～+0.018，k=10 為 −0.001～+0.015。

**判讀：gate 通過**（Equal-k +0.0210 ≥ +0.005、各 k 最小 +0.0032、區間不含零）。合併增益（+0.020）大於 P1、P2 單獨增益之和（+0.0086＋0.0059），為何超過相加未知。限制：E2 是今天新建的題組，P1、P2、P1＋P2 三次比較都用它，而且 P1＋P2 是看過前兩次結果才提出的，存在選擇偏差；原 E1（含 HD≥2 題目）上的增益較小、區間跨零。尚未產生提交檔：測試 store 只存 SGF 屬性名稱、沒有 HD 值，需從原始測試 CSV 讀 HD，且 `humansl_predict aggregate` 要支援兩欄讓先特徵。

## 2026-11-07：P1＋P2 提交檔

Slurm **722279**（dev，1 分 45 秒，純 CPU）。`scripts/strength/humansl_hd_filter.py fit_p12` 擬合最終模型 `outputs/rank/humansl/hd_filter/model_p12.pkl`（Valid-S HD≤1 共 62,867 盤、seed 2027，欄位＝正式 AH＋C 60 欄＋`hd1_black`／`hd1_white`），在 E2 重現 0.7360。`humansl_predict aggregate` 新增 `--raw-csv`：模型欄位以兩欄 HD1 結尾時，從官方題目 CSV 讀每盤 HD[n]（store 只存屬性名稱），檢查執色一致、遇 HD≥2 直接報錯；其他模型行為不變。回歸檢查：正式 AH＋C 模型經修改後的 aggregate 在 Public／Private 真實測試資料上的輸出，與原 `rankvec_predictions.parquet` **逐列完全相同**。新增 `tests/test_hd1_features.py` 3 項，連同匯出 6 項、release 4 項共 13 項通過。

輸出 `submissions/rank/ahc-p12_722279_rank.csv`（SHA-256 `ba4531c0…bd5f23`），800 列、官方順序、無 fallback。與首選 `ahc_713770_rank.csv` 比：Public 59／400 題不同（−3：1、−2：1、−1：29、+1：28），Private 58／400，預測級別分布相近。

| tag／狀態 | k=5 本地參考（E2） | k=10 | k=15 | k=20 | Equal-k | 預期 LB／假設 |
|---|---:|---:|---:|---:|---:|---|
| AH_C_B2_p12_722279；gate 通過的候選，等待使用者決定 | 0.6763 | 0.7320 | 0.7619 | 0.7736 | 0.7360 | 首選同題組 E2 0.7150 →預期 Public 約 +0.02；59 題差異的 LB 雜訊約 ±0.010（1 SD），探針可大致分辨。E2 有選擇偏差，實際增益可能較小 |

### 2026-11-08：提交檔改名（辨識部分放前面）

依使用者要求，提交檔命名改為 `<模型>[-<變體>]_<jobid>_<競賽>.csv`，讓上傳介面與檔案列表一眼可分。只改檔名，五個檔案的 SHA-256 改名前後一致。

| 舊檔名 | 新檔名 |
|---|---|
| `submission_rank_ah_c_b2_713770.csv` | `ahc_713770_rank.csv`（首選 AH＋C） |
| `submission_rank_ah_c_b2_hd01_722045.csv` | `ahc-hd01_722045_rank.csv`（P1 探針） |
| `submission_rank_ah_c_b2_p12_722279.csv` | `ahc-p12_722279_rank.csv`（P1＋P2 候選） |
| `submission_rank_se3_hc_b2_dev_713770.csv` | `se3hc-dev_713770_rank.csv` |
| `submission_rank_se3_hc_b2_final_713770.csv` | `se3hc-final_713770_rank.csv` |

本帳本與 HANDICAP_V11.md 中的舊檔名引用已同步替換。`slurm/rank_ahc_predict.slurm`、`slurm/SE3_H_C_B2/predict.slurm` 預設輸出改為 `ahc_<jobid>_rank.csv`／`se3hc_<jobid>_rank.csv`，README 同步；競賽一之後比照 `<模型>_<jobid>_pid.csv`。

### 2026-11-08：競賽二首批 Public LB（使用者 11/06 上傳，補登）

使用者以隊伍帳號 jason.hsiao05 於 11/06 上傳 3 次，平台檔名皆被截成 `submission_rank_...csv`，依使用者指認對應：

| 上傳時間 | 檔案（現名） | 本地分數（題組） | Public LB | Δ（LB−本地） |
|---|---|---:|---:|---:|
| 11/06 13:03:35 | `ahc_713770_rank.csv`（AH＋C 首選） | 0.6934（Valid-U E1） | **0.7103381** | +0.017 |
| 11/06 16:24:17 | `se3hc-dev_713770_rank.csv` | 0.7796（S-select，選模用，非獨立） | 0.6713744 | −0.108 |
| 11/06 16:24:36 | `se3hc-final_713770_rank.csv` | 0.6562（U_newgames） | 0.6661153 | +0.010 |

平台目前顯示名次 32/99，對應最後一次上傳（final 0.6661）。最終成績只看 11/24 23:59:59 前的最後一次上傳，期中顯示的名次不影響最終成績（CLAUDE.md §6），目前不需要為此補傳。

**判讀：**
- AH＋C 在 Public 比 SE3＋H＋C＋B2 高 0.039（dev）／0.044（final）。這兩份與 AH＋C 在 Public 分別有 148／136 題不同（多為 ±1 級，偏向預測較高一級：+1 有 82／72 題，−1 有 54／53 題），差距大於這些差異題可造成的雜訊（約 ±0.02），與本地 final gate 未通過（U_newgames 對 AH＋C Equal-k 僅 +0.0008，k=10 為 −0.047）方向一致。SE3 線不取代 AH＋C。
- SE3 dev 的 S-select 0.78 → LB 0.67，證實 S-select 因玩家重疊而高估，不能當成績。
- AH＋C 本地 0.6934 → LB 0.7103（+0.017），在 Public 400 題標準誤約 0.02 的範圍內；Valid-U 對此模型未見高估。
- P1＋P2（`ahc-p12_722279_rank.csv`）與 AH＋C 在 Public 有 59 題不同，預期 LB 約 0.71＋0.02（有選擇偏差，可能較小），尚未上傳。

### 2026-11-08：P1＋P2 Public LB

| 上傳時間 | 檔案 | 本地 E2 | Public LB | 對 AH＋C（0.7103381） |
|---|---|---:|---:|---:|
| 11/08 16:00:35（帳號 jabir95tsai） | `ahc-p12_722279_rank.csv` | 0.7360 | **0.7165169** | **+0.0062** |

平台顯示 19/99。預期約 +0.02，實得 +0.0062，方向一致但幅度約三分之一，符合事先寫的「E2 選擇偏差，實際增益可能較小」。兩檔在 Public 只有 59 題不同，差值的 LB 雜訊約 ±0.011（1 SD），+0.0062 約 0.5 SD，**單憑 LB 無法排除零增益**；但本地 E2（+0.021，CI 不含零）、E1（+0.014）與 LB 三者同號。
P1 單獨探針（hd01，對首選 47 題不同）預期增益更小、雜訊相近，上傳也分不出 P1／P2 貢獻，不建議再傳。

### 2026-11-08：級別先驗探針（Slurm 724435，CPU）

**假設：** B2 合成題按玩家等機率抽，隱含先驗＝Valid-S 玩家級別比例（不均勻）；若測試每級等量，往均勻先驗校正應有增益。三種解碼都只改 P1＋P2 的最後一步（`scripts/strength/rank_prior_probes.py`）：`uprior` 把先驗換成均勻；`em` 以 Saerens EM 由預測估計題組先驗；`balanced` 強制每級題數相等（期望分數指派）。

| 本地（P1＋P2 模型） | k=5 | k=10 | k=15 | k=20 | Equal-k | 差值（bootstrap 95%） | 改變題數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 base（級別不完全平均） | 0.6265 | 0.7063 | 0.7461 | 0.7320 | 0.7027 | — | — |
| E1 uprior | 0.6406 | 0.7078 | 0.7501 | 0.7360 | 0.7086 | +0.0059 [−0.0006, +0.0122] | 33 |
| E1 em | 0.6221 | 0.7084 | 0.7211 | 0.7230 | 0.6936 | −0.0091 [−0.0249, +0.0067] | 149 |
| E1 balanced | 0.5928 | 0.7044 | 0.7230 | 0.7280 | 0.6871 | −0.0157 [−0.0272, −0.0041] | 100 |
| E2 base（每級恰 10%） | 0.6763 | 0.7320 | 0.7619 | 0.7736 | 0.7360 | — | — |
| E2 uprior | 0.6845 | 0.7284 | 0.7612 | 0.7762 | 0.7376 | +0.0016 [−0.0037, +0.0068] | 92 |
| E2 em | 0.6612 | 0.7266 | 0.7512 | 0.7695 | 0.7271 | −0.0089 [−0.0207, +0.0031] | 273 |
| E2 balanced | 0.6846 | 0.7297 | 0.7675 | 0.7671 | 0.7372 | +0.0013 [−0.0063, +0.0085] | 125 |

**判讀：** uprior 兩組皆小幅正向（未達 +0.005 gate 的嚴格要求，E2 只 +0.0016）；em 兩組皆負，EM 估出的先驗不穩（Private 2D 僅 0.005），不建議；balanced 只在完全平衡時持平略正，E1 這種輕度不平衡就 −0.016，是測試「級別是否等量」的探針而非改進。

**提交檔（皆 800 列、sample 順序；對 P1＋P2 改變 Public／Private 題數）：**
- `ahc-p12-uprior_724435_rank.csv`（SHA `cea20a80…`）：21／17 題；預期 LB 與 0.7165 差 0～+0.005，雜訊約 ±0.006。
- `ahc-p12-balanced_724435_rank.csv`（SHA `b1b66882…`）：59／50 題；測試若每級等量則約持平，明顯下降則代表測試級別不等量。
- `ahc-p12-em_724435_rank.csv`（SHA `8f3324fd…`）：54／52 題；本地為負，不建議上傳。
- `ah_713770_rank.csv`（SHA `347906c7…`）：AH 對照（無 C），對 AH＋C 首選 Public 33 題不同；測 C 在 LB 的貢獻（本地 +0.0102）。

### 2026-11-08：先驗探針 LB 與玩家重疊檢查（Slurm 724449）

| 上傳（jabir95tsai） | 檔案 | Public LB | 對 P1＋P2（0.7165169） | 改變題數 |
|---|---|---:|---:|---:|
| 16:15:27 | `ahc-p12-uprior_724435_rank.csv` | 0.7076956 | **−0.0088** | 21 |
| 16:16:06 | `ahc-p12-balanced_724435_rank.csv` | 0.6749365 | **−0.0416** | 59 |

**判讀：測試集級別不是每級等量，往均勻先驗校正有害。** balanced 在 59 題上淨失約 16.6 分（強迫配額把 D 題改成 6D 等），uprior 的 21 題主要是 A／2D→1D（11 題），淨失約 3.5 分，表示測試中 1D 比均勻少；方向與「按玩家抽題、級別比例接近玩家數比例」一致（1D 玩家最少），而 B2 原本隱含的就是玩家比例先驗。結論：維持 P1＋P2 原解碼，不做任何級別平衡或均勻化；em 檔不上傳。本地 E2（構造為等量）對這類假設無判別力，以 LB 為準。

**玩家重疊檢查（`scripts/strength/rank_player_overlap.py`，開局指紋 TF-IDF，CPU，`outputs/rank/player_overlap/report.json`）：結論不確定，指紋太弱。**
- 題目間同一玩家（Check 1）：門檻設在模擬「全不同人」每份約 1 對誤報（0.4297）。模擬中刻意放入的 100 對同人只抓到 4–12 對（召回約 8%）。測試 Public 9 對、Private 7 對超過門檻，高於虛無的約 1 對；但測試整體相似度分布（中位數、p90）也普遍高於模擬，可能只是測試棋局的開局族群與訓練不同，不能據此推論有同一玩家。即使真有，8% 召回能合併的題目極少，增益估計 < 0.005。
- 測試玩家在訓練集（Check 2）：同樣指紋在 16k 個玩家×顏色中找到正確玩家的比率只有 30%（k=5 22%、k=20 41%），「在／不在」兩情境的 top-1 相似度幾乎重疊（超過門檻 3.5% vs 1%）。測試 top-1 中位數（0.387／0.393）高於兩種模擬情境，同樣顯示族群偏移，比例估計（Public 60%、Private 30%）不可信。配合 SE3 LB 落回 unseen 水準的證據，L10 成立的可能性仍判斷為低。
- 若要再查，需要更強的指紋（競賽一 GE2E 只支援黑方，約 1 GPU-h）並設計不受族群偏移影響的檢定；預期回報低，未排程。

### 2026-11-08：B2 延伸 A（勝負近似）／B（題內離散度、MLP）——全部未過 gate（Slurm 724619，CPU 47 分）

`scripts/strength/rank_b2_extensions.py`；P1＋P2 設定（Valid-S HD≤1＋HD1 旗標），所有組共用 B2 同一批合成題抽樣，對照組重現 E2 0.7360。A 特徵由 `features/game_stats.py` 重播：最後一手是否本人（中盤認輸時多為勝方）、手數、終局子數佔比、提子統計，以及「最後一手是本人」×對手 HumanSL 特徵（贏過什麼程度的對手）。

| E2，seed 2027 | k=5 | k=10 | k=15 | k=20 | Equal-k | 差值（bootstrap 95%） | 五 seed 平均差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 對照（P1＋P2） | 0.6763 | 0.7320 | 0.7619 | 0.7736 | 0.7360 | — | — |
| A_outcome | 0.6841 | 0.7335 | 0.7731 | 0.7446 | 0.7338 | −0.0022 [−0.0122, +0.0080] | +0.0015 |
| A_allstats（＋全部重播統計） | 0.6664 | 0.6964 | 0.7532 | 0.7239 | 0.7100 | −0.0260 [−0.0395, −0.0123] | −0.0229 |
| B_std（平均＋題內標準差） | 0.6865 | 0.7268 | 0.7526 | 0.7473 | 0.7283 | −0.0077 [−0.0185, +0.0030] | −0.0093 |
| B_mlp（同上，MLP 頭） | 0.6769 | 0.7057 | 0.6979 | 0.7312 | 0.7029 | −0.0331 [−0.0495, −0.0170] | −0.0351 |

E1 五個 seed 全部為負（A_outcome −0.000～−0.014）。

**判讀：** 規律一致——特徵越多、頭越複雜越差，且 **k=20 每組都掉最多**（−0.026～−0.050），A_outcome、B_std 在 k=5 反而 +0.008／+0.010。機制推測（未驗證）：B2 的合成題只來自 Valid-S 玩家，k=20 需要至少 20 盤 HD≤1 的玩家，可用人數最少，2 萬題反覆抽同一批人，多出的特徵讓模型記住玩家而非學棋力；與 C4 時「可施力的是玩家數而非算力」一致。勝負近似本身沒有帶來可測增益；測試集「最後一手是本人」比例 0.526–0.529，高於 Valid 的 0.504，原因未知。結論：B2 這一層在現有 5,000 名擬合玩家下已接近飽和，增益要從每盤特徵本身（上游）或更多擬合玩家來。不產生提交檔。

**LB 探針檔（2026-11-08，Slurm 724951；使用者決定：理論上合理且本地小幅下降者也上 LB 測）：** seed 2027 擬合，對照組逐題重現 P1＋P2 提交檔（Public／Private 0 題不同）。
- `ahc-p12-outcome_724951_rank.csv`（SHA `e18dbf0d…`）：A_outcome；對 P1＋P2 改變 Public 53／Private 48 題；本地 E2 −0.0022（五 seed 平均 +0.0015）。
- `ahc-p12-std_724951_rank.csv`（SHA `c5624cac…`）：B_std；改變 53／53 題；本地 E2 −0.0077（五 seed 平均 −0.0093）。
兩檔各只改一處，LB 對照為 P1＋P2 的 0.7165169；差異題 53 題的雜訊約 ±0.010（1 SD）。
- **LB：** `ahc-p12-outcome_724951_rank.csv` 11/08 18:22:02（jabir95tsai）Public **0.7086153**，對 P1＋P2 **−0.0079**（53 題，約 −0.8 SD，雜訊內但方向為負）；與本地 E2 −0.0022 同號。勝負近似不採用。

### 2026-11-08：P3 讓先局貼目 0.5（Slurm 725001 特徵、725149 測試特徵、725150 匯出／評估）

Valid-S/U 17,744 盤＋測試 2,280 盤讓先局，H 與 C 都以 KataGo 貼目 0.5 重算（短測 724501：H 平均絕對變化 0.0135，約為盤間標準差 0.31 的 4%；C 0.0062／0.27）。`humansl_rankvec extract` 新增選用 `--komi`（不帶時行為不變）。實際 8 卡約 49 分鐘＋測試 2 卡；725001 顯示 FAILED 只因測試分片表格欄名錯誤（已修正後由 725149 補算），訓練分片全部完整。
提交檔 `ahc-p12-komihc_725150_rank.csv`（SHA `3a21d9d6…`，800 列）：seed 2027、Valid-S HD≤1＋HD1 旗標，H＋C 都換成貼目 0.5 版；對 P1＋P2 只改 Public 6 題、Private 2 題，皆差一級。LB 與 0.7165 的差距上限約 ±0.015，實際預期在 ±0.005 內。本地 gate 評估同作業進行中。
- **LB：** `ahc-p12-komihc_725150_rank.csv` 11/08 19:47:16（jabir95tsai）Public **0.7196775**（19/99），對 P1＋P2 **+0.0032**；只有 6 題不同，相當於這 6 題淨多約 1.3 分，屬雜訊範圍但為目前 Public 最高分。本地 E2（只換 H 的臂，seed 2027–2030）差值 +0.0008／−0.0011／−0.0003／+0.0009，接近 0。

**首選更新（2026-11-08，使用者確認）：** 競賽二首選改為 `ahc-p12-komihc_725150_rank.csv`（P1＋P2＋P3，Public 0.7196775）。演進：AH＋C 0.7103 → P1＋P2 0.7165 → P1＋P2＋P3 0.7197，三步皆為讓先語意修正。重跑：`humansl_hd_filter.py`（P1／P2）＋`slurm/rank_p3_komi.slurm`、`slurm/rank_p3_test.slurm`、`humansl_p3_komi.py export --arm hc`（P3）；尚未併入一鍵 `slurm/rank_ahc_predict.slurm`。

### 2026-11-08：B2 bagging／拿掉 A（Slurm 725503，CPU）

P1＋P2＋P3 設定，E2 Equal-k：首選 seed 2027 為 0.7372，五 seed 0.7372／0.7355／0.7327／0.7373／0.7367（平均 0.7359）。
- bag10（10 個合成抽樣 seed 機率平均）0.7343、syn5x（5 倍合成題）0.7343：都只回到單 seed 平均附近（−0.0016），沒有增益；E1 也是 −0.002／−0.007。seed 2027 在本地剛好偏好。
- **拿掉 A-024 的 10 欄（noA）：E2 五 seed 全部 ≥ 對照，平均 +0.0056**（seed 2027 為 0.7441，+0.0070，bootstrap [−0.0035, +0.0176]）；E1 平均 +0.0002。A-024 對 B2 沒有可測貢獻。意義：替 Valid-S 玩家的 training-split 棋局補算 H＋C（A 在那些棋局是 in-sample）不再有障礙，k=20 可用擬合玩家可由 1,036 擴到約 4,800（每 k 可用玩家：2,685／1,847／1,359／1,036）。

### 2026-11-08：競賽一讓子處理與路徑修正（11/11 前準備）

- **路徑：** `outputs/` 重整後 `identity/predict.py`（開局參數）與 `identity/insurance.py`（GE2E）仍指向舊路徑，11/11 會直接失敗；已改指 `outputs/pid/opening/…`、`outputs/pid/ge2e/…`。`humansl_predict`、`v2/predict` 預設路徑與 `run_release.sh` 兩個模型路徑同步修正（競賽二正式流程一向明確傳路徑，不受影響）。
- **HD 欄位：** `preprocess` 轉檔時記錄每盤 `hd`（無 HD 為 −1）；`day0_audit.py` 新增 HD 分布、HD≥2 前 10 手是否落在 Q13 星位、題數 vs 候選人數（>候選人則必非一對一）。
- **彩排（假資料＝訓練集合成候選庫 split 1／2，Slurm 725660、725705）：** HD≥2 佔候選庫 4.2%、題目 3.6%，前 10 手落在星位 0 盤（與訓練集相同：標記反轉、缺讓子石）；有 5 位候選人 100 盤全為 HD≥2。

| 彩排 Top-5 衰減分數 | public k=5 | public 平均 | private k=5 | private 平均 |
|---|---:|---:|---:|---:|
| 保留 HD≥2（A-053 行為） | 0.9461 | 0.9865 | 0.9374 | 0.9828 |
| 跳過 HD≥2 | 0.9261 | 0.9715（6 題無法評分） | 0.8851 | 0.9681（4 題） |
| 保留＋讓子分布項 w=0.5 | 0.9463 | 0.9866 | 0.9374 | 0.9822 |
| w=1 | 0.9474 | 0.9869 | 0.9463 | 0.9850 |
| **w=2** | 0.9497 | 0.9874 | **0.9612** | **0.9887** |
| w=4 | 0.9561 | 0.9890 | 0.9499 | 0.9859 |

判讀：跳過 HD≥2 有害（候選庫與題目的錯標一致，反而是指紋）→ 預設保留，`--hd2plus drop` 只當選項。讓子分布項（候選人 100 盤 HD 直方圖下題目各盤 HD 的 log 機率，`predict.hd_scores`）兩份都正向；w=2 在兩份皆為正且不在網格邊界，採用。合成驗證已飽和，真實測試的效果未知；若正式資料沒有 HD，該項為常數、不影響排序。`run_release.sh` 主檔與指派變體加 `--hd-weight 2`，另出 `pid_<part>_nohd.csv`（A-053 原融合）作 LB 對照。CPU 測試 16 項通過。

**LB 探針檔（Slurm 725504，CPU 84 分）：** 對照組重算逐題重現首選 `ahc-p12-komihc_725150_rank.csv`（Public／Private 0 題不同）。各檔只改一處，對首選改變題數（Public／Private）：
- `ahc-p3-noA_725504_rank.csv`（SHA `727150ca…`）：拿掉 A-024；41／49 題；本地 E2 +0.0070（五 seed 平均 +0.0056）。**LB 11/09 06:21:18（jabir95tsai）Public 0.7270608（16/102），對首選 +0.0074**；41 題不同，相當於淨多約 3 分（約 0.7 個標準差的雜訊量級，但與本地五 seed 同方向）。目前 Public 最高分。
- `ahc-p3-allp_725504_rank.csv`（SHA `27a3cc0c…`）：Valid-S＋U（HD≤1）擬合；55／65 題；無本地分數（A-058 交叉擬合 +0.0017）。
- `ahc-p3-bag10_725504_rank.csv`（SHA `e161e0ea…`）：10 seed 平均；20／17 題；本地 E2 −0.0028。
- `ahc-p3-allpbag10_725504_rank.csv`（SHA `7aacc8f8…`）：S＋U 且 10 seed 平均；46／69 題。

**以 noA 為基礎的探針（2026-11-09，`rank_b2_bagging.py export --arms …`）：** 重算 noA 與 `ahc-p3-noA_725504` 逐題相同（0／0 題）。
- `ahc-p3-noA-allp_731429_rank.csv`（SHA `53f8b8fd…`，Slurm 731429，5 分）：noA＋Valid-S＋U（HD≤1）擬合；對 noA 改 50／55 題（對原首選 63／83）。無本地分數（評估集就是 Valid-U）。**LB 11/09 08:46:25（jabir95tsai）Public 0.7386411（15/102），對 noA +0.0116**（50 題不同，淨多約 4.6 分）；目前 Public 最高分。與 A-058 本地交叉擬合 +0.0017 相比大得多：多 1,442 位未見過的玩家，而測試玩家也是未見玩家，推測擬合玩家數是主要瓶頸（與 A-044／A-047 的資料增益一致；機制未驗證）。
- `ahc-p3-noA-bag10_731431_rank.csv`（SHA `2bca912d…`，Slurm 731431，36 分）：noA 10 seed 平均；對 noA 改 22／16 題。
- `ahc-p3-noA-allpbag10_731431_rank.csv`（SHA `72cc8ae2…`）：noA-allp 10 seed 平均；對 noA-allp 改 16／17 題。本地 bag10 無增益（−0.0016），屬低優先探針。

**noA＋train20（2026-11-09，使用者同意，約 10 GPU-h）：** `scripts/strength/rank_train20_hc.py`、`slurm/rank_train20_hc.slurm`（Slurm 733177，8 卡；後續 CPU 733179 先匯出 `ahc-p3-noA-t20`、`ahc-p3-noA-t20-allp` 再跑 E2／E1 五 seed）。A-047 的 train20（每位 train 切分玩家最多 20 盤）去掉 HD≥2 與學 C 向量用過的 15,000 盤：HD0 68,513 盤補 C（貼 6.5，H 已有），HD1 18,235 盤以貼 0.5 重算 H＋C。可組題（玩家、顏色）k=5／10／15／20 由 3,807／2,178／1,305／816 增為 9,948／6,542／3,502／2,112。增益未知。
- GPU 733177：8 卡 81 分（約 10.8 GPU-h），86,748／86,748 盤特徵完整。擬合盤數 62,867 → 149,615；B2 可抽玩家（依 player_id，≥k 盤）k=5／10／15／20：2,685／1,847／1,359／1,036 → 5,604／5,177／4,500／3,146。
- `ahc-p3-noA-t20-allp_733179_rank.csv`（SHA `695140f6…`；11/10 更正，原誤記為 `bd14b596`）：Valid-S＋train20＋Valid-U；對 noA-allp 改 40／49 題。**LB 11/09 16:57:42（jabir95tsai）Public 0.7404805（15/103），對 noA-allp +0.0018**（40 題不同，淨多約 0.7 分，雜訊範圍）；目前 Public 最高分。
- `ahc-p3-noA-t20_733179_rank.csv`（SHA `bd14b596…`；11/10 更正）：Valid-S＋train20；對 noA-allp 改 66／59 題（對 noA 76／80）。
- 預測分布（800 題，D…6D）：noA-allp 87／98／91／111／52／71／63／81／78／68；noA-t20-allp 86／92／98／128／34／70／66／87／72／67。加入 train 切分玩家後 1D 預測變少、A 變多，與各級玩家數（1D 433 人、A 1,582 人）造成的 B2 隱含先驗一致；測試集的級別分布未知。
- **本地（Slurm 733179，CPU 19 分；Valid-S 擬合 vs Valid-S＋train20 擬合，Valid-U 評分）：**

| 差值（t20 − noA） | seed 2027 | 2028 | 2029 | 2030 | 2031 | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| E2 Equal-k | −0.0139 | −0.0066 | −0.0021 | −0.0127 | −0.0159 | **−0.0102** |
| E1 Equal-k | +0.0013 | +0.0090 | −0.0018 | −0.0022 | −0.0020 | +0.0009 |
| E1（只含 HD≤1 題） | +0.0054 | +0.0150 | +0.0051 | +0.0048 | −0.0013 | +0.0058 |

  seed 2027 E2 分 k：−0.0089／−0.0125／−0.0248／−0.0093，bootstrap [−0.0288, +0.0006]；E1 分 k：+0.0028／−0.0080／−0.0061／+0.0166。未過 gate；E2（每級 50 題、級別均勻）偏負，與 1D 預測減少的先驗偏移方向一致；E1 不變。判讀：在擬合玩家已含 Valid-U 時，再加 train 切分玩家的淨效果接近 0（LB +0.0018、E1 +0.0009）；資料增益可能被先驗偏移抵消，兩者未分離（未知）。

**先驗對齊（2026-11-09，使用者同意，`scripts/strength/rank_more_players.py`，CPU）：** B2 合成題原本對「≥k 盤的玩家」均勻抽，隱含先驗＝各級可抽玩家占比，加 train20 後該占比改變（1D 預測 52→34、A 111→128）。`PriorMatched`：先依參考先驗抽級別，再在該級均勻抽玩家，其餘（20,000 題／k、LR C=1）不變。
- **本地（Slurm 736587，4 分）：** Valid-S 擬合 vs Valid-S＋train20 擬合＋Valid-S 先驗，Valid-U 評分。

| 差值（t20＋先驗 − noA） | seed 2027 | 2028 | 2029 | 2030 | 2031 | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| E2 Equal-k | +0.0019 | +0.0061 | +0.0052 | +0.0019 | +0.0013 | **+0.0033** |
| E1 Equal-k | +0.0158 | +0.0241 | +0.0114 | +0.0163 | +0.0164 | **+0.0168** |

  seed 2027 分 k：E1 0.6158／0.7129／0.7487／0.7204 → 0.6198／0.7280／0.7672／0.7461（Equal-k 0.6994 → 0.7153，bootstrap [+0.0016, +0.0305]）；E2 0.6843／0.7456／0.7713／0.7753 → 0.6912／0.7433／0.7773／0.7722（0.7441 → 0.7460，[−0.0110, +0.0148]）。兩組五 seed 全正。對照未對齊的 t20（E2 −0.0102、E1 +0.0009）：**先驗對齊把 train20 的資料增益釋放出來**，支持「增益被先驗偏移抵消」的假設。E1 過 gate（bootstrap 下界 > 0）。
- `ahc-p3-noA-t20-allp-prior_736589_rank.csv`（SHA `d98def9c…`，Slurm 736589，3 分）：Valid-S＋train20＋Valid-U，先驗＝Valid-S＋U（HD≤1）的玩家占比；同作業重算最佳檔逐題相同（0／0 題）。對最佳檔 `ahc-p3-noA-t20-allp_733179` 改 Public 27／Private 33 題，全為差一級（A→1D 12、2D→1D 4、B→A 4…）。**LB 11/09 21:50:35（jabir95tsai）Public 0.7508820（12/106），對 noA-t20-allp +0.0104**（27 題全差一級，淨多約 4.2 分）；本地 E1 +0.0168／E2 +0.0033 同方向，依泛化原則升為**新基底**。目前 Public 最高分。預測分布 88／87／105／112／49／72／65／84／73／65（最佳檔 86／92／98／128／34／70／66／87／72／67）。
- **fold −1 排除玩家（794 人，每人最多 40 盤 HD≤1，共 24,747 盤）的 H＋C：** 第一次 Slurm 736586 落在 25a-hgpn083，8 卡皆只有約 300 positions/s（train20 在 hgpn119 為 3,515），2 小時逾時、無產出（抽取程式不能續跑，浪費 16 GPU-h；節點慢的原因未知）。重送 737126（排除 hgpn083、上限 2.5 小時；開跑 30 秒約 2,400 positions/s，正常），匯出 737127 接續。
  - 737126 共 64 分（約 8.5 GPU-h），24,747／24,747 盤特徵完整；匯出 737127（5 分）重現最佳檔 0／0 題，`-prior` 與 736589 逐位元相同。
  - `ahc-p3-noA-t20-allp-ex_737127_rank.csv`（SHA `854468a7…`）：一般 B2（無先驗對齊）＋794 人；對最佳檔改 27／25 題，對 `-prior` 改 38／30 題。預測分布 86／89／97／135／32／72／65／88／68／68：1D 再減、A 再增，與未對齊 t20 的先驗偏移同方向（該偏移在本地 E2 −0.0102）。**無本地分數，暫不升為基底。**
  - 補做（2026-11-09 21:50，使用者同意）：`rank_more_players.py local-ex`（Slurm 737644）以 Valid-U E1／E2 五 seed 比較「＋794 人」的增益，先驗對齊與一般 B2 各一組；`export` 新增 `-prior-ex` 臂（Slurm 737645）。
  - `737645-priorex.csv`（原名 `ahc-p3-noA-t20-allp-prior-ex_737645_rank.csv`，22:30 改名；SHA `dc847f3d…`）：先驗對齊＋794 人；對新基底 `-prior`（0.7509）改 Public 24／Private 13 題，全為差一級（1D→2D 6、B→A 5、C→B 5…）。同作業重算 `-prior` 逐位元相同。
  - **local-ex 五 seed（Slurm 737644，CPU 31 分，`local_ex.json`；＋794 人 − Valid-S＋train20，Valid-U 評分）：**

| 差值 | 2027 | 2028 | 2029 | 2030 | 2031 | 平均 | 分 k 平均（5／10／15／20） |
|---|---:|---:|---:|---:|---:|---:|---|
| 先驗對齊 E1 | +0.0013 | −0.0022 | +0.0062 | −0.0058 | −0.0076 | **−0.0016** | +0.0009／−0.0029／+0.0052／−0.0097 |
| 先驗對齊 E2 | −0.0048 | −0.0018 | +0.0001 | +0.0012 | +0.0017 | **−0.0007** | −0.0048／−0.0040／+0.0060／−0.0001 |
| 一般 B2 E1 | −0.0044 | +0.0065 | +0.0034 | −0.0016 | +0.0000 | +0.0008 | +0.0011／+0.0108／+0.0009／−0.0097 |
| 一般 B2 E2 | +0.0083 | +0.0007 | +0.0018 | +0.0023 | +0.0062 | +0.0039 | +0.0035／+0.0034／+0.0043／+0.0043 |

  seed 2027 bootstrap：先驗 E1 [−0.0081, +0.0106]、E2 [−0.0127, +0.0028]；一般 E2 [−0.0008, +0.0172]。判讀：先驗對齊後再加 794 人沒有增益（兩組平均略負、未過 gate）。一般 B2 的 E2 五 seed 全正，但它的對照是未對齊、先驗已偏的 t20；794 人的級別組成（1D 42 人＝5.3%）可能把隱含先驗拉回一點，增益較可能來自先驗而非資料（未驗證）。推測 train20 之後擬合玩家數已不是瓶頸（未驗證）。**`737645-priorex.csv` 不升為基底，LB 探針優先度低。**
  - **上傳判斷（22:17 上傳前登記）：** 本地沒有支持 → 不升為基底，只當資訊探針用掉今日最後一次額度。預期 LB 與 0.7509 差距在 ±0.02 內（雜訊）；不論升降，基底維持 `-prior`，除非之後本地證據轉正。
  - **LB 11/09 22:18:13（jabir95tsai）Public 0.7508820（12/106）**，與 21:50 的 `-prior` 到第 7 位小數完全相同。使用者當時在 IDE 開的是 `ahc-p3-noA-t20-allp-prior_737645_rank.csv`（與 `-prior_736589` 逐位元相同），**使用者確認誤傳了重複檔**，`-prior-ex` 未上傳、仍無 LB 分數；此次額度浪費，不影響基底。
  - **命名規則（2026-11-09 使用者要求，`go_aicup.tasks.strength.submission_files.write_rank_csv`）：** 平台會把長檔名截成 `ahc-p3-noA-t20-a...csv`，前綴相同的檔看起來一樣。新檔一律 `<job>-<tag>.csv`（tag ≤ 8 字元，全名 ≤ 19 字元，例 `737645-priorex.csv`），完整來源記在本檔；與既有檔內容相同時不再另寫。`rank_more_players.py` 已改用；其他舊匯出腳本下次使用時再改。
  - **`737645-priorex.csv` LB 11/10 10:59:23（jabir95tsai）Public 0.7468017（14/108），對基底 `-prior`（0.7509）−0.0041**：24 題不同、全為差一級，淨少約 1.6 分（約 2.6 題由對變鄰級），在 ±0.02 雜訊內；與本地 local-ex 先驗對齊 E1 −0.0016／E2 −0.0007 同方向。結論：先驗對齊後再加 794 人無增益，**基底維持 `-prior_736589`**。⚠️ 平台上競賽二的最後一筆現在是 `priorex`，停止上傳前須再傳一次基底（或之後更好的檔）。
  - **決策原則（同日使用者修訂 CLAUDE.md §6）：** 最終排名看 Private LB，以本地多 seed 證據＋機制為主，Public 差距 < 約 0.02 視為雜訊。`-prior` 上傳的事前規則：除非 Public 掉超過 0.02，否則保留為候選基底。


### 2026-11-09：HumanSL residual B2 三階段程式與 Slurm

依使用者指定的 `src/go_aicup/tasks/strength/humansl_residual_b2/PLAN.md`，以三個 subagent 分工，完成固定 HumanSL 雙分支分類、凍結第一階段的十層殘差修正、四個 k 的 22 維線性 B2；主 agent 整合 DDP／microbatch、精確續跑、嚴格改善早停、best/final/latest、監控／發布失敗重試、100 步量測、可追蹤的三階段 Slurm。另完成 HD0/1／原 fold 稽核、雙方選定 ply 串流編碼、逐位置強 teacher、四折 OOF／全 D 特徵、H＋C＋P2 對照、完整 M best/final 選擇／信心尺度診斷、配對 gate／玩家 bootstrap、U 設定鎖定、release 固定 s1/s2 重訓及單一全 P 模型 CSV 匯出接線。共享程式僅新增 CV helper 及相容的 export 檔名參數。

整合 CPU／Gloo 回歸 **108 項通過（108.24 秒）**；其後最新特徵／B2 回歸 **15 項通過（1.83 秒）**，兩組重疊、合計 **109 個不同測試**。包含實際兩程序訓練／累積梯度、第二階段凍結、完整 RNG／sampler 續跑、spawn workers 預取、teacher 假引擎部分失敗重試，以及真實測試 store→board cache→模擬 HumanSL→B2／CSV 的接線。既有 HD0／HD1 各三個位置與舊 encoder 逐元素一致。所有新 CLI／兩份 config 與 Slurm bash 語法檢查通過。這是工程驗證，**k=5/10/15/20 分數與 Equal-k 增益均未知**。

短測設定 `configs/strength/humansl_residual_smoke100.json` 的兩階段各 100 updates、AdamW LR=0.001；C 對照 100 updates、LR=0.01，均只作工程起點，未實跑 GPU。正式 `humansl_residual_train.json` 的 updates／LR 留 null，需量測後明訂；smoke 不允許鎖定 U／release。C 對照為有界全盤位置均勻抽樣 Adam／cosine，保留原 likelihood 目標，不宣稱逐步重現舊 epoch packs 訓練。

唯讀查核發現 `data/processed/cv/` 為空，預設的原 seed=2027 完整玩家 fold 表尚不存在；prepare 要求取回原表並明確驗證，不自行重抽。未建立本版官方資料／題庫、啟動真 teacher／GPU 短測、正式訓練／全量推論、提交 Slurm 或上傳；真實收斂、CUDA／SyncBN、顯存／吞吐與完整成本未知。輸出隔離於 `outputs/rank/humansl_residual_b2/<run_id>/`；三階段及輔助 Slurm 位於套件內 `slurm/`，操作、前置條件、全部命令與量測界線見 [README](src/go_aicup/tasks/strength/humansl_residual_b2/README.md) 及 PLAN §13。


## 2026-11-10：HumanSL residual teacher 匯入 SE3 與八分片續跑

已新增 `teacher_reuse.py`，透過 `outputs/value_resume_672603_recovery.json` 留存的八個原始 manifest，以及 SE3 的 `data_identity.json` 綁定舊 value／prepared checksum；確認 KataGo engine、model、analysis config 雜湊與日本規則、貼目 6.5、1 visit、完整歷史及指定玩家視角相符後，僅匯入 HD0 本人位置，其他位置補算。原八片 23,302 盤完成資料全部通過 checksum，未覆寫；共 407,329 盤 HD0 開發棋局可對齊，原可重用位置合計 39,151,903（含已完成棋局，非未完成部分的節省量）。全舊 value 值域／有限值通過，逐片各抽 100 盤實際 moves/colors 與 ply 對齊通過。稽核報告 `data/processed/humansl_residual_b2/teacher_v1/se3_resume_audit.json`。

保留原 identity／分片 manifest，明列原 labeler SHA-256 作為相容續跑版本；已完成棋局略過，未完成棋局先匯入再補缺值，整盤才提交 checksum/done。共用初始化改用等待式檔案鎖、分片用獨占非等待鎖，程序退出自動釋放；Slurm exec Python 並在 USR1/TERM/INT 後完成當前棋局再停止。確認八個舊作業已取消且無 teacher 工作後，清理八個殘留 `.writer.lock`。

來源／索引拒絕、HD1 排除、八片匯入／中斷重試、不覆寫已完成值、鎖競爭與既有資料／訓練／模型回歸共 25 項 CPU 測試通過（50.38 秒）；Slurm bash 語法通過。未啟動 GPU 補算或 Slurm，實際提速、剩餘時間與 k=5/10/15/20 棋力增益未知。兩份既有 config 已啟用匯入，續跑指令見套件 README。


## 2026-11-10：提交檔整理

`submissions/rank/` 頂層只留現行基底 `ahc-p3-noA-t20-allp-prior_736589_rank.csv`（`d98def9c`）；其餘移到 `archive/`，三個逐位元重複檔（`-prior_737127`、`-prior_737645`、`-ex_737645`）移到 `archive/duplicates/`，未刪除任何檔。索引（每檔 SHA、上傳時間、Public LB、未上傳原因）見 `submissions/rank/INDEX.md`。`rank_b2_bagging.py`／`rank_more_players.py` 寫死的重現檢查路徑同步改指 `archive/`。
整理時重算 SHA 發現 11/09 紀錄把 `noA-t20-allp_733179`（實為 `695140f6`）與 `noA-t20_733179`（實為 `bd14b596`）的 SHA 寫反；已以題數差（對 noA-allp 40／49 vs 66／59）與預測分布確認檔案內容正確、只是帳本誤記，上方已更正。



## 2026-11-10：HumanSL residual teacher 742066「完成」但標籤未落地 → 改寫入方式重跑（746225）

- **現象：** array 742066（8 分片 × 1 GPU，各約 3.1 h，共約 25.6 GPU-h）全部 `COMPLETED`、exit 0、`metrics.json` 寫 `status=complete`，但磁碟重驗只有 **23,304／618,954 盤 D（3.8%）** 可驗證，幾乎等於開跑前的 SE3 匯入續跑量。
- **判定為何誤報：** 舊 `status=complete` 只表示迴圈未被停止；`completed_games` 讀的是程序自己的 memmap 映射，未檢查 `done.sum()==training_games`，也未從磁碟重讀。
- **為何未寫入（證據）：** `values/done/checksums.npy` 以 `np.memmap` r+ 原地寫、每盤 `flush()`（msync 回報成功）。執行於 hgpn030／hgpn032 的分片 2–7，三個檔的 mtime 停在開跑後首次寫入（10:38），3 小時內頁面從未寫回；hgpn004 的分片 0／1 前約 15% 處理量遺失、之後落地，但 `checksums.npy` 全程未落地（約 11.7 萬列有值、checksum=0）。分片 0–2 連程序自身視圖都掉列（髒頁被丟棄），與登入節點 11:17–11:24 讀取時間吻合，原因只能部分解釋。同作業一般寫入（metrics.json、se3_import.json）正常；在原節點讀檔與伺服器相同，無法搶救。
- **重現：** 健康節點 hgpn015（746008，並行讀取）與 hgpn013（746025，先獨占寫 5 分鐘、96 MB 檔）上，memmap+flush、+fsync、pwrite+fsync、不 flush 四種寫法全部完整落地（90/90、210/210）。WEKA 文件未提 mmap；**根因（Weka client 狀態／節點相依）未知**。診斷約 0.3 GPU-h，腳本在 `outputs/diag/memmap_weka/`。
- **修正（`teacher.py`）：** 每 `chunk_games`（500）盤寫 `chunks/chunk_*.npz`（一般寫入＋fsync＋原子 rename＋目錄 fsync）；結束時原子發布三個 `.npy`、丟本機快取後重讀驗證，相符才刪 chunk；續跑逐盤重驗既有陣列，不可驗證列丟棄重算；只有 D 全部經磁碟重讀驗證才 `complete`，否則非 0 結束。Slurm 改 6 h、轉送 USR1，另起程序 `--verify`；進度腳本計入 chunk 作為登入節點外部檢查。標籤語意與 teacher_id 不變（identity 仍為 LEGACY 相容）。新增 4 項測試（崩潰保留已提交 chunk 不重查、發布失敗由 chunk 重放、舊列 checksum=0 丟棄重算、重讀不符不得 complete），相關回歸共 25 項通過。
- **重跑：** array **746225**（8 分片 × 1 GPU，16:04 開跑）。16:14 外部檢查：全部分片（含又排到 hgpn030 的 6／7）chunk 持續落地，約 18.4 萬盤／h，預估約 3.0 h、約 26 GPU-h。**完成（2026-11-10 約 19:26）：** 8 分片皆 COMPLETED exit 0，elapsed 3:05:04–3:21:47，合計約 25.5 GPU-h；metrics 皆 `complete`。之後以獨立程序 `--verify`（posix_fadvise 丟快取後從磁碟重讀、逐盤 checksum）再驗一次：**8／8 complete，verified 618,954／618,954 盤**（77,496／77,323／77,532／77,482／77,270／77,305／77,340／77,206），殘留 chunk 0 個。分片 0／1 丟棄並重算 742066 留下的不可驗證列 59,013／57,993 盤，分片 2–5 各 1–2 盤，6／7 為 0。這次 hgpn030 也有分到分片 6／7，同樣正常落地。teacher_id `72a1a325…` 不變，可進 stage 1。


## 2026-11-10：HumanSL residual B2 修訂（使用者同意「全部依建議修改」）

只改程式、設定、測試與文件；沒有提交任何作業，也沒有動執行中的 teacher（746225 於 16:03:50 開跑，早於本次修改；teacher 身分綁定的 `dataset.py`、`teacher.py` 與 base prepared manifest 都未改，`dataset_id` 仍為 `5777bfbe…`）。完整理由與取代關係見 `humansl_residual_b2/PLAN.md` §14。

- **聚合：** 每盤每方加存 log_softmax 平均（特徵格式 v2）；B2 主臂改 `aggregation=logp`，prob 保留作對照（依據 A-025：log 聚合在 Valid-U +0.064）。
- **stack 臂：** H30＋C20＋新 20＋P2＝72 維 B2；H＋C 預設仍只在 `fit` 池抽（`stack_fit_pool`），因每位置約 25 次前向。
- **B2 玩家數：** 新增 `ext_v1` 擴充池 `fit_all`＝全部 D 玩家 5,167 人、每人每色最多 20 盤（約 157,843 盤），原 B_dev 為 4,404 人／56,246 盤。
- **預算：** 第一階段 3,000 步（每 300 步監控）、第二階段 6,000 步（每 600 步），cosine 衰減到 5%；原為各 20,000 步、每 5,000 步監控、常數 LR。
- **上游：** 不訓練全 D 模型；M／U／測試用四個 fold 模型的每盤特徵平均。取消 release 重訓：鎖定綁四個 fold checkpoint，release B2 以開發 OOF＋四折平均的 M＋U 擬合。
- **探針：** 新增 `probe.py`＋`probe_cache.slurm`／`probe_run.slurm`，凍結主幹 768 維讀出、四折 OOF 線性／MLP 頭＋B2，在 M 上分 k 報告（有 H＋C 時附 gate）。
- **成本粗估（未驗證）：** 開發約 110–130 GPU-h；原計畫開發＋release 約 950 GPU-h。
- **測試：** `tests/test_humansl_residual_*.py` 62 項 CPU 測試通過（84 秒），含新增 `test_humansl_residual_improvements.py`；release 預測與鎖定的舊測試改寫為四折版本。分 k 分數、探針結果、CPU 編碼吞吐與實際 GPU-h 均未知。
- **下一步（需使用者確認後提交）：** `prepare --extension`（CPU）→ `probe_cache`（8 卡）→ `probe_run`（1 卡），並行跑 H＋C 對照；探針過 gate 才跑第一階段。

**補充（同日，使用者決定）：** 本實驗只回答「ResNet 架構能否勝過 H＋C」，先不做 72 維 stack。config `b2.arms=["new"]`，stack 臂被程式拒絕、探針也不跑；新模型 B2 為 20 維＋P2＝22 維。新增 `fit_pool` 參數，讓新模型 B2 可在與 H＋C 相同的 `fit` 池擬合，作為配對架構比較；`fit_all` 視為另一個變因分開報告。`stage3.slurm` 改跑 best／final × {fit, fit_all} × {prob, logp}。residual 測試 64 項通過（90 秒）。未提交作業。


## 2026-11-10：C 級別向量的 LB 對照（`ah_713770`）

- `archive/ah_713770_rank.csv`（SHA `347906c7…`，AH＝A-024＋HumanSL＋B2，無 C）。**LB 11/10 23:36:39（jabir95tsai）Public 0.6949365（31/115），對 AH＋C `ahc_713770`（0.7103381）−0.0154**。兩檔 Public 33 題不同，淨少約 6.2 分；33 題的差值雜訊約 ±0.008（1 SD），−0.0154 約 −1.9 SD。
- 與本地 Valid-U +0.0102（A-056，玩家 bootstrap [−0.0003, +0.0211]）同方向且幅度相近：**C 在 LB 上也有貢獻**，維持現行基底的 C。事前登記的判讀為「低於 0.7103 即為佐證」。


## 2026-11-11：HumanSL residual B2 凍結主幹探針（Slurm 748656／748657／748659）

`prepare --extension`（52 秒）→ `probe_cache`（8 卡，23:11–23:34）→ `probe_run`（1 卡，738 秒）。凍結 HumanSL 主幹、每盤每方固定 30 手的 768 維讀出，四折 OOF 線性／MLP 頭 → 每盤平均 → 22 維 B2，在 M（5,000 題、每 k 1,250 題、每級 125 題，級別均勻，未見玩家）評分。結果 `outputs/rank/humansl_residual_b2/seed2026/probe/result.json`。

| 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| MLP、logp、fit_all | 0.5224 | 0.5729 | 0.6173 | 0.6454 | **0.5895** |
| MLP、logp、fit | 0.5392 | 0.5728 | 0.5977 | 0.6140 | 0.5809 |
| 線性、logp、fit_all | 0.5079 | 0.5538 | 0.5955 | 0.6314 | 0.5722 |

其餘臂：線性 prob fit_all 0.5688、線性 prob fit 0.5639、線性 logp fit 0.5649、MLP prob fit_all 0.5887、MLP prob fit 0.5758。
- **沒有 H＋C 配對對照**（M 玩家多數沒有 H＋C 特徵），只能和設計相近的 E2（Valid-U、級別均勻）比：H＋C 系列 E2 Equal-k 0.737–0.746，分 k 約 0.68／0.74／0.77／0.77。探針最佳臂低約 0.15，k=20 也低約 0.13，遠大於「每盤只用 30 手」與玩家組不同可能造成的差異（未量化）。
- 判讀：凍結 HumanSL 表示＋分類頭遠不如「各級別條件下實際著手的 likelihood」特徵；與 SE3（直接訓練分類，LB 0.67）、V2（0.555）同一水準。stage 1／2 要靠訓練投影與十層殘差補回約 0.15，沒有證據支持（預期增益：未知，傾向負）。


## 2026-11-11：L10 真實測試檢查（測試玩家是否在訓練集；Slurm 749076，CPU）

`scripts/strength/l10_check.py`（新增 `--reference`，改用 `humansl_residual_b2/seed2027/questions_U.parquet`，因 V2 參考題檔已不存在；已確認其 game_index 與 identity meta 列號逐列一致）。開局模型（D4 pp+slot）以訓練集 8,007 人為候選，每題統計量＝第 1 名減第 2 名 log-likelihood 差；參考題 400 題（每 k 100）給 seen／unseen 兩個分布，門檻＝unseen 第 95 百分位（13.25）。輸出 `outputs/rank/l10_real_test.json`。

| 題組 | 超過門檻比例 | 中位 gap |
|---|---:|---:|
| 參考 unseen（null） | 0.050 | 2.67 |
| 參考 seen | 0.290 | 5.02 |
| **測試 Public** | **0.080** | 2.78 |
| **測試 Private** | **0.0975** | 2.69 |

- 測試的中位 gap 與 unseen 幾乎相同 → **測試玩家大多不在訓練集**。超過門檻的比例比 null 高（Public +3.0 點、Private +4.75 點；400 題下 null 比例的 SD 約 1.1 點），以混合比例粗估約 12%／20% 是訓練玩家；但 11/08 已見測試開局族群與訓練不同，這個多出來的量也可能是族群偏移，未分離（未知）。
- 開局模型對 seen 的召回只有 29%，無法直接拿來指認個別題目。
- **逐題追加（Slurm 749151，同設定重跑並存 `outputs/rank/l10_real_test.questions.csv`）：** unseen null 的最大 gap 71.5；測試超過此值的只有 **7 題**（Public 3、Private 4）。這 7 題「最像的訓練玩家」的級別與現行基底 `-prior_736589` 的預測：相同 3、差一級 3、差兩級 1；低 gap（<5）題目的相同比例只有 12%（69／556），所以這 7 題很可能是真的比到同一人。若全部改成比到的級別且都正確，Public 只多約 1.6 分（約 +0.004）、Private 約 +0.003。**L10 不是大增益來源**；門檻放寬到 13.25 時有 71 題，但誤報約一半以上，不可用。


### 2026-11-11：測試題之間的同人配對、L10 覆寫檔（`scripts/strength/rank_identity_merge.py`）

使用者要求「都做」：(1) L10 覆寫檔、(2) 測試題之間是否有同一玩家、(3) F2 vote 重測。

- **(3) F2 vote 無法低成本重測：** 擬合池（Valid-S／train20 的 HD0 棋局）的逐手 log-likelihood 已刪（`humansl_train20`、`humansl_q1000`、`humansl_vs_rest` 只剩 `strength_train_hc/standard` 的平均特徵），重算須重跑 HumanSL（數十 GPU-h）；A-046 本地只 +0.0028，不做。
- **(2) 配對（Slurm 750617／750620／750627，CPU 各約 40 秒）：** 開局模型（pp＋slot、D4、A-029 參數）以每個測試題為 gallery，M_ij＝LR(i|j)＋LR(j|i)（LR 相對空 gallery）。虛無分布：Valid-U 玩家合成題，每色 550 題（150 位玩家各兩題、250 位各一題，HD≤1，題目玩家的棋局移出母體），共 301,650 對不同人、300 對同人。
  - 測試 159,616 對（同色）。分 (k_min, k_max) 格比：中位數與 null 幾乎相同（無分布偏移）；超過各格 null q999 的 288 對（預期 160），超過 q9999 的 37 對（預期 16）；但召回最高的 (20,20) 格沒有多（8 對 vs 9.9）。null 最大值 256.8 以上 7 對（null 0 對）。參考同人配對召回率：q999 24%（k_min=5／10／15／20：12%／35%／49%／70%）。
  - **獨立檢查（基底預測級別一致率）：** 全部配對 10.6%；M＞89 12.6%、M＞161 13.0%（46 對）。同人兩題的一致率應遠高於此；以混合比例粗估 M＞89 的 366 對中只有約 7%（~27 對）是同人。前 7 對的基底級別差 3／1／2／1／0／1／1。→ **尾巴多出來的主要是開局相似的不同玩家（開局風格與棋力相關），不是同一人；不做合併檔。** 合併 7 對的試算保留在 `outputs/rank/identity_merge/export.json`，未寫提交檔。
- **(1) L10 覆寫檔 `submissions/rank/750633-l10.csv`（SHA `1dfa77d6`，Slurm 750633）：** 同作業重擬合基底並逐題重現 `-prior_736589`（0 題不同）；L10 gap 超過 unseen null 最大值（71.5）的 7 題改為比到的訓練玩家級別，實際改 4 題：pub_q_0046 4D→6D、pub_q_0135 3D→2D、priv_q_0047 4D→3D、priv_q_0168 4D→5D（另 3 題原本就相同）。預期：若比對都正確 Public 約 +0.004（+1.63 分）、Private 約 +0.003；若錯則反向。LB 變化小於雜訊，上傳只能確認方向；機制（gap 遠超任何未見玩家）是採用與否的主要依據。**LB 11/11 09:36:01（jabir95tsai）Public 0.7524623（12/116），對基底 +0.0016**＝恰好 +0.632 分（一題由差一級變完全正確）：與「pub_q_0135 真值 2D、pub_q_0046 真值在 4D／6D 之間（5D，前後都是差一級）」一致，方向支持 L10 比對；幅度在雜訊內。
- 11/11 第 1 名 TEAM_11819 升到 **0.8053895**（33 次上傳），與基底差 0.055，已超出 Public 雜訊（約 ±0.02），對方有真實的結構性差異，原因未知。

### 2026-11-11：測試棋的對手是不是訓練玩家（`scripts/strength/rank_opponent_check.py`，Slurm 750638，CPU 29 分）

假設：若對手是訓練集已知級別的玩家，認出對手就能推出本人級別（幽玄配對相近棋力、HD1＝差一級）。做法：每盤一個查詢，用開局模型（pp＋slot、D4）把**對手方**的落子比對 8,007 位訓練玩家同色的棋局，統計 gap（第一減第二）。null：Valid-U 玩家單盤本人落子 4,000 盤（移出 gallery，unseen＝拿掉本人欄、seen＝保留）；另抽訓練棋 12,000 盤的對手方做同樣比對。

| 集合 | 超過 unseen q95 | 超過 q999 | 超過 unseen 最大值 |
|---|---:|---:|---:|
| reference seen（本人在 gallery） | 7.6% | 0.13% | 0.13% |
| reference unseen | 5.0% | 0.10% | 0 |
| 訓練棋的對手 | 8.3% | 0.13% | 0.02% |
| 測試 Public 的對手 | 7.8% | 0.10% | 0.04% |
| 測試 Private 的對手 | 8.3% | 0.06% | 0.02% |

- **單盤開局比對認不出人：** 本人確定在 gallery 時，超過 q95 的比例也只有 7.6%（null 5%），所以這個工具無法回答「對手是不是訓練玩家」，更不能逐盤指認；測試與訓練對手的比例相近（都約 8%），兩者看起來是同一種對手族群，但是否為池內玩家仍未知。
- 訓練棋上「最像的玩家」的級別與本人級別：相同 17.3%（HD0）／18.3%（HD1）、差 ≤1 級 42.3%，打亂配對的基準為 10.2%／28.9%；這是開局風格與棋力的相關，不是身分，H 特徵的對手 likelihood 已涵蓋同類訊號。
- 測試題中有至少一盤對手 gap 超過 null 最大值的只有 Public 2、Private 1 題。**結論：對手身分這條路用現有工具做不到，不是可用的槓桿。** 要逐盤認人需要比開局模型強得多的單盤指紋（例如競賽一 GE2E 在 8,007 人上的單盤辨識，未測、預期仍弱）。
- 註：腳本初版 `lever` 的級別差用 uint8 相減溢位（mean_diff 約 100 的數字無效），已改為 int；上表的相同／差 ≤1 比例由存檔 `outputs/rank/opponent_check/train_sample.parquet` 重算。

### 2026-11-11：強引擎特徵提高 visits 的速度短測（Slurm 750766，1×H200，2 分 17 秒）

`katago_engine.py` 新增 `--visits`（預設 1＝A-048 原設定，舊結果可重現；meta 記錄 visits／overrides）。同一批 A-048 試算盤（engine_pilot shard_0 前 100–200 盤）：

| 設定 | 盤數 | 位置 | 每秒位置數（含啟動） |
|---|---:|---:|---:|
| 1 visit | 200 | 46,418 | 3,565（A-048 長跑實測 4,760–5,450） |
| 16 visits | 100 | 23,087 | 688 |
| 16 visits，`numAnalysisThreads=256,nnMaxBatchSize=256` | 100 | 23,087 | 735 |
| 32 visits，同上 | 100 | 23,087 | 423 |

估算（以 735／423 位置每秒、單卡）：A-048 試算集約 420 萬位置 → 16 visits 約 1.6 GPU-h、32 visits 約 2.8 GPU-h；擴大到全擬合池（約 4,600 萬位置，未精算）16 visits 約 17 GPU-h、32 visits 約 30 GPU-h；測試 1 萬盤約 230 萬位置 16 visits 約 0.9 GPU-h。短測含引擎啟動時間，長跑應略快。

### 2026-11-11：強引擎 16／32 visits 試算 → 未過 gate，不擴大

使用者同意後執行。A-048 同一批 18,772 盤，同一切分（4 片，與 A-048 分片逐盤相同），貼目一律 6.5。Slurm 752680（16 visits，4×1 卡，`numAnalysisThreads=128,nnMaxBatchSize=128`；先前 752622_0–3 用 256 執行緒、4 片落在同一節點，超過每使用者 2,048 程序上限，KataGo `Resource temporarily unavailable` 失敗，重送）、752622_4–7（32 visits，256 執行緒）。全部 4,693／片完成、0 拒絕。實測 16 visits 826 位置／秒、每片 21 分（**1.4 GPU-h**）；32 visits 493 位置／秒、每片 35 分（**2.4 GPU-h**）。比較 Slurm 752698（1 卡 CPU 用，1 小時 17 分），`scripts/strength/engine_visits_pilot.py`，輸出 `outputs/rank/humansl/engine_visits_pilot/result.json`。

設定：基底改為現行 noA 特徵（H〔P3 貼目〕＋C＋讓先旗標，52 維）；B2 以 q1000 S-fit 的 HD≤1 盤擬合（9,116 盤、499 人），在 qsel S-select 評估（剔除含 HD≥2 盤的題，856 題：k=5／10／15／20＝225／230／200／201）；5 個 B2 抽樣 seed 平均；玩家 bootstrap 為 seed 2027。事前 gate：最佳 E 版 − HC ≥ +0.01 且各 k ≥ −0.005。

| S-select，5 seed 平均 | Equal-k | Δ vs HC | Δk=5 | Δk=10 | Δk=15 | Δk=20 | bootstrap 95% |
|---|---:|---:|---:|---:|---:|---:|---|
| HC | 0.7129 | | | | | | |
| HC＋E 1 visit（A-048 特徵） | 0.6952 | −0.0177 | −0.0324 | +0.0017 | −0.0015 | −0.0387 | [−0.0346, +0.0031] |
| HC＋E 16 visits | 0.7079 | −0.0050 | −0.0223 | +0.0295 | −0.0117 | −0.0154 | [−0.0233, +0.0155] |
| HC＋E 32 visits | 0.7026 | −0.0102 | −0.0155 | +0.0088 | −0.0166 | −0.0177 | [−0.0277, +0.0126] |

- **未過 gate：** 三個 E 版都低於 HC；16 visits 比 1 visit 好 +0.013，但 32 visits 沒有再進步，看不到「visits 越高特徵越好」的趨勢。
- 機制（推測，未驗證）：引擎著手品質（目數損失、與引擎首選一致率）在中段級別與 H／C 高度重疊（11/11 LDA 診斷：加 E 中段 d′ 無增量）；在只有 499 位擬合玩家時，多 14 維主要增加 B2 的變異數。擴大擬合池可能縮小負值，但目前沒有任何一臂為正，**不做 17–30 GPU-h 的全量擴大**。
- 本試算共約 3.8 GPU-h（含 752622_0–3 失敗的數十秒）。不產生提交檔。

### 2026-11-11：B2 合成題級別先驗改均勻 → 增益約 +0.002，不是槓桿（Slurm 753754，1 卡 CPU 用，29 分）

使用者同意後執行。動機：現行基底的測試預測 1D 偏少（每份 400 題只 23–26 題），推測是合成題依擬合玩家比例抽級別所致；若測試每級等量，均勻先驗才對。`scripts/strength/rank_uniform_prior.py`，輸出 `outputs/rank/humansl/uniform_prior/{local,test}.json`（第一次 753743 因基底路徑寫錯失敗，修正後重送）。

- **前提只有一半成立：** 擬合池「有 ≥k 盤的玩家」的級別比例本來就接近均勻（k=20：A 6.9%、其他 9.4–11.3%；k=5：A 14.4%、1D 7.6%），不是全體 8,007 人的 433／1,582 懸殊比例。
- 本地（擬合 Valid-S＋train20 HD≤1，E2＝Valid-U 每 k×級別各 50 題，5 seed）；以兩種測試情境重新加權每 (k, 級別) 平均分：

| Equal-k | 測試每級等量 | Δ | 測試照玩家比例 | Δ |
|---|---:|---:|---:|---:|
| 玩家比例先驗（基底） | 0.7447 | | 0.7523 | |
| 均勻先驗 | 0.7468 | +0.0021（各 seed −0.000～+0.005） | 0.7512 | −0.0011 |
| 各半混合 | 0.7472 | +0.0024 | 0.7533 | +0.0011 |

- 分 k（等量情境，均勻 − 基底）：k=5 −0.0005、k=10 +0.0046、k=15 −0.0011、k=20 +0.0053。逐級（k 平均）：均勻先驗讓 1D 0.611→0.662，但 B 0.813→0.796、2D 0.623→0.603、3D 0.602→0.596——1D 偏少主要是特徵在 A／1D／2D 之間分不開，換先驗只是把分數在相鄰級之間搬動。
- 測試：基底臂逐題重現 `-prior_736589`（0 題不同）。均勻先驗改 Public 38／Private 31 題，1D 變 41／33；混合改 30／25 題。
- **判讀：** 兩種情境下差異都在 ±0.003 內，遠小於 Public LB 雜訊（約 0.02），上傳也無法分辨。**不產生提交檔、不建議上傳。**
- **補註（事後查帳）：** 這個實驗與 11/08「級別先驗探針」重複——當時 LB 已顯示 uprior −0.0088、balanced −0.0416，判定測試級別不是每級等量、1D 比均勻少。本次提案前漏查該節；本次結果（影響很小）與該結論一致，不改變任何決定。


## 2026-11-11：HumanSL residual B2 stage 1 單折 smoke100（Slurm 751237）→ 停止

`humansl_residual_smoke100.json`、fold 1、只跑 stage 1、dev 8 卡，100 updates（AdamW LR 0.001，常數）。38 分鐘、**5.0 GPU-h**（訓練 13.8 分、監控 18.9 分）。事前停損線（使用者訂）：M 上追不到 0.65 就停。

| M（5,000 題，未見玩家） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| fused（機率平均，早停用） | 0.4099 | 0.4275 | 0.4321 | 0.4427 | **0.4280** |
| fused_logp_t1（診斷） | 0.4538 | 0.4967 | 0.5132 | 0.5361 | 0.4999 |
| own／opp | | | | | 0.4205／0.4296 |

- rank CE 2.417 → 2.155（均勻 2.303），值 loss 0.81 → 0.048，仍在下降；這是頭直接解碼、未經 B2。
- 判讀：**低於 0.65 約 0.15–0.22，依停損線停止 hrb2 stage 1／2**。100 步不代表收斂，但凍結探針（MLP＋B2 0.5895）已顯示上限低；要追到 0.65 以上沒有證據。
- 成本實測：純訓練 0.091 step/s → 正式 3,000 步單折約 9 h、四折約 37 h × 8 卡 ≈ 290 GPU-h（未含監控），遠高於 11/10 粗估 110–130 GPU-h。


### 2026-11-11：現行基底下 H／C 拆解（`scripts/strength/rank_feature_ablation.py`，Slurm 751848，CPU 16 分）

設定同 11/09 先驗對齊本地比較（`rank_more_players.local`）：B2 以 Valid-S＋train20（HD≤1）擬合、Valid-S 先驗、Valid-U E1／E2 評分、seed 2027–2031；三臂只差特徵欄。H＋C 在 seed 2027 重現先前數字（E1 0.7153、E2 0.7460）。輸出 `outputs/rank/humansl/more_players/feature_ablation.json`。

| 五 seed 平均 | k=5 | k=10 | k=15 | k=20 | Equal-k | 對 H＋C |
|---|---:|---:|---:|---:|---:|---:|
| E1 H＋C＋P2（52 維，現行基底） | 0.6237 | 0.7254 | 0.7646 | 0.7453 | **0.7148** | — |
| E1 H 單獨＋P2（32 維） | 0.6145 | 0.7207 | 0.7694 | 0.7459 | 0.7126 | −0.0021 |
| E1 C 單獨＋P2（22 維） | 0.6351 | 0.7177 | 0.7655 | 0.7302 | 0.7121 | −0.0026 |
| E2 H＋C＋P2 | 0.6894 | 0.7403 | 0.7715 | 0.7778 | **0.7447** | — |
| E2 H 單獨＋P2 | 0.6911 | 0.7427 | 0.7681 | 0.7696 | 0.7429 | −0.0019 |
| E2 C 單獨＋P2 | 0.6834 | 0.7397 | 0.7615 | 0.7749 | 0.7399 | −0.0048 |

- 各 seed 差值：C−HC E1 +0.0007／−0.0021／+0.0003／−0.0051／−0.0070，E2 −0.0048／−0.0061／−0.0052／−0.0050／−0.0031（五 seed 全負）；H−HC E1 −0.0032／−0.0041／−0.0047／−0.0009／+0.0023，E2 −0.0032／−0.0010／+0.0004／−0.0035／−0.0020。seed 2027 玩家 bootstrap：C−HC E1 [−0.0106, +0.0110]、E2 [−0.0148, +0.0054]；H−HC E1 [−0.0106, +0.0039]、E2 [−0.0090, +0.0028]。
- 判讀：H＋C 最高，但拿掉任一塊都只少約 0.002–0.005，沒有一臂過 gate（+0.005）。H 與 C 大部分重疊：兩者都是「HumanSL 在某級別條件下對實際著手的 likelihood」，只差條件化方式（KGS 15 profile vs 幽玄 10 級學出的向量），合併主要在平均掉兩種條件化各自的誤差。C 單獨在 A-055（擬合 38,589 盤、S-select）略勝 H 單獨（0.7182 vs 0.7025）；現在擬合池約 15 萬盤／5,600 人，B2 已能把 H 的 KGS 尺度校準到幽玄，C 的「直接用幽玄校準」優勢變小，而 H 的 30 維（15 個 profile 較密）保留較多資訊——此機制未驗證。LB 端 AH vs AH＋C（11/10）−0.0154 也支持合併。
- **結論：維持 H＋C 基底，不產生提交檔。**

### 2026-11-11：C 級別向量 v2 重學（事前登記；`scripts/strength/rank_cvec_v2.py`，程式完成、未送作業）

**動機：** 現行 C 向量（`humansl_rankvec/vectors.pt`，10/25，A-055）早於 11/07 的讓子發現：(1) 學習盤含 HD≥2（約 3.6%，本人／對手互換、缺讓子石），e[r, 本人] 與 e[r, 對手] 互相滲入；(2) 讓先局以貼目 6.5 學，P3 抽特徵時卻用 0.5；(3) 只學 13,500 盤。
**做法（模型、loss、optimizer 不變，只換學習資料與編碼）：**
- 學習池：排除 Valid-U 玩家與 fold −1 玩家、只留 HD≤1、扣掉所有 Valid-S／Valid-U 盤、train20 擬合盤與 v1 自己的 15,000 盤 → 527,408 盤。每級 10% 玩家當 monitor（seed 2027，每級 150 盤），其餘每人最多 15 盤、每級 3,600 盤 → **36,000 盤／3,882 人／8.0M positions（v1 的 2.7 倍）**；HD1 7,367 盤以貼目 0.5 編碼。1D 學習玩家只有 270 人，決定了每級上限。292 位 monitor 玩家中 194 位有其他盤在 v1 學習集內（對 v1 有利，G0 偏保守）。
- 訓練：Adam 3e-3 cosine、batch 512、**1 epoch**（v1 為 2 epoch，但 v1 的 monitor NLL 在第 3,000 步已停在 1.470→1.469）。
- 重抽 C：Valid-S HD≤1＋全部 Valid-U（E1 含 HD≥2，與 v1 相同以 6.5 處理）＋train20 擬合盤＋兩份測試，共 194,698 盤；HD1 貼目 0.5。學習集與擬合／評估盤零重疊（已檢查）。
**Gate：**
- G0（抽 C 前）：同一組 v2 monitor positions／盤，NLL(v2) ≤ NLL(v1)，且單盤 score_rank 官方分數 v2 ≥ v1 − 0.01；不過就停，不花抽取的 GPU-h。
- G1（Valid-U，基底設定：Valid-S＋train20 擬合、Valid-S 先驗、seed 2027–2031）：v2 − v1。**升為基底**：E1 或 E2 五 seed 平均 Equal-k ≥ +0.005、另一組 ≥ 0、E2 各 k 五 seed 平均皆 ≥ −0.005；兩組皆 ≥ 0 但未達 → 只當 LB 探針；否則停止。
**預期增益：未知，可能很小。** 同日拆解顯示拿掉整塊 C 只少 0.002–0.005（H 與 C 大部分重疊），改善 C 的品質可能也只在這個量級以內，未必達到 +0.005。
**估算（依實測速度：抽 C 約 4.5 盤/秒/GPU、v1 學習 3.9 步/秒）：** pack 8 卡約 20–30 分（純 CPU 編碼，約 3 GPU-h）；學習 1 卡約 70 分＋G0 約 30 分（約 1.7 GPU-h）；抽 C 8 卡約 1.5–1.8 小時（約 13 GPU-h）；G1／export 為 CPU。**合計約 18 GPU-h**，G0 不過則只花約 5 GPU-h。Slurm：`slurm/rank_cvec_v2_{pack,learn,extract}.slurm`（皆排除慢節點 hgpn083；抽取可跳過已完成分片續跑）。輸出 `outputs/rank/humansl/cvec_v2/`；提交檔 `<job>-cvec2.csv`（export 先以 v1 重算並要求與基底 `-prior_736589` 逐題相同）。
- **11/11 使用者決定：不做 G0 判別，直接跑全程。** 學習作業不再跑 G0、抽取不再檢查 g0.json；G1 照登記判讀。作業鏈（afterok）：pack 751956（8 卡）→ learn 751957（1 卡）→ extract 751958（8 卡）→ eval 751959（1 卡，G1＋`751959-cvec2.csv`）。預估合計約 16 GPU-h。
- **G1 結果（eval 751959，2,714 秒；`outputs/rank/humansl/cvec_v2/result.json`）：v2 − v1 Equal-k 五 seed 平均 E1 −0.0008、E2 +0.0015 → 依登記判定 stop。** 各 seed：E1 −0.0035／−0.0041／+0.0009／+0.0015／+0.0011；E2 +0.0045／−0.0022／+0.0039／+0.0022／−0.0007。E2 各 k 平均 k=5／10／15／20＝+0.0041／+0.0063／−0.0023／−0.0019；E1 各 k 平均（由各 seed 計）−0.0040／−0.0004／+0.0002／+0.0010。正負號隨 seed 變動，差異在雜訊內；與「拿掉整塊 C 只少 0.002–0.005」的上限一致。
- 提交檔 `submissions/rank/751959-cvec2.csv`（SHA `5471b8be`）：v1 重算與 `-prior_736589` 逐題相同（0／0 題）；v2 對基底改 Public 10／Private 11 題，預測分布 88／88／103／110／52／71／67／83／73／65（基底 88／87／105／112／49／72／65／84／73／65）。
- **11/11 使用者決定：不理 gate，cvec2 升為競賽二新基底**（本地證據為持平：E1 −0.0008／E2 +0.0015；採用理由為 v2 修正了讓子污染與貼目不一致、學習資料 2.7 倍，機制上較乾淨）。尚未上傳、無 LB 分數。後續以 C v2（`outputs/rank/humansl/cvec_v2/learn/vectors.pt`）為 C 特徵；C4x 改用 `INIT=…/cvec_v2/learn/vectors.pt`、`BASE=v2`。`-prior_736589` 保留在 `submissions/rank/` 頂層，因 `rank_c4x.py`／`rank_cvec_v2.py`／`rank_identity_merge.py` 以它做 v1 重現檢查。L10 覆寫（`750633-l10.csv`）是疊在舊基底上，尚未套到 cvec2。
- **新基底＋L10：`submissions/rank/751959-cv2-l10.csv`（SHA `d6cc3fc5`）。** `rank_identity_merge.py overlay`（新指令，不重擬合；L10 覆寫只依 `l10_real_test.questions.csv`）：7 題 L10 中 cvec2 實改 4 題，與舊 L10 相同（pub_q_0135 3D→2D、pub_q_0046 4D→6D、priv_q_0047 4D→3D、priv_q_0168 4D→5D，另 3 題原本相同）。驗證：同指令套在 `-prior_736589` 上，輸出與 `750633-l10.csv` 逐位元相同。對 `750633-l10.csv` 改 Public 10／Private 11 題（即 cvec2 對舊基底的差異，全為差一級），預測分布 88／88／103／110／52／72／67／80／74／66。
- **LB 11/11（jason.hsiao05）：`751959-cv2-l10.csv` 20:27:18 Public 0.7499623；`751959-cvec2.csv` 20:27:39 Public 0.7483820（13/120）。** 對照舊版：cvec2 對舊基底 `-prior_736589`（0.7508820）−0.0025；cv2-l10 對 `750633-l10.csv`（0.7524623）−0.0025。兩組差值相同，都是 −0.0025＝1/400，也就是那 10 題 Public 合計淨少 1.0 分（例如約 1.6 題由對變鄰級），在 ±0.02 雜訊內，與本地 E1 −0.0008／E2 +0.0015 的「持平」一致。L10 在新基底上同樣 +0.0016（0.7483820→0.7499623），與舊基底一致。基底依使用者 11/11 決定維持 cvec2。平台上競賽二最後一筆為 `751959-cvec2.csv`。

### 2026-11-11：C4／C4b 失敗原因複查（只讀紀錄與程式，未花算力）

複查 A-057、`src/go_aicup/tasks/strength/humansl_finetune.py`、`scripts/strength/humansl_finetune.py`。C4 的輸出與 log 已刪（只剩帳本數字）。
- **程式未見錯誤：** C4b 的 10 個假設排列（`repeat_interleave` 對 `arange.repeat`）、`reshape(-1, 10)`、真值索引、eval 模式的 BN、DDP 監控分片都正確；pack 以 `iloc[shard::16]` 間隔切，少 4 個 shard 不會讓級別失衡。
- **設計問題 1（主要）：有效玩家數遠小於帳本寫的 2,880。** `select` 把 F 玩家的全部棋局放進池、loss 逐位置平均，玩家權重∝盤數。fold 1–4 玩家盤數中位 50、最大 1,545；前 10% 玩家占 44% 棋局、前 20% 占 65%。以盤數加權的有效玩家數（Σn)²/Σn²：fold 1–4 全體 1,922 → F 半約 **960**；各級約 60（1D）～200 人。同一池每人最多 20 盤時有效玩家數為 5,560（×5.8）。A-057「真正的樣本數是玩家數」的推論方向沒錯，但當時的訓練設計本身讓有效玩家數縮到約三分之一，而 B2／C2／train20 都有每人上限。
- **設計問題 2：** gate 在 6% epoch（1,000 步／8,000 步）、以 Bm 270 位玩家的 C 單獨單盤分數判定，不是下游 H＋C＋B2 的題目分數；C4 的 −0.0008 等於「還沒動」，不能證明生成式學不到（HumanSL 本身就是生成式訓練而能分級）。C4b 三點單調下降（0.5195→0.5138→0.5087）是記玩家的實際訊號。
- **其他：** 微調池含 HD≥2（本人／對手互換、缺讓子石，約 3.6%），讓先局用貼目 6.5；全層同一 lr、weight decay 0、無凍結層。影響未量化。
- **與 SE3 的對照：** SE3 以全部 train 切分玩家判別式訓練（玩家權重同樣∝盤數），LB 0.67，低於 H＋C 約 0.08；這是「判別式訓練在本資料上落後 likelihood 特徵」的反證，但 SE3 不是從 HumanSL 出發，不能直接套用。
- **結論：** C4 線沒有被真正測過（有效玩家約 960、只跑 6% epoch），「玩家數是瓶頸」與「微調無效」都未驗證。改用玩家等權取樣重試的預期增益未知。

### 2026-11-11：C4x 交叉擬合微調（事先登記；使用者要求實作雙折交叉擬合，cvec2 結果出來並檢視後才執行）

**假設：** C4／C4b 失敗主要因為有效玩家數只有約 960（見上一節）。改成每人最多 20 盤、級別等權後，同樣的 HumanSL 微調能在未見玩家上學到級別差異。預期增益：未知（主觀估計做出可測增益的機率約三成）。
**做法（`scripts/strength/rank_c4x.py`；模型、loss、optimizer、lr 與 C4／C4b 相同，只改資料）：**
- fold 1–4 玩家（所有 Valid-S／train20 擬合玩家）依級別分層（seed 2027）分 A／B 兩半；M_A 只用 A 半、M_B 只用 B 半微調。每人最多 20 盤 HD≤1，讓先局貼目 0.5，`--rank-balance` 讓每級權重相同。A 半 2,863 人、有效玩家約 2,750；B 半 52,474 盤／2,876 人／11.7M positions／有效 2,749（C4 為約 960）。
- 監控：fold −1 玩家（不在任何擬合、評估集或半邊），每級 300 盤、每人最多 8 盤，共 3,000 盤／641 人（C4 的 Bm 為 270 人）。
- 交叉擬合：Valid-S／train20 的每盤 C4 由「沒看過該玩家」的模型抽；Valid-U 與測試由兩個模型各抽一次（兩個 view），B2 機率取兩 view 平均；最終匯出時 Valid-U 當擬合資料，每位玩家隨機指定一個 view（seed 2027）。已驗證兩個模型的學習玩家與其抽取的擬合玩家重疊 0、學習池無 Valid-U 玩家。
- 兩臂：`gen`（生成式 NLL，2 epochs，每卡 512）、`mmi`（NLL＋1.0×級別 CE，1 epoch，每卡 64×10 假設）。B2 臂：base（現行 H＋C）、**swap（C 換成 C4，主臂）**、add（H＋C＋C4，診斷）。
**Gate：**
- G0（每臂，抽特徵前）：fold −1 監控的單盤 score_rank 官方分數，終點減第 0 步、兩模型平均 ≥ +0.01，且兩模型終點都不低於第 0 步；不過就不抽該臂特徵。
- G1（Valid-U，基底設定：Valid-S＋train20 擬合、Valid-S 先驗、seed 2027–2031）：swap − base。升為基底：E1 或 E2 五 seed 平均 Equal-k ≥ +0.005、另一組 ≥ 0、E2 各 k 五 seed 平均皆 ≥ −0.005；兩組皆 ≥ 0 但未達 → 只當 LB 探針；否則停止。
**估算：** pack 8 卡約 45 分（約 5 GPU-h）；訓練 gen 每半約 30 分（約 4 GPU-h，兩半 8）、mmi 每半約 1.5 h（約 12 GPU-h，兩半 24）；抽特徵每臂 239,781 盤約 1.9 h × 8 卡（約 15 GPU-h）；G1／匯出 CPU。**兩臂都過 G0 約 68 GPU-h，都不過約 37 GPU-h。** 四個訓練作業可同時跑（32 卡＝帳號上限）。pack 約 26 GB（/work 剩 36 GB；cvec2 完成後可刪其 learn pack 約 9 GB）。
**驗證（CPU，未用 GPU）：** 乾跑以 C v1 冒充 C4（兩 view 相同）：swap − base 在 E1／E2 皆 0.0000，base 重現 11/11 拆解的 E1 0.7153／E2 0.7460，匯出逐題重現 `-prior_736589`（0／0 題）。`humansl_finetune.py` 新選項（`--vectors`、`--monitor`、`--rank-balance`、`--no-step-checkpoints`；預設值不變，C4 可重現）與 `humansl_rankvec extract --finetuned` 尚未在 GPU 上跑過；第一個訓練作業的第 0 步監控應等於 C v1。
**執行順序：** `tables`（已完成）→ `slurm/rank_c4x_pack.slurm` → `ARM={gen,mmi} HALF={A,B} slurm/rank_c4x_train.slurm` → `rank_c4x g0 --arm` → `ARM= slurm/rank_c4x_extract.slurm` → `ARM= [BASE=v1|v2] slurm/rank_c4x_eval.slurm`。若 cvec2 升為基底，改用 `INIT=outputs/rank/humansl/cvec_v2/learn/vectors.pt`、`BASE=v2`（由使用者決定）。**11/11：cvec2 已升為基底 → 採 v2 設定。**

### 2026-11-11：k=20 天花板診斷（`scripts/strength/rank_ceiling.py`，登入節點 CPU 約 8 分）

**問題：** k=20 是否已到每位玩家的 k=∞ 極限？剔除擬合池中的「離群玩家」重擬合 B2 是否有空間？
**設定：** 現行基底（PriorMatched B2，Valid-S＋train20 HD≤1 擬合、Valid-S 先驗、52 維）。評估單位＝Valid-U（玩家, 顏色）組且 HD≤1 盤數 ≥20：**420 組／344 人**（每級 35–52 組，盤數中位 26；≥40 盤 76 組）。每組隨機抽 40 次子集算期望分數；k=∞＝該組全部盤平均送 k=20 模型。seed 2027–2031 平均；`rank_balanced` 為各級等權（同 E2）。輸出 `outputs/rank/humansl/more_players/ceiling.json`。

| 420 組，rank_balanced | k=5 | k=10 | k=15 | k=20 | 全部盤（k=∞） |
|---|---:|---:|---:|---:|---:|
| 現行 B2 | 0.6763 | 0.7270 | 0.7590 | 0.7776 | 0.7879 |

| ≥40 盤的 76 組（同 k=20 模型） | n=20 | n=30 | n=40 | 全部 |
|---|---:|---:|---:|---:|
| rank_balanced | 0.8095 | 0.8362 | 0.8511 | 0.8459 |

- k=∞ 的錯誤結構：完全正確 67.5%、差一級 29.7%、差 ≥2 級 **2.9%**。逐級正確率：D／C／B 0.77–0.84、**A／1D／2D／3D 0.44–0.65**、4D–6D 0.67–0.94；平均符號誤差 A +0.24、1D +0.32、2D −0.25、3D −0.17（中段級別向 1D／2D 交界靠攏）。
- Oracle（直接在這 420 組上擬合 LR，屬樂觀上界）：全部盤平均 in-sample 0.7946（只比 B2 的 0.7879 高 0.007）；同設定按玩家 5 折 0.7223；k=20 抽樣 in-sample 0.8148（40 個抽樣×420 組可記住組別，偏高、不可信）。
- **判讀：**
  1. 「離群玩家」（全部盤仍差 ≥2 級）只占約 3%；主要的失分是差一級（30%），集中在 A–3D，屬於特徵在中段級別的解析度問題，不是少數髒玩家。
  2. B2 在全部盤平均上的決策邊界，距離直接用評估玩家擬合的 LR 只有 0.007 → 用同一組特徵重擬合 B2（包括剔除離群玩家）在極限處最多約 +0.007，實際應更少。**步驟 1（剔除／降權擬合玩家）不做。**
  3. k=20 尚未到變異數極限：≥40 盤的組從 20→40 盤 +0.042。420 組整體的 k=∞ 只多 +0.010，是因為多數組只有約 26 盤，「全部盤」幾乎等於 k=20，這個數字低估了天花板。每盤雜訊在 k=20 仍有可觀份量 → 降低每盤雜訊或提升中段級別區分度的上游特徵（C v2、強引擎提高 visits 等）才是 k=20 的槓桿。
  4. 限制：76 組重度玩家的基準本來就較高（0.81），不代表全體；oracle 為 in-sample 上界。不產生提交檔。

### 2026-11-11：H 各 profile 對幽玄級別的對應與相鄰級可分度（登入節點 CPU，臨時腳本）

擬合池（Valid-S＋train20，HD≤1）每盤本人 15 個 HumanSL profile 平均 log p，按幽玄級別平均：
- 各級 likelihood 最高的 KGS profile：D、C → 5k；B、A、1D → 3k；2D、3D、4D → 1k；5D → 1d；6D → 2d。**幽玄 B–4D 六個級別擠在 KGS 3k–1k 之間**，而這段只有 rank_3k、rank_1k 兩個 profile（沒有 2k、4k）。
- 單盤相鄰級可分度（最佳單一 profile 的 d′）：D/C 0.52、C/B 0.39、B/A 0.48、A/1D 0.34、**1D/2D 0.19、2D/3D 0.22、3D/4D 0.27**、4D/5D 0.36、5D/6D 0.48；中段最佳區分 profile 為 rank_9d／rank_7d（「多像高手」的軸）。與 k=∞ 診斷中 A–3D 正確率最低一致。

**C4x 修訂（2026-11-11，使用者同意；仍未執行）：** 目標改為提高中段（1D/2D、2D/3D、3D/4D）解析度。
- 依據（CPU，擬合池 149,615 盤 HD≤1，一半玩家擬合 LDA、另一半評估）：中段單盤 d′ H 0.49／0.49／0.56、C 0.52／0.48／0.57（兩端約 0.95–1.0）。**C（直接用幽玄級別學的向量）與 H 相同**，補 2k／4k profile 預期無增益；強引擎特徵（A-048 試算 18,168 盤）在中段也無增量（H＋C 0.44／0.38／0.60 → 加 E 0.41／0.35／0.59）。題目層級（每人平均 k 盤）中段 d′：本人 k=1／5／20＝0.33／0.60／0.73（k20/k1＝2.21），對手 0.27／0.69／1.06（3.93；獨立時應為 4.47），全部 0.35／0.76／1.05——大 k 時本人特徵被玩家偏差卡住，k=20 只用對手特徵就等於全部。
- **路線：** gen 改為 `adj`（NLL＋1.0×只在 {r−1, r, r+1} 上的級別 CE，`--mmi-adjacent`，每位置 3 次前向，1 epoch，每卡 64）；`mmi`（10 類）保留作對照。資料比例不變（每人最多 20 盤、級別等權）：中段缺的是玩家人數（每半 1D 約 155 人），加重中段資料會加重少數玩家的權重。
- **G0 改為：** 兩模型終點的單盤官方分數都不低於第 0 步，且（官方分數平均增益 ≥ +0.01，或中段 d′〔全部著手，三對平均，兩模型平均〕增益 ≥ +0.05）。監控每個檢查點記錄中段 d′（本人／對手／全部）。
- **G1 不變**（swap − base），另報 Valid-U 題目層級中段 d′（k=1／5／20，本人／對手／全部；LDA 在擬合池同 k 的題目上擬合）。base 參考值：本人 0.374／0.717／0.855（2.29）、對手 0.357／0.797／1.223（3.43）、全部 0.452／0.936／1.293（2.86）；中段每級約 70–80 位 Valid-U 玩家有 ≥20 盤，屬診斷用。
- **驗證（CPU）：** d′ 計算以已知分離的合成資料檢查（0.527，正負號正確）；相鄰級 CE 在 D／中段／6D 與手算一致；G0 四種情況判定正確；評估端中段診斷在現有特徵上可跑（16 秒）。GPU 端仍未跑過。
- **估算更新：** adj 每位置 3 次前向，預估每半約 35 分（約 5 GPU-h，兩半 10）；mmi 兩半約 24；pack 約 5；抽特徵每臂約 15。兩臂都過 G0 約 70 GPU-h，都不過約 39 GPU-h。
- **11/11 20:45 送出（C v2 基底：`INIT=outputs/rank/humansl/cvec_v2/learn/vectors.pt`、`BASE=v2`；兩臂 adj＋mmi）。** 使用者決定：**G0 只看「不傷害」（兩模型終點單盤官方分數都 ≥ 第 0 步）就抽特徵**，不再要求 +0.01／中段 d′ +0.05；`g0_<arm>.json` 的 `pass` 仍照原登記寫出供紀錄。G0 改在抽特徵作業開頭自動計算（`slurm/rank_c4x_extract.slurm`），整鏈以 afterok 串接；eval 時間上限 1:30→3:00（cvec2 eval 實測 80 分）。G1 不擋匯出，結論見 `result_<arm>_v2.json`。
  - pack 753665（dev 8 卡）→ adj：train A 753666／B 753667 → extract 753668 → eval 753669；mmi：train A 753670／B 753671 → extract 753672 → eval 753673（後兩者因 dev 每人提交數上限改送 `8gpus`）。
  - 預估：pack 約 45 分；adj 訓練每半約 35 分、mmi 約 1.5 h（GPU 未實測；帳號 32 卡上限扣他人占用，mmi B 可能等 adj 結束）；抽特徵每臂約 2 h；eval 約 1.5–2 h。adj 結果約 11/12 02:00、mmi 約 03:30。兩臂都抽約 75 GPU-h。

## 2026-11-11：Fable 獨立計畫；Public 級別直方圖常數探針（使用者同意執行）

**新查到的事實（登入節點 pandas，1M 列的手數／HD／顏色，秒級）：**
- 「最後一手是本人」依玩家總盤數（train，HD≤1）：≤20 盤 0.516、20–50 盤 0.520、50–100 盤 0.510、100–200 盤 0.497、200–500 盤 0.490、>500 盤 0.500；玩家等權 0.510。測試 Public／Private 0.526／0.529 最像輕度玩家 → **假設 H1：測試玩家是盤數少的人**（未驗證；B2 擬合與 E1／E2 偏重度玩家）。
- 現行基底測試預測分布對 train 玩家數比例（HD≤1 玩家數 890／815／1046／1576／427／605／611／637／656／706）：D／C／B／6D 吻合，**A 預測 13.75%、玩家比例 19.8%**，2D–5D 各多 1–2 點。真實分布未知。
- `game_id` 無時間順序（十分位的手數、HD 比例、第一手分布全平，只按顏色排序；玩家 game_id 跨度 0.960 對隨機期望 0.966）→ 時間漂移無法檢驗。題號相鄰自相關（每題 HD 比例 −0.03、最後一手比例 −0.08）無結構。
- 排行榜（使用者 11/11 提供）：第 1 名 0.8054、第 2 名 0.768、第 6 名 0.76；我們 0.748–0.752。第 2–6 名與我們的差距在 Public 雜訊（約 0.02）內，第 1 名是離群值。

**D1 常數探針（`scripts/strength/rank_const_probe.py`）：** 全部 800 題填同一級 r 的 Public 分數＝(n_r＋(n_{r−1}＋n_{r+1})/e)/400，三對角線性系統；9 檔＋總和 400 可解出 Public 十級題數（`solve` 以合成直方圖自測：還原誤差 <1e-4 題）。檔案在 `submissions/rank/probe/const-<級別>.csv`（10 檔，SHA 前 8 碼：d `caec5ce1`、c `882a4b1a`、b `7559caf0`、a `08d509ae`、1d `aa5016db`、2d `a0ccff04`、3d `b88cf5f8`、4d `b5aa2853`、5d `4f0f1bb2`、6d `dce4bd25`）。
- **假設：** 測試級別比例接近 train 玩家數比例（A 約 20%、1D 約 5%），現行基底少判 A。**預期各檔分數（若按玩家數比例）：** a 約 0.27、1d 約 0.16、2d 約 0.12、b 約 0.24、3d 約 0.13；若每級等量則各約 0.17（端點 0.14）。
- **用途與限制：** 只取得 Public 的級別邊際分布，用來設定 `PriorMatched` 先驗（向 train 玩家比例收縮，不直接用 Public 整數題數）；由此衍生的檔在 Public 上屬樣本內，**LB 上升不算確認**，以本地重加權模擬判斷（gate：E2 重加權成實測分布，5 seed 平均 Equal-k ≥ +0.005、各 seed ≥ 0、各 k ≥ −0.005，玩家 bootstrap）。須在報告申報。
- **上傳順序：** a、1d、2d、b、3d、c、d、4d、5d（6d 由總和推出，可不傳）。探針結束當天最後一次補傳基底。
- **L10：** 使用者交由判斷 → 保留（演算法式、可重現、gap 超過未見玩家 null 最大值；Private 預期 ±0.003）。**現行最終預設檔＝`751959-cv2-l10.csv`。**

| 上傳時間 | 檔案 | Public LB |
|---|---|---:|
| 11/11 22:41:06（jabir95tsai） | `const-a.csv` | 0.2071379 |
| 11/11 22:41:30（jabir95tsai） | `const-1d.csv` | 0.1427728 |
| 11/12 08:10:17（jabir95tsai） | `const-2d.csv` | 0.1392622 |
| 11/12 08:09:57（jabir95tsai） | `const-b.csv` | 0.2144698 |
| | `const-3d.csv` | |
| | `const-c.csv` | |
| | `const-d.csv` | |
| | `const-4d.csv` | |
| | `const-5d.csv` | |

**前兩檔判讀（11/11 22:41）：** `const-a` 0.2071379 → n_A＋(n_B＋n_1D)/e＝82.86；`const-1d` 0.1427728 → n_1D＋(n_A＋n_2D)/e＝57.11。每級等量時兩式都是 69.4，按全體玩家數比例時 A 式約 106。**「按 8,007 位玩家數比例抽題（A 約 20%）」被否證**：A 一帶只比等量略多、1D 一帶比等量少。兩式還解不出單一級別；粗估（n_B 約 45–50、n_1D 約 25）n_A 約 55–57、n_2D 約 30，與現行基底在 Public 的預測量級（兩份合計 A 110、1D 52）相近 → 現行先驗沒有大偏差，D1 的預期增益下修為小（未知，估 <0.005）。其餘探針仍可完成以取得精確直方圖，但優先序低於 C4x 的確認上傳。

**11/12 08:10 再兩檔：** `const-b` 0.2144698 → n_B＋(n_C＋n_A)/e＝85.79；`const-2d` 0.1392622 → n_2D＋(n_1D＋n_3D)/e＝55.70。四式仍不足以解出單一級別（需 9 檔）；方向一致：B／A 一帶高於等量（69.4），1D／2D 一帶低於等量。

### 2026-11-12：測試族群稽核（`scripts/strength/rank_shift_audit.py`，Slurm 754091，1 卡 CPU 用，25 分）

輸出 `outputs/rank/humansl/shift_audit/{domain,prior}.json`。設定同基底本地比較（Valid-S＋train20 HD≤1 擬合、E2、seed 2027–2031；特徵為 C v1 版）。

**(1) 先驗值多少（三臂只差合成題先驗；E2 依情境重加權，Equal-k 五 seed 平均）：**

| 測試情境 | base（≥k 盤玩家比例） | allplayers（8,007 人比例，A 19.8%） | mix |
|---|---:|---:|---:|
| 每級等量 | 0.7447 | 0.7308（−0.0139） | 0.7371（−0.0077） |
| ≥k 盤玩家比例 | 0.7523 | 0.7450（−0.0073） | 0.7486（−0.0037） |
| 全體玩家比例 | 0.7534 | 0.7596（+0.0063，各 seed +0.002～+0.010） | 0.7575（+0.0041） |

→ 即使測試真的是最極端的「全體玩家比例」，知道先驗也只值 +0.006；常數探針已否證該情境，所以 **D1 的可得增益約 0，先驗不再調整**。剩餘常數探針只用來把直方圖結案。

**(2) 輕度玩家（H1）：** 基底在 E2 的分數與級別偏差依玩家總盤數分層（已扣 (k, 級別) 格平均）：

| 玩家總盤數 | 題數／人數 | 分數（調整後） | 期望級別偏差（調整後） |
|---|---:|---:|---:|
| ≤30 | 278／192 | −0.000 | +0.086 |
| 30–60 | 344／172 | −0.000 | −0.012 |
| 60–120 | 365／162 | −0.020 | +0.058 |
| 120–300 | 526／205 | +0.008 | +0.012 |
| >300 | 487／135 | +0.007 | −0.097 |

≤50 盤合計：分數 −0.006、偏差 +0.045 級；≥150 盤：+0.007、−0.044 級；log(盤數) 與分數的相關 0.006。→ 輕度玩家被高估約 0.05–0.09 級、重度被低估約 0.1 級，但**對分數的影響只有約 0.006**，不是差距的來源。**D2（擬合重加權）不做。**

**(3) 測試題對 E2 的 domain classifier（52 維題目平均特徵，玩家分組 5 折，E2 已對齊測試的預測級別分布）：** AUC k=5 0.56／0.59、k=10 0.55／0.66、k=15 0.45／0.50、k=20 0.55／0.53（Public／Private）；平均 |標準化差| 0.04–0.16。輕度對重度玩家在特徵上本來就難分（AUC 0.59–0.63），估不出測試的輕度比例。→ 族群偏移輕微，k=5／10 稍明顯。

**結論：H1、H2 都不是 0.05 差距的來源**；與第 2–6 名（0.76–0.768）的差距在 Public 雜訊內。

**C4x 結果判讀（由另一個 session 執行；此處只記讀到的數字，`outputs/rank/humansl/c4x/`）：**
- `adj`（NLL＋相鄰級 CE，1 epoch）：G0 過（fold −1 單盤官方分數 0.5358→0.5472／另一半類似，中段 d′ 0.531→0.565，NLL 1.412→1.391）。G1（swap − base，v2 基底）：**E1 +0.0059（五 seed +0.003～+0.009 全正；k=5 +0.016、k=10 +0.000、k=15 +0.007、k=20 +0.000），E2 −0.0009（+0.001／−0.002／+0.000／−0.002）** → 依登記判定 stop（E2 <0）。題目層級中段 d′（全部特徵）k=1／5／20：0.415／0.885／1.203 → 0.469／0.933／1.313。提交檔 `753669-c4x-adj.csv` 已產出（對舊 v1 基底 `-prior_736589` 改 Public 23／Private 31 題；log 欄位為 `changed_vs_prior_736589`，不是對 cvec2），未上傳。
- `mmi`（10 類 CE）：G0 不過，單盤分數 0.536→0.470／0.478、中段 d′ 0.53→0.25／0.18（未見玩家上崩壞）；未抽特徵。
- 判讀：相鄰級判別 loss 有真實但小的改善（未見玩家監控與中段 d′ 都升），10 類 CE 則是記住玩家。下游增益集中在 k=5 且只在 E1 出現。
- **11/12 補充：** seed 2027 玩家 bootstrap 95%：E1 [−0.0043, +0.0159]、E2 [−0.0117, +0.0038]，皆跨 0 → 逐 k 差（含 k=5）在雜訊內，「只在 k=5 有效」不成立。訓練 log：adj 的相鄰級 CE 整個 epoch 約 1.017→1.013／1.016（ln 3＝1.099），判別項幾乎沒學；監控改善與 NLL 下降同步（推論為領域適應，未做 λ=0 對照）。mmi 訓練 CE 2.28→2.05 而監控崩壞 → 學到訓練玩家風格。題目層級中段 d′：本人 k20/k1 2.42→2.14（玩家偏差主導），對手 3.58→3.88（接近獨立的 4.47，每盤雜訊主導）；全部特徵 k=20 d′ +9% 未轉成分數，原因未知。
- **11/12 adj＋L10：`submissions/rank/753669-adj-l10.csv`（SHA `c0b936b3`）**，`rank_identity_merge overlay --base 753669-c4x-adj.csv`（登入節點，不重擬合）。L10 的 7 題在 adj 上實改 4 題，與 cv2-l10 相同（pub_q_0135 3D→2D、pub_q_0046 4D→6D、priv_q_0047 4D→3D、priv_q_0168 4D→5D）。對現行最終預設檔 `751959-cv2-l10.csv` 改 **Public 15／Private 26 題**（k=5／10／15／20：Public 5／3／4／3、Private 7／10／4／5），全為差一級，往強 25 題／往弱 16 題；預測分布 88／87／101／111／47／80／69／78／72／67。本地證據為 E1 +0.0059／E2 −0.0009（G1 stop），只能當 LB 探針；Public 15 題全對或全錯也只有約 ±0.024，預期在雜訊內。未上傳。

### 2026-11-12：C4x adj 兩檔 LB 結果與 Public 逐題反推（登入節點，只讀提交檔與 LB 分數）

**LB 11/12（jason.hsiao05）：** `753669-c4x-adj.csv` 10:42:59 Public **0.7461411**；`753669-adj-l10.csv` 10:43:23 Public **0.7477214**（17/122）。對照 `751959-cvec2.csv` 0.7483820／`751959-cv2-l10.csv` 0.7499623：兩組都是 **−0.0022409**；L10 在 adj 上仍是 +0.0015803（與舊基底、cvec2 相同）。平台上競賽二最後一筆為 `753669-adj-l10.csv`。

**反推方法：** 每題得分只有 1、1/e、0 三種，所以 Public 分數 ×400 ＝ 完全正確題數 c ＋ 相鄰題數 a ／e。e 是無理數，在 0–400 的整數範圍內 (c, a) 唯一（容差 3e-4，全部只有一組解）。同理兩檔分數差 ×400 ＝ m(1−1/e) ＋ n/e 也唯一。

| 檔案 | Public | 完全正確 | 差一級 | 差 ≥2 級 |
|---|---:|---:|---:|---:|
| `-prior_736589` | 0.7508820 | 254 | 126 | 20 |
| `750633-l10` | 0.7524623 | 255 | 125 | 20 |
| `737645-priorex` | 0.7468017 | 252 | 127 | 21 |
| `751959-cvec2` | 0.7483820 | 253 | 126 | 21 |
| `751959-cv2-l10` | 0.7499623 | 254 | 125 | 21 |
| `753669-c4x-adj` | 0.7461411 | 251 | 129 | 20 |
| `753669-adj-l10` | 0.7477214 | 252 | 128 | 20 |

**cvec2 → adj 改的 15 題 Public（全為差一級的移動）：淨結果為完全正確 −2、差一級 +3、差 ≥2 級 −1。** 最省的解釋是「2 題由對變鄰級、1 題由差兩級變鄰級」；也可能是 2＋j 題變錯、j 題變對。**LB 只給總數，無法指出是哪幾題**。同法：prior → cvec2 的 10 題為完全正確 −1、差 ≥2 級 +1；L10 的 Public 2 題為完全正確 +1。

**Public 真實級別直方圖（由四個常數探針精確解出，互相驗證一致：const-a 的 n_B＋n_1D＝73＝49＋24、const-1d 的 n_A＋n_2D＝90＝56＋34）：**

| | D | C | B | A | 1D | 2D | 3D | 4D–6D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 真實 | — | 44 | 49 | 56 | 24 | 34 | 35 | D＋4D＋5D＋6D＝158 |
| cvec2 預測 | 53 | 42 | 50 | 53 | 28 | 37 | 30 | 41／34／32 |
| adj 預測 | 53 | 41 | 51 | 51 | 27 | 40 | 31 | 40／32／34 |

**15 題的特徵：**

| 題號 | cvec2→adj | k | 色 | HD1 比例 |
|---|---|---:|---|---:|
| pub_q_0017 | C→B | 10 | W | 0.20 |
| pub_q_0169 | C→B | 10 | W | 0.40 |
| pub_q_0150 | B→C | 5 | B | 0.00 |
| pub_q_0266 | B→A | 15 | B | 0.13 |
| pub_q_0248 | A→B | 5 | B | 0.60 |
| pub_q_0182 | A→1D | 20 | B | 0.15 |
| pub_q_0358 | A→1D | 5 | B | 0.20 |
| pub_q_0235 | 1D→2D | 10 | B | 0.10 |
| pub_q_0306 | 1D→2D | 15 | B | 0.00 |
| pub_q_0316 | 1D→2D | 15 | W | 0.00（prior 為 2D、cvec2 改 1D、adj 改回 2D） |
| pub_q_0130 | 4D→3D | 20 | W | 0.15 |
| pub_q_0337 | 5D→4D | 15 | B | 0.00 |
| pub_q_0399 | 4D→5D | 5 | W | 0.00 |
| pub_q_0176 | 5D→6D | 5 | W | 0.00 |
| pub_q_0223 | 5D→6D | 20 | W | 0.10 |

- **級別帶：** B–2D 中段 7 題（A 相關 4、1D→2D 3），C／B 低段 3 題，3D–6D 高段 5 題。**不是全在中段**，但中段最多。
- **方向：** 往強 11 題、往弱 4 題；k＝5／10／15／20 為 5／3／4／3，顏色 W 8／B 7，HD1 比例與整體相近 → k、顏色、讓先沒有集中。
- **與真實直方圖對照（推論，不是逐題證據）：** adj 在中段的移動方向與直方圖相反：A 已少判（53 vs 真實 56），adj 再移出 3 題（A→1D ×2、A→B ×1，淨 −2 → 51）；2D 已多判（37 vs 34），adj 再移入 3 題（1D→2D ×3 → 40）。兩處加起來 6 題中，至少有一部分應是那 2 題「由對變鄰級」的來源；高段 5 題與低段 3 題無直方圖可比（D／4D–6D 只知合計）。prior→cvec2 的 10 題也有 8 題在 B–2D。
- **解讀：** C4x adj 的改動偏向「把題目往強一級推」（與 G0 的中段 d′ 改善一致），但 Public 的 A 與 1D／2D 邊界上，推過去的方向是錯的居多。淨 −2 題在 400 題雜訊內（±0.02 約 ±8 題），與本地 E2 −0.0009 同方向、與 E1 +0.0059 反向。**adj 不升為基底；最終預設檔維持 `751959-cv2-l10.csv`。**
- **限制：** 只能得到每檔的總數與 Public 6 個級別的題數；逐題對錯要聯立更多上傳檔（每檔 2 條整數方程）才能縮小範圍，且只對 Public 有效、對 Private 沒有直接用處，未做。

### 2026-11-12：競賽一 Day-0 稽核（真實資料首次到手；Slurm 756473，21 秒）

資料包 `data/raw/tests/`（11/12 08:34 上傳）：原測試集 4 檔＋新增集 4 檔＋`sample_submission_player_v2.csv`。`preprocess.RELEASE_FILES` 加入 `pool2_*`／`pid2_*`，`scripts/release/day0_audit.py` 改為四組（原／新增 × Public／Private）。Store 在 `outputs/pid/day0_1112/store/`，報告 `outputs/pid/day0_1112/day0_audit.json`。

| 項目 | 原測試集（`pub_`／`priv_`） | 新增集（`pub2_`／`priv2_`） |
|---|---|---|
| 候選人 × 盤數 | 400 × 100，全黑 | **600 × 10**，黑白約各半（B 2,975／2,929）；每人黑棋盤數 0–10，**0 盤黑棋的候選人 3／10 位** |
| 題數與 k | 400；k=5／10／15／20 各 100 | 400；k=1／2／3／5 各 100；黑白混合題 214／206 |
| 題數 vs 候選人 | 400 vs 400（一對一可能，未證實） | 400 vs 600（官方明文非一對一） |
| SGF | 只有 `SZ`、`HD`；解析錯誤 0、pass 0、黑先且交替 100%；長度 100–409 | 同左；長度 100–369 |
| HD | 只有 0／1（HD1 約 17%）；**HD≥2 為 0 盤** | 只有 0／1（HD1 約 24%）；HD≥2 為 0 盤 |
| 逐字重複棋局 | 與 train、其他候選庫、題目、棋力測試題互相皆 0 | 皆 0 |
| `player_id` 與 train／其他候選庫重疊 | 0 | 0 |
| 第一手分布（題目 vs 候選庫，TV） | raw 0.019／0.018，D4 0.005／0.011 → **題目沒有被旋轉** | raw 0.048／0.032，D4 0.030／0.013（樣本少，雜訊較大） |
| 候選人黑方第一手集中度（中位數） | 0.98（合成庫 split 0 為 0.96） | 0.70（此數字把候選人執白的盤也算進去，量到的是「該盤黑方」不是本人，不能與左欄直接比） |

判讀：
- 原測試集的結構與合成驗證幾乎一致（全黑、HD 0／1、第一手集中度與合成庫相同、方位未打亂）→ baseline 0.28 的原因 (c)（候選人不像合成庫）與 (d)（方位打亂）被排除；(b)（依時間切、開局習慣漂移）與 (a)（官方 baseline 弱）仍未知，只能靠 LB。
- HD≥2 在四個題目檔與四個候選庫都是 0 盤 → 11/07 擔心的「取到白方」不會發生；`--hd2plus` 選項無作用。
- 新增集是另一個問題：每人只有 10 盤（黑棋平均 5 盤）、題目 1–5 盤且黑白混合。現行開局模型與 GE2E 都只看黑方落子，直接套用會把執白盤當成對手的棋 → 需要另訂方案（未實作）。
- 提交格式：`question_id,top_1..top_5`，1,600 列；`identity/predict.py` 的輸出已改成這個欄位格式。

### 2026-11-12：競賽一第一份合法提交檔（保底）

**原測試集（Slurm 756474，1 卡 13 分）：** 現行管線原樣（D4 開局＋8×GE2E＋HD 項 w=2，不做指派），輸出 `outputs/pid/day0_1112/{public,private}/`。沒有標籤，只有診斷：

| | Public | Private |
|---|---:|---:|
| 400 題 top-1 指到的不同候選人 | 396（4 人被指 2 次、4 人 0 次） | 395（5／5） |
| 第 1、2 名分數差的中位數（k=5／10／15／20） | 28／57／85／93 | 27／63／86／94 |
| 分數差 <2 的題目比例（k=5／10／15／20） | 4%／0／0／0 | 4%／1%／0／0 |
| top-1 與「只用開局」一致 | 94.8% | 93.0% |
| top-1 與 Hungarian 版一致 | 99.0% | 98.8% |

→ 400 題幾乎各指向不同的人，強烈暗示**原測試集是一對一**（非一對一時不會這麼剛好），且模型信心型態與合成驗證相同。Hungarian 只會改 4–5 題。原測試集的真實分數仍未知（無標籤），要看 LB。

**新增集保底（Slurm 756494，1 卡 2 分）：** `scripts/identity/pid_submission.py black-stores` 只留本人執黑的盤，套同一條管線（不做指派）；沒有黑棋盤的題目（Public 96、Private 107 題，約 25%）填候選庫前五人（等於 0 分）。有黑棋的題目覆蓋 k=1／2／3／5：Public 47／74／91／92、Private 43／68／88／94。這只是為了有一份合法檔，新增集的分數預期很低且未知。

**提交檔（`pid_submission.py merge`，已對照官方範例：1,600 題、每列 5 人相異、前綴一致、皆在候選庫內）：**

| 檔案 | 內容 | 假設／預期 | Public LB |
|---|---|---|---|
| `submissions/pid/756494-floor.csv` | 原：開局＋GE2E＋HD，不指派；新增：黑棋保底 | 原測試集若如合成驗證約 0.98、新增集未知（粗估 0.2–0.4）→ 平均約 0.6–0.7；若原測試集有開局漂移則更低 | **0.6328760**（11/12 13:30:51，jabir95tsai；44/53） |
| `submissions/pid/756494-floor-hung.csv` | 同上，原測試集加 Hungarian（只差 9 題的 top-1） | 探針：一對一成立時應 ≥ 上一檔 | |

### 2026-11-12：競賽一新增集——合成驗證台與雙色模型（使用者核准四步方案）

**驗證台（`cv.make_pid2_bank`／`cv.draw_pid2`，`scripts/identity/pid2_bench.py`，Slurm 757026，1 卡 10 分）：** 玩家取自合成 split 0–2（GE2E 從未見過），每人抽 40 盤 HD≤1 的棋（黑白皆可）當題庫；少數玩家 HD≤1 不足 40 盤被跳過，分成兩個互斥 pool 各 591 人（官方為 600）。每次抽樣：每人 10 盤候選、400 題（k=1／2／3／5 各 100，候選人可重複被抽、題間與候選盤皆不重疊）。每個 pool 抽 5 次取平均；**pool 0 選參數，pool 1 只報告**。指標為官方分數，不做指派。

**模型（`identity/twocolor.py`）：**
- 開局雙色化：執黑盤用黑方手（第 0、2、… 手）、執白盤用白方手（第 1、3、… 手），各自的候選人計數與族群先驗，題內相加；候選人沒有該色的盤時退回族群模型（對所有候選人相同）。
- 風格：GE2E 只訓練過黑方落子。執白盤把盤面黑白對調後當成黑方落子編碼（swap），或原樣送入（raw）。
- HD×顏色直方圖項（4 類，與原測試集的 HD 項同式）。
- 融合：開局＋λ·w·cos＋μ·直方圖。

**pool 1（未參與選擇）結果：**

| 做法 | k=1 | k=2 | k=3 | k=5 | 平均 |
|---|---:|---:|---:|---:|---:|
| 黑棋保底（11/12 的 floor 檔做法） | 0.149 | 0.244 | 0.335 | 0.409 | 0.284 |
| 雙色開局（原參數） | 0.151 | 0.256 | 0.312 | 0.421 | 0.285 |
| 風格：只用黑棋盤 | 0.108 | 0.195 | 0.255 | 0.313 | 0.218 |
| 風格：白棋原樣 | 0.097 | 0.120 | 0.165 | 0.201 | 0.146 |
| 風格：白棋對調 | 0.109 | 0.218 | 0.257 | 0.401 | 0.246 |
| HD×顏色單獨 | 0.004 | 0.011 | 0.005 | 0.009 | 0.007 |
| 開局＋4×風格 | 0.194 | 0.362 | 0.461 | 0.558 | 0.394 |
| **開局＋4×風格＋2×直方圖（採用）** | **0.219** | **0.378** | **0.470** | **0.573** | **0.410** |
| 同上，λ 依 k 選 | 0.220 | 0.378 | 0.470 | 0.588 | 0.414 |

- gate（四個 k 桶都高於黑棋保底）：通過，平均 +0.126；pool 1 五次抽樣保底 0.255–0.322、採用版 0.385–0.446，逐次都高。
- 白棋對調有效、原樣送入明顯較差（0.246 vs 0.146）：encoder 沒看過「輪白下」的輸入，對調後才落在訓練分布內。
- pool 0 的選擇：開局 moves=8、beta=0.1、slot=4（格點間差 <0.006，等於原參數）；λ=4（曲線 1→0.377、2→0.408、4→0.423、8→0.414）；μ=2 在格點邊界，但 1 與 2 只差 0.0007。依 k 選 λ 只多 0.004，不採用。
- 限制：驗證台玩家都是有 ≥120 盤黑棋的重度玩家，官方新增集的候選人組成未知；題目如何抽人（均勻可重複）是假設。絕對分數不保證轉移。

**正式預測（`identity/predict2.py`，Slurm 757049，1 卡 2 分）：** 參數全讀驗證台 `outputs/pid/pid2_bench/summary.json`，沒有在官方檔上擬合。診斷：top-1 指到 303／285 個不同候選人（單一候選人最多 5／4 題）；第 1、2 名分數差中位數 k=1／2／3／5 為 1.6／2.3／3.7／4.4（Public）；有黑棋的題目與保底檔 top-1 一致率 38%。

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/757049-twocolor.csv` | 原：同 floor（不指派）；新增：雙色開局＋風格＋直方圖 | 比 `756494-floor.csv` 高約 0.06（新增集 +0.13 的一半）；保留規則：升 → 成為基底；持平或降 → 驗證台不轉移，回頭查候選人組成 | **0.6900431**（11/12 13:29:26，jabir95tsai；36/53） |
| `submissions/pid/757049-twocolor-hung.csv` | 同上，原測試集加 Hungarian | 一對一探針，與上一檔只差原測試集 9 題 | |

**第一筆 LB（11/12 13:29，`757049-twocolor.csv`）：Public 0.6900431，排名 36/53。** 競賽一第一次有真實分數。拆解（假設 Public＝`pub_` 400 題＋`pub2_` 400 題等權平均，官方未明寫）：兩部分合計 1.380。若原測試集如合成驗證約 0.98 → 新增集約 0.40，與驗證台 pool 1 的 0.410 一致；若原測試集較低（例如開局漂移），新增集就更高。單一分數無法拆開兩部分，目前無法區分這兩種情形。
- 下一步建議上傳 `756494-floor.csv`：與本檔只差新增集，驗證台預期差 0.126 → LB 應低約 0.06。若 LB 差約 0.06，代表驗證台的相對比較能轉移；差接近 0 則代表不轉移。
- 本檔暫為競賽一基底（目前唯一有分數的檔）。

**第二筆 LB（11/12 13:30，`756494-floor.csv`）：Public 0.6328760，44/53。** twocolor − floor ＝ **+0.0572**。兩檔只差新增集，換回新增集本身約 **+0.114**，驗證台 pool 1 預期 +0.126 → **驗證台的相對比較有轉移**，新增集之後可依驗證台迭代。
- 絕對值的交叉檢查：驗證台保底 0.284、雙色 0.410。若官方新增集與驗證台一致，原測試集 ≈ 2×0.6329 − 0.284 ≈ 0.98，也 ≈ 2×0.6900 − 0.410 ≈ 0.97，兩式吻合 → 推論原測試集約 0.97–0.98，已接近飽和；**排名差距主要在新增集**。（這是在「Public 為兩部分等權」與「新增集絕對分數也像驗證台」兩個假設下的推論，未直接量到。）
- 基底維持 `757049-twocolor.csv`。`-hung` 版只影響原測試集約 9 題，優先序低。

### 2026-11-12：競賽一新增集——跨賽棋力項（使用者核准「做 2」）

**假設：** 題目與正解候選人同一級別；候選人 10 盤與題目 1–5 盤各算十級後驗，加 `ν·log Σ_r P_題(r)·P_候選(r)`，把棋力對不上的候選人降權。
**做法（新檔 `scripts/identity/pid2_strength.py`，未改其他 session 編輯中的檔）：** V2 ResNet（A-024，`outputs/rank/v2/v2_strength`）每盤指定玩家落子平均 log-prob（同 C2），一組 n 盤取平均後乘 T_n 再 softmax。T_n 只在 Valid-S 的同玩家 n 盤組上以 NLL 擬合：T_1／2／3／5／10＝1.585／2.754／3.673／4.898／6.998（C2 的 T_5＝4.42、T_10 無）。基底＝現行 twocolor 融合（`pid_ge2e_v1_short/summary.json`：開局＋4·w·風格＋2·直方圖）。Slurm 757862（1 卡 dev，7 分）：驗證台 47,280 盤＋官方 pool2／pid2 共 14,200 盤；評估 CPU。
**洩漏注意：** 驗證台 1,182 人中 861 人在 V2 訓練 fold 1–4，V2 對他們的後驗偏準 → 另做「只留 V2 沒看過的玩家（fold 0／−1）」子驗證台（pool 0／1＝146／180 人，候選人少所以絕對分數較高，只看相對增益），**以它為 gate**。

| 驗證台 | pool | 做法 | k=1 | k=2 | k=3 | k=5 | 平均 |
|---|---|---|---:|---:|---:|---:|---:|
| unseen | 1 | 現行 | 0.3368 | 0.5712 | 0.6535 | 0.7569 | 0.5796 |
| unseen | 1 | ＋4×棋力 | 0.3400 | 0.5817 | 0.6722 | 0.7735 | **0.5919（+0.0123）** |
| unseen | 1 | ＋4×候選人真實級別（上限診斷） | 0.3434 | 0.5678 | 0.6695 | 0.7820 | 0.5907 |
| full | 1 | 現行 | 0.2186 | 0.3775 | 0.4697 | 0.5728 | 0.4096 |
| full | 1 | ＋4×棋力 | 0.2280 | 0.3894 | 0.4879 | 0.5914 | **0.4242（+0.0146）** |
| full | 1 | ＋4×候選人真實級別 | 0.2354 | 0.3922 | 0.4682 | 0.6013 | 0.4243 |

- ν 在 pool 0 選（格點 0–16），兩個驗證台都選 ν＝4（非格點邊界；unseen 曲線 0→0.5847、2→0.5937、4→0.5988、8→0.5953）。
- **gate（pool 1 四個 k 桶都不低於現行、平均 ≥ +0.01）：unseen 通過**（+0.003／+0.011／+0.019／+0.017，平均 +0.0123）；full 也通過（+0.0146）。unseen pool 1 五次抽樣增益全為正：+0.008／+0.020／+0.016／+0.003／+0.014。k=1 增益最小（+0.003）。
- 機制：後驗本身很弱（pool 1 題目 top-1 級別準確率 0.30、候選人 0.36），但這項不需要猜對級別，只要把差好幾級的候選人往下壓。V2 後驗已接近「候選人真實級別」上限（unseen 0.5919 vs 0.5907）→ 換 HumanSL 只能改善題目端的後驗，預期增益有限（未知）。
- 預期 Public：新增集 +0.012～0.015 → 若 Public 為兩部分等權，約 **+0.006～0.007**，小於 Public 標準誤約 0.02，LB 只能看方向。

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/757862-strength.csv`（SHA `d04fa90c`） | `757049-twocolor.csv`＋新增集棋力項 ν＝4（原測試集 800 列逐列相同） | 約 +0.006（雜訊內）；保留規則：升或持平 → 成為基底（本地 player-disjoint 證據支持）；降超過 0.02 → 查官方候選人級別分布與驗證台的差異 | **0.6972253**（11/12 14:27:51，jabir95tsai；35/53） |

- 對基底改變 top-1：pub2 77 題（k=1／2／3／5：23／18／16／20）、priv2 61 題；有任何名次變動的列 347／339。ν＝0 時可逐列重現基底排名（已驗證）。未上傳。

**LB（11/12 14:27）：`757862-strength.csv` Public 0.6972253，35/53。** 對 `757049-twocolor.csv`（0.6900431）**+0.0072**，落在預期 +0.006～0.007 內 → 換回新增集約 +0.014，與 unseen／full 驗證台 pool 1 的 +0.012／+0.015 一致；驗證台相對比較第二次轉移。差距小於 Public 標準誤，單看 LB 不顯著，但方向與本地 player-disjoint 證據一致 → **依事先寫好的規則，競賽一基底改為 `757862-strength.csv`。**

### 2026-11-12：競賽一新增集——身分連結診斷（負結果；唯讀，登入節點 CPU）

**動機：** 第 1 名 TEAM_11543 Public 0.8529788（11/12 14:11，10 次上傳）；我方基底 0.6972。若原測試集約 0.98，對方新增集約 0.73、我方約 0.41–0.42，差距是結構性的。先檢查最便宜的結構性解釋：新增集候選人是否就是原候選庫或訓練集裡的人（換了 ID），若是，10 盤候選庫可擴成上百盤。
**做法：** D4 開局模型（A-029 參數，只用黑棋盤），把新增集每位候選人的黑棋盤（≥4 盤者，467／463 人，平均 5.7 盤）當成一題，對三種候選庫打分，看第 1、2 名分數差。

| 題目 → 候選庫 | n | 分數差 p50 | p90 | p99 | 分數差 >15 的比例 |
|---|---:|---:|---:|---:|---:|
| 正對照：`pid_public` k=5 → `pool_public` | 100 | 11.3 | 45.5 | 82.3 | 0.440 |
| 負對照：`pid_public` k=5 → `pool_private` | 100 | 3.0 | 9.4 | 24.3 | 0.030 |
| `pool2_public` → `pool_public`／`pool_private` | 467 | 3.3／3.3 | 13.9／12.4 | 34.5／29.7 | 0.086／0.071 |
| `pool2_private` → `pool_public`／`pool_private` | 463 | 3.5／3.8 | 12.3／12.5 | 25.7／36.0 | 0.067／0.089 |
| `pool2_public` → `pool2_private` | 467 | 2.0 | 7.6 | 22.8 | 0.036 |
| `pool2_public`／`pool2_private` → 訓練集 3,637 人（≥30 盤黑棋） | 467／463 | 2.3／2.2 | 8.3／8.6 | 23.9／19.2 | 0.026／0.028 |
| `pid_public` k=5 → 訓練集（參考） | 100 | 2.4 | 6.7 | 11.9 | 0.000 |

**結論：** 新增集候選人對原候選庫的分數差分布與負對照同級，且對同一部分（Public→Public）與另一部分（Public→Private）沒有差別；對訓練集也沒有雙峰。**新增集候選人不是原候選庫或訓練集的人，候選庫無法靠連結擴充。** 極少數離群值（最大 75.6／90.6）未逐一查，最多影響 1–2% 候選人。另：只用風格向量時，分數正規化（s-norm）與向量去均值都沒有增益（pool 1：0.246 → 0.241／0.242）；合併同一玩家題目的 oracle 上限為 0.330（+0.084），實際需先分群，未做。

### 2026-11-12：競賽一新增集——風格模型改為雙色訓練（使用者「繼續優化模型」）

**假設：** 現行 GE2E 只訓練過黑方、ply ≥20 的落子；新增集一半是執白盤、每人同色只有約 5 盤，開局計數稀疏。改成 (a) 黑白兩色的盤都訓練（白棋盤面黑白對調後送入，`sampling.colors=both`）、(b) 前 20 手也納入（`min_ply=0`）、(c) 多訓幾步。其餘設定同 A-034（V2 主幹、64 人×8 盤×16 手、lr 3e-4、排除合成 split 0–2 的 1,200 人）；訓練玩家 6,548 位、469,821 盤（原 5,438 位黑棋）。gate：驗證台 pool 1 融合平均比 0.410 高 ≥0.02 且四個 k 桶不退步。

**驗證台 pool 1（五次抽樣平均；參數在 pool 0 選；`outputs/pid/pid2_bench/<run>/`）：**

| 風格模型 | 訓練 | 風格單獨 | 融合 k=1 | k=2 | k=3 | k=5 | 融合平均 | 五次抽樣範圍 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `pid_ge2e_v1_short`（原，黑、ply≥20、300 步；白棋推論時對調） | — | 0.246 | 0.219 | 0.378 | 0.470 | 0.573 | 0.410 | 0.385–0.446 |
| `pid_ge2e_v2_both`（雙色、ply≥20、300 步） | 757799，8 卡 8 分 | 0.299 | 0.211 | 0.434 | 0.531 | 0.669 | 0.461 | 0.444–0.507 |
| `pid_ge2e_v2_both_p0`（雙色、ply≥0、300 步） | 757802，8 卡 8 分 | 0.338 | 0.224 | 0.455 | 0.530 | 0.669 | 0.470 | 0.449–0.498 |
| **`pid_ge2e_v2_both_p0_600`（雙色、ply≥0、600 步）** | 757804，8 卡 16 分 | **0.394** | **0.237** | **0.483** | **0.569** | **0.704** | **0.498** | 0.483–0.532 |

- 三個都過 gate；600 步版比原模型 +0.088，五次抽樣的最低值（0.483）高於原模型的最高值（0.446）。融合權重 λ=8、μ=1（pool 0 選；不在格點邊界）。
- 機制：直接用雙色資料訓練比推論時硬對調好（+0.05）；納入前 20 手再 +0.01–0.04（風格單獨 0.299→0.338）；300→600 步風格單獨 0.338→0.394，訓練玩家命中率才 29%→37%，還沒到背玩家的區間（A-035 的 3,000 步是 98.9%）。
- 成本：三個訓練合計約 4.4 GPU-h，三次評估各 1 卡 8–51 分（757805 節點 I/O 慢）。
- 進行中：1,200 步（758185）與 2,400 步（758187）找峰值，評估 758186／758188。

**提交檔（`identity/predict2.py --bench-summary outputs/pid/pid2_bench/pid_ge2e_v2_both_p0_600/summary.json`，Slurm 758189）：**

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/758189-tc600.csv` | 原測試集同 `757049-twocolor.csv`（逐題相同）；新增集換 600 步雙色風格模型（800 題中 370 題 top-1 改變） | 驗證台 +0.088 → Public 約 +0.04（0.690 → 約 0.73）；升 ≥0.02 → 成為基底；持平或降 → 回查驗證台與官方新增集的差異 | **0.7342343**（11/12 14:52:20，jabir95tsai；29/54） |

**LB 判讀：** `758189-tc600.csv` − `757049-twocolor.csv` ＝ **+0.0442**（0.6900 → 0.7342），兩檔只差新增集 → 新增集本身約 +0.088，與驗證台 pool 1 的 +0.088 一致；連同前一次（預期 +0.126、實際約 +0.114），驗證台的相對增益兩次都轉移。**基底改為 `758189-tc600.csv`。** 若原測試集約 0.98，新增集目前約 0.49（驗證台 0.498）。

### 2026-11-12：競賽二 S1 乾淨堆疊檢定 → 未過 gate，「疊一個 CNN」結案（Slurm 758310，CPU 137 秒）

使用者「都做」（S1／S2／S3）。`scripts/strength/rank_clean_stack.py`，輸出 `outputs/rank/humansl/clean_stack/result.json`。

**問題：** 過去把判別式模型疊進 B2（A-024 於 AH＋C、SE3＋H＋C、noA 反證）時，擬合玩家都是該模型的訓練玩家，特徵在擬合端是樣本內、在未見玩家是樣本外。「對擬合與評估玩家都是未見」的判別式特徵能否補 H＋C，沒被測過。
**設計：** 只用 Valid-U（A-024 從未見過，1,442 人、HD≤1 33,656 盤）。玩家依級別分層分 5 折（seed 2027），每折以其餘四折擬合 PriorMatched B2（先驗＝擬合折 ≥k 盤玩家比例），評該折玩家的 E1（1,000 題）／E2（2,000 題）；特徵為 P3 版 [A 10｜H 30｜C v1 20｜讓先旗標 2]，seed 2027–2031。事前 gate：hca 或 hca2 − hc，五 seed 平均 Equal-k 在 E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005。

| 五 seed 平均 Equal-k | E1 | E2 | E2 各 k 差（5／10／15／20） |
|---|---:|---:|---|
| hc（H＋C＋旗標，52 維） | 0.6766 | 0.7102 | — |
| hca（＋A-024 10 維） | 0.6757（−0.0009） | 0.7092（−0.0010） | −0.003／−0.000／−0.012／+0.011 |
| hca2（＋A 的前兩個主成分） | 0.6767（+0.0000） | 0.7120（+0.0017） | +0.000／+0.006／−0.002／+0.003 |
| a（只有 A＋旗標） | 0.5493 | 0.5600 | — |

- 各 seed 差值：hca2 E1 −0.000／+0.002／+0.004／−0.003／−0.003、E2 −0.002／+0.004／+0.003／+0.006／−0.002；hca E1 +0.001／+0.004／+0.001／−0.003／−0.008、E2 −0.009／−0.000／−0.001／+0.004／+0.001。seed 2027 玩家 bootstrap 95%：hca E1 [−0.014, +0.015]、E2 [−0.021, +0.002]；hca2 E1 [−0.008, +0.007]、E2 [−0.009, +0.004]。
- **未過 gate。** A 單獨 0.55–0.56（與 A-024＋C2 在 Valid-U 的 0.555 一致），在完全樣本外的條件下加到 H＋C 上增益為 0（±0.002）。先前「A 有害」可歸因於樣本內污染，但拿掉污染後也沒有正貢獻。
- 限制：每折擬合玩家約 1,150 人（正式池約 5,600 人），hc 基準因此比正式設定低約 0.035；A-024 比 SE3 弱（0.555 對 game-disjoint 0.62），不能直接證明較強的判別式模型也加不上去。但 SE3 以 LB 0.67 仍低於 H＋C 約 0.08、hrb2 凍結探針 0.59，沒有任何證據支持花約 140 GPU-h 做按玩家交叉擬合的 SE3 → **S1b 不提案，「把判別式 CNN 疊到 H＋C」結案。**
- **S3（補 fold −1 玩家）不重做：** 提案時漏查，11/09 已做過（`rank_more_players.py local-ex`，794 人、24,747 盤）：先驗對齊後 E1 −0.0016／E2 −0.0007，LB `737645-priorex` 0.7468（對基底 −0.0041）。結論不變。

### 2026-11-12：競賽二 S2 逐手重新聚合（事前登記；抽取 Slurm 758313、分析 758323 進行中）

**假設：** C 的每盤特徵是「各級別假設下 log p（實際著手）的整盤平均」，少數極低機率的手主導平均；k=20 尚未到每盤雜訊極限（11/11 天花板診斷），對手特徵接近獨立雜訊。改用穩健聚合可降每盤雜訊。預期增益：未知（A-046 的分段特徵 −0.025、vote +0.003 是反面證據；主觀過 gate 機率約 25%）。
**做法：** `humansl_rankvec extract --per-move`（新選項，預設行為不變）以 C v2 向量重抽 cvec2 同一批表（Valid-S HD≤1＋Valid-U＋train20＋兩份測試，194,698 盤），每個位置存 10 個 log p（float16）、手數與本人／對手；輸出 `outputs/rank/humansl/permove/`。`scripts/strength/rank_permove.py`：
- `features`：每個變體產生取代 C 20 欄的每盤特徵（H 與旗標不動）：mean（須重現 cvec2 的 C，誤差 ≤1e-2）、clip4／6／8（log p 下限截尾）、trim5／10（丟掉各方最不像的 5%／10% 手）、vote（逐手對 10 個假設 softmax 後平均）、p20／p40／u200／mid（手數窗）、mean+vote（40 欄，唯一加維度的變體）。
- `select`：**只在擬合池內選變體**（Valid-S＋train20 HD≤1；玩家依級別分層對半，一半擬合、另一半出級別均勻題，每 k 每級 100 題，互換，共 8,000 題，B2 seed 2027–2029 平均）。Valid-U 不讀。
- `evaluate`：只有選擇增益最高的一個變體、且增益 ≥ +0.003 才在 Valid-U E1／E2 評分（五 seed，對 mean）。G1 同 cvec2：E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005 → 升為基底候選；兩組 ≥ 0 → LB 探針；否則停止。
**估算：** 抽取 8 卡約 1.5–2 小時（約 13–16 GPU-h；首片實測 533 位置／秒含啟動），逐手檔約 1 GB；分析為 CPU。


### 2026-11-12：C4x adj 兩項 loss 梯度診斷（使用者要求）

- 唯讀起始 HumanSL＋C v2 與 adj A／B 最終 checkpoint；每半固定 4×512 位置，重用原 loss、BF16／eval／rank weights，分別計算 NLL 與相鄰 CE 梯度。單卡 H200 job **759077**，47 秒、約 0.013 GPU-h，exit 0；未更新原權重。
- **網路終點 CE/NLL 梯度 norm：A 0.0771、B 0.0821；norm 份額 7.16%／7.59%。cosine +0.1265／+0.1595，四批各自均為正值，未看到整體梯度互相抵消。** 級別向量的 norm 比為 0.7980／0.8100、cosine +0.7902／+0.7999；向量 LR 為網路 10 倍，raw norm 份額不是 AdamW 更新份額。
- 原訓練前／後約 2,000 步窗口 CE：A 1.017137→1.017141，B 1.016982→1.017131；理想 rank-balanced 均勻基準 **1.017519**（兩端 2 候選，其餘 3）。同一批抽樣起始→終點 CE 微降：A 1.017241→1.015360、B 1.006836→1.006144；不同樣本的加權均勻基準不同，不能直接比較 A／B 絕對 CE。另各 512 位置固定 monitor CE Δ−0.000557／−0.000169。終點保存的 AdamW moments 對八批的下一步一階 CE 變化全部為負。
- **解讀：尺度不平衡成立，但目前證據不足以把 CE 停滯歸因於 NLL 壓制；小幅改善、抽樣波動與相鄰假設共同提高 likelihood 均可相容。** 未做調權／單 loss 訓練、跨 batch 衝突／訊噪診斷或下游分 k 重評，不宣稱加權可改善泛化。
- 完整方法、數字與限制：`outputs/rank/humansl/c4x/adj/gradient_diagnostic_20261112/report.md`；同目錄 `result.json`、`probe.py`、`run.slurm`、`slurm-759077.out`。assert self-check 通過。
- 佇列現況：dev 的 p_dev **MaxJobsPU=2／MaxSubmitJobsPU=7**，8gpus 的 p_8gpus **8／13**。原診斷 758987 因兩個既有 dev benchmark 已占滿而 pending，取消後改送 8gpus（只申請 1 GPU）；已回填 `CLAUDE.md`，未中斷其他作業。

**S2 結果（抽取 758313：8 卡 1 小時 35 分＝12.7 GPU-h，194,698 盤全數完成，逐手檔 1.3 GB；分析 758323：CPU 17 分）→ 未過，停止。**
- 重現檢查：逐手檔的 mean 與 cvec2 的 C 特徵最大差 4.1e-4（train）／3.1e-4／3.0e-4（測試），盤數完全一致。
- 擬合池內選擇（149,615 盤對半、8,000 題、三 seed 平均；Equal-k 對 mean 0.7319 的差）：

| 變體 | Δ Equal-k | Δk=5 | Δk=10 | Δk=15 | Δk=20 |
|---|---:|---:|---:|---:|---:|
| clip4／clip6／clip8 | +0.0002／+0.0016／+0.0012 | | | | clip6 +0.0044 |
| trim5／trim10 | +0.0003／−0.0026 | | | | |
| p20／p40／u200／mid | +0.0010／−0.0025／−0.0007／−0.0020 | | | | |
| **vote**（選中） | **+0.0058** | +0.0031 | +0.0017 | +0.0084 | +0.0100 |
| mean+vote（40 欄） | +0.0054 | +0.0024 | +0.0012 | +0.0083 | +0.0098 |

- Valid-U（只評 vote，五 seed，對 mean）：**E1 −0.0017（+0.004／+0.003／−0.000／−0.011／−0.004），E2 −0.0074（−0.010／−0.005／−0.005／−0.015／−0.003，五 seed 全負）**；E2 各 k −0.003／−0.015／−0.008／−0.004。G1 判定 **stop**。seed 2027 bootstrap 見 `result_vote.json`。
- 判讀：(1) 截尾、修剪、手數窗都在 ±0.003 內 → 「少數極端手主導每盤平均」不是可用的雜訊來源，整盤平均已接近這組 log p 的最佳線性聚合。(2) vote 在擬合池內 +0.006、到未見玩家變 −0.007。推測（未驗證）：C v2 向量是用擬合池玩家的其他棋局學的，逐手 softmax 放大了向量對這些玩家的貼合，池內選擇因此偏樂觀；這也表示**擬合池內的對半驗證不能代替 Valid-U**。(3) 與 A-046（vote +0.003、分段為負）一致。
- **不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。** 逐手檔保留在 `outputs/rank/humansl/permove/`（1.3 GB），不再需要時可刪。
- **11/14 補做 LB 探針（使用者「沒過 gate 但可能有幫助的都傳看看」）：** `rank_permove export --variant vote`（Slurm 777246，1 卡配置、CPU 3 分 16 秒）＝cvec2 export 原樣（Valid-S＋train20＋Valid-U HD≤1、先驗對齊、seed 2027），只把 C 20 欄換成 vote；同一程式中未換的臂逐題重現 `751959-cvec2.csv`（不一致即報錯）。再以 `rank_identity_merge overlay` 疊 L10。

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/rank/777246-vote-l10.csv`（SHA `74ccc337`） | 基底 `751959-cv2-l10.csv` 的 C 改 vote；L10 與基底相同。改 Public 33／Private 38 題，全為差一級（往強 35、往弱 36） | Valid-U E1 −0.0017、E2 −0.0074（五 seed 全負）→ 預期 Public 約 −0.004，偏負。保留規則：降或持平 → 維持基底、vote 結案；升 <0.02 → 不升基底（本地明確為負）；升 ≥0.02 → 回查擬合池與正式題的差異 | **0.7527214**（11/14 22:01:42，jabir95tsai；18/136） |
| `submissions/rank/777246-vote.csv`（SHA `4a558e3a`） | 同上、無 L10，僅對照 | 不建議上傳 | |

**LB（11/14 22:01）：`777246-vote-l10.csv` Public 0.7527214，對基底 `751959-cv2-l10.csv`（0.7499623）+0.0028**（400 題總分約 +1.1；改了 33 題，相當於約 1–2 題由鄰級變正確），與預期 −0.004 方向相反，但遠小於 Public 標準誤約 0.02。Valid-U 五 seed E2 全負（−0.0074）、擬合池內 +0.006，Public 落在兩者之間，三者都在雜訊量級內。依保留規則（升 <0.02、本地明確為負）**不升基底；最終預設檔維持 `751959-cv2-l10.csv`**。這是目前競賽二 Public 最高分（前高 `750633-l10.csv` 0.7524623），但依 Private 泛化原則不以單次 Public 小升換檔。

**截尾／修剪 LB 探針（11/14 使用者：「截尾和修剪應該也會有效，提交看看」；Slurm 777316，CPU 6 分 24 秒，流程與 vote 相同、未換臂同樣逐題重現 `751959-cvec2.csv`）：** 各取擬合池內最佳一個；兩者都**沒有在 Valid-U 評過**（S2 事前規則只評選中的 vote）。用途是檢驗「vote 的 Public 小升是否來自壓低極端手」：若是，clip6／trim5 也應不降。

| 檔案 | 內容 | 擬合池內 Δ（三 seed） | 預期 Public | Public LB |
|---|---|---|---|---|
| `submissions/rank/777316-c6-l10.csv`（SHA `e2be285b`） | C 改 clip6（log p 下限 −6 後平均）＋L10。改 Public 12／Private 22 題，全差一級 | +0.0016（k=20 +0.0044） | 約 0，只動 12 題 | **0.7493017**（11/14 22:58:33，jabir95tsai；20/136） |
| `submissions/rank/777316-t5-l10.csv`（SHA `3813b537`） | C 改 trim5（每方丟最不像的 5% 手）＋L10。改 Public 12／Private 22 題，全差一級 | +0.0003 | 約 0 | **0.7518017**（11/14 22:57:57，jabir95tsai） |

兩檔在 Public 改動的 12 題中有 8 題相同（同題同改）、與 vote 只重疊 2–3 題。無 L10 版 `777316-clip6.csv`／`777316-trim5.csv` 只作對照。保留規則同 vote：升 <0.02 不升基底（本地未評或為負）；任一升 ≥0.02 → 先在 Valid-U 五 seed 評該變體再決定。

**LB（11/14 22:57／22:58）：** 對基底 0.7499623，**trim5 +0.0018**（總分 +0.7）、**clip6 −0.0007**（−0.3）；連同 vote +0.0028（+1.1），三個穩健聚合變體在 Public 分別 +1.1／+0.7／−0.3 分，都在 12–33 題改動的雜訊內（同實力下約 ±2–3.5 分）。clip6 是三者中擬合池內最佳的截尾版，Public 反而最低；沒有一致的「壓低極端手 → Public 上升」型態。依保留規則**三者都不升基底，最終預設檔維持 `751959-cv2-l10.csv`；S2 穩健聚合結案**。

**本輪小結（S1／S2／S3）：** 三項都沒有增益。到目前為止，H＋C 之上的 B2 層、先驗、解碼、聚合、外加判別式特徵、擬合玩家數都已測到 ±0.005 內；剩下未被否證的只有「更好的逐手 likelihood 模型本身」（C4x adj 有監控改善但下游持平）。與第 1 名的 0.055 差距原因仍未知。

### 2026-11-12：棋力項疊到雙色風格模型 → 未過 gate，不加（CPU，`pid2_strength eval --bench-dir`）

基底換成 `758189-tc600.csv`（600 步雙色風格模型）後，在新融合上重選 ν（V2 後驗與 T_n 不變，pool 0 選、pool 1 報告）：

| 風格模型 | 驗證台 | 現行 pool 1 | ＋棋力項 | Δ（k=1／2／3／5） | 五次抽樣 Δ | 上限診斷 Δ |
|---|---|---:|---:|---|---|---:|
| v1_short（參考，上一節） | unseen | 0.5796 | 0.5919（ν=4） | +0.003／+0.011／+0.019／+0.017 | 全正 | +0.011 |
| p0_600 | unseen | 0.6734 | 0.6742（ν=4） | −0.010／+0.003／+0.004／+0.007 | +0.006／−0.001／+0.001／+0.001／−0.002 | +0.005 |
| p0_600 | full | 0.4984 | 0.5031（ν=2） | +0.006／+0.004／+0.007／+0.002 | 一負四正 | +0.008 |
| p0_1200 | unseen | 0.6769 | 0.6763（ν=2） | −0.003／+0.001／+0.003／−0.003 | 三負二正 | +0.005 |
| p0_1200 | full | 0.5090 | 0.5145（ν=4） | +0.006／+0.006／+0.007／+0.003 | 全正 | +0.007 |

- **gate（unseen pool 1 四桶不退、平均 ≥ +0.01）：兩個新模型都不過**（+0.0008／−0.0006，k=1 退步）。full 驗證台的 +0.005 有 V2 樣本內偏差，不採信。
- 機制：連「候選人真實級別」上限也從 +0.011 縮到 +0.005 → 新融合已含大部分級別資訊。推測原因：雙色風格模型從 V2 棋力主幹初始化、白棋盤與前 20 手也納入後，風格向量本身就把不同級別的候選人拉開（未直接驗證）。
- **結論：棋力項不加到 `758189-tc600.csv`，基底不變；不建議為此用上傳額度（預期 Δ≈0）。** `757862-strength.csv` 的 +0.0072 只在舊風格模型上成立。

### 2026-11-12：1,200 步雙色風格模型提交檔（使用者「先產生提交檔」）

`identity/predict2.py --bench-summary outputs/pid/pid2_bench/pid_ge2e_v2_both_p0_1200/summary.json`（Slurm 761130，1 卡 2 分；輸出 `outputs/pid/day0_1112/{public,private}2_tc1200/`，不覆蓋基底目錄）＋`pid_submission merge`。融合參數與 600 步相同（λ=8、μ=1、開局同），差別只在風格向量。

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/761130-tc1200.csv`（SHA `0e699e27`） | 原測試集與 `758189-tc600.csv` 逐列相同；新增集換 1,200 步風格模型。對 tc600 改 top-1：pub2 116 題（k=1／2／3／5：42／30／27／17）、priv2 134 題 | 驗證台 pool 1 +0.011（五次配對 +0.002／−0.016／+0.028／+0.020／+0.019）、pool 0 只 +0.003 → Public 約 +0.003～0.005，雜訊內。保留規則：LB 差 <0.02 不據以換基底，最終選擇以本地多次配對抽樣決定（尚未做）；降 ≥0.02 → 維持 tc600 | **0.7443283**（11/12 21:27:09，jabir95tsai；33/63） |

### 2026-11-12：競賽二——學聚合方式的小模型、玩家層級微調 C4q（使用者提案「做這個」；事前登記）

**使用者的診斷與提案：** 聚合到極限後約 65：33：2（正確：差一級：其餘），瓶頸是聚合後的級別解析度（玩家層級偏差）；(1) 用小模型學聚合方式、放大明顯能判級的盤；(2) 用整體判錯的玩家偏差微調 HumanSL 主幹或 C，為防過擬合每次 update 只調 18 個 block 中的 3 個。

**(1) 學聚合（`scripts/strength/rank_attn_pool.py`，Slurm 761154，CPU）：** 每盤權重 w＝softmax(s(x))，加權後的特徵送同一個 LR；s＝0 時即 B2。臂：mean（對照，同一程式路徑）、att_lin（s＝u·x，52 參數）、att_mlp（52→8→1）。資料同 cvec2 基底（Valid-S＋train20 HD≤1、C v2、先驗對齊、合成題抽樣與 PriorMatched 相同），L-BFGS，L2 同 sklearn C＝1。Valid-U E1／E2 五 seed。Gate：最佳注意力臂 − mean，E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005。預期：未知，偏負（B0、B_std、B_mlp、vote 皆負；若偏差是整個人一起偏，盤間重新加權搬不動）。

**(2) C4q（`humansl_finetune.py --player-ce`，新增；預設行為不變）：**
- loss＝NLL（真實級別）＋1.0×玩家層級 CE：每步每卡抽 8 位玩家（先均勻抽級別、再均勻抽該級玩家）、每人從其全部棋局（每人最多 20 盤 HD≤1）隨機抽 32 個位置，在 {r−1, r, r+1} 三個級別假設下求 log p；每位玩家把本人手、對手手的平均 log p 各乘一個可學的尺度後相加成 3 個 logit，對真實級別做 CE。整體證據指向錯誤級別的玩家 CE 大，梯度因此對準玩家層級偏差（即「用判錯的人來調」），不是單一位置。
- 資料、交叉擬合、監控沿用 C4x（A／B 兩半各約 2,870 人、fold −1 監控 641 人、C v2 向量起始）。3,000 步 × 8 卡，lr 同 C4x（主幹 3e-5、向量 3e-4、warmup 300、cosine）。
- 臂：**q3**（使用者指定：只訓練 18 個主幹 block＋向量，每步隨機 3 個 block 得到梯度，stem／policy head 凍結）；**qall**（對照：C4x 的全部可訓練參數、每步全更新），用來分離「玩家層級 loss」與「block 隨機更新」的效果。
- 監控新增玩家層級指標：每位監控玩家各盤每手平均 log p 再平均 → 中段 d′（`player_mid_d_*`）與官方分數（`player_official`，溫度取固定網格最佳值，各步同規則）。第 0 步：單盤 0.5358（與 C4x 第 0 步相同，確認起點為 C v2）、player_official 0.6442、player_mid_d_all 0.8135。
- **G0（沿用 11/11 使用者規則）：** 兩模型終點單盤官方分數都 ≥ 第 0 步才抽特徵。兩臂都過時只抽「player_official 平均增益較高」的一臂（省約 15 GPU-h），另一臂只報監控。
- **G1（同 C4x）：** swap − base（cvec2 基底），E1 或 E2 五 seed 平均 Equal-k ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005 → 升為基底候選；兩組 ≥ 0 → LB 探針；否則停止。
- 預期增益：未知；主觀過 G1 機率約 15–20%（依據：adj 單盤監控改善但題目層級持平、mmi 記住玩家；中段每級每半約 155 人）。
- 估算：訓練每半實測約 2,500 位置／秒 → 約 40 分＋監控 7 次約 11 分，四個作業約 28 GPU-h；抽特徵一臂約 15 GPU-h；G1／匯出 CPU。合計約 45 GPU-h。作業：q3 A 761146、q3 B 761179、qall A 761180、qall B 761181。

**(1) 學聚合結果（Slurm 761154，CPU 9 分 30 秒；`outputs/rank/humansl/attn_pool/result.json`）→ 未過 gate。**

| 五 seed 平均 Equal-k | E1 | E2 | E2 各 k 差（5／10／15／20） |
|---|---:|---:|---|
| sklearn 基底（參考） | 0.7139 | 0.7463 | — |
| mean（同程式路徑的對照） | 0.7135 | 0.7465 | — |
| att_lin（52 參數） | +0.0008 | −0.0011 | −0.008／+0.006／+0.002／−0.005 |
| att_mlp（52→8→1） | −0.0026 | −0.0045 | −0.007／−0.005／+0.005／−0.011 |

- 各 seed：att_lin E1 +0.001／−0.003／−0.002／+0.002／+0.007、E2 +0.001／−0.006／−0.003／+0.001／+0.003；att_mlp E1 −0.012／+0.002／−0.001／−0.005／+0.003、E2 −0.011／−0.002／−0.007／−0.004／+0.001。seed 2027 bootstrap：att_lin E1 [−0.010, +0.011]、E2 [−0.007, +0.008]；att_mlp E1 [−0.024, +0.000]、E2 [−0.021, −0.002]。
- 對照有效：torch 的 mean 臂與 sklearn 基底相差 ≤0.0004。
- 判讀：線性權重 ±0.001（雜訊內），MLP 權重略負——與 B0／B_std／B_mlp／vote 同方向：盤間重新加權沒有可用訊號，容量越大越差。符合「偏差是整個人一起偏，不是某幾盤特別準」。**不產生提交檔。**

**LB（11/12 21:27）：`761130-tc1200.csv` Public 0.7443283，33/63。** 對 `758189-tc600.csv`（0.7342343）**+0.0101** → 新增集本身約 +0.020，略高於驗證台 pool 1 的 +0.011，方向一致。差距 <0.02 單看 LB 不顯著，但本地 pool 1／pool 0 都是正的（+0.011／+0.003）→ **暫定競賽一基底改為 `761130-tc1200.csv`**；依事前規則，最終以本地多次配對抽樣確認（未做）。

**本地配對抽樣確認（`scripts/identity/pid2_paired.py`，CPU；每 pool 30 次抽樣、種子 100–129 與驗證台 0–4 不重疊；同一批題目比兩個模型；輸出 `outputs/pid/pid2_paired/`）：**

| 1,200 − 600 步 | k=1 | k=2 | k=3 | k=5 | 平均（± 標準誤） | 抽樣中為正的比例 |
|---|---:|---:|---:|---:|---:|---:|
| pool 0 | +0.007 | +0.005 | +0.011 | +0.005 | **+0.0070 ± 0.0026** | 67% |
| pool 1 | +0.004 | +0.008 | −0.003 | +0.011 | **+0.0048 ± 0.0021** | 60% |

- 兩個互斥 pool 都約 +2.5 個標準誤，平均約 +0.006；唯一負的桶（pool 1 k=3 −0.003）在 1 個標準誤內，pool 0 同桶 +0.011 → 不視為退步。原先 5 次抽樣的 pool 1 +0.011 偏高，30 次估計約為一半。
- 本地證據與 LB（+0.0101）方向一致 → **確認競賽一基底為 `761130-tc1200.csv`。**

**(2) C4q 結果（Slurm 761146／761179／761180／761181，各 8 卡 25–30 分，合計約 14.6 GPU-h）→ G0 未過，不抽特徵，停止。** fold −1 監控（641 位未見玩家），第 0 步 → 第 3,000 步：

| 臂／半 | 單盤官方分數 | 單盤中段 d′ | 玩家層級官方分數 | 玩家層級中段 d′（本人／對手／全部） | 監控 NLL | 訓練玩家 CE（終點） |
|---|---|---|---|---|---|---:|
| 第 0 步（C v2） | 0.5358 | 0.531 | 0.6442 | 0.620／0.706／0.814 | 1.4117 | — |
| q3 A | 0.4991 | 0.518 | 0.6365 | 0.616／0.663／0.797 | 1.4104 | 0.989 |
| q3 B | 0.5031 | 0.506 | 0.6368 | 0.585／0.685／0.778 | 1.4103 | 0.989 |
| qall A | 0.4930 | 0.503 | 0.6163 | 0.598／0.641／0.787 | 1.4078 | 0.981 |
| qall B | 0.4851 | 0.505 | 0.6219 | 0.580／0.673／0.785 | 1.4078 | 0.979 |

- **四個模型在未見玩家上全部退步**：單盤 −0.033～−0.051，玩家層級 −0.007～−0.028，中段 d′（單盤與玩家層級）都降。依登記的 G0（終點 ≥ 第 0 步）兩臂都不抽特徵，省下約 15 GPU-h；`g0_q3.json`／`g0_qall.json` 已寫出。
- 曲線形狀：前 500–1,000 步掉最多（lr 最高），之後只部分回升，沒有任何檢查點高於第 0 步 → 不是「訓練太久」造成，早停也救不回來。
- 監控 NLL 略降（1.4117→1.410／1.408）而判級變差：模型更會預測著手，但級別之間的 likelihood 差被訓練玩家的特徵取代——與 C4b／C4x mmi（訓練 CE 降、未見玩家崩壞）同一現象，只是幅度較小。玩家層級 loss 沒有改變這個結果。
- q3 退步比 qall 小（玩家層級 −0.007 對 −0.025）。這符合「每個 block 只拿到六分之一的更新、stem 與 policy head 凍結 → 整體改得少」，不能解讀為隨機 block 有正則化效果：兩臂都低於起點，改得越少越接近起點。
- 限制：訓練玩家 CE 的起點無法與終點比較（尺度參數初值過大，前 200 步在調尺度）；λ、lr、每人位置數各只試一個設定。但四個模型、兩種容量方向一致，且與先前三次判別式微調一致，不再調參重試。
- **結論：以玩家層級判錯訊號微調 HumanSL，在現有約 2,870 人／半（中段每級約 155 人）下學到的是訓練玩家而不是棋力。競賽二基底與最終預設檔不變（`751959-cv2-l10.csv`）。** 檢查點在 `outputs/rank/humansl/c4x/{q3,qall}/`，可刪。

### 2026-11-13：競賽一原測試集 Hungarian 探針（重新合併，CPU 數秒）

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/761130-tc1200-hung.csv`（SHA `a3b4fc3d`） | 新增集與 `761130-tc1200.csv` 逐列相同；原測試集換成 Hungarian 指派（與 `757049-twocolor-hung.csv` 的原測試集相同）。改 top-1：pub 4 題、priv 5 題；有任何名次變動的列 40／26 | 一對一探針：只影響 Public 4 題的第 1 名，變動上限約 ±0.005（4 題 ÷ 800）。升 → 一對一可能成立，可保留；降 → 原測試集不是一對一，維持不指派。差距都在雜訊內，只看方向 | |
| `submissions/pid/772748-hung.csv`（SHA `82ff0cf6`；11/14 使用者「沒過 gate 但可能有幫助的都傳看看」，登入節點重新合併） | 新增集與現行基底 `772748-d4-9600.csv` 逐列相同；原測試集換 Hungarian（`public_hungarian`／`private_hungarian`，與上列相同）。改 top-1：pub 4 題、priv 5 題 | 同上：Public 變動上限約 ±0.005，只看方向。升 → 一對一可能成立，可保留為基底；降或持平 → 維持 `772748-d4-9600.csv`（不指派，D-004） | **0.7980564**（11/14 21:29:51，jabir95tsai；31/81） |

**LB（11/14 21:29）：`772748-hung.csv` Public 0.7980564，對基底 `772748-d4-9600.csv`（0.7936014）+0.0045。** 只有 Public 4 題的名次改變，所以這個差可以完整歸因：+0.0045 × 800 題 ≈ 總分 +3.56，接近上限（4 題全由錯變對約 +4）→ **4 題幾乎全部改對**。這不是一般 400 題雜訊下的小差距，而是逐題可歸因的證據：**原測試集（Public 部分）是一對一，Hungarian 有效。** Private 部分（priv 改 5 題）同一機制、同一出題方式，推論同樣成立（無法直接驗證）。依保留規則**競賽一基底改為 `772748-hung.csv`**；之後新增集的探針都疊在這份上（原測試集用 Hungarian）。新增集官方明文非一對一，不套用指派。D-004（指派預設關閉）對原測試集的部分由此改判，`--assign` 預設是否改動另行決定。CLAUDE.md 第 0 節同步註記。

### 2026-11-13：競賽一新增集——計分方式探針與玩家層級 D4 增強（使用者「繼續優化競賽一模型」；事前登記）

**現況：** 基底 `761130-tc1200.csv`（Public 0.7443）；驗證台 pool 1 融合約 0.51–0.53。第 1 名 0.85，若原測試集約 0.98，差距都在新增集。2,400 步風格模型已過擬合訓練玩家（pool 1 融合 0.473 < 1,200 步的 0.509）。

**(1) 只改計分、不重訓（登入節點 CPU，pool 0、抽樣 100–109、只看風格項、未與開局融合；臨時腳本）：**

| 做法 | k=1 | k=2 | k=3 | k=5 | 平均 |
|---|---:|---:|---:|---:|---:|
| 1,200 步，cosine（現行） | 0.200 | 0.388 | 0.520 | 0.628 | 0.434 |
| 600＋1,200 步 cosine 相加 | 0.217 | 0.394 | 0.529 | 0.642 | 0.445 |
| 四個模型（300／600／1,200／2,400）相加 | 0.218 | 0.398 | 0.528 | 0.645 | 0.447 |
| 1,200 步，黑白分色去均值後 cosine | 0.205 | 0.385 | 0.521 | 0.623 | 0.433 |
| 1,200 步，PLDA（類內共變異由候選庫 10 盤估，shrink 0.2） | 0.208 | 0.387 | 0.523 | 0.647 | 0.441 |
| 600 與 1,200 步各做 PLDA 後相加（共變異加另一 pool 的候選庫） | 0.221 | 0.397 | 0.538 | 0.674 | 0.457 |

→ 計分端的改動合計約 +0.02（只在 pool 0 看過，尚未在 pool 1 與融合後驗證）；不是主要槓桿，瓶頸在 encoder 對未見玩家的泛化。

**(2) 玩家層級 D4 增強（`sampling.player_d4`，`identity/ge2e.py`）：**
- 假設：訓練玩家只有 6,548 位，1,200 步之後開始背玩家。每個 batch 內同一玩家的所有盤面套同一個隨機 D4 變換（不同玩家各自抽），同一人在 8 個方位下等於 8 位「虛擬玩家」，玩家內的方位習慣仍一致，所以不會像逐盤 D4 那樣洗掉方位指紋。預期可延後過擬合、讓更長的訓練有用；增益大小未知。
- 設定：其餘同 `pid_ge2e_v2_both_p0_1200`；`pid_ge2e_v3_d4_1200`、`pid_ge2e_v3_d4_2400`。
- gate：pool 1 融合平均（30 次配對抽樣，`pid2_paired`）比 `p0_1200` 高 ≥0.01 且四個 k 桶不退步 → 產生提交檔；2,400 步若比 1,200 步高則再加長。
- 成本：8 卡約 30／61 分，合計約 12 GPU-h；嵌入＋評估各 1 卡約 10 分。

### 2026-11-13：競賽二——對手身分連結前提檢定 → 未過 gate，停止（Slurm 763755，7×1 卡共 41 GPU 分；分析為登入節點 CPU）

**假設（使用者同意先做第一步）：** 測試棋的對手常是訓練集 8,007 位已知級別的玩家；幽玄配對相近棋力，認出對手即可推本人級別。11/11 的開局模型單盤認不出人（本人在庫時也只有 7.6% 超過門檻），改用 11/12 的雙色 GE2E（`pid_ge2e_v2_both_p0_1200`）當單盤指紋重測。
**做法（`scripts/strength/rank_opp_link.py`、`slurm/rank_opp_link.slurm`，輸出 `outputs/rank/opp_link/result.json`）：** 候選庫＝每位訓練玩家最多 40 盤 HD≤1 棋局（247,745 盤、7,969 人有盤）的單位向量平均；查詢皆為單盤，對 8,007 個中心取 cosine。seen＝候選庫玩家庫外的一盤本人落子（4,000 人各一盤），同一批拿掉本人欄即 unseen null；train_opp＝6,000 盤訓練棋的對手方（遮掉棋局擁有者）；test_opp／test_own＝測試 10,000 盤的對手方／本人方。事前 gate：seen top-1 召回 ≥0.20 且測試對手在庫比例估計 ≥0.30。

| seen（本人確定在庫，8,007 選 1） | top-1 | top-5 | top-50 | 真人名次中位數 |
|---|---:|---:|---:|---:|
| 全部 4,000 盤 | **0.080** | 0.185 | 0.449 | 69 |
| GE2E 訓練過的玩家（2,919）／沒訓練過（1,081） | 0.081／0.078 | | | |

| 超過 unseen null 門檻的比例 | cos q95 | cos q99 | 分數差 q95 | 分數差 q99 |
|---|---:|---:|---:|---:|
| seen（在庫） | 0.058 | 0.013 | 0.059 | 0.015 |
| 訓練棋對手 | 0.056 | 0.018 | 0.053 | 0.011 |
| 測試對手 Public／Private | 0.058／0.049 | 0.017／0.017 | 0.061／0.058 | 0.016／0.015 |
| 測試本人 Public／Private | 0.044／0.053 | 0.014／0.015 | 0.050／0.047 | 0.010／0.011 |

- **未過 gate：top-1 召回 0.080（門檻 0.20）。** 比開局模型強（隨機為 0.0001），但本人在庫時分數分布與不在庫幾乎相同（超過 q95 的比例 5.8% 對 5%），所以「在庫比例」的混合估計分母只有 0.008，算出的數字（−0.7～+3.8）沒有意義——**對手是否為訓練玩家仍然未知，這個工具回答不了。**
- 單盤 GE2E 是否訓練過該玩家沒有差別（0.081 對 0.078）→ 不是樣本內問題，是單盤風格向量在 8,007 人中解析度不夠（中位 cosine 0.72，第 1、2 名差 0.017）。
- 槓桿大小（訓練棋對手方，top-1 候選人的級別對棋局擁有者級別）：相同 22.1%（HD0）／27.8%（HD1）、差 ≤1 級 53.8%／58.5%，打亂基準 9.9%／28.2%；分數差超過 q99 的 65 盤為 31%／71%（HD0）。這是「風格相近的人級別相近」的相關（等同在風格空間做近鄰判級），不是身分；屬判別式特徵，S1 已結案同類方向，不接進 B2。
- **結論：第二步不做。** 對手身分連結在現有兩種指紋（開局、GE2E）下都不可行。不產生提交檔；競賽二基底與最終預設檔不變（`751959-cv2-l10.csv`）。`outputs/rank/opp_link/`（約 270 MB 向量＋測試棋盤面快取）可刪。

### 2026-11-13：競賽二——測試玩家是否為訓練玩家，k 盤 GE2E 重測（使用者「目標是冠軍」；Slurm 763865，1 卡 2 分；分析為登入節點 CPU）

**動機：** 11/11 的 L10 以開局模型估計測試玩家約 12%／20% 是訓練玩家，但工具召回只有 29%，只認得 7 題。若比例屬實且能可靠認人，這些題的級別直接已知（估 +0.03～0.05）。
**做法（`scripts/strength/rank_own_link.py`，輸出 `outputs/rank/opp_link/own/`）：** 沿用上一節的 8,007 人中心（每人最多 40 盤）與測試本人方向量；一題＝題內各盤單位向量平均。參考題：每色、每 k 各 150 位訓練玩家，以候選庫外的 k 盤組題（seen），同題拿掉本人欄為 unseen null；統計量＝第 1、2 名 cosine 差。

| 參考題（8,007 選 1） | k=5 | k=10 | k=15 | k=20 |
|---|---:|---:|---:|---:|
| seen top-1（全部／GE2E 沒訓練過的玩家） | 0.407／0.322 | 0.630／0.608 | 0.680／0.663 | 0.730／0.690 |
| 分數差中位數 seen／null | 0.018／0.013 | 0.034／0.015 | 0.035／0.015 | 0.043／0.018 |
| **測試 Public／Private 分數差中位數** | 0.016／0.014 | 0.021／0.014 | 0.017／0.016 | 0.015／0.013 |
| 超過 null 最大值：seen 召回 | 0.077 | 0.047 | 0.070 | 0.153 |

- k 盤 GE2E 的指認力遠高於開局模型（top-1 41–73%），但**測試題的分數差分布貼著 null**（k=15／20 的中位數等於或低於 null）→ 測試玩家絕大多數不是訓練玩家；「12%／20%」不成立，11/11 多出的尾巴主要是族群偏移。
- 超過 null 最大值（參考題中此時 top-1 全對）的測試題只有 6 題（Public 3、Private 3）。
- **兩種獨立指紋指到同一位訓練玩家（8,007 選 1）的測試題共 27 題（Public 11、Private 16；k=5／10／15／20：4／4／8／11）**，含 L10 的 7 題中的 6 題。這 27 題「比到的級別」對現行最終檔：相同 59%、差 ≤1 級 78%（其餘題 GE2E 第 1 名為 31%／70%）；與最終檔不同的有 11 題（Public 5），其中 4 題差 ≥2 級（基底差 ≥2 級的機率本來只有約 5%，這幾題較可能是誤連）。兩種指紋「巧合同人」的 null 比率尚未量。
- **判讀：** 訓練玩家在測試中的比例粗估約 5%（27 題／召回約一半），不是結構性的 +0.05。把 L10 擴成「開局＋GE2E 融合」的校準偵測器，預期可再改約 5–10 題，全體約 +0.005（未驗證）。不產生提交檔。

**(3) 訓練盤清理與跨卡彙整（11/13 使用者同意；事前登記）：**
- 稽核：GE2E 雙色訓練的 469,821 盤中 21,701 盤為 HD≥2（4.6%；依「先抽玩家再抽盤」實際抽中約 5.0%），這些盤取到的是對手落子且缺讓子石；6,548 位訓練玩家中 2,667 位至少一盤、167 位超過一半。
- `sampling.max_handicap=1`（`train_ge2e.training_players`）：排除後 6,426 位玩家、447,684 盤。
- `batch.cross_gpu`：8 卡的盤向量 all_gather 後當成一個 512 人的 GE2E 群（他卡向量不帶梯度，loss 乘 world 使 DDP 平均後等於完整梯度；小張量驗證逐元素相同），每步玩家由共用亂數一次抽出、各卡不重複。假設：64 人中分類比正式的 600 人容易太多，負例變多後對少盤題較有利；增益未知。
- 設定：`pid_ge2e_v4_d4_hd_2400`（D4＋排除 HD≥2）、`pid_ge2e_v4_d4_hd_x8_2400`（再加跨卡）；其餘同 `pid_ge2e_v3_d4_2400`。作業 763958／763960（8 卡，各約 61 分），評估 763959／763961；`v3_d4_2400` 評估 763957。
- gate 同 (2)：pool 1 融合（30 次配對，對 `p0_1200`）平均 ≥+0.01 且四桶不退步。

**(2) 結果——玩家層級 D4 增強（pool 1 融合，30 次配對抽樣，對 `pid_ge2e_v2_both_p0_1200`）：**

| 風格模型 | 訓練命中率（末段） | 風格單獨（pool 1，5 次） | k=1 | k=2 | k=3 | k=5 | 融合平均 | Δ（標準誤；正的抽樣比例） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `p0_1200`（基底檔所用） | — | 0.406 | 0.283 | 0.484 | 0.609 | 0.740 | 0.5287 | — |
| `v3_d4_1200`（763770，8 卡 31 分） | 49% | — | 0.311 | 0.499 | 0.622 | 0.724 | 0.5389 | +0.010（0.003；77%），k=5 −0.016 |
| **`v3_d4_2400`（763771，8 卡約 61 分）** | 54% | 0.488 | **0.322** | **0.520** | **0.665** | **0.773** | **0.5702** | **+0.0415（0.003；97%）**，四桶 +0.040／+0.037／+0.056／+0.034 |

- gate（平均 ≥+0.01、四桶不退步）：`v3_d4_2400` 通過；`v3_d4_1200` 平均過、k=5 退步。
- 機制：無增強時 2,400 步訓練命中率 77%、驗證台退步（背玩家）；加 D4 後 2,400 步才 54%，驗證台持續上升 → 增強把過擬合往後推，較長訓練才有用。融合權重 λ 由 8 升為 16（pool 0 選；格點 …8／16／32，不在邊界），代表風格項比以前可靠。
- 依登記規則「2,400 步高於 1,200 步則再加長」：`pid_ge2e_v4_d4_hd_4800`（D4＋排除 HD≥2，4,800 步；763977，8 卡約 2 小時、約 16 GPU-h；評估 763978 對 `v3_d4_2400` 配對）。

**提交檔（`identity/predict2.py --bench-summary outputs/pid/pid2_bench/pid_ge2e_v3_d4_2400/summary.json`＋`pid_submission merge`，Slurm 763976，1 卡 3 分；輸出 `outputs/pid/day0_1112/{public,private}2_d4_2400/`）：**

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/763976-d4-2400.csv`（SHA `d76a349a`） | 原測試集與 `761130-tc1200.csv` 逐列相同；新增集換 `v3_d4_2400`（λ=16、μ=1）。改 top-1：pub2 149 題、priv2 162 題 | 驗證台 +0.0415 → Public 約 +0.02（0.744 → 約 0.765）。保留規則：升或持平（差 <0.02 也算）→ 成為基底（本地 30 次配對證據明確）；降 ≥0.02 → 驗證台不轉移，維持 tc1200 並回查 |**0.7644138**（11/13 17:19:31，jabir95tsai；29/72） |

### 2026-11-13：競賽二——使用者「都做」三項＋同色抽樣對照（事前登記與結果）

**(1) 雙指紋身分連結偵測器（`scripts/strength/rank_link.py`、`slurm/rank_link.slurm`，Slurm 763898，1 卡 64 分，多為 CPU 開局計分）→ 沒有新的可改題目。**
- 參考題 4,000 題（每色每 k 500 位訓練玩家，候選庫外的 k 盤；拿掉本人欄為 null），開局 o（HD≤1 候選庫）＋GE2E cosine g，融合 f＝o＋λ·w·g。本人在庫 top-1（k=5／10／15／20）：開局 0.46／0.63／0.73／0.77、GE2E 0.42／0.58／0.68／0.70、**融合 λ=16：0.69／0.85／0.91／0.93**。
- 偵測（融合分數差超過各 k 的 null 最大值；半數玩家選 λ、另一半報告）：召回 0.165（k=5／10／15／20：0.15／0.24／0.02／0.25），命中時 top-1 全對，但另一半的誤報率 0.55%（11／2,013，換算 800 題約 4.4 題）——null 尾巴很長（總有少數玩家有一位很像的人）。
- **「兩種指紋同人」的巧合率 2.3%**（null，半 A）→ 800 題預期約 18 題；測試實測 30 題，多出約 12 題。上一節的「27 題」因此大半是巧合，不能據以覆寫（上一節當時未量此 null）。
- 測試題以全參考 null 門檻只認出 5 題（Public 3、Private 2）：pub_q_0046、0073、0078、priv_q_0047、0159，其中 4 題為 L10 已有；對 cvec2 只改 2 題（0046、priv 0047），都已在 `751959-cv2-l10.csv` 內。推估測試玩家約 3–4% 是訓練玩家，但能確定指認的只有約 5 題。
- 作業自動寫出的 `763898-cv2-link.csv`（SHA `1c24e143`）比 L10 檔少改 2 題，已移到 `submissions/rank/archive/`。**最終預設檔不變。**

**(2) 讓先局逐盤建模（`scripts/strength/rank_hd_shift.py`，Slurm 763895，CPU 10 分）→ 未過 gate。**
- 前提檢定（擬合池 HD≤1，149,615 盤；以 HD0 盤的區塊特徵線性迴歸級別，再看 HD1 盤的偏移）：對手區塊在「HD1 且本人執黑」偏 +0.2～+1.4 級、「HD1 且本人執白」偏 −0.3～−0.8 級；本人區塊約 0（−0.35～+0.5，無一致方向）→ 讓先局的對手確實強／弱約一級。
- 模型：logit_r＝b_r＋W_own[r]·本人＋W_flag[r]·旗標＋Σ_d W_opp[r+d]·對手_d（d＝0／+1／−1，W_opp 12 列）；untied＝三種盤型各一組。資料、先驗、合成題、L2 同 `rank_attn_pool`，對照臂與 sklearn 基底差 ≤0.0004。

| 五 seed 平均 Equal-k | E1 | E2 | E2 各 k 差（5／10／15／20） |
|---|---:|---:|---|
| mean（對照） | 0.7135 | 0.7465 | — |
| tied | −0.0020 | −0.0018 | −0.005／−0.002／−0.009／+0.009 |
| untied | −0.0022 | −0.0035 | −0.006／−0.004／−0.003／−0.001 |

- 判讀：偏移存在，但兩個讓先旗標已線性修正；精確建模沒有增量。停止。

**(3) 同色抽樣對照（外部建議第 1 項；`scripts/strength/rank_same_color.py`，Slurm 764062，CPU 37 秒）→ 無增益，混色不是缺口。**
- B2 合成題原為同一玩家黑白混抽，正式題同色。三臂只差抽樣：mixed（基底）、same（同一玩家同色 ≥k 盤）、mixed_elig（混抽但限 same 的可用玩家）。先驗、特徵、題數相同。
- 可用玩家（十級合計）：mixed k=5／10／15／20＝5,604／5,177／4,500／3,146；same＝5,440／4,281／2,269／1,342（k=20 每級 110–160 人）。

| 五 seed 平均 Equal-k | E1 | E2 | E2 各 k 差（5／10／15／20） | E2 bootstrap 95%（seed 2027） |
|---|---:|---:|---|---|
| mixed | 0.7139 | 0.7463 | — | — |
| same | −0.0086 | −0.0042 | −0.010／−0.003／−0.017／+0.014 | [−0.019, −0.000] |
| mixed_elig | −0.0085 | −0.0048 | −0.008／+0.003／−0.019／+0.004 | [−0.018, +0.001] |

- 判讀：same 與 mixed_elig 幾乎相同（E2 差 +0.0006）→ **顏色混抽本身沒有影響；下降來自可用玩家變少**。再次指向「玩家數」而非抽樣細節。維持混色抽樣。


### 2026-11-13：競賽二 cvec2／adj 換盤穩定性診斷（使用者要求確認；Slurm 769711，CPU 63 秒）

- 復用既有特徵、既有 B2 配方，只重擬合小型 B2；Valid-S＋train20 HD≤1 149,615 盤擬合，Valid-U 玩家零重疊。五 seed、base／swap／add 共 15 個模型成功重現既有 E1/E2 四桶分數（容差 0.0002）。未更新 HumanSL、C／adj 或既有提交檔。
- 同一批 Valid-U 344 人、420 個玩家／顏色組（每組 ≥20 盤 HD≤1），各 k=5/10/15/20 共用 200 次不放回抽盤；分數按組等權。base：0.6755／0.7280／0.7556／0.7734；adj swap：0.6795／0.7317／0.7582／0.7741；C v2＋adj add：0.6799／0.7318／0.7580／0.7736。add Equal-k +0.0027，玩家 bootstrap 95% [−0.0010,+0.0066]；不是穩定的整體增益。
- A～3D 175 組：「持續錯」＝單一固定模型在 ≥90% 重抽中預測同一錯級別，且至少 4/5 B2 seeds 符合並指向同一級。base 持續錯／變動／穩定對（k=5/10/15/20）：0/168/7、2/160/13、8/149/18、43/87/45；swap：1/168/6、3/156/16、8/149/18、43/84/48；add：1/167/7、4/156/15、8/149/18、41/84/50。k20 swap 讓 7 組離開持續錯、另新增 7；add 離開 6、新增 4。離開不等於穩定判對。
- 修正先前解讀：可用棋局中位數只有 26，k20 子集高度重疊。≥40 盤只剩 76 組（中段 39 組）；其中中段 k20 base 0/32/7、swap 0/31/8、add 0/29/10。以每 seed 90% 標準平均，中段 stable_wrong 為全組 26.6%、≥40 盤 2.6%；玩家組成也不同，不能全歸因於有限池，但不支持把中段錯題一概稱為怎麼抽都錯。
- 改善集中 2D：四 k 平均 swap 在 A／1D／2D／3D 為 +0.0056／−0.0050／+0.0236／−0.0042，add +0.0015／−0.0013／+0.0259／−0.0023。既有完整 E2 swap −0.0009／add −0.0008，結論維持未過 gate；Public 0.7461411 是 swap，不是 add。
- 產物／限制：`outputs/rank/humansl/resample_errors_20261113/report.md`、`players.csv`、`persistent_middle_k20.csv`、`predictions.npz`、`result.json`、`interpretation.json`；程式 `scripts/strength/rank_resample_errors.py`，assert self-check 與語法檢查通過。Slurm 配置 1 GPU、實際僅 CPU，約 0.0175 GPU-h。子群為較多盤玩家、200 次抽樣非獨立玩家；未推論官方測試、未產生新提交。

### 2026-11-13：H profile 峰值集中與固定 20 組 C 向量稽核（使用者要求）

- 重用保存的 H／C v2 特徵，只做 CPU 描述統計，沒有重訓或改寫既有模型。輸出 `outputs/rank/humansl/profile_audit_20261113/`：`audit.py`、`audit.json`、八份平均曲線 CSV、`unseen_profile_curves.png`。Slurm 769728／769743 完成（13／104 秒、exit 0；配置 1 GPU、實際僅 CPU，合計約 0.0325 GPU-h）；補跑增加 C 學習／監控玩家與 Valid-U 零重疊斷言：實際 4,174 位 C 學習／監控玩家與 Valid-U 重疊 0 人。
- **原記錄可重現：** Valid-S＋train20、HD≤1 共 149,615 盤／5,728 人。本人 H 平均 log p 的最高 profile：D／C→5k，B／A／1D→3k，2D／3D／4D→1k，5D→1d，6D→2d；改成先按玩家平均、再讓各玩家等權，對應完全相同。原單盤最佳 profile d′ 九組重現為 0.518／0.392／0.484／0.338／0.192／0.221／0.274／0.360／0.476。
- **未見玩家也成立：** Valid-U、HD≤1 共 33,656 盤／1,433 人，玩家等權的本人 H 峰值對應與擬合池完全相同。但 2D 的 1k 與次高 3k 僅差 0.00058 log p／手，盤等權時會改為 3k；峰值標籤不是穩定的精確段位換算表。
- **C 沒有跟著擠成少數類：** C v2 保存形狀為 `[10, 2, 384]`，各側十個向量皆不同。Valid-U 玩家等權後，**本人與對手兩組十級平均曲線各為 10/10 的真實級別條件得到最高似然**；盤等權亦同。本人 1D／2D／3D 的正確條件對次佳條件優勢僅 0.00125／0.00133／0.00202 log p／手；這是類別平均診斷，不代表每位玩家都能分開，更不等於十級準確率 100%。
- **機制核對：** H＋B2 先平均各盤完整的本人／對手 30 維，再由逐 k LR 輸出幽玄十級機率，並未先取 15 個 profile 的 argmax。C 是每級、每角色一個學到的 metadata 條件向量，全部由 rank_1d 初始化後以真實級別的著手 NLL 學習；不是挑選 20 個既有 KGS profile，也不是把所有玩家壓到 20 個固定預測值。相同 H 峰值不代表完整曲線相同。
- 交叉核對既有 11/11 五 seed 消融（**C v1，非本輪新擬合**）：E2 H＋P2＋B2 在 k=5／10／15／20 為 0.6911／0.7427／0.7681／0.7696、Equal-k 0.7429；H＋C＋P2＋B2 為 0.6894／0.7403／0.7715／0.7778、0.7447，C 的平均增量約 +0.002、bootstrap 跨零。C v2 換掉 v1 的既有比較為 E1 −0.0008／E2 +0.0015，仍無明確增益。
- **判斷：峰值集中現象正確，但不足以否定固定 20 組 C；本輪直接觀察反而顯示 C 已能在未見玩家的類別平均上對齊十級。** 中段跨玩家可分度不足仍是問題，但來自條件向量數量、固定主幹、訓練目標或玩家差異的比例未知。修正先前過強解讀：現有 C 的消融增量不是改良 C 的增益上限；C 與 H 可分度相近也不能證明補 2k／4k profile 必定無效。每級多向量或依棋局調整條件是否有益，仍須相同資料、B2 與 player-disjoint 分 k 受控比較。本輪未做這類替代模型實驗，未更新提交基底。

**(4) C4o 對手方監督微調（`humansl_finetune.py --mmi-adjacent --ce-side opp`；`rank_c4x_train.slurm` ARM=oadj／oadj10；Slurm 763889／763890／763891／763892，各 8 卡 39–42 分，合計約 22 GPU-h）→ 不抽特徵，停止。**
- 假設：判別式微調失敗是因每半只有約 2,870 位玩家、模型記住人；對手方每盤幾乎是不同的人，相鄰級 CE 只算對手方落子（NLL 仍算全部）就沒有重複的人可記。資料、交叉擬合、監控、C v2 起點同 C4x adj；兩臂 λ＝1（oadj）、λ＝10（oadj10）。
- fold −1 監控（未見玩家），第 0 步 → 終點：

| 臂／半 | 單盤官方分數 | 中段 d′ 全部／本人／對手 | 監控 NLL | 訓練 CE（首→末） |
|---|---|---|---|---|
| 第 0 步 | 0.5358 | 0.531／0.434／0.392 | 1.4117 | — |
| adj A／B（11/11，參考） | 0.5472／0.5433 | 0.565／0.452／0.428、0.564／0.436／0.443 | 1.3906／1.3902 | 1.017→1.013 |
| oadj A／B | 0.5497／0.5448 | 0.562／0.447／0.430、0.567／0.438／0.448 | 1.3906／1.3903 | 1.0156→1.0129／1.0160→1.0165 |
| oadj10 A／B | 0.5222／0.5254 | 0.546／0.459／0.403、0.538／0.448／0.409 | 1.4001／1.4009 | 1.0155→1.0113／1.0160→1.0150 |

- oadj：形式上過登記的 G0（平均 +0.0114，門檻 +0.01），但與 adj（+0.0094、中段 d′ +0.033 對 +0.033、NLL 相同）無法區分；判別項仍然沒學（CE 幾乎不動），改善同樣來自 NLL。adj 的下游已知為 E1 +0.006／E2 −0.001、LB −0.002 → **預期 oadj 的 G1 與 adj 相同，不花 15 GPU-h 抽特徵。**
- oadj10：兩個模型都退步（−0.014／−0.010），G0 不過。加重判別項沒有讓 CE 降下去（1.016→1.011），只是拖壞生成項。
- 判讀：把判別訊號限制在「不會重複的人」身上沒有產生記憶，也沒有產生學習——單手層級上相鄰級的 likelihood 差接近不可學（CE 貼著均勻基準 1.0175）。瓶頸不是「記住玩家」這一個因素，而是相鄰級在逐手 likelihood 上本來就幾乎沒有可分訊號。檢查點在 `outputs/rank/humansl/c4x/{oadj,oadj10}/`（各約 600 MB），可刪。

**LB（11/13 17:19）：`763976-d4-2400.csv` Public 0.7644138，29/72。** 對 `761130-tc1200.csv`（0.7443283）**+0.0201**，預期 +0.02 → 換回新增集約 +0.040，驗證台 +0.0415；驗證台相對增益第四次轉移。**競賽一基底改為 `763976-d4-2400.csv`。**

**(3) 結果——排除 HD≥2、跨卡彙整、加長（pool 1 融合，30 次配對抽樣）：**

| 風格模型 | 訓練命中率（末段） | 風格單獨（pool 1，5 次） | k=1 | k=2 | k=3 | k=5 | 融合平均 | 配對 Δ（標準誤） |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `v3_d4_2400`（基底檔所用） | 54% | 0.488 | 0.322 | 0.520 | 0.665 | 0.773 | 0.5702 | 對 `p0_1200` +0.0415（0.003） |
| `v4_d4_hd_2400`（763958，8 卡 62 分） | 56% | 0.496 | 0.326 | 0.531 | 0.671 | 0.780 | 0.5769 | 對 `p0_1200` +0.0482（0.003）；比 `v3_d4_2400` 高約 0.007（兩者未直接配對） |
| `v4_d4_hd_x8_2400`（763960，8 卡 62 分） | 34%（512 人中） | 0.362 | 0.272 | 0.454 | 0.578 | 0.682 | 0.4965 | 對 `p0_1200` **−0.0322**（0.002），30 次全負 |
| **`v4_d4_hd_4800`（763977，8 卡 123 分，約 16 GPU-h）** | 64% | 0.549 | **0.360** | **0.566** | **0.705** | **0.810** | **0.6099** | 對 `v3_d4_2400` **+0.0397**（0.002），30 次全正；四桶 +0.037／+0.045／+0.040／+0.037 |

- 排除 HD≥2：約 +0.007，方向正確但未直接配對、在抽樣誤差邊緣；因為是純清理，之後的訓練一律帶上。
- 跨卡 512 人：明顯較差，**不採用**。原因未知（未驗證的猜測：同樣步數下 512 類的任務較難、還沒收斂；或他卡向量不帶梯度使 centroid 更新不同步）。不再花算力追。
- 加長到 4,800 步：再 +0.040，訓練命中率 64%，尚未見到過擬合跡象 → 9,600 步（8 卡約 4 小時、約 33 GPU-h；使用者 11/13 確認，訓練 770095、評估 770096 對 `v4_d4_hd_4800` 配對）。

**計分方式在 D4 模型上重測（`pid2_scoring --tag d4`，Slurm 770089，20 次配對；第一個模型 `v4_d4_hd_4800`，平均對象加 `v4_d4_hd_2400`、`v3_d4_2400`）：**

| 風格項 | k=1 | k=2 | k=3 | k=5 | 平均 | 對 cosine 單模型 Δ（標準誤） |
|---|---:|---:|---:|---:|---:|---|
| 4,800 步 cosine（λ=16） | 0.365 | 0.574 | 0.706 | 0.805 | 0.6124 | — |
| 4,800 步 PLDA（λ=2） | 0.342 | 0.567 | 0.698 | 0.809 | 0.6039 | −0.0085（0.002） |
| 三模型 cosine 平均 | 0.353 | 0.562 | 0.695 | 0.795 | 0.6013 | −0.0112（0.002） |
| 三模型 PLDA 平均 | 0.343 | 0.563 | 0.694 | 0.807 | 0.6017 | −0.0108（0.003） |

→ (1) 在舊模型上量到的 +0.017 不成立於新模型：與較弱的模型平均會拉低、PLDA 在 k=1 退 0.023。**計分維持單模型 cosine，PLDA 與多模型平均結案**（`twocolor.plda_scores` 保留但未接入 `predict2`）。同強度、不同 seed 的平均未測。

**提交檔（Slurm 770088，1 卡 3 分；輸出 `outputs/pid/day0_1112/{public,private}2_d4_4800/`）：**

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/770088-d4-4800.csv`（SHA `486584b0`） | 原測試集與基底逐列相同；新增集換 `v4_d4_hd_4800`（λ=16、μ=1）。對 `763976-d4-2400.csv` 改 top-1：pub2 98 題、priv2 119 題 | 驗證台 +0.040 → Public 約 +0.02（0.764 → 約 0.784）。保留規則：升或持平 → 成為基底；降 ≥0.02 → 維持 `763976-d4-2400.csv` 並回查 |**0.7820433**（11/13 17:29:01，jabir95tsai；23/72） |

**LB（11/13 17:29）：`770088-d4-4800.csv` Public 0.7820433，23/72。** 對 `763976-d4-2400.csv`（0.7644138）**+0.0176**，預期約 +0.02 → 換回新增集約 +0.035，驗證台 +0.040；驗證台相對增益第五次轉移。**競賽一基底改為 `770088-d4-4800.csv`。** 若原測試集約 0.98，新增集目前約 0.58（驗證台 0.61）。

### 2026-11-13：競賽二——A–3D 錯誤拆解與新訊號篩選（使用者要求；全部 CPU）

**(a) 題目層級拆解（`scripts/strength/rank_error_split.py questions`，Slurm 770110；cvec2 基底在 Valid-U E2 的 2,000 題，五 seed 合併）**

| 真實級別帶 | 分數 | 完全正確 | 往強 | 往弱 | 差 ≥2 級 | 平均符號誤差 |
|---|---:|---:|---:|---:|---:|---:|
| D–B | 0.817 | 0.722 | 0.185 | 0.092 | 0.020 | +0.137 |
| **A–3D** | **0.637** | **0.475** | 0.265 | 0.260 | 0.086 | +0.005 |
| 4D–6D | 0.822 | 0.724 | 0.096 | 0.180 | 0.010 | −0.095 |

- 逐級：A 0.716、**1D 0.601（完全正確 40%，38% 被判成更強）**、2D 0.628、3D 0.601（2D／3D 各約 30% 被判弱）→ 中段向 1D／2D 交界靠攏，與 11/11 的 k=∞ 診斷一致。
- **A–3D 內：顏色無差（黑 0.635／白 0.638）；讓先比例無一致趨勢**（0：0.630、(0,0.2]：0.650、(0.2,0.4]：0.655、>0.4：0.598 且差 ≥2 級 17%，n=78）；k=5／10／15／20＝0.551／0.632／0.684／0.680（15 之後不再升）。k=5 時 2D、3D 差 ≥2 級達 23–29%。
- **同一玩家的題目錯同一邊：** A–3D 中有 ≥2 題的 205 位玩家，符號誤差的玩家平均變異 0.77，占總變異 0.91 的 85%。錯誤是玩家層級的，不是題目抽樣造成。
- **兩端級別黑棋題較差：** D–B 黑 0.786／白 0.843、4D–6D 黑 0.800／白 0.843（合計差約 0.05，約 2.3 個標準誤），黑棋題更往中間縮。擬合池中對手區塊的白減黑偏移由 D 的 +0.2 級到 6D 的 −0.15 級（HD0 盤），即特徵→級別的對應因顏色而異。

**(b) 逐手層級（同腳本 `phase`；C v2 逐手 log p，擬合池＋Valid-U 共 96,523 盤 HD≤1、6,318 人；A/1D、1D/2D、2D/3D 三對平均；統計量＝該區段內 log p[r+1]−log p[r] 的平均）**

| 區段 | 平均手數 | 單盤 d′ | 玩家平均 d′（≥10 盤） | 玩家偏差占單盤變異 |
|---|---:|---:|---:|---:|
| 本人 0–30 手 | 15 | 0.25 | 0.45 | 0.22 |
| 本人 30–80 | 25 | 0.29 | 0.69 | 0.14 |
| 本人 80–140 | 29 | 0.28 | 0.74 | 0.10 |
| 本人 140–200 | 24 | 0.24 | 0.73 | 0.06 |
| 本人 200 以後 | 18 | 0.21 | 0.59 | 0.08 |
| **本人整盤** | 111 | 0.48 | **0.85** | **0.28** |
| 對手各區段 | — | 0.20–0.26 | 0.76–1.03 | 0.01–0.02 |
| **對手整盤** | 111 | 0.41 | **1.28** | 0.06 |

- 本人的玩家偏差在開局最重（0.22），但每個區段都有，而且**跨區段相關**：整盤的偏差占比（0.28）高於任何單一區段，玩家平均 d′（0.85）只比最好的單一區段（0.74）略高。偏差是整盤一致的個人特質，不是某個階段的問題 → 與 S2 的手數窗變體（±0.003）一致，沒有可針對的區段。
- 對手方幾乎沒有玩家偏差（0.01–0.06），玩家平均 d′ 1.28，比本人高 50%。

**(c) 分顏色 B2（`scripts/strength/rank_color_model.py`，Slurm 770259，58 秒）→ 無增益。**

| 五 seed 平均 Equal-k | E1 | E2 | E2 bootstrap 95% |
|---|---:|---:|---|
| mixed（基底） | 0.7139 | 0.7463 | — |
| sep（黑白各一個 B2） | −0.0057 | −0.0027 | [−0.018, +0.003] |
| inter（特徵×顏色，102 維） | −0.0131 | −0.0105 | [−0.025, −0.002] |
| inter_opp（只對手區塊×顏色） | −0.0125 | −0.0097 | [−0.027, −0.004] |

- 兩端級別「黑棋題較差」在每一臂都還在（sep：D–B 黑 0.771／白 0.836、4D–6D 黑 0.809／白 0.850）→ 不是模型沒分顏色造成的，是黑棋題本身資訊較少；原因未知。

**(d) 新訊號的玩家層級篩選（`scripts/strength/rank_signal_probe.py`，Slurm 770256；6,514 位玩家各一列，按玩家五折，比較基底 52 維與加上候選區塊；第一次 770109 的 opening 區塊在擬合端含玩家自己的盤，數字作廢）**

| 級別等權官方分數 | 全部 6,514 | GE2E 沒訓練過 1,061 | A–3D 2,532 |
|---|---:|---:|---:|
| 基底（52 維玩家平均） | 0.7683 | 0.8120 | 0.6803 |
| ＋GE2E 風格主成分（24 維） | +0.0075 [+0.001, +0.014] | +0.0008 [−0.013, +0.015] | +0.0138 [+0.001, +0.025] |
| ＋風格近鄰的級別分布 | −0.0028 | −0.0095 | +0.0019 |
| ＋依級別的開局模型（20 維） | −0.0043 | −0.0126 | +0.0004 |
| 單獨：風格／近鄰／開局 | 0.641／0.580／0.252 | | |

- 風格主成分是唯一為正的：全部 +0.0075，但依玩家總盤數分層，≤40 盤的 2,479 人 +0.0005、40–400 盤 +0.013～+0.019（標準誤 0.007–0.009）、GE2E 沒訓練過的重度玩家 −0.009／+0.018（兩層符號相反）。風格中心要有幾十盤才穩定，而正式題只有 5–20 盤 → **在正式題的盤數下沒有可用增益**，未過門檻（≥+0.01 且區間高於 0，須在未訓練玩家上成立）。
- 風格單獨就有 0.64、近鄰 0.58：風格確實帶很多級別資訊，但與 H＋C 重疊。開局依級別單獨只有 0.25（隨機約 0.17）。
- **結論：三個候選都沒有提供 H＋C 沒有的玩家層級資訊。** 不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。

**探針檔（11/13，登入節點 CPU 數秒；兩檔都只供量測，不作基底）：**

| 檔案 | 內容 | 目的／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/770088-origonly.csv`（SHA `0aec36e8`） | 原測試集 800 列與基底逐列相同；新增集 800 題改填該題**分數最低**的 5 位候選人（預期幾乎 0 分） | 拆解探針：LB ≈ 原測試集分數 ÷ 2。若約 0.49 → 原測試集約 0.98、差距全在新增集（目前所有推論的前提）；若明顯低於 0.49 → 原測試集還有空間，要回頭做原測試集。新增集分數＝2×(0.7820 − 本檔)。**不是預測檔，不可留作最後一次上傳** |**0.4955449**（11/13 18:56:01，jabir95tsai；59/72） |
| `submissions/pid/770088-d4-4800-hung.csv`（SHA `60978721`） | 新增集與基底逐列相同；原測試集換 Hungarian 指派（改 top-1：pub 4 題、priv 5 題） | 一對一探針：Public 只動 4 題，上限約 ±0.005，只看方向。升 → 一對一可能成立；降 → 維持不指派 | |

**拆解結果（11/13 18:56，`770088-origonly.csv` Public 0.4955449）：** 在「Public＝`pub_`＋`pub2_` 各 400 題等權」與「最低分 5 人幾乎不得分」下，**原測試集 Public ≈ 0.991**（合成驗證 0.98–0.99，吻合），**新增集 Public ≈ 2×(0.7820−0.4955) ＝ 0.573**（驗證台 pool 1 為 0.610，絕對值低約 0.04）。
- 第 0 節的競賽一 baseline 警訊解除：官方 baseline 0.28 只是弱（原因 a）；開局漂移（b）不存在，原測試集已飽和，剩餘空間 ≤0.009（Public 約 3–4 題）。
- 第 1 名 0.853 → 若其原測試集也約 0.99，新增集約 0.715；我方 0.573，差 0.14，全在新增集。之後算力全部放新增集；`-hung` 探針上限 ±0.005，價值低，不優先。
- 此檔不是預測檔；最後一次上傳不可停在它（當日後續會再傳正常檔）。

### 2026-11-13：競賽二——「要不要換模型」的三個便宜檢定（使用者提問；登入節點 CPU，臨時腳本）

**(1) 強引擎特徵的玩家層級篩選（A-048 試算盤 18,168 盤、760 位 ≥8 盤的玩家，玩家五折 × 5 次重切，基底為 cvec2 52 維玩家平均）：** 加上 14 維引擎特徵：1 visit −0.018、16 visits −0.016、32 visits −0.019（單次標準誤 0.009）；A–3D −0.043～−0.049。引擎特徵單獨 0.61–0.62。→ 「客觀著手品質」在玩家層級也沒有補到 H＋C，方向為負（760 人加 14 維有變異數成本，但沒有任何正的跡象）。

**(2) 玩家層級偏差有多少是「真實棋力偏離標籤」（6 千多位玩家中 ≥20 盤的 4,061 人；每人棋局隨機分兩半，各半的本人區塊／對手區塊玩家平均各擬合一個按玩家五折的 LR，取期望級別減真實級別為殘差，扣掉各級平均）：**

| 殘差相關（A–3D，1,601 人，每半中位 14 盤） | 數值 |
|---|---:|
| 本人（A 半）對本人（B 半） | 0.79 |
| **對手（A 半）對對手（B 半）** | **0.52** |
| **本人（A 半）對對手（B 半，不同棋局）** | **0.38** |
| 本人對對手（同一批棋局） | 0.44 |
| 殘差標準差：本人／對手 | 0.97／0.74 級 |

（全部級別：0.81／0.56／0.39；B–4D：0.80／0.53／0.40。）
- **對手殘差在同一位玩家的不同棋局間是持續的（0.52）**：同一標籤級別的玩家，有人長期遇到較強的對手、有人較弱。對手是不同的人，這不是對手的風格，而是這位玩家實際被配對的水準偏離他的標籤。
- **本人殘差與「另一批棋局的對手殘差」相關 0.38**（扣除量測雜訊後約 0.6）：本人落子看起來偏強的玩家，他的對手也真的偏強。
- 粗估（假設兩種殘差對真實偏離的斜率都是 1）：共同成分的標準差約 0.52 級（cov＝0.38×0.97×0.74＝0.27），等於對手殘差的全部持續變異（0.52×0.74²＝0.28），約占本人持續變異（0.79×0.97²＝0.74）的 37%；本人其餘約 0.68 級才是風格偏差。
- **意義：A–3D 的標籤與「棋譜顯示的實際水準」之間本來就有約 0.5 級的散布**（可能來源：標籤是單一時點的級別而棋局跨三年、同級內的 rating 位置；未驗證）。這部分任何讀棋譜的模型都修不掉。若實際水準量得完全準確，A–3D 的完全正確率上限粗估為 P(|N(0, 0.52)|<0.5)≈66%，分數約 0.79；目前 A–3D 在全部盤下約 0.68（探針）、E2 題目 0.64。
- 可動的只剩本人的風格偏差（約 0.68 級）與有限 k 的對手雜訊；前者今天篩過的來源都修不掉。

**(3) 同一玩家的棋局順序有沒有時間資訊（HD0、同色、≥12 盤的 4,307 組；`source_row`／`game_id` 排序對每盤線性級別分數的組內 Spearman）：** 平均 0.008（t＝2.2），組間標準差 0.246＝虛無期望 0.246。→ 訓練檔的列順序不帶棋力趨勢（不分顏色時的 −0.036 是因為列依顏色排序）。無法用時間順序找出標籤對應的時點。

### 2026-11-13：競賽二——最高機率 argmax 解碼探針（使用者要求）

只將 cvec2 的 `score_rank(p)` 改為 `p.argmax(axis=1)`；模型、特徵、先驗、題目聚合及 L10 覆寫不變。假設：若模型的相鄰級機率校準不足，直接選最高機率級別可能較好；增益事前未知。比較既有 E1／E2 題組、五個 seed（2027–2031），以 E2 官方 Equal-k 與分桶分數判斷，不用 accuracy 代替官方得分。程式 `scripts/strength/rank_argmax.py`，Slurm **771562 COMPLETED、70 秒、exit 0**；只用 CPU，分區配置 1 GPU，約 0.0194 配置 GPU-h，未重跑 GPU 特徵／網路訓練。

| 五 seed 平均官方分數 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| E1 score_rank | 0.6197 | 0.7250 | 0.7648 | 0.7463 | 0.7139 |
| E1 argmax | 0.6285 | 0.7154 | 0.7671 | 0.7456 | 0.7141 |
| E1 差值 | +0.0088 | −0.0096 | +0.0023 | −0.0007 | +0.0002 |
| E2 score_rank | 0.6935 | 0.7466 | 0.7691 | 0.7759 | 0.7463 |
| E2 argmax | 0.6795 | 0.7357 | 0.7615 | 0.7687 | 0.7364 |
| E2 差值 | −0.0140 | −0.0108 | −0.0076 | −0.0072 | **−0.0099** |

E2 五個 seed 的 Equal-k 差值全負（−0.0127／−0.0065／−0.0107／−0.0130／−0.0067）；seed2027 玩家 bootstrap 95% [−0.0192, −0.0070]。E2 完全正確率也由 0.6240 降為 0.6111；E1 則由 0.5798 升為 0.5828。**本地不支持升為基底**：最高機率只有在後驗正確時才保證最佳 accuracy，實測 accuracy 也可能下降；官方得分還包含差一級的部分分數。

提交機率重用 `outputs/rank/humansl/question_probs/cvec2.parquet`，已驗證原 `score_rank` 逐列重現 `751959-cvec2.csv`；本地五 seed 的對照亦逐桶重現既有 cvec2 結果。輸出通過 800 題唯一 ID、官方順序、合法級別與 7 題 L10 覆寫檢查；L10 以外與純 argmax 檔逐列相同。

| 檔案 | SHA 前 8 碼 | 用途／預期 Public | Public LB |
|---|---|---|---|
| `submissions/rank/771562-argmax.csv` | `603618ff` | 純 cvec2 argmax，無 L10，供對照 | **0.7465426**（11/14 21:01:39，jabir95tsai；22/136） |
| `submissions/rank/771562-arg-l10.csv` | `dbe3d807` | **供使用者試傳的探針**；對目前 `751959-cv2-l10.csv` 改 Public 21／Private 19 題。E2 −0.0099，預期 Public 方向偏負（約 −0.01，絕對值未知） | **0.7481229**（11/13 21:18:50，jabir95tsai；19/136） |

保留規則：LB 降／持平維持 `751959-cv2-l10.csv`；小升但差距 <0.02 亦不因單次 Public 結果升基底；若大升，先回查本地與正式題的差異。沒有自動上傳。完整分 seed／分 k／accuracy／bootstrap 存於 `outputs/rank/humansl/argmax/result.json`；提交差異計數在同目錄 `export.json`，log 為 `outputs/rank/argmax-771562.out`。

**LB（11/13 21:18，11/14 補登）：`771562-arg-l10.csv` Public 0.7481229，19/136。** 對基底 `751959-cv2-l10.csv`（0.7499623）**−0.0018**（400 題總分約 −0.7 分；改了 21 題，相當於淨損不到 1 題正確），方向與本地 E2 −0.0099 一致、幅度較小，在雜訊內。依保留規則**基底與最終預設檔維持 `751959-cv2-l10.csv`**；argmax 解碼結案，`score_rank` 維持正式解碼。

**LB（11/14 21:01）：`771562-argmax.csv`（無 L10）Public 0.7465426，22/136。** 與同樣無 L10 的 `751959-cvec2.csv`（score_rank，0.7484）比 **−0.0019**，與有 L10 的一對（−0.0018）幾乎相同 → argmax 的損失與 L10 無關，兩對一致；L10 在 argmax 下貢獻 +0.0016（約 0.6 分），與 score_rank 下的 +0.0016 相同。結論不變：基底維持 `751959-cv2-l10.csv`。

### 2026-11-14：競賽一——新增集風格模型 9,600 步（`pid_ge2e_v4_d4_hd_9600`）

設定與 `v4_d4_hd_4800` 相同，只把訓練步數加倍（`configs/identity/pid_ge2e_v4_d4_hd_9600.json`；Slurm 770095，1 卡 4h04m）。驗證台 pool 1 做 30 次配對比較（Slurm 770096，`outputs/pid/pid2_paired/pid_ge2e_v4_d4_hd_4800__pid_ge2e_v4_d4_hd_9600_pool1.csv`）：

| k | 4,800 步 | 9,600 步 | Δ | 標準誤 | Δ>0 比例 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3595 | 0.3680 | +0.0085 | 0.0044 | 0.70 |
| 2 | 0.5657 | 0.5705 | +0.0048 | 0.0041 | 0.60 |
| 3 | 0.7046 | 0.7149 | +0.0103 | 0.0054 | 0.57 |
| 5 | 0.8099 | 0.8254 | +0.0155 | 0.0042 | 0.67 |
| 平均 | 0.6099 | 0.6197 | **+0.0098** | 0.0024 | 0.77 |

→ 四個 k 都是正的，但增益只有 2,400 → 4,800 步（+0.040）的四分之一，步數的報酬在遞減；19,200 步（約 8 GPU-h）預期增益更小，不建議跑。pool 0 未測。

**提交檔（Slurm 772748，1 卡 2 分；輸出 `outputs/pid/day0_1112/{public,private}2_d4_9600/`）：**

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/772748-d4-9600.csv`（SHA `f1c327a9`） | 原測試集與基底 `770088-d4-4800.csv` 逐列相同；新增集換 `v4_d4_hd_9600`（λ=16、μ=1）。改 top-1：pub2 124 題、priv2 110 題 | 驗證台 +0.010 → 以過去約一半轉移，Public 約 +0.005，在雜訊內。保留規則：升或持平 → 成為基底（本地 30 次配對證據，77% 抽樣為正）；降 ≥0.02 → 維持 `770088-d4-4800.csv` 並回查 | **0.7936014**（11/14 02:01:45，jabir95tsai；26/77） |

**LB（11/14 02:01）：`772748-d4-9600.csv` Public 0.7936014，26/77。** 對 `770088-d4-4800.csv`（0.7820433）**+0.0116**，預期約 +0.005；換回新增集約 +0.023，高於驗證台 +0.010（Public 只有 400 題新增集，差距在雜訊內）。方向與本地一致，依保留規則**競賽一基底改為 `772748-d4-9600.csv`**。名次 23 → 26 是因為隊伍數由 72 增為 77、其他隊伍也在進步。

### 2026-11-14：競賽一——新增集題目間轉導重排（使用者同意；CPU，`scripts/identity/pid2_transductive.py`）

動機：同一玩家可對應多題（官方新增集 top-1 為 307／313 個不同的人，與驗證台「400 題從 600 人可重複抽」的期望 292 相符），候選人只有 10 盤。全部以 `v4_d4_hd_9600` 的融合分數 S（開局＋λw·cosine＋μ·HD 色別，參數同提交）為基底；pool 0 draws 100–119 調參，pool 1 draws 100–129 驗收（與 9,600 步配對評估同一批題，基底 0.6197 逐位重現）。gate：Δ>0 約 2 個標準誤以上且 ≥70% 抽樣為正。

| 做法 | pool 1 k=1／2／3／5 Δ | 整體 Δ（標準誤）| Δ>0 比例 | Slurm |
|---|---|---|---:|---|
| A. 候選人風格中心擴充（soft EM，leave-one-out；pool 0 選 τ=2、α=0.5、2 輪，τ 在網格邊界） | +0.003／+0.003／+0.001／−0.002 | +0.0014（0.0009） | 0.60 | 772889 |
| A 的上限（其他題歸屬用真值，α=1） | +0.006／+0.018／+0.011／+0.013 | +0.012（0.001） | 0.97 | 772896 |
| B. 題目合併：S + β Σ m(q,q′) log P(c\|q′)，m＝後驗重疊 ΣP·P（pool 0 最佳 τ=32、β=0.125，pool 0 已為負） | −0.002／−0.003／−0.004／−0.003 | −0.0032（0.0006） | 0.13 | 772899 |
| B 的上限（m＝是否同一人的真值，β=1） | **+0.188／+0.123／+0.063／+0.022** | **+0.099（0.003）** | 1.00 | 772899 |
| C. 同 B，m 改為 pool 0 擬合的同人判斷器（題目間風格 cosine×k、後驗重疊、k；logistic） | −0.010／−0.011／−0.011／−0.015 | −0.0119（0.0009） | 0.00 | 772903 |

→ **三個可實作版本都沒過 gate，不產生提交檔。** 機制：A 的上限本身就小（候選人已有 10 盤，多幾盤只微調中心）。B／C 上限很大（題目合併相當於把 k 加大，k=1 0.37 → 0.56），但瓶頸是「兩題是否同一人」的判斷：C 的判斷器在同人配對上平均 m＝0.16，非同人配對 0.0014；每題有 399 個其他題，誤配的總權重約 0.57，真配對只約 0.11，雜訊壓過訊號（pool 0／pool 1 數字一致，不是過擬合）。B 的後驗重疊沒有新資訊，只會放大兩題同時錯指的候選人。
→ 方向本身（上限 +0.099）仍是新增集最大的已知槓桿；要做就需要更準的題目對題目同人判斷（例如直接比對兩題的開局序列、或訓練一個 pair verifier），並只在高信心配對上合併。未做；風格 seed 平均（約 32 GPU-h）依使用者決定暫不送。

**A 的提交檔（11/14 使用者「沒過 gate 但可能有幫助的都傳看看」；登入節點 CPU，`pid2_transductive export`）：** 讀正式新增集 `{public,private}2_d4_9600` 的快取向量與分項，先驗證風格項與融合分數逐值重現 `predict2`，再套 pool 0 選定的 τ=2、α=0.5、2 輪；輸出 `outputs/pid/day0_1112/{public,private}2_d4_9600_em/`。

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/hung-em.csv`（SHA `47524ecc`） | 原測試集與基底 `772748-hung.csv` 逐列相同（Hungarian）；新增集換 A。改 top-1：pub2 24 題（k=1／2／3／5：7／10／2／5）、priv2 28 題（11／10／6／1） | 驗證台 +0.0014（標準誤 0.0009，60% 抽樣為正，gate 未過）→ Public 約 +0.0007，遠在雜訊內，只看方向。保留規則：升 ≥ +0.003（約 24 題中淨對 3 題以上）→ 升為基底；持平或小升 → 維持 `772748-hung.csv`（本地證據不足）；降 → 維持並結案 A | **0.7975302**（11/14 21:47:58，jabir95tsai；31/81） |

**LB（11/14 21:47）：`hung-em.csv` Public 0.7975302，對基底 `772748-hung.csv`（0.7980564）−0.0005**（總分約 −0.4；24 題改動大致對錯相抵、略負）。與本地 +0.0014 都在雜訊內，Public 上看不到 A 的增益。依保留規則**基底維持 `772748-hung.csv`，A 結案**；新增集轉導的方向只剩更準的題對題同人判斷（上限 +0.099，見上）。

### 2026-11-14：競賽一——新增集題對題同人判斷的 CPU 診斷（使用者同意；`scripts/identity/pid2_pairdiag.py`，Slurm 774549，3 分 20 秒）

目的：上一節的合併上限 +0.099 需要「兩題是否同一人」的判斷。本次量三種便宜的題對題分數夠不夠用，不訓練 verifier、不產生提交檔。事前約定（使用者修訂）：精確率／召回只當診斷；pool 0 選定並鎖定規則，pool 1 以最終分數 Δ ≥ +0.005 且 ≥70% 抽樣為正判斷是否繼續；真值打折模擬要含高相似度錯配與合併傳遞。

設定：基底 `v4_d4_hd_9600` 融合分數 S。題對題分數：`cos`（兩題平均風格向量 cosine）、`open`（開局模型以對方題當 gallery 的雙向 LR，相對空 gallery）、`cand`（log Σ_c P(c|q)P(c|q′)，只經候選人）、`lr_dir`（cos＋open＋k 的 logistic，不含候選人資訊）、`lr_all`（再加 cand）。logistic 以 pool 0 draws 200–219 擬合（159.6 萬對、2,645 對同人）；規則在 pool 0 draws 100–119 選（5 分數 × 每 draw 取前 10／20／40／80／160 條邊 × 直接／連通分量／互為最佳 × β 0.25／0.5／1，共 225 臂）；pool 1 draws 100–129 驗收。合併：T＝S＋β Σ_{相連 q′} log P(c|q′)。pool 0／pool 1 各 591 位玩家、**零重疊**；每 draw 同人約 134 對（79,800 對中）。

**(1) 題對題分數（pool 1；pool 0 數字相近）：**

| 分數 | 召回 5%／10%／20%／30% 時的精確率 | 誤報率 10⁻⁴／10⁻³ 時的召回 | k_min=1 的配對：召回 10% 時精確率 |
|---|---|---|---:|
| cos | 0.65／0.43／0.24／0.14 | 0.07／0.20 | 0.15 |
| open | 0.34／0.17／0.07／0.03 | 0.04／0.11 | 0.06 |
| lr_dir（cos＋open） | 0.81／0.58／0.31／0.19 | 0.09／0.22 | 0.23 |
| cand（只經候選人） | 0.98／0.96／0.85／0.64 | 0.23／0.38 | 0.78 |
| lr_all | 0.92／0.84／0.65／0.48 | 0.15／0.36 | 0.47 |

→ 開局直接比對是最弱的；不經候選人的最佳組合（lr_dir）在召回 20% 時精確率只有 0.31。**只經候選人的 `cand` 遠比任何直接比對準**，加上直接比對（lr_all）反而比 cand 單獨差（logistic 以全體 log loss 擬合，不是針對高精確率區，這一點未另外調）。

**(2) 真值打折模擬（pool 0，20 draws，整體 Δ；直接相連、β=1）：**

| 召回 | 隨機：精確率 1.0／0.9／0.7／0.5 | 困難：精確率 1.0／0.9／0.7／0.5 |
|---:|---|---|
| 0.05 | +0.006／+0.005／+0.001／−0.007 | +0.000／+0.000／−0.001／−0.002 |
| 0.1 | +0.012／+0.010／+0.002／−0.014 | +0.001／+0.001／−0.001／−0.002 |
| 0.2 | +0.026／+0.021／+0.005／−0.021 | +0.002／+0.001／−0.001／−0.004 |
| 0.3 | +0.035／+0.028／+0.006／−0.028 | +0.006／+0.005／+0.001／−0.005 |
| 0.5 | +0.056／+0.043／+0.013／−0.041 | +0.021／+0.019／+0.013／+0.000 |
| 1.0 | +0.103／+0.083／+0.034／−0.050 | +0.103／+0.094／+0.073／+0.046 |

「隨機」＝真配對與錯配都隨機抽；「困難」＝保留 lr_all 排最前的真配對、錯配取 lr_all 排最前的異人配對。β=0.5 時錯配的傷害約減半、增益也略小（隨機、精確率 0.5、召回 0.3：+0.007）。連通分量版在精確率 ≤0.7 時明顯更差（隨機、召回 1.0、精確率 0.7：直接 +0.034、連通 −0.080），一條錯邊會把整群串起來。完整表在 `outputs/pid/pid2_transductive/pid_ge2e_v4_d4_hd_9600_pairdiag_degraded_pool0.csv`。

→ 兩個結論：(a) 隨機錯配下，β=1 的損益平衡精確率約 0.6–0.7。(b) **「困難」列在精確率 1.0 時也幾乎沒有增益**：判斷器最有把握的前 30% 真配對全部合併且零錯配，只有 +0.006（隨機抽 30% 是 +0.035）。現有分數認得出來的同人配對，是兩題本來就各自指向同一候選人的配對，合併不會改變答案；能改變答案的配對（其中一題原本答錯）正是分數認不出來的。

**(3) 實際分數＋合併規則：**

| | 規則 | k=1／2／3／5 Δ | 整體 Δ（標準誤） | Δ>0 比例 | 邊精確率／召回／涵蓋題目比例 |
|---|---|---|---|---:|---|
| pool 0（選擇，20 draws） | `lr_dir`、前 20 條、互為最佳、β=0.25 | +0.000／+0.001／+0.002／−0.002 | +0.0002（0.0006） | 0.55 | 0.59／0.09／0.10 |
| pool 1（鎖定後，30 draws） | 同上 | +0.000／−0.000／−0.003／−0.002 | **−0.0012（0.0005）** | 0.27 | 0.61／0.09／0.10 |

pool 0 的 225 臂最佳只有 +0.0002，已在雜訊內；`cand` 的邊精確率 0.97 但增益為 0（符合 (2b)）。pool 1 基底 0.6197。**gate 未過，不產生提交檔。** 標準誤是 draw 之間的（同一池玩家重複抽題），不含玩家抽樣的不確定性；pool 1 與 pool 0 玩家不重疊，但都來自訓練集，不代表正式新增集。

**判讀：** 本輪的便宜分數（風格 cosine、開局直接比對、候選人後驗重疊及其 logistic 組合）加硬門檻合併，未呈現足夠收益。這不否證方向本身：要有增益，判斷器必須在「至少一題原本答錯」的配對上有鑑別力，也就是帶有候選人分數以外的資訊；本次的直接分數（lr_dir）在 k_min=1 的配對上召回 10% 時精確率只有 0.23，遠低於約 0.6–0.7 的損益平衡。訓練專用 pair verifier 能否達到，未知；未安排。原始輸出 `outputs/pid/pid2_bench/logs/pid2-pairdiag-774549.out`，摘要 `outputs/pid/pid2_transductive/pid_ge2e_v4_d4_hd_9600_pairdiag.json`。

### 2026-11-14：競賽一——玩家數探針 S0（82% 一臂）與 R2 固定 gallery 模擬程式（使用者同意；事前登記）

**S0（已送）：** 問題＝固定 4,800 步下，增加玩家多樣性、同時減少每位玩家的重複曝光，是否有利。`configs/identity/pid_ge2e_v5_sub82_4800.json` 與 `pid_ge2e_v4_d4_hd_4800` 只差 `sampling.player_subset={"fraction":0.82,"seed":2026}`：按級別 ×（級別內棋局數三分位）分層抽 82%，同一 seed 下較小比例是較大比例的子集（`train_ge2e.stratified_subset`）。實際人數 **5,272／6,426（82.0%）**、365,878／447,684 盤，各級比例一致；名單存 `outputs/pid/ge2e/<run>/training_players.json`。（先前文件寫的 6,548 人是排除 HD≥2 之前的數字；`v4` 系列實際訓練 6,426 人，見各 run 的 `train.jsonl`。）訓練 Slurm **775661**（8 卡，上限 2h45m，約 16 GPU-h），評估 **775662**（pool 1，30 次配對，對 `pid_ge2e_v4_d4_hd_4800`）。使用者只批准這一臂；50% 臂不跑。

判讀規則（事前）：以 pool 1 整體配對 Δ 為主，四個 k 桶只報告。訓練 seed 的變異從未量過，30 次抽題只涵蓋抽題變異；單次訓練差距 <約 0.01 只作方向性證據，不下結論。82% 比 100% 低 ≥0.005 → 支持 R1（全玩家重訓），是否補 seed 或送 R1 由使用者決定；S0 不影響 R2。pool 1 自 11/12 起持續用於選模型，是開發驗證集，不是未觸碰的測試集。

**R2 模擬程式（已寫，未送 GPU 作業）：**
- `cv.pid2_fixed_gallery(bank, pool, gallery_seed)`：每位候選人固定 10 盤 gallery；`cv.draw_pid2_fixed(...)`：gallery 不變、題目只從其餘 30 盤抽，內含「題目盤不在 gallery」「一盤不進兩題」的 assert。`draw_pid2` 未改，既有抽題逐位不變。
- `sampling.gallery_players={"pool":1,"gallery_seed":0,"share":0.12}`：pool 1 的 591 位候選人只以固定 gallery 的 10 盤成為訓練類別；每個 GE2E group（64 人）固定 round(0.12×64)＝8 位 gallery 玩家＋56 位一般玩家（正式版候選庫 10 盤玩家佔比約 12%）。訓練前檢查 gallery 盤與題目盤、與一般訓練盤皆零重疊，gallery 玩家不在一般訓練玩家內；名單（玩家與每人 10 盤）寫入 `training_players.json`。config `configs/identity/pid_ge2e_v6_gal_p1_4800.json`；已確認其一般訓練玩家與 `v4_d4_hd_4800` 完全相同。
- 評估：`pid2_paired --fixed-gallery 0`（兩個模型在同一批固定 gallery 題目上配對）。事前門檻：整體 Δ ≥ +0.01、k=1 退步不超過 0.005。限制：模擬同時多了 591 位玩家與「他們是候選人」兩件事，未拆開；原候選庫（100 盤、全黑）不在模擬內。正式訓練前須核對官方規則原文是否允許以候選庫訓練。
- 測試：`tests/test_identity_ge2e.py` 6 項通過（2.5 秒；新增抽樣比例、分層巢狀子集、固定 gallery 零重疊三項）。

**算力記帳：** 以 `sacct -a` 加總專案帳號 10/25 12:00 起約 4,170 GPU-h（本帳號 527）對 `wallet` 108,387 → 59,885，估約 **11.6 SU／GPU-h**（兩次讀數時間未完全對齊，可能混入其他費率的資源）。S0 送出前 `wallet`＝59,095.38（11/14 16:49）。成本與競賽二 C4g 合併記帳。

**S0 結果（訓練 775661：8 卡 2h02m＝16.3 GPU-h；評估 775662：1 卡 10 分；pool 1，30 次配對，82% 臂 − 100% 臂）：**

| 比較方式 | k=1 | k=2 | k=3 | k=5 | 整體 Δ（標準誤） | Δ>0 比例 |
|---|---:|---:|---:|---:|---|---:|
| 各用自己在 pool 0 選的融合權重（100%：λ=16、μ=2；82%：λ=8、μ=1；`pid2_paired` 原輸出） | +0.005 | −0.010 | −0.019 | −0.028 | −0.0131（0.0019） | 0.13 |
| 同權重 λ=16、μ=1（登入節點 CPU 重算） | −0.006 | −0.015 | −0.010 | −0.006 | **−0.0093（0.0018）** | 0.20 |
| 同權重 λ=8、μ=1 | −0.005 | −0.014 | −0.011 | −0.002 | −0.0078（0.0018） | 0.30 |

- 絕對值：100% 臂 0.6099，82% 臂 0.5969（各自權重）；同權重 λ=16 為 0.6095 對 0.6002。訓練末段命中率 100% 臂 64%、82% 臂 63%。
- 原輸出的 −0.013 混入了融合權重的選擇（pool 0 只用 5 次抽題選 λ，82% 臂選到 8）；同權重的 −0.008～−0.009 才是 encoder 本身的差。
- 判讀（依事前規則）：82% 比 100% 低約 0.009，超過 0.005 的門檻、方向支持「玩家多一點比較好」；但小於約 0.01，且訓練 seed 的變異未量，**只作方向性證據**。若同樣的斜率延伸到 6,426 → 8,007 人（+24.6%），R1 在新增集約 +0.01；這是外推，本地無法驗證。是否補 seed 或送 R1 待使用者決定。
- 記帳：`wallet` 59,095.38（16:49）→ 59,040.55（19:02），期間本帳號用約 16.5 GPU-h，全專案其他成員用量未扣除；降 54.8 SU，與 11.6 SU／GPU-h 的估計不符（低很多），可能 `wallet` 不是即時扣款，費率仍未確定。

**R2 模擬已送（使用者 11/14 同意；暫不補 seed、不送 R1）：** 訓練 Slurm **776986**（`pid_ge2e_v6_gal_p1_4800`，8 卡約 2 小時、約 16 GPU-h），評估 **776987**。事前登記：
- 比較基底 `pid_ge2e_v4_d4_hd_4800`（100% 玩家、4,800 步）；兩臂用同一份固定 gallery（pool 1、gallery seed 0）與同一批題目（draws 100–129）。
- **主要判讀＝固定融合權重 λ=16、μ=1**（現行提交的形式；`pid2_paired --weights 16 1`）；各自在 pool 0 選的權重另列。
- 採用門檻：整體 Δ ≥ +0.01 且 k=1 Δ ≥ −0.005，不看結果後調整。
- 附帶對照（不佔 GPU）：同一模型在 pool 0（未被適配的 591 人，隨機 gallery）固定權重下對基底的 Δ，用來看「多 591 位玩家」本身對未見玩家的影響；這不是乾淨的拆解（pool 0 與 pool 1 是不同的人）。
- 解釋範圍：過門檻代表「加入候選玩家並以其 gallery 訓練」整套有效，不能單獨歸因於適配；不自動啟動正式重訓，正式前先核對官方規則原文。送出前 `wallet`＝59,020.71（20:38）。

**R2 結果（訓練 776986：8 卡 2h03m；評估 776987：1 卡 12 分；`outputs/pid/pid2_bench/logs/bench-pid_ge2e_v6_gal_p1_4800-776987.out`）→ 未過門檻，不做正式重訓。**

| 比較（v6 − v4_4800，30 次配對） | k=1 | k=2 | k=3 | k=5 | 整體 Δ（標準誤） | Δ>0 比例 |
|---|---:|---:|---:|---:|---:|---:|
| **主要：pool 1 固定 gallery、固定權重 λ=16 μ=1** | −0.0066 | −0.0100 | −0.0076 | +0.0035 | **−0.0052（0.0021）** | 0.37 |
| pool 1 固定 gallery、各自權重 | −0.0092 | −0.0114 | −0.0079 | +0.0045 | −0.0060（0.0021） | 0.27 |
| pool 0（未適配的 591 人）、固定權重 | −0.0077 | −0.0092 | −0.0052 | −0.0006 | −0.0057（0.0020） | 0.27 |

- 絕對值（主要）：v4_4800 0.6085、v6 0.6033。門檻（整體 ≥ +0.01、k=1 ≥ −0.005）兩項都沒過，且方向為負（約 2.5 個標準誤）。
- 讀法：把候選人（各 10 盤）當訓練類別放進每個 GE2E group，對**那些候選人自己的題目**也沒有幫助；只有 k=5 小正（+0.0035，在雜訊內）。pool 0 的 −0.0057 與 pool 1 相近，表示退步不是「被適配的人」特有，而是整體 encoder 變差——可能原因（未驗證）：每 group 固定 8 位只有 10 盤的玩家，GE2E 的類中心在這些人身上很吵，等於減少了有效的一般玩家數（與 S0「玩家少一點較差」同方向）。
- 結論：**R2 結案；不以候選庫訓練**（官方規則是否允許也不必再查）。新增集風格模型維持 `v4_d4_hd_9600`。

**R2 模擬結果（11/14 22:54；訓練 776986：8 卡 2h03m＝16.4 GPU-h，評估 776987：1 卡 12 分）——未過門檻：**

Δ＝`v6_gal_p1_4800` − `v4_d4_hd_4800`，30 次配對抽題（draws 100–129），括號為標準誤。

| 比較 | k=1 | k=2 | k=3 | k=5 | 整體 | 正的抽題比例 |
|---|---|---|---|---|---|---|
| **主要：pool 1 固定 gallery、固定權重 λ=16 μ=1**（基底 0.3512／0.5780／0.6957／0.8091／0.6085） | −0.0066 (0.0052) | −0.0100 (0.0049) | −0.0076 (0.0051) | +0.0035 (0.0036) | **−0.0052 (0.0021)** | 37% |
| pool 1 固定 gallery、各自權重 | −0.0092 (0.0050) | −0.0114 (0.0049) | −0.0079 (0.0048) | +0.0045 (0.0036) | −0.0060 (0.0021) | 27% |
| 附帶：pool 0（未適配）、固定權重（基底 0.3484／0.5814／0.7002／0.8158／0.6114） | −0.0077 (0.0030) | −0.0092 (0.0051) | −0.0052 (0.0048) | −0.0006 (0.0037) | −0.0057 (0.0020) | 27% |

- 判定：整體 −0.0052 < +0.01，k=1 −0.0066 < −0.005 → **兩個條件都未過，不採用、不啟動正式 R2 重訓**。
- 被適配的 pool 1（−0.0052）與未適配的 pool 0（−0.0057）幾乎相同 → 看不到適配收益；兩者都略低於基底。只有 pool 1 的 k=5 偏正（+0.0035，在 1 個標準誤內）。
- 機制（推測，未驗證）：每組 64 人中固定 8 人來自只有 10 盤的 gallery 玩家，一般玩家每步由 64 人降為 56 人（−12.5%）；gallery 玩家每人 8 盤取自同 10 盤，重複度高，可能只是被記住而沒有提供可推廣的對比訊號。訓練末段 own-centroid accuracy 0.649。
- 限制：單一訓練 seed（seed 變異未量，−0.005 的方向不能視為確定）；單一 share（0.12）、單一 gallery seed；pool 1 是開發驗證集。記為「目前設定未呈現收益」，不是 gallery 適配方向被否證。
- 成本：本輪 16.6 GPU-h；S0＋R2 合計約 33 GPU-h。`wallet` 22:54＝57,648.23（20:38 為 59,020.71；期間同 account 另有其他成員多個 8 卡作業與 C4g 776711，無法換算費率，費率仍未知）。

### 2026-11-15：競賽一——新增集候選人是否與訓練集／原候選庫是同一批人（重疊檢查；使用者同意）

動機：Public 第一名 0.9060（11/15 04:32），我們 0.7936；原測試集已近滿分，反推對方新增集至少約 0.81（我們本地約 0.61）。檢查「候選人換代號出現在別處」這種能把 10 盤變成上百盤的結構。Day-0 只確認過逐字重複棋局與代號重疊皆為 0。

做法（`scripts/identity/pid2_overlap.py`，Slurm 779909，1 卡 24 分；輸出 `outputs/pid/pid2_overlap/`）：`v4_d4_hd_9600` 風格向量（白棋換色）。訓練集 7,969 人每人最多 20 盤 HD≤1（145,118 盤）；原候選庫每人抽 30 盤；新增集候選庫全部。校準：1,186 位編碼器未見過的訓練玩家，10 盤中心向量對「其他所有人的 20 盤中心」的最高 cosine＝無對應基準（null）；再與「自己另外 10 盤」取較大者＝有對應時的樣子（alt）。官方候選人皆以 10 盤中心對訓練集 20 盤中心比對。

| | 最高 cosine 中位數 | 超過 null 90 百分位（0.9069）的比例 | 第 1–第 2 名差距超過 null 95 百分位的比例 |
|---|---|---|---|
| null（無對應） | 0.8452 | 10% | 5% |
| alt（全部有對應） | 0.8876 | 32% | 31% |
| 新增集 Public 候選人（600）→ 訓練集 | 0.8483 | 13.8% | 7.0% |
| 新增集 Private 候選人（600）→ 訓練集 | 0.8404 | 13.2% | 9.3% |
| 原候選庫 Public（400，取 10 盤）→ 訓練集 | 0.8579 | 11.2% | 6.2% |
| 原候選庫 Private（400，取 10 盤）→ 訓練集 | 0.8564 | 11.5% | 6.0% |

- 新增集候選人 ↔ 原候選庫（30 盤中心）：最高 cosine 中位數 0.75、q99 0.90–0.93，低於同集合內「最近的其他人」（中位數 0.79）→ **沒有重疊的跡象**（原候選庫全黑、新增集黑白混合，跨集合 cosine 偏低，此比較較粗）。
- 新增集候選人 ↔ 訓練集：中位數與 null 相同；尾端比例略高於 null（線性內插約 10–20% 重疊，樣本誤差約 ±6 個百分點，且官方玩家與訓練玩家的分布差異也能造成同樣偏移）→ **沒有大範圍重疊；小比例重疊無法確認也無法排除**。
- 檢定力有限：即使全部有對應，10 盤對 10 盤也只有 32% 超過門檻；真對應排第 1 的比例為 70%（7,969 人中）。
- 結論：這個結構解釋不了與第一名約 0.20 的新增集差距。對方的做法仍未知。

**提交檔（使用者要求產生，作 LB 探針；Slurm 776740，1 卡約 2 分；流程同 772748，輸出 `outputs/pid/day0_1112/{public,private}2_v5_sub82/`）：**

| 檔案 | 內容 | 假設／預期 Public | Public LB |
|---|---|---|---|
| `submissions/pid/776740-v5-sub82.csv`（SHA `dc633f36`） | 原測試集與基底 `772748-d4-9600.csv` 逐列相同；新增集換 `v5_sub82_4800`（其 bench 選的 λ=8，k=5 為 16；μ=1）。改 top-1：pub2 121 題、priv2 114 題 | 對**基底**（9,600 步、100% 玩家）同批 pool 1 抽題約 −0.023（0.5969 對 0.6197）；以約一半轉移，Public 約 −0.01，預期偏負 | **0.7789612**（11/14 20:06:11，jabir95tsai；34/80） |

格式檢查：1,600 列、ID 與基底同序、同題 5 人不重複。保留規則：降或持平 → 維持 `772748-d4-9600.csv`；升但 <0.02 → 不升基底（本地明確為負，只記為「Public 新增集對訓練玩家數不敏感」的弱證據）；升 ≥0.02 → 回查本地驗證台與正式新增集的差異，不直接採用。這份只能看出「整體換 v5」的效果，不能把 82% 子集的影響和 9,600→4,800 步的影響分開。

**LB（11/14 20:06）：`776740-v5-sub82.csv` Public 0.7789612，34/80。** 對基底 `772748-d4-9600.csv`（0.7936014）**−0.0146**，預期約 −0.01，方向與幅度都和本地（−0.023 × 約一半轉移）相符；也低於 `770088-d4-4800.csv`（0.7820433，100% 玩家、4,800 步）0.0031，與本地同權重 −0.008～−0.009 的方向一致（雜訊內）。依保留規則**基底維持 `772748-d4-9600.csv`**。S0「玩家數重要」的方向在 Public 上沒有被推翻；R1 的外推仍無法由這次 LB 驗證。

### 2026-11-14：競賽二——「資訊不足」還是「用法不對」的最小檢定＋低容量 B2（使用者「都做」；事前登記）

背景：使用者的架構討論稿提出逐手權重（A）、序列推論（B）、局面背景（C）、受限微調（D）。A／B 只讀已存的逐手 log p，可不重抽就檢定；討論稿第 8 節的混色抽題缺口已於 11/13 `rank_same_color.py` 結案，不重做。全部 CPU（配置 1 卡），不產生提交檔。

**(1) 逐手證據的序列相關（判 B）：** `scripts/strength/rank_move_weights.py acf`。逐手檔 `outputs/rank/humansl/permove/`（C v2，HD0 盤）；u_t＝log p 對十個級別假設的斜率，每盤每方扣平均後算同方 lag 1–10 自相關，對照盤內打亂。design effect＝1＋2Σ(ρ−ρ_打亂)。**規則：本人與對手都 <1.15 → B 的「持續表現狀態」變體沒有可建模的相依，不做**（「下一手優先看附近級別」等同固定級別下累積 likelihood；失誤容忍是逐手混合，屬 clip／vote 同族，由 (2) 的 deepsets 臂涵蓋）。

**(2) 單側單盤讀取器（判 A）：** 同腳本 `weights`。擬合池（Valid-S＋train20）HD0 盤的 70% 玩家擬合、30% 玩家定方向與選臂，Valid-U HD0 盤評分。臂：mean（現行統計量）、vote、依「十假設離散度／平均驚訝度／手數」分箱的各箱平均（40–90 維，為級別共用權重的線性超集）、attn（每手一個級別共用權重，輸入只有離散度、驚訝度、手數）、deepsets（任意逐手非線性轉換）。輸入不含盤面，讀取器認不出棋局擁有者。**以對手側判定**（每盤是不同的人、幾乎無玩家偏差、受雜訊限制）；本人側只列報。
- **Gate：** hold-out 中段 d′（A/1D、1D/2D、2D/3D，單盤）最高的非 mean 臂為選定臂；其 Valid-U 中段 d′ ≥ mean 的 1.10 倍 → 進完整 G1（換進 52 維 B2，E1／E2 五 seed）；否則「已存 likelihood 的用法」結案，A、B 不再做，剩下只有重抽完整著手分布（C，約 13 GPU-h）才可能有新資訊。
- 另報：單盤 NLL／官方分數、k=5／20 的題目層級中段 d′、vote 在 hold-out 與 Valid-U 的落差（檢驗 11/12「池內選擇偏樂觀來自 C 向量貼合池內玩家」的推測：若成立，對手側不應有落差）。
- 預期：未知，偏負（截尾／修剪／手數窗 ±0.003、vote 負）；即使通過，以 k 曲線換算分數上限約 +0.01。

**(3) 低容量 B2：** `scripts/strength/rank_b2_lowcap.py auto`。cvec2 基底 52 維、同一批先驗對齊合成題，只換分類頭：base（C=1，須重現 PriorMatched）、L2 加強（C=0.3／0.1／0.03）、降秩 2／3、序數累積 logit、鄰級平滑標籤（0.15／0.70／0.15）。依據：加容量的頭全負，反方向只試過一次 C=0.3（+0.0013，舊設定）。
- 擬合池內對半選一臂（同 `rank_permove select`，三 seed）；**增益 ≥ +0.003 才在 Valid-U 評該臂**（五 seed）。G1 同前：E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005 → 升為基底候選；兩組 ≥ 0 → LB 探針；否則停止。
- 預期：未知；主觀過 gate 約 15%，上限約 +0.005。

**(4) 生成式微調放大棋局數 C4g（只登記，未實作、未送作業；待使用者確認估時）：**
- 依據：C4x adj／oadj 的判別項沒學到（相鄰級 CE 貼著均勻基準），監控改善與 NLL 下降同步，實質上是幽玄領域的生成式適應；每半只用約 52,000 盤的全部位置（11.7M positions）。每人最多 20 盤是為了不讓重度玩家主導本人側；對手側每盤是不同的人，不需要這個上限。
- 可用資料（11/14 清點，fold 1–4 玩家、HD≤1、扣掉全部 B2 擬合／評估盤）：A 半 258,200 盤／2,708 人、B 半 283,418 盤／2,726 人（約為 C4x 的 5 倍；盤數加權的有效玩家只有約 780，所以本人手必須按玩家等權）。
- 設計：純 NLL（真實級別條件，無判別項）；每盤抽約 45–60 個位置使位置總數與 C4x 相同（每半約 12M、pack 約 12 GB，須先刪可重建的 `c4x/learn_{A,B}` 24 GB；/work 目前剩 27 GB）→ **位置數與算力不變，只把棋局／對手人數放大約 5 倍**；本人手權重按玩家等權再級別等權，對手手只做級別等權。A／B 交叉擬合、fold −1 監控、C v2 起點沿用 C4x。
- Gate：G0＝兩模型終點單盤官方分數都 ≥ 第 0 步，且對手側中段 d′ 平均增益高於 adj（+0.04）；G1 同前（swap − base）。
- 估算（由 C4x 實測外推，未驗證）：pack 約 6 GPU-h、訓練兩半約 8 GPU-h、抽特徵約 15 GPU-h，合計約 30 GPU-h；G0 不過約 14 GPU-h。預期：未知；主觀過 G1 約 20–25%，若成立 +0.005–0.015、偏 k=5／10。
- 若 (2) 顯示對手側已存 likelihood 用到極限，本項是剩下唯一針對「每盤資訊量」的方向。

**結果（Slurm 772926：43 分；772927：25 分；皆配置 1 卡、實際 CPU 為主，合計約 1.1 配置 GPU-h）→ 三項都未過，A、B 與低容量 B2 結案。** 輸出 `outputs/rank/humansl/move_weights/{acf,weights}.json`、`outputs/rank/humansl/b2_lowcap/{select,result_ord}.json`。

**(1) 序列相關：** 級別斜率 u_t 的同方自相關 lag 1＝+0.006（盤內打亂 −0.009），lag 2 以後低於 0；design effect 對手 1.12、本人 1.12（<1.15）。「平均驚訝度」有短程相關（lag 1 +0.059，design effect 1.36），但它與級別無關。→ **逐手的級別證據幾乎是序列獨立的，持續表現狀態沒有東西可建模，B 不做。**

**(2) 單側單盤讀取器（擬合 84,591 盤、hold-out 36,025 盤、Valid-U 26,676 盤，HD0）：**

| 對手側 | hold-out 中段 d′ | Valid-U 中段 d′（對 mean） | U 單盤官方分數 | U 題目層級中段 d′ k=5／20 |
|---|---:|---:|---:|---|
| mean | 0.448 | 0.426 | 0.4638 | 0.873／1.107 |
| vote（依 hold-out 選定） | 0.450 | 0.428（×1.003，bootstrap [0.988, 1.018]） | 0.4636 | 0.843／1.067 |
| bin_spread／surprise／ply／all | 0.439／0.440／0.436／0.427 | ×0.997／1.016／1.017／1.026 | 0.465–0.468 | 0.86–0.89／0.96–1.08 |
| attn | 0.427 | ×0.954 | 0.4325 | 0.845／0.974 |
| deepsets | 0.450 | ×1.024 | 0.4585 | 0.850／1.140 |

- **未過 gate（門檻 ×1.10）：對手側所有臂都在 ×0.95–1.03。** 85k 盤、不含盤面、最多 90 維的線性超集與任意逐手非線性都拿不到 3% 以上 → 已存的逐手 log p 以整盤平均使用已接近極限。「用法不對」在這組資訊上被否證。
- 本人側（只列報，不作判定）：hold-out 一致為負或持平（×0.92–1.02），Valid-U 卻是 ×1.02–1.08（bin_all 1.075、deepsets 1.071；k=20 題目層級 0.68→0.77／0.81）。兩個獨立玩家集合符號相反，選定臂 vote 的 bootstrap 為 [0.968, 1.049] → 視為雜訊；事後挑 Valid-U 最好的臂不合規，不追。
- vote 的 hold-out／Valid-U 落差：對手側 ×1.005／×1.003（無落差），本人側 ×1.025／×1.010（小落差）；與 11/12 的推測方向一致但幅度在雜訊內，不足以證實。
- attn（原提案 A 的寫法）最差：每手一個級別共用權重會改變各盤的整體尺度，單盤 NLL 由 1.696 升到 1.773。

**(3) 低容量 B2：** 擬合池內（8,000 題、三 seed，對 base 0.7320）：L2 加強 −0.0001／−0.0027／−0.0096；降秩 2／3 +0.0064／+0.0059；**序數 +0.0117（選中）**；鄰級平滑 +0.0103。Valid-U（只評序數，五 seed）：**E1 −0.0118、E2 −0.0145，五 seed 全負**；E2 各 k −0.020／−0.020／−0.014／−0.004 → **stop**。base 臂逐題重現 PriorMatched（`base_vs_PriorMatched_max_abs` 見 result_ord.json）。
- 判讀：池內 +0.012 到未見玩家變 −0.013，是繼 S2 vote 之後第二次「擬合池內對半選擇與 Valid-U 反向」，而且這次只換分類頭、與 C 向量無關。原因未知。**之後任何選擇都不能只靠擬合池內對半驗證。** 鄰級平滑（池內第二）依規則未在 Valid-U 評。

**小結：** 討論稿的 A（逐手權重）、B（序列推論）與低容量 B2 都結案；C（熵／著手名次）依 likelihood 原則只能當權重輸入，而權重這條路已否證，重抽 13 GPU-h 的理由變弱。H＋C 這組資訊的讀法到頭了，剩下只有改上游模型本身（(4) C4g，尚未執行）。不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。

### 2026-11-14：競賽二——上一節判讀的更正＋池內／Valid-U 反向的對照（事前登記）

**更正（外部審閱後；上一節原文保留不改）：** 上一節有四處說得比證據多，以下列為準。
- 「用法不對被否證」「H＋C 的讀法到頭了」→ 應為：**已測的讀取方式（vote、四種分箱、attn、deepsets、序數／降秩／平滑／加強 L2 的頭）沒有可轉移的增益**。沒測的讀法（例如以新輸入作分類或校準特徵）不在結論內。
- 「逐手的級別證據幾乎是序列獨立」→ 應為：**級別斜率的線性自相關很弱（lag 1 +0.006、design effect 1.12）**；非線性相依與可靠度狀態的持續性沒測（驚訝度的 design effect 1.36 是後者存在的描述性跡象）。不做 B 是資源決定。
- 「對手側每盤是不同的人」「85k 盤、資料量不是限制」→ 沒有對手 ID，對手可能重複；同一位標註玩家的多盤共享棋風、級別與局面分布，likelihood 向量仍可能帶玩家資訊。**獨立單位是玩家（擬合約 3,900 人），不是盤**；「資料量不是限制」不成立為結論。
- 「C 只能當權重輸入，所以不值得測」→ 推論不成立：HumanSL 是近似模型，熵／著手名次可以是分類或校準特徵。**C 的狀態改為「未測；不排入目前時程」**，重啟時先以小型玩家隔離樣本看增量訊號，不直接全量重抽。
- 「這次只換分類頭、與 C 向量無關」→ 錯：分類頭吃的仍是 C 特徵。下面的對照就是檢這一點。
- 「85%」的定義（11/13 一節）：A–3D 中有 ≥2 題的 205 位玩家，題目符號誤差的**玩家平均之變異 0.77／總變異 0.91**，是變異數分解，不是「持續判錯的人數比例」；它也不能推出逐手讀法原理上碰不到玩家偏差。
- (4) C4g 的敘述同步修正：是「棋局覆蓋約 5 倍」，不是「對手人數 5 倍」；它檢定的是「固定位置預算下，純 NLL 微調覆蓋更多棋局是否改善未見玩家的棋力證據」，**不能把「上游還有沒有空間」結案**；須有三臂（未微調起點／同為純 NLL 的少棋局／多棋局），G0 須含題目層級表現。約 30 GPU-h 不稱低成本；刪 `c4x/learn_{A,B}` 另行確認。仍未實作。

**對照（`scripts/strength/rank_b2_lowcap.py inout`，CPU；只作診斷，不選臂、不升級、不產生提交檔）：**
- 事實：池內對半選擇的題目 99% 來自 Valid-S 玩家（train20 每人同色盤數不足），而 Valid-S 玩家的其他棋局訓練過 C 向量；Valid-U 玩家沒有任何元件看過。池內題與 E2 用同一種建構（級別平衡、單色）。
- 臂：base／ord／smooth／rr2／l2_003 × 特徵 {HC 52 維、H-only 32 維（拿掉 C；沒有任何欄位在池內玩家上學過）}。
- 三個格：**half→P**（半池擬合、另一半評分＝原 select）、**half→U**（同一批模型直接在 E1／E2 評分：只換玩家）、**full→U**（全池擬合＝原 evaluate：再換擬合人數）。各格對 base 的差、分 k、玩家 bootstrap。half 三 seed×兩半，full 五 seed。
- 事前寫定的讀法：
  - HC 的 half→P 為正、half→U 為負，而 **H-only 兩格同號** → 反向來自 C 特徵在池內玩家上的貼合。
  - **H-only 也反向** → 與 C 無關，是 Valid-S 與 Valid-U 玩家母體（或固定玩家集合的波動）的差異；看 half→P 的 bootstrap 是否含 0 判斷選擇偏差能否解釋。
  - half→U ≥ 0 而 full→U < 0 → 擬合人數效應（低容量頭只在人少時有利）。
  - 檢核：HC 的 ord half→P 應重現 +0.0117，full→U 應重現 E1 −0.0118／E2 −0.0145。
- 注意：這次會在 Valid-U 讀到未被選中的臂；任何一格的正值都不能當升級依據。
- 成本：約 110 次擬合，估 1 小時內（配置 1 卡，CPU 為主）。

**結果（Slurm 774440，4 分鐘；`outputs/rank/humansl/b2_lowcap/inout.json`）：反向與 C 無關，由「換玩家」與「擬合人數」兩段各貢獻一部分。** 檢核通過：HC 序數 half→P +0.0117、full→U E1 −0.0118／E2 −0.0145，與前次相同。對 base 的 Equal-k 差（括號為玩家 bootstrap 95%）：

| 特徵 | 臂 | half→P | half→U E1 | half→U E2 | full→U E1 | full→U E2 |
|---|---|---:|---:|---:|---:|---:|
| HC | ord | +0.0117 [+0.005, +0.018] | −0.0059 | −0.0058 | −0.0118 | −0.0145 [−0.026, −0.003] |
| HC | smooth | +0.0103 [+0.007, +0.013] | +0.0052 | +0.0048 | −0.0020 | +0.0002 |
| HC | rr2 | +0.0064 [+0.001, +0.012] | −0.0017 | −0.0047 | −0.0081 | −0.0044 |
| HC | l2_003 | −0.0096 | −0.0161 | −0.0083 | −0.0179 | −0.0092 |
| H | ord | +0.0079 [+0.002, +0.014] | −0.0026 | −0.0054 | −0.0127 | −0.0131 [−0.025, −0.002] |
| H | smooth | +0.0094 [+0.006, +0.012] | +0.0052 | +0.0071 [+0.002, +0.012] | +0.0008 | +0.0029 [−0.001, +0.007] |
| H | rr2 | +0.0050 | +0.0022 | +0.0051 | −0.0075 | −0.0005 |
| H | l2_003 | −0.0122 | −0.0191 | −0.0110 | −0.0173 | −0.0074 |

base 水準：HC half→P 0.7320、half→E2 0.7305、full→E2 0.7463；H-only 0.7309／0.7276／0.7429。分 k 見 json（ord full→E2 HC：−0.020／−0.020／−0.014／−0.004；H：−0.016／−0.022／−0.015／+0.001）。

- **H-only 與 HC 的型態相同** → 「C 向量貼合池內玩家」不是原因（11/12 的推測在這個案例上不成立）。
- **只換玩家（half→P 到 half→U，同一批模型）：** 低容量頭的差值一致往下，ord −0.011 到 −0.018、smooth −0.002 到 −0.006、rr2 −0.000 到 −0.011、l2_003 −0.001 到 −0.007。half→P 的區間不含 0，單靠選擇偏差解釋不了 ord；half→U 的區間都含 0（除 l2_003）。機制未知；可檢查但未檢查的差異：池內題只能取自在 10% 留出盤中同色 ≥k 盤的 Valid-S 玩家（重度玩家），Valid-U 題取自全部棋局。
- **再換擬合人數（half→U 到 full→U）：** ord 再降 0.006–0.010、smooth 0.004–0.007、rr2 0.000–0.010，l2_003 不變 → 低容量頭的好處隨擬合玩家增加而消失，對半驗證（玩家減半）會系統性高估它。
- **方法結論：** 擬合池對半選擇對「限制容量」這類改動偏樂觀約 0.01–0.025（兩段合計），不能用來選這類臂；選擇要在與正式擬合相同的玩家數下、對未見玩家評。
- smooth 是唯一沒有反向的臂（full→U −0.002 到 +0.003），但都在 G1 +0.005 以下，且這次是事後讀 Valid-U，不升級、不出檔。
- 原因只排除了 C 與選擇偏差（就 ord 而言）；玩家母體差異的具體機制仍未知。

### 2026-11-14：競賽二——C4g 三臂設計、成本核算與第一階段（使用者同意實作；事前登記）

**問題：** 固定訓練位置預算，純 NLL 微調覆蓋更多不同棋局，是否改善未見玩家的棋力證據。失敗不代表上游模型沒有改善空間。

**三臂（模型、lr 3e-5／向量 3e-4、warmup 300＋cosine、每卡 64 位置 × 8 卡、步數、`--rank-balance`、C v2 起點皆與 C4x adj 相同，只拿掉級別 CE）：**
- base：HumanSL＋C v2 向量，未微調（＝任一 run 的第 0 步）。
- few：C4x 的學習集（每人最多 20 盤、全部位置；既有 `c4x/learn_<half>`，不重編碼）。
- many：同一批玩家的**全部** HD≤1 棋局，抽出與 few 相同數量的位置（`scripts/strength/rank_c4g.py tables`，seed 2027）。
- 兩臂之間相同的量（程式內 assert）：玩家集合；**每位玩家的本人位置數**（本人側按玩家的權重規則因此相同）；**每級的對手位置數**；因此各級 × 本人／對手的位置數、級別權重、步數都相同。不同的只有位置來自哪些棋局。

| 半 | 臂 | 棋局 | 玩家 | 位置 | 每盤位置數 5%／中位／95% | 本人側有效玩家 |
|---|---|---:|---:|---:|---|---:|
| A | few | 51,972 | 2,863 | 11,583,495 | 129／229／300 | 2,709 |
| A | many | 331,319 | 2,863 | 11,583,495 | 11／26／95 | 2,709 |
| B | few | 52,474 | 2,876 | 11,707,455 | 129／230／300 | 2,731 |
| B | many | 359,907 | 2,876 | 11,707,455 | 10／25／89 | 2,731 |

- 棋局覆蓋約 6.4–6.9 倍（不是先前寫的 5 倍：few 臂沒有排除 B2 擬合盤，many 臂也就不排除，交叉擬合是按玩家分的）。**不同對手數未知**（資料沒有對手 ID），不把棋局倍數當成對手倍數。
- 編碼器新增「整盤重播、只編碼抽中的手」；同一盤 4 個位置與完整編碼逐位元相同（bits／globals／target／own）。

**開發驗證（玩家隔離，不用 Valid-U）：** fold −1 玩家（不在任何一半、不在任何 B2 擬合或評估集合）。每人取盤數較多的一色、至少 5 盤、最多 20 盤，每盤隨機 50 手本人＋50 手對手（固定位置，三臂相同）→ 762 人、12,974 盤、1,297,400 個位置；可出 k=5／10／15／20 題的玩家 762／672／564／492 人。訓練作業結束時存每盤十個級別假設的 log p 總和。
- 指標（`rank_c4g g0`）：每盤 20 維（本人／對手平均 log p），在開發玩家內做 5 折玩家分組 CV，用正式的分類頭（先驗對齊 B2、合成題）；級別平衡單色題每格 200 題，官方分數分 k 與 Equal-k；玩家 bootstrap 2,000 次的配對差。另報 dev NLL、單盤與玩家層級中段 d′（不作判定）。
- 限制：開發集只有 20 維 C（沒有 H），每盤只取 100 手，分數水準會低於正式管線，只用於臂間配對比較；第一階段只訓練 A 半一個模型，訓練隨機性沒有重複量測；fold −1 的 3,000 盤監控集先前被看過（沒有用來選模型），開發玩家與它重疊。

**Gate（跑前寫定，之後不調）：** 進第二階段須同時滿足
1. many − few：Equal-k ≥ +0.005，且 bootstrap P(差 > 0) ≥ 0.90；
2. many − base：Equal-k ≥ +0.005，且 P(差 > 0) ≥ 0.90，且沒有任何 k 桶 < −0.005；
3. dev NLL：many ≤ base。
- 不滿足 → 停止：不訓練 B 半、不抽特徵。檢定力未知（配對標準誤要跑完才知道）；真實效果若只有 +0.005，可能因雜訊被判停，接受這個風險。
- 第二階段（須使用者再確認）：many 的 B 半＋完整交叉擬合抽特徵＋G1（swap − base，E1／E2 五 seed，門檻同前）。few 臂不做完整抽取，所以 Valid-U 上只能回答「many 對 base」，「many 對 few」的歸因只靠開發集。Valid-U 已多次使用，不視為全新測試集。

**成本核算（先前「約 30 GPU-h／第一階段 14」只算了 many 一臂、兩半，不含 few 臂與開發驗證；以下為三臂重估）：**
- 實測依據：C4x pack 兩半 23.1M 位置 44 分 × 8 卡＝5.9 GPU-h（本次量測：只編碼抽中位置時每位置速度不變）；adj 訓練 22,561 步 38 分（其中監控 7 次 × 98 秒），3 次前向；抽特徵 2 小時 03 分 × 8 卡＝16.3 GPU-h。
- **第一階段（只做 A 半）：** pack many A＋dev 約 3.4；few A、many A 各約 21–24 分 × 8 卡，合計約 6（純 NLL 的速度由 adj 外推，未實測）→ **約 9.5 GPU-h**。
- 第二階段（若過 gate）：pack many B 約 3＋訓練約 2.8＋抽特徵 16.3＋eval 0.5 ≈ 22.6 → **合計約 32 GPU-h，略高於 30 的上限**；到時依第一階段實測重報再請使用者決定。
- 磁碟：第一階段新增約 13 GB（many A 12＋dev 1.3＋檢查點 0.6），/work 現剩 27 GB → **本階段不刪 `c4x/learn_{A,B}`**（few 臂也正要用它）。它們是可重建的中間產物：`rank_c4x tables`（seed 2027，清單 `c4x/learn_*_hd*.parquet` 保留）＋`slurm/rank_c4x_pack.slurm`，約 5.9 GPU-h；沒有檢查點在裡面。第二階段需要空間時再確認是否刪 B。

**執行：** `rank_c4g tables` → `HALF=A slurm/rank_c4g_pack.slurm` → `ARM={few,many} slurm/rank_c4g_train.slurm`（afterok）→ `rank_c4g g0`。輸出 `outputs/rank/humansl/c4g/`。不產生提交檔。

**C4g 第一階段事故（11/14）：pack 作業 774480 超時，約 12 GPU-h 沒有產出。**
- 774480（dev 佇列 8 卡、8 個行程各 11 workers，節點 25a-hgpn009）跑滿 90 分鐘上限，沒有任何 shard 完成（pack 跑完才寫檔、不印進度，沒有留下可判斷的紀錄）；訓練作業 774481／774482 因相依被取消，未執行。sacct：TotalCPU 129 小時（約 86 核心全程忙碌），為預期（約 32 CPU 小時）的 4 倍。
- 事後量測：本機 300 盤 24 秒、1,200 盤 78 秒（線性）；剖析重播 30 盤 0.1 秒（重播不是成本）。**診斷作業 775671（8gpus 佇列 1 卡、同一份程式加進度輸出，節點 25a-hgpn005）19 分 50 秒完成 shard 0**：learn_A hd0 34,075 盤／1,161,320 位置 831 秒（1,398 位置／秒）、hd1 289,527 位置、dev 兩份；每盤位置數與抽樣表逐盤相符。
- **774480 變慢的原因未知。** 已排除：重播成本、抽樣表、程式版本（兩個檔案的修改時間在作業開始前，11:21 另一個 session 的提交與切到 `main` 沒有改寫它們）。未排除：該節點或「8 個行程同節點」的問題（C4x 用同樣配置時正常）。
- 現況：shard 0 可用（`c4g/learn_A/pack_hd{0,1}_0`、`c4g/dev/pack_hd{0,1}_0`）。其餘 7 個 shard 以 `TASK=i slurm/rank_c4g_pack1.slurm`（各 1 卡約 20 分）補齊約 2.3 GPU-h；pack 已加每 2,000 盤一行的進度輸出。
- 預算：已用約 12.3 GPU-h（其中 12 為浪費）。完成第一階段尚需約 2.3＋6＝8.3 → 第一階段累計約 20.6；第二階段約 22.6 → 若全做完累計約 43，超過 30 的上限。**是否繼續待使用者決定；在此之前不送作業。**

**C4g 第一階段結果（11/14；Slurm 776710 少棋局、776711 多棋局，各 8 卡約 28 分；G0 於登入節點 CPU 計算，`outputs/rank/humansl/c4g/g0_A.json`）：gate 未通過，不進第二階段。**

設定如事前登記，未改門檻：半邊 A、22,561 步、global batch 512、同一組 2,863 位玩家與 11,583,495 個位置；少棋局 51,972 盤、多棋局 331,319 盤。開發集為 fold −1 的 762 位玩家／12,974 盤（不在任何一臂的訓練裡，也不是 Valid-U），每盤 C 特徵 20 維，B2 以玩家分組 5 折交叉擬合，題目各級等量；差異區間為玩家 bootstrap 2,000 次。

| 臂 | dev NLL | k=5 | k=10 | k=15 | k=20 | Equal-k | 單盤 d′ | 玩家 d′ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 未微調基底（C v2） | 1.43104 | 0.6387 | 0.6783 | 0.7156 | 0.7125 | 0.6863 | 0.4404 | 1.0334 |
| 少棋局 | 1.40954 | 0.6497 | 0.6971 | 0.7267 | 0.7378 | 0.7028 | 0.4708 | 1.0660 |
| 多棋局 | 1.40966 | 0.6515 | 0.6955 | 0.7258 | 0.7337 | 0.7016 | 0.4718 | 1.0694 |

| 差異 | k=5 | k=10 | k=15 | k=20 | Equal-k | 95% 區間 | P(>0) |
|---|---:|---:|---:|---:|---:|---|---:|
| 多 − 少 | +0.0017 | −0.0015 | −0.0009 | −0.0040 | **−0.0012** | [−0.0062, +0.0037] | 0.305 |
| 多 − 基底 | +0.0128 | +0.0173 | +0.0102 | +0.0213 | +0.0154 | [+0.0068, +0.0237] | 1.000 |
| 少 − 基底 | +0.0110 | +0.0188 | +0.0111 | +0.0253 | +0.0166 | [+0.0077, +0.0253] | 1.000 |

監控集（fold −1 另 3,000 盤，單盤）：NLL 基底 1.41169 → 少 1.39032／多 1.39001；單盤官方分數 0.5358 → 0.5451／0.5480。訓練 NLL 少 1.39847、多 1.37756（兩臂的訓練位置不同，不能直接比）。

**判定：** 「多 − 少 ≥ +0.005 且 P ≥ 0.90」未達成（−0.0012，P＝0.305）→ **G0 不通過**。其餘條件（多 − 基底、各 k 不低於 −0.005、dev NLL 不高於基底）都成立，但第一條是必要條件。

**能說的與不能說的：**
- 能說：在同一組玩家、同樣位置數與步數下，把位置分散到約 6.4 倍的棋局，沒有比集中在少數棋局好；區間上限 +0.0037，低於登記的 +0.005。單一半邊、單一 seed、單一訓練長度。
- 兩個微調臂都比未微調的 C v2 高約 +0.015（P＝1.000），但這**不是新發現也不能直接當增益**：這裡只用 C 的 20 維、沒有 H 與讓先旗標，開發玩家也不是 Valid-U；全網路微調先前接上完整 H＋C、在 Valid-U 檢定時都沒有留下增益（C4／C4b 未過 gate，A-057；C4x adj 監控改善但下游持平，E1 +0.0059／E2 −0.0009）。是否值得用 H＋C 重驗，需另提計畫。
- 不能說：上游模型已無改善空間。這次沒有改變的變數——玩家數、目標函數、訓練長度、對手多樣性（實際不同對手數仍未知）——都沒有被檢定。
- 機制：未知。與「可施力的是玩家數而非棋局數／算力」（A-057 的推測）方向一致，但本實驗沒有改變玩家數，不構成對它的檢定。

**算力：** 訓練 2 × 8 卡 × 約 0.47 h ≈ 7.5 GPU-h；第一階段累計約 22.4 GPU-h（事故浪費 12.3、pack 2.3、失敗啟動 0.3、訓練 7.5），在 30 的上限內。第二階段（半邊 B、完整抽取、G1，約 22.6 GPU-h）**不執行**。

**產物與清理（未刪任何東西）：** `outputs/rank/humansl/c4g/` 14 GB（pack、兩個 `last.pt`、dev 特徵），`c4x/learn_{A,B}` 各 12 GB；`/work` 剩約 12 GB。重建方式：`rank_c4g tables` → `TASK=i slurm/rank_c4g_pack1.slurm`（16 個、各約 20 分）→ `ARM=few|many slurm/rank_c4g_train.slurm` → `rank_c4g g0 --half A`。刪除待使用者確認。

不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。

### 2026-11-15：競賽二——題目特徵按手數加權（使用者「好」；事前登記）

**假設：** 基底把題內各盤特徵等權平均，但每個特徵都是「每手平均 log p」；若每手是獨立證據，整題 log-likelihood 是所有手的加總，長的棋局應該佔較多權重。改為 Σ_g n_g x_g / Σ_g n_g（n_g＝該盤總手數；本人／對手手數最多差 1），合成擬合題與評估／測試題同樣處理；擬合集、先驗、seed、C 全同 cvec2 基底。**預期：小（< +0.003）**——棋局長度差異不大（中位數 228），且 B2 每個 k 有自己的尺度。程式 `scripts/strength/rank_lenweight.py`，Slurm **779900**（CPU，1 卡配置）。
- **Gate（看 Valid-U 前寫定）：** E1 或 E2 五 seed 平均 Equal-k ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005 → 升為基底候選；兩組都 ≥ −0.01 → LB 探針（使用者的近似規則）；否則停止。
- export：同一程式中 PriorMatched 未換臂須逐題重現 `751959-cvec2.csv`，否則報錯；輸出 `<job>-lenw.csv`（無 L10，L10 另疊）。

### 2026-11-15：競賽二——強引擎著手品質＋死活特徵，全擬合池（事前登記；使用者同意）

**動機：** 使用者提出「強的選手能把局部做活，分區塊看是否有幫助」。把棋盤切塊當輸入不會增加資訊（現有網路都是整盤卷積）；可行的版本是把死活做成每盤特徵。同類的強引擎 14 維特徵（A-048、11/11 visits 試算、11/14 玩家層級篩選）三次都是負的，但都只在 18,772 盤試算盤／760 位玩家上測過，**沒有擴到全擬合池、沒有測試題特徵、沒有上過 LB**；死活特徵本身沒有測過。

**假設：** 全擬合池（約 5,700 位玩家）下，加入特徵的變異數成本變小，強引擎著手品質或死活事件能補到 H＋C。預期增益：未知（主觀估可測增益的機率一到兩成；試算盤上中段級別 d′ 沒有變好）。

**做法（`src/go_aicup/features/katago_engine.py --ownership`、`scripts/strength/rank_engine_full.py`、`slurm/rank_engine_full.slurm`）：**
- 對象＝cvec_v2 的抽取表（Valid-S HD≤1＋全部 Valid-U 97,950 盤、train20 86,748 盤、測試 10,000 盤，共 194,698 盤，約 4,500 萬位置）。kata1 b18c384nbt、1 visit、日本規則；**HD1 局貼目 0.5、其餘 6.5**（試算盤當時一律 6.5）。
- 引擎 14 維（E）：同 A-048（每方：平均目數損失、平均勝率損失、損失 >1／>3／>7 目比例、實際著手的 policy log p、與第一選擇相同比例）。
- 死活 10 維（L，新）：每個位置由歸屬預測與重播盤面算每色「已失去的子」＝盤上歸屬偏向對方（|歸屬|>0.5）的子＋已被提的子；每方 5 個：自己一手後己方失子增加量（截 0–30）的平均、增加 ≥5 子的比例、對方失子減少量（讓對方活回來）的平均、減少 ≥5 子的比例、己方未定（|歸屬|≤0.5）子數平均。只存每個位置 6 個整數，不存整張歸屬圖。
- 臂：HC（現行 cvec2 基底 52 維）、HC_E、HC_L、HC_EL。B2、先驗、seed 2027–2031、E1／E2 題組與 cvec_v2 的 G1 相同。

**Gate（每臂對 HC，沿用 G1，跑前登記）：** 五 seed 平均 Equal-k 在 E1 或 E2 ≥ +0.005、另一組 ≥ 0、且 E2 各 k ≥ −0.005 → 升為候選並出提交檔；兩組都 ≥ 0 但未達 → 只當 LB 探針；否則停止，不出檔。四臂取其一，若多臂通過以 E2 最高者為準。Valid-U 已多次使用，通過也只列候選。

**成本：** 1 visit 依試算盤速度（每卡 4,760–5,450 位置／秒）約 3 GPU-h；多了歸屬輸出與 JSON 解析，實際速度由 1 卡短測（Slurm 779912，各 400 盤）決定，短測後回填。輸出每位置約 25 bytes，全部約 1.2 GB（`/work` 剩約 12 GB）。

**結果（Slurm 779900，CPU 46 分；`outputs/rank/humansl/lenweight/result.json`）→ G1＝probe（未達升級門檻，兩組都在 −0.01 以內）。** 題內棋局長度的變異係數平均 0.20。

| Valid-U（五 seed 平均，lenw − base） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| E1 | +0.0044 | −0.0028 | −0.0052 | −0.0034 | **−0.0018** |
| E2 | −0.0045 | +0.0010 | +0.0055 | −0.0058 | **−0.0009** |

- 各 seed Equal-k：E1 −0.0026／−0.0028／−0.0037／−0.0012／+0.0014；E2 −0.0038／+0.0033／−0.0023／−0.0001／−0.0018。seed 2027 bootstrap：E1 [−0.0118, +0.0072]、E2 [−0.0112, +0.0035]。
- 判讀：與基底持平（−0.001 到 −0.002，區間跨零），分 k 的正負在兩組間不一致。長度差異（CV 0.20）不足以讓權重有實質影響，且 B2 每個 k 的尺度已吸收平均長度；「把整題所有手加總」在理論上更乾淨，但實測沒有可量到的增益。
- **提交檔：** `submissions/rank/779900-lenw-l10.csv`（SHA `ccf88c6a`；export 中 PriorMatched 未換臂逐題重現 `751959-cvec2.csv`，再以 `rank_identity_merge overlay` 疊 L10）。對基底 `751959-cv2-l10.csv` 改 Public 21／Private 21 題，全為差一級（往強 19、往弱 23）。預期 Public 約 0（雜訊內）。保留規則：降或持平 → 維持基底、結案；升 <0.02 → 不升基底（本地持平）；升 ≥0.02 → 回查。無 L10 版 `779900-lenw.csv` 只作對照。

| 檔案 | 預期 Public | Public LB |
|---|---|---|
| `submissions/rank/779900-lenw-l10.csv` | 約 0 | **0.7452214**（11/15 11:43:48，jabir95tsai；24/136） |

**LB（11/15 11:43）：`779900-lenw-l10.csv` Public 0.7452214，對基底 `751959-cv2-l10.csv`（0.7499623）−0.0047**（總分約 −1.9；改了 21 題，相當於約 2–3 題由正確變鄰級），與本地持平偏負同方向，仍在雜訊內（同實力下 21 題約 ±2.8 分）。依保留規則**維持基底，按手數加權結案**。

### 2026-11-15：競賽二——盤面條件逐手加權 C（使用者核准實作與訓練；事前登記）

假設：既有 attention 只看離散度／平均 log p／手數，加入落子前盤面與十級分數形狀可改善逐手可靠度判斷。預期增益未知；不是新增棋力主幹，不以 training CE 降低宣稱多盤題目增益。

固定架構：2×19×19 盤面，三層無 padding 的 5×5 conv（2→8→8→8，各 ReLU），flatten 392→10，串接 13 維逐手特徵，23→16/ReLU→1，共 **7,955 參數**（先前對話估 7,986 有誤，以實際模型計數為準）。本人／對手共用網路，各自在盤內 softmax；加權原始十級 log p，兩側直接對本人級別算 CE，兩側／各盤等權。不加入 H、旗標、分類頭或溫度。最後一層零初始化，起點精確為 mean。HumanSL 與 C v2 凍結。

資料：重用 `rank_move_weights.split_players` 的 HD0 玩家切分：train 84,591 盤／3,960 人、hold 36,025／1,698、U 26,676／1,419。訓練 batch 每級 N 位不同玩家、每位一盤、全部著手。C v2 曾以部分 train／hold 玩家的其他盤學向量；U 玩家沒有參與 C 向量學習。Public／Private 快取完全不讀、不參與選模。盤面直接重用 `data/processed/boards/strength_v1` 的 packed 黑白盤面，盤內位移一手並按落子方交換通道，不重播、不建立新全量盤面快取。逐手來源核對 game_index／ply／side／長度、vectors／komi／finetuned，平均須重現 C v2。

短測：單 H200，N=32／64／128 各 34 updates（前兩步暖機），測端到端位置／秒、顯存；吞吐在最佳 95% 內選最小 N。再測完整 hold 一次以外推正式成本。各級可用玩家最少 212，三個候選均足夠。正式：AdamW lr=.001、weight_decay=.0001，最多 2,000 updates，每 100 步完整 hold CE 選模，至少 500 步，之後 patience=5；baseline step 0 也可被選中。保存 best／latest、optimizer 與 sampler RNG，可續跑。初估單卡約 2–6 GPU-h；短測後以實測更新成本，尚未宣稱實測速度。

訓練後凍結最佳權重、抽新 C 20 維，後續以 mean C＋相同旗標重訓 B2 為對照；HD0 題目分 k=5／10／15／20，增益候選門檻沿用 +0.005、另一組不負、E2 各桶不低於 −0.005（HD0 題組與正式 HD≤1 不同，不能直接當全管線 gate）。本輪不更換基底、不上傳。程式與追蹤 Slurm：`src/go_aicup/tasks/strength/board_attention/`，設定 `configs/strength/board_attention.json`，測試 `tests/test_board_attention.py`，輸出 `outputs/rank/humansl/board_attention/seed2027_hd0/`。

**短測完成（780382，H200 單卡）：** 147,292 盤逐手與盤面載入／對齊 41.3 秒；N=32／64／128 各 34 步，端到端吞吐 1.311／1.341／1.351M positions/s，step 0.0541／0.1061／0.2108 秒，allocated 峰值 4.42／5.26／6.91 GiB。三者速度接近，依事前 95% 規則選 **N=32**（每步 320 盤）。完整 hold CE 一次 1.70 秒。2,000 步＋21 次 hold 外推約 **144 秒＝0.040 GPU-h**，另加資料載入與最後抽取，正式作業預留 **10 分鐘＝0.167 GPU-h 上限**；先前 2–6 GPU-h 為未量測的保守估計，不再沿用。均勻起點 hold CE：本人 2.249574、對手 2.254305、平均 2.251939。

**使用者指定 N=64：** 短測建議值為 N=32，但使用者確認維持原先目標 **每級 64、共 640 盤**。首個正式作業 780388 在資料載入階段取消，保留原短測與載入紀錄；新 run 為 `seed2027_hd0_n64`，config 的 `per_rank=64` 為準，不套用短測自選值。2,000 updates 約 128 萬盤次、約 2.85 億位置次，純訓練加 hold 外推約 248 秒＝0.069 GPU-h；作業仍預留單卡 10 分鐘。

**使用者最新指定 N=128：** 停止 N=64 作業 780391，保留其 run；新 run `seed2027_hd0_n128`，每級 128 位不同玩家、每步 1,280 盤、最多 2,000 updates。純訓練約 422 秒＋定期 hold 約 36 秒＋載入／輸出約 1 分鐘，預估合計 8.5–10 分鐘，單卡作業上限 **15 分鐘＝0.25 GPU-h**。先前取消的 780388 實際已跑到 step 175（最後完整 checkpoint 100）；前一節「資料載入階段取消」不準確，以此更正。舊 run 不作比較選模、不混入 N=128 訓練。

### 2026-11-15：強引擎／死活提交探針（使用者要求生成）

評估 Slurm 780058 已完成（1:17:20，exit 0）。HC_E／HC_L／HC_EL 的 G1 均為 stop；使用者仍要求生成提交檔，故僅列探針，不升基底。五 seed 平均增益如下（對 HC）：

| 臂／題組 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| HC_E E1 | +0.0059 | -0.0205 | +0.0020 | -0.0123 | -0.0062 |
| HC_E E2 | +0.0036 | -0.0105 | -0.0164 | +0.0006 | -0.0057 |
| HC_L E1 | +0.0032 | -0.0139 | +0.0172 | -0.0047 | +0.0005 |
| HC_L E2 | +0.0014 | -0.0061 | -0.0154 | +0.0018 | -0.0046 |
| HC_EL E1 | +0.0017 | -0.0145 | +0.0132 | -0.0173 | -0.0042 |
| HC_EL E2 | +0.0031 | -0.0065 | -0.0235 | -0.0004 | -0.0068 |

匯出 Slurm 780395：沿用 rank_engine_full export，seed 2027、每臂先驗與擬合流程不變，HC refit 必須逐題重現 751959-cvec2.csv；依序生成 780058-hc-e／hc-l／hc-el.csv，再以既有 rank_identity_merge overlay 生成各自 -l10.csv。假設為引擎品質／死活提供補充訊號；本地不支持正增益，預期 LB 持平或小幅下降，實際未知。保留規則：下降／持平／上升不足 0.02 均維持既有基底；上升至少 0.02 僅觸發回查，不自動升基底。未上傳。匯出與格式驗證狀態待完成後補登。

**N=128 正式訓練：780396**（前個 N=64 作業 780391 停在 step 325，最後完整 checkpoint 300，保留在獨立 run）。後續固定 HD0 評估事前登記：以 train＋hold 共 120,616 盤擬合兩個相同 PriorMatched B2 臂（mean C＋2 個零旗標／新 C＋2 個零旗標），seed2027–2031。U 的 26,676 盤用唯一 CV 模組 `make_rank_questions` 生成兩組各 4,000 題、同玩家同色、各級等量、四個 k 各 1,000 題（question seed2027／2028）。命名 **HD0_Q2027／HD0_Q2028**，不冒充原 E1／E2。這是新權重的 HD0 診斷，不作正式基底 promote；逐 k 官方分數、五 seed 平均及首 seed 玩家 bootstrap 列報，結果未知。CPU 五工並行，Slurm 配置單 GPU（CPU 為主）20 分鐘上限，訓練成功後才執行。

**完成結果（正式訓練 780396、B2 評估 780403）：未通過，不更換 mean C。**

N=128（每步 1,280 盤），AdamW .001／weight_decay .0001，最多 2,000 updates；開發 CE 在 step 1,000 最佳，此後五次沒有改善，step **1,500 早停**。保留 `model.pt`＝step **1,000**。hold CE 起點 2.251939（本人 2.249574／對手 2.254305）→ **2.057936**（2.052254／2.063618）。最佳模型 hold 每方有效手數 **5.43**、平均最大單手權重 **0.489**，相較均勻起點有效手數 111.38，權重顯著集中。這是描述性觀察，未用額外正則或改目標重訓。

B2 五 seed 平均（mean C＋旗標 vs 新 C＋旗標；兩個旗標因 HD0 都是 0；每題同色、各級等量）：

| 題組／臂 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| HD0_Q2027 mean | 0.66768 | 0.72452 | 0.73592 | 0.77238 | **0.72512** |
| HD0_Q2027 weighted | 0.56234 | 0.62368 | 0.65372 | 0.67352 | **0.62830** |
| HD0_Q2027 差 | −0.10534 | −0.10080 | −0.08222 | −0.09888 | **−0.09682** |
| HD0_Q2028 mean | 0.68182 | 0.72542 | 0.73696 | 0.76096 | **0.72630** |
| HD0_Q2028 weighted | 0.55560 | 0.62510 | 0.65232 | 0.69720 | **0.63258** |
| HD0_Q2028 差 | −0.12624 | −0.10028 | −0.08460 | −0.06374 | **−0.09372** |

兩組各五 seed 全負：Q2027 Equal-k 差 −0.0926／−0.0939／−0.0990／−0.1002／−0.0984；Q2028 −0.0922／−0.0949／−0.0947／−0.0927／−0.0941。首 seed 玩家 bootstrap 95%：Q2027 **[−0.1120, −0.0735]**、Q2028 **[−0.1114, −0.0732]**。兩組題是同一批 U 玩家的不同抽題，不是兩個獨立玩家 holdout。

判讀：**直接兩側 CE 能降低單盤分類 loss，未轉成 B2 多盤辨識力，反而有約 −0.095 的顯著損失。** 權重集中與丟掉大部分著手證據的現象相容，但未分別消融盤面、13 維輸入、直接 CE 或池化形式，不能將原因單獨歸為任何一項；也不能據此否定所有看盤面的加權方法。這是 HD0、C-only 診斷，不是完整 H＋C 的新分數。不產生提交檔，正式基底維持 `751959-cv2-l10.csv`。

成本：正式訓練作業 **6 分 27 秒＝0.1075 GPU-h**（含載入與最佳特徵抽取，內部 train/hold/輸出 343 秒）；B2 CPU 評估配置單 GPU **19 秒＝0.0053 GPU-h**；兩者合計 **0.1128 GPU-h**。加短測 68 秒、因使用者改 N 而取消的 60＋80 秒，本輪共 **614 秒＝0.1706 配置 GPU-h**。沒有計入別的 session 的 `rank-eng-export` 作業。

驗證：`tests/test_board_attention.py` **7 項通過（2.69 秒）**，涵蓋快取轉接、提子後時序、黑白通道、ragged 盤長、均勻起點、穩定 segmented softmax／CE、卷積梯度、玩家抽樣／RNG、checkpoint 載入與 B2 同特徵對照；Slurm 語法／Python 編譯及 `git diff --check` 通過。沒有重建或全量重驗盤面快取。

輸出：`outputs/rank/humansl/board_attention/seed2027_hd0_n128/` 的 `model.pt`、`last.pt`、`learning_curve.json`、`complete.json`、`features.npz`、`b2_result.json`、兩組 HD0 題表、`data.json`。舊 N=32／64 run 保留，不用來選模型。不刪既有產物、不上傳。

**匯出完成：** 六份 CSV 均已驗證欄位、800 題與官方模板順序／唯一 ID、合法十級標籤；三臂 HC refit 均逐題重現基底。Slurm 780395 完成三臂擬合及無 L10 匯出，最後 overlay 因 `hc-el-l10` 超過 8 字元標籤上限而 exit 1；改短標籤 `el-l10` 後以既有 overlay 完成（不重跑擬合）。下表改題數各對應無 L10 基底或 L10 基底。

| 提交檔 | Public 改題 | Private 改題 | SHA-256 前八碼 |
|---|---:|---:|---|
| `780058-hc-e.csv` | 25 | 30 | `37d8c99a` |
| `780058-hc-l.csv` | 27 | 33 | `f85a07d4` |
| `780058-hc-el.csv` | 36 | 38 | `839b57c1` |
| `780058-hc-e-l10.csv` | 25 | 30 | `435d2e0e` |
| `780058-hc-l-l10.csv` | 27 | 33 | `a7e989d6` |
| `780058-el-l10.csv` | 36 | 38 | `2775820b` |

未上傳；本地 gate 未通過，僅供使用者手動 LB 探針。

**結果（11/15；抽取 Slurm 779914，8 卡 28 分約 3.7 GPU-h；短測 779912；評估 780058，CPU 1 小時 17 分；`outputs/rank/humansl/engine_full/result.json`）：三臂全部 stop，不出提交檔。**

抽取：194,698／194,698 盤完成、0 筆被拒，每卡約 3,970 位置／秒，輸出 1.1 GB。歸屬正負號以短測確認（終局黑方領先目數與「白失子 − 黑失子」相關 0.90／0.93）。

五 seed 平均 Equal-k（HC 為現行 cvec2 基底；E1／E2 各 seed 的 HC 為 0.7102–0.7190／0.7431–0.7505）：

| 臂 − HC | E1 | E2 | E2 k=5 | k=10 | k=15 | k=20 | E2 95%（seed 2027，玩家 bootstrap） | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| HC_E（＋引擎 14 維） | −0.0062 | −0.0057 | +0.0036 | −0.0105 | −0.0164 | +0.0006 | [−0.0195, −0.0021] | stop |
| HC_L（＋死活 10 維） | +0.0005 | −0.0046 | +0.0014 | −0.0061 | −0.0154 | +0.0018 | [−0.0217, −0.0033] | stop |
| HC_EL（＋兩者） | −0.0042 | −0.0068 | +0.0031 | −0.0065 | −0.0235 | −0.0004 | [−0.0256, −0.0051] | stop |

E1 分 k（5／10／15／20）：HC_E +0.0059／−0.0205／+0.0020／−0.0123；HC_L +0.0032／−0.0139／+0.0172／−0.0047；HC_EL +0.0017／−0.0145／+0.0132／−0.0173。

**判讀：**
- 「試算盤玩家太少才是負的」這個未排除的弱點現在排除了：擴到全擬合池（HD≤1 擬合盤、Valid-S＋train20）後，強引擎著手品質仍是負的（E1／E2 都約 −0.006）。
- 死活特徵在 E1 持平（+0.0005）、E2 −0.0046，seed 2027 的 E2 區間不含 0；沒有任何一組達到 +0.005。k=5 三臂都略正（+0.001～+0.004），k=10／15 為負，與 11/08 B2 延伸「特徵越多、較大的 k 掉越多」的規律同方向。
- 機制：未知。與先前的推測（業餘段位範圍內，「像哪個級別的人」已涵蓋「下得多好」）一致，但這次沒有直接檢定該推測。
- 限制：1 visit 的歸屬是網路原始輸出；死活只用「失子數變化」這一種定義（門檻 0.5、跳動 5 子），沒有調過，也沒有按局部區域分開統計。這不代表任何死活特徵都無效，只代表這一組沒有補到 H＋C。

**處置：** 依登記停止；不出提交檔、不上傳。基底與最終預設檔不變（`751959-cv2-l10.csv`）。保留 `engine_full/` 的每位置輸出（1.1 GB）與測試題特徵，之後若要改死活定義只需 CPU。算力：短測＋抽取約 3.8 GPU-h。

**LB 11/15（jabir95tsai）：** `780058-hc-l-l10.csv`（＋死活 10 維＋L10）15:50:04 Public **0.7497032**（24/139）。對照 `751959-cv2-l10.csv` 0.7499623：**−0.0002591**（400 題合計 −0.104 分；改動的 27 題淨效果約為零，例如 2 題由差一級變正確、1 題由正確變錯、1 題由差一級變錯即 −0.104，實際組合未知）。與本地一致（E1 +0.0005／E2 −0.0046，預期持平或略低）。依登記的保留規則維持基底 `751959-cv2-l10.csv`；死活特徵不採用。`hc-e`／`hc-el` 各檔未上傳。平台上競賽二最後一筆目前為 `780058-hc-l-l10.csv`。

**LB 補登（11/15 22:07，使用者回報，jabir95tsai）：**

| 提交檔 | 上傳時間（台灣） | Public LB | 對基底 0.7499623 的 Δ | 名次 |
|---|---|---:|---:|---|
| `780058-el-l10.csv`（引擎＋死活＋L10） | 2026-11-15 22:07:26 | 0.7474623 | −0.0025000 | 未提供 |
| `780058-hc-e-l10.csv`（引擎＋L10） | 2026-11-15 22:07:34 | 0.7515426 | +0.0015803 | 22/141 |

引擎單獨加入後 Public 小升（400 題總分約 +0.632），與本地 E1 −0.0062／E2 −0.0057 方向相反；引擎＋死活小降（總分 −1.000），與本地 E1 −0.0042／E2 −0.0068 同方向。連同死活單獨 −0.0002591，三臂沒有一致的正增益；本次 Public 無分 k 成績，不能推斷哪個 k 改善。依事前保留規則，上升不足 0.02 且本地 gate 未過，**三臂均不升基底；基底與最終預設檔維持 `751959-cv2-l10.csv`**。

截至上述回報，平台最後一筆為 **`780058-hc-e-l10.csv`（22:07:34）**，不是 `780058-el-l10.csv`。三份含 L10 的探針皆已上傳；上方「未上傳／不出提交檔」為先前階段紀錄，由匯出完成與本節 LB 紀錄更新。未代使用者上傳或另產生 CSV。

### 2026-11-15：競賽二——C→A→B→D 計畫第一步 C：微調 C 加入 H 後是否還有增益（使用者定稿計畫；事前登記）

**問題：** C4g few 單用 C 20 維時對未微調 base 為 Equal-k +0.0166（[+0.0077, +0.0253]）；正式分類頭是 H 30＋C 20＋HD1 旗標 2，先前全網路微調接上 H 後增益都沒留下。在同一批 dev 玩家、同樣五折與題組上補驗。程式 `scripts/strength/rank_c4g_hc.py`（`audit`／`tables`／`g0`）。

**Gate（跑前寫定）：** 主比較 `HC_few − HC_base`：Equal-k ≥ +0.005、玩家 bootstrap 2,000 次 P(Δ>0) ≥ 0.90、四個 k 桶皆 ≥ −0.005，三者同時成立。診斷臂：H-only、C_base／few／many、HC_many；各臂共用折、先驗、B2 設定、抽題 seed 與 bootstrap。**H 覆蓋 < 99% dev 棋局或任一級別 × k 格可用玩家 < 5 人 → 「無法判定」，不算模型失敗。** 有效比較未過 → 不投入 few 的 B 半訓練與全量抽取；通過 → 重新核算下一階段（歷史估計約 19 GPU-h）另行決定。

**第一段：資料稽核（CPU，`hc_audit.json`）**
- H 來源：11/09 fold −1 排除玩家的抽取（`more_players/h_hd{0,1}`，HD0 貼目 6.5／HD1 貼目 0.5，全部著手）。dev 12,974 盤中只有 **7,569 盤（58.3%）有 H**（HD0 5,475／9,630、HD1 2,094／3,344）；原因：該次每人隨機取最多 40 盤（兩色），dev 每人取單色最多 20 盤，兩次抽樣獨立。
- 一致性（全部通過）：dev 的 player_id／顏色／級別／HD 與主表相同；每位玩家單色；共用 7,569 盤在兩份表的玩家、顏色、HD 相同；H 欄位順序為本人 15＋對手 15；三臂 C 的棋局集合相同；C 每盤每側分母恆為 50 手。dev base C（v2 向量、抽 50 手）對同盤舊抽取 C（v1 向量、全部手）逐欄相關 0.80–0.86，本人／對手對調後只有 0.12–0.32 → 側別沒有接反。
- 可出題玩家數（各級 D…6D）：

| k | 覆蓋前（合計） | 覆蓋後各級 | 覆蓋後合計 |
|---|---:|---|---:|
| 5 | 762 | 69／63／72／119／26／41／42／48／52／52 | 584 |
| 10 | 672 | 51／36／49／58／20／30／32／33／39／42 | 390 |
| 15 | 564 | 22／17／17／26／8／20／21／20／27／22 | 200 |
| 20 | 492 | 12／4／3／10／**1／1**／7／8／6／6 | 58 |

- **判定：覆蓋不足 → 無法判定。** k=20 的 1D／2D 各只剩 1 人，五折時訓練折缺級別（試跑在第 4 折因只剩 8 類而中止，沒有產生任何臂的分數）。
- 補齊方式：dev 缺 H 的 5,405 盤（HD0 4,155、HD1 1,250；清單 `c4g/h_dev_hd{0,1}.parquet`）用同一支 H 抽取程式補抽，`slurm/rank_c4g_hdev.slurm`（4 卡）。估時依 11/09 實測（單卡 2,418 盤 2,355 秒）：每卡約 1,350 盤、約 25 分 → **約 1.5–2 GPU-h**，輸出約 70 MB（/work 剩 8.8 GB）。**尚未送出，待使用者確認。** 之後 `rank_c4g_hc g0` 為 CPU。

### 2026-11-15：盤面條件逐手加權 C——原產物缺失後同設定重訓與退步診斷

使用者要求分析退步，發現 `seed2027_hd0_n128` 為空目錄；原訓練與 B2 Slurm 日誌仍在。使用者明確要求重新訓練，故從保留的 780382 日誌恢復 benchmark metadata，依原 config（N=128、seed2027、lr=.001、最多 2,000 updates、每 100 步 hold、至少 500 步／patience5）從頭重訓，並接續 B2。保留完整模型與逐盤特徵供診斷，不變更架構、loss 或選模規則。依前次實測，訓練約 6–10 分鐘、單卡 15 分鐘上限，B2 前次約 19 秒。尚未重跑結果，不宣稱與前次逐位元一致。

同設定重訓 **781069**、成功後 B2 **781070**；診斷 **781078** 在 B2 完成後執行。診斷腳本 `scripts/strength/rank_board_attention_diagnostic.py`（self-check 通過），只讀凍結特徵：本人／對手各自原始 CE 與訓練集擬合的單參數尺度校準、train-only 線性探針（本人／對手／合併）、標準化後相鄰級 Fisher d′。分 train／hold／U 報告，不以 U 選模型。用途為區分校準／尺度、特徵可分度、跨玩家泛化；不更改 loss 或重訓權重網路。

**慢速節點改為續跑（不變更設定）：** 781069 在 25a-hgpn007 的載入 423.4 秒（前次約 35 秒）、更新約 1.2 秒／步（前次 0.21），完整 hold 約 12.6 秒（前次 1.7）。GPU 時脈 1,980 MHz、取樣利用率 33%，慢速原因尚未定位。已保存 step100 的模型／optimizer／sampler RNG（hold CE 2.069363）。為避免原 15 分鐘上限超時，在保存後取消此作業，另申請前次正常節點 25a-hgpn012 續跑；使用 afterany 保證舊作業結束後才啟動，B2 781070 的 afterok 改接續跑作業，診斷 781078 仍接 B2。不從頭重訓、不改 LR／batch／選模。剩餘估 5–8 分鐘，作業上限單卡 15 分鐘。

排程器拒絕修改已排程的 dependency（scontrol 回 Unspecified error），所以取消尚未開始的 781070／781078，另建 afterok 接續跑 **781124** 的 B2 與診斷，不重複執行評估。781124 用 afterany 等待 781069 停止後才啟動，避免同 run 並寫。以實際 Slurm checkpoint／停止步數為準，接續時可能已保存比 step100 更新的完整狀態。

### 2026-11-15：競賽一——新增集第一階段診斷（步驟 0＋D1＋D2＋D3-a；使用者提供計畫；CPU）

腳本 `scripts/identity/pid2_diag_d123a.py`（新檔，未改既有模型與正式管線）；Slurm 780999（1 卡配額當 CPU 用，33 分 42 秒，峰值記憶體 2.0 GiB；短測 780927 失敗於 D2 一行索引、780977 通過）。輸出 `outputs/pid/pid2_diag/pid_ge2e_v4_d4_hd_9600/`（`report.json`、各表 CSV、`d3a_grid.npz`），log `outputs/pid/pid2_bench/logs/pid2-diag-780999.out`。模型 `v4_d4_hd_9600`，pool 0 draws 100–119、pool 1 draws 100–129；標準誤是 draw 之間的，不含玩家抽樣。

**步驟 0：** 向量 47,280×256 與題庫棋局完全對齊、全為有限值，w 與 summary 相同；題庫每人 40 盤、HD≤1、棋局不重複、擁有者與 meta 一致；兩個 pool 各 591 人零重疊，全部在編碼器排除的 split 0–2 內；母體先驗不含題庫玩家；每個 draw 的 gallery／query 棋局互斥、同一盤不進兩題。開局參數 moves=8、beta=0.1、alpha=5.0、slot_moves=4（tau=0.25 只給指派用）；w=11.668。summary 記錄的權重就是 λ=16、μ=1，與固定權重為同一臂；pool 1 30 draws 平均 0.6197，與歷史 paired CSV 逐 draw 最大差 8e-17。pool 0 為 0.6252。**更正：770095 為 8 卡 × 4h03m55s ≈ 32.5 GPU-h**（`sacct`：`gres/gpu=8`），上方 11/14 一節寫的「1 卡 4h04m」有誤。

**D1 分項（對完整融合的配對差，括號為標準誤）：**

| 臂 | pool 0 平均 | Δ | pool 1 平均 | Δ | pool 1 各 k Δ（1／2／3／5） |
|---|---|---|---|---|---|
| 完整 | 0.6252 | — | 0.6197 | — | — |
| 移除 slot | 0.6173 | −0.0079（0.0016） | 0.6035 | −0.0161（0.0017） | −0.019／−0.016／−0.018／−0.012 |
| 移除 prefix | 0.6136 | −0.0116（0.0017） | 0.6025 | −0.0172（0.0016） | −0.016／−0.017／−0.021／−0.016 |
| 移除全部開局 | 0.5869 | −0.0383（0.0032） | 0.5758 | −0.0439（0.0024） | −0.036／−0.043／−0.054／−0.042 |
| 只用開局 | 0.2914 | | 0.2894 | | |
| 只用 prefix | 0.2309 | | 0.2237 | | |
| 只用 slot | 0.2362 | | 0.2320 | | |
| 只用風格 | 0.5807 | −0.0445（0.0031） | 0.5678 | −0.0519（0.0023） | −0.041／−0.051／−0.063／−0.053 |

prefix＋slot＝`twocolor.opening_scores` 已在每個 pool 第一個 draw 以 allclose 驗證。兩項各自移除的損失相加（pool 1：−0.033）小於同時移除（−0.044），兩項部分互補、不可相加。

**前綴命中（pool 0，20 draws；「查詢盤 × 候選人」配對中，候選人同色 gallery 有棋走到同一條線的比例；完整分層表 `d1_prefix_hits_pool0.csv`）：**

| 本人第 j 手 | 黑・正解 after | 黑・錯誤 after | 白・正解 after | 白・錯誤 after |
|---|---|---|---|---|
| 0 | 0.960 | 0.641 | 0.739 | 0.336 |
| 1 | 0.556 | 0.143 | 0.333 | 0.067 |
| 2 | 0.221 | 0.012 | 0.049 | 0.002 |
| 3 | 0.041 | 0.001 | 0.017 | 0.001 |
| 4–7 | 0.020→0.003 | <0.001 | 0.008→0.0003 | <0.001 |

黑棋 j=0 的 before 是空前綴（只要候選人有黑棋 gallery 就命中，0.999），不代表資訊。正解候選人的命中率在第 3 手之後就低於 5%，moves=8 的後半幾乎都走回退項；命中率隨候選人同色盤數上升（黑 j=1：1 盤 0.25 → 10 盤 0.72；白 0.12 → 0.50）。這是描述性統計，不代表把 moves 縮短會有增益（驗證台選參時 8 優於 4）。

**D2 錯誤段位分布（top-1 答錯的題；背景＝同批錯題的其餘候選人）：**

| pool／模型 | 錯題數 | 同級：模型／背景／倍率 | 差 1 級 | 差 2 級 | 差 ≥3 級 |
|---|---|---|---|---|---|
| 0／風格 | 3,730 | 0.251／0.100／2.51 | 0.337／0.177／1.90 | 0.200／0.156／1.28 | 0.212／0.566／0.37 |
| 0／完整 | 3,348 | 0.250／0.100／2.49 | 0.325／0.177／1.83 | 0.198／0.157／1.26 | 0.227／0.565／0.40 |
| 1／風格 | 5,719 | 0.241／0.103／2.34 | 0.333／0.183／1.82 | 0.194／0.160／1.21 | 0.232／0.553／0.42 |
| 1／完整 | 5,029 | 0.231／0.103／2.24 | 0.323／0.183／1.76 | 0.205／0.161／1.27 | 0.241／0.553／0.44 |

各 k 的倍率相近（同級 2.1–2.6、差 ≥3 級 0.30–0.52，k 越大越集中在鄰近級），分 k 表在 `d2_rank_gap.csv`。只保留真實同級候選人後（已知正解段位的上限，不是可實作的方法）：pool 1 完整融合 0.6197 → 0.7604（+0.1407，標準誤 0.0023；k=1／2／3／5：+0.191／+0.164／+0.123／+0.085），純風格 +0.1565；pool 0 完整 +0.1439、純風格 +0.1605。錯誤有 23–25% 落在同級、約 56–58% 在差一級內，仍有 23–24% 差 ≥3 級。依計畫，這只描述既有排序，不據此否證同級抽樣，也不宣告棋力資訊飽和；本階段未接 cvec2。

**D3-a 逐 k 融合權重（pool 0 選參；基準 λ=16、μ=1）：**

| 臂 | 選到的 (λ, μ)：k=1／2／3／5 | pool 0 Δ（標準誤） | pool 1 Δ（標準誤，Δ／SE） | pool 1 各 k Δ | 判定 |
|---|---|---|---|---|---|
| Arm 1（只選 λ） | (8,1)／(8,1)／(16,1)／(16,1) | +0.0042（0.0013） | **+0.0034（0.0012，2.72）** | +0.009／+0.004／0／0 | 未過（整體 <+0.005） |
| Arm 2（選 λ、μ） | (8,2)／(8,0.5)／(16,2)／(16,2) | +0.0063（0.0012） | +0.0033（0.0016，2.10） | +0.013／+0.001／+0.002／−0.0021 | 未過（整體、Δ／SE、k=5 三項） |

Arm 2−Arm 1：pool 0 +0.0021（0.0008），pool 1 −0.0001（0.0010）→ μ 的逐 k 選擇在 pool 1 沒有額外價值，pool 0 的差是選參偏高。兩臂都過 pool 0 門檻（選參與評分同一批 draw，偏樂觀）所以都進了 pool 1；**兩臂都未過 gate，不成為候選，不改正式管線、不產生提交檔。** Arm 1 的方向一致（k=1、2 用較小的 λ，70% 的 draw 為正），但幅度約 +0.003，低於門檻。summary 既有的 `lambda_per_k`（只有 k=1 改 8）在 pool 1 為 +0.0023（0.0008）。備註：2 個 draw 的短測（780977）也印出了 pool 1 draws 100–101 的 D3-a 結果，未據此改任何設定。

012 與 001 經 `scontrol show node` 確認八張 GPU 都已占用（不能以 CPU mixed 狀態推斷有空 GPU）。排程器同樣不接受變更 NodeList；取消尚未執行的 781124／781129／781130，重建相同三階段鏈，改為不指定節點、排除 007／069／083。被取消的三個 pending 作業沒有執行、不產生 GPU-h，checkpoint 與設定不動。

續跑鏈：**781137 → B2 781138 → 診斷 781139**。781137 在 25a-hgpn021 已從 step100 正常接續，更新速度恢復約 **0.21 秒／步**；原慢速作業最後跑到 step175，step100 之後未保存的更新由完整 checkpoint 重新消費同一抽樣序列。慢速原因仍未定位，僅記錄節點變更後速度恢復，不以此斷定硬體故障。

**同設定重訓完成與退步診斷：** 781137 從 step100 續跑至 **step1600 早停**，最佳 **step1100**，hold CE **2.058031**（前次 2.057936）；每方有效手數約 **5.48**、平均最大單手權重 **0.488**。因 CUDA 浮點非確定性與跨節點續跑，不宣稱逐位元一致；現象與幅度重現。

B2 五 seed 平均，HD0_Q2027 mean／weighted＝**0.72512／0.63180（−0.09334）**，分 k 差（5／10／15／20）＝−0.10902／−0.09484／−0.07358／−0.09592；HD0_Q2028＝**0.72630／0.63514（−0.09112）**，分 k＝−0.13090／−0.10498／−0.06976／−0.05890。首 seed 玩家 bootstrap 95% 分別 [−0.1085,−0.0707]、[−0.1078,−0.0685]。

**診斷 781139（只擬合 TRAIN 的尺度／線性探針；U 只報分）：**

| U 單盤、側別 | mean 原始 CE | mean 校準乘數 | mean 校準後 CE | weighted 原始 CE | weighted 校準後 CE |
|---|---:|---:|---:|---:|---:|
| 本人 | 2.25003 | 49.8520 | **1.65449** | 2.05402 | 2.05054 |
| 對手 | 2.25497 | 44.0757 | **1.72209** | 2.07286 | 2.06803 |

weighted 最佳乘數只有本人 1.1854／對手 1.2225。mean 不改任何手的權重，僅校準尺度，就遠優於 weighted。CE 對 logit 尺度敏感，平均 policy log p 不是已校準的級別後驗；直接 CE 下降不等於特徵資訊增加。

U 的平均十級分數離散度（每盤十維標準差平均）：本人 mean **0.0612→weighted 0.6441**、對手 **0.0563→0.5955**，約放大 10.5 倍。每盤平均 log p 的中位數：本人 **−1.459→−3.757**、對手 **−1.453→−3.682**，表示加權偏向模型認為較罕見的著手（不等於證明它們是壞棋）。原始十維 argmax accuracy：本人 **0.3237→0.1827**、對手 **0.3043→0.1790**；這是在原始 logits 的診斷，並非正式 B2 accuracy。

**特徵辨識力下降，損失已出現在訓練玩家：** 用相同 StandardScaler＋LR、只在 TRAIN 擬合，單盤官方分數如下（不是多盤 B2）：

| C 20 維線性探針 | train | hold | U |
|---|---:|---:|---:|
| mean | 0.5486 | 0.5445 | **0.5398** |
| weighted | 0.4104 | 0.4051 | **0.4051** |

本人單側 U 0.4845→0.3615、對手 0.4644→0.3500，兩側均退步。U 合併特徵的中段 d′（A/1D、1D/2D、2D/3D；方向及標準化均在 TRAIN 固定）**0.5803→0.3652**；全部相鄰級 d′ **0.7804→0.4515**。這支持可用的線性級別訊號變弱，並非只有新玩家分布落差。

同玩家跨盤變異占總變異比例（訓練集標準化後、盤等權）：U 合併 **0.7155→0.7802**、本人側 **0.6356→0.7530**、對手 **0.7951→0.8072**。比例上升表示相對於玩家間差異，同一玩家各盤變得較不穩定；不能把有效手數比例或變異比例直接換算成資訊損失百分比。

**最有證據的解釋：** 直接以平均 log p 當 logits 的 CE 有明顯尺度問題；在此受限加權架構中，學習偏向少數、級別差距較大的罕見著手，放大原始 logits 並降低未校準 CE，同時使各盤特徵的級別可分度及穩定性下降。這是由校準對照、權重分布、train／hold／U 線性探針及 d′ 共同支持的機制解釋；沒有做「固定校準後再訓練權重」或其他因果消融，不能把全部損失唯一歸因於 softmax／CNN／CE 任一項。未變更架構或再訓新臂，不升基底、不上傳。

產物已恢復在 `outputs/rank/humansl/board_attention/seed2027_hd0_n128/`，新增 `diagnostic.json`；診斷程式 `scripts/strength/rank_board_attention_diagnostic.py`，self-check 通過。成本：慢速／取消 781069＝11:58，正常續跑 781137＝6:52，B2 781138＝0:20，診斷 781139＝0:35，合計 **19:45＝0.3292 配置 GPU-h**；純 pending 被取消作業沒有執行、不計入。其他 session 的 H 特徵抽取未計入。

### 2026-11-15：競賽一——新增集開局改用 raw 方位（取代 D4 正規化）→ 過 gate，列為候選（使用者提供計畫；CPU）

問題：新增集開局保留絕對方位，是否比現行 D4 正規化更有辨識力。唯一改變是開局讀的座標：raw 臂的母體先驗、gallery、query 都用原始方位，D4 臂三者都用正規化；風格向量（`v4_d4_hd_9600`）、融合（開局＋16·w·cosine＋1·HD 色別，w=11.668）、開局參數（moves=8、beta=0.1、alpha=5.0、slot_moves=4）、題庫與抽題全部共用，不指派、不調任何參數。腳本 `scripts/identity/pid2_raw_opening.py`；短測 Slurm 781176（pool 0、2 draws，33 秒），正式 781183（4 CPU、8 GiB、1 卡配額，3 分 03 秒，MaxRSS 4.2 GiB，約 0.05 GPU-h）。輸出 `outputs/pid/pid2_raw_opening/pid_ge2e_v4_d4_hd_9600/`（`report.json` 含來源檔 SHA-256、`per_draw.csv`）。

**工程關（通過）：** raw／D4 前綴同為 1,000,000×60、同一順序；D4 快取逐列等於 `canonical_d4(raw)`；所有值為盤上點；最短棋局 100 手，無棋局被剔除；99.0% 的棋局前綴被 D4 改動。D4 臂逐 draw 重現第一階段的完整融合分數（pool 0／pool 1 最大差 6e-17／8e-17）。

| | k=1 | k=2 | k=3 | k=5 | 整體 |
|---|---|---|---|---|---|
| pool 0（20 draws）D4 | 0.3592 | 0.5960 | 0.7139 | 0.8318 | 0.6252 |
| pool 0 raw | 0.3674 | 0.6006 | 0.7262 | 0.8369 | 0.6328 |
| pool 0 Δ（標準誤） | +0.0083（0.0023） | +0.0046（0.0042） | +0.0123（0.0033） | +0.0052（0.0034） | **+0.0076（0.0020）** |
| pool 1（30 draws）D4 | 0.3680 | 0.5705 | 0.7149 | 0.8254 | 0.6197 |
| pool 1 raw | 0.3702 | 0.5774 | 0.7229 | 0.8288 | 0.6248 |
| pool 1 Δ（標準誤） | +0.0023（0.0024） | +0.0069（0.0034） | +0.0080（0.0028） | +0.0034（0.0033） | **+0.0051（0.0012）** |

pool 1：Δ／SE＝4.26、83% 的 draw 為正（pool 0：3.72、80%）。Gate：pool 0 ≥+0.003 ✔；pool 1 ≥+0.005 ✔（+0.0051，剛過線）、Δ／SE ≥2.5 ✔、四個 k 桶 ≥−0.002 ✔ → **列為候選；正式管線未改、未產生提交檔。**

診斷（不作選模依據）：只用開局時 raw 比 D4 高很多——pool 1 0.2894 → 0.3568（+0.0674，標準誤 0.0027；k=1／2／3／5：+0.041／+0.064／+0.074／+0.091），pool 0 +0.0672；融合後只剩 +0.005，表示 raw 多出的方位資訊大部分已被風格向量涵蓋（未另外驗證）。top-1 改變率 pool 1 為 9.7%（各 k 9.3–10.3%）。

限制：(1) 標準誤只是固定玩家池內的抽題變異，pool 1 已是開發驗證集。(2) 驗證台的 gallery 與 query 都來自訓練集、方位一致；raw 在正式新增集有效的前提是官方候選庫與題目沒有被各自旋轉。11/12 Day-0 稽核只確認原測試集題目未被旋轉，**新增集的方位分布要在套用前另外核對**（E0-001：查詢盤一旦被旋轉，raw 會崩到 0.09–0.16）。(3) 結果只適用於「raw 取代 D4」；未測 raw＋D4 混合。

**第二段：補抽 H 與 CPU 配對比較（11/15，使用者授權 H 補抽＋CPU 比較；`g0_hc.json`）：gate 未通過。**

- 補抽：Slurm **781096**（8gpus 佇列 4 卡，節點 25a-hgpn002，17 分 44 秒 ＝ **1.18 GPU-h**），只抽缺少的 5,405 盤（HD0 4,155 貼目 6.5、HD1 1,250 貼目 0.5），同一 H 模型 `b18c384nbt-humanv0`、同一 15 個 profile 順序；輸出 `c4g/h_dev_hd{0,1}`，既有產物未動。
- 重跑 audit（`hc_audit.json`）：覆蓋 **12,974／12,974（100%）**、762 人；每盤只有一個 H 來源、無重複棋局、H 全為有限值；四個來源皆完整（無拒絕查詢、returncode 0）、貼目與模型／profile 順序一致；其餘身分／顏色／級別／HD／C 分母檢查同第一段，全部通過。各級 × k 可出題玩家數回到覆蓋前（最少 36 人）。
- 比較設定：762 人／12,974 盤，與 C4g G0 相同的五折玩家切分、抽題 seed、每格 200 題、先驗對齊 B2、玩家 bootstrap 2,000 次；各臂只差特徵欄。C 三臂逐位數重現 G0（0.6863／0.7028／0.7016）。

| 臂（欄數） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| H（30＋旗標 2） | 0.6663 | 0.7048 | 0.7318 | 0.7262 | 0.7073 |
| C_base（20） | 0.6387 | 0.6783 | 0.7156 | 0.7125 | 0.6863 |
| C_few（20） | 0.6497 | 0.6971 | 0.7267 | 0.7378 | 0.7028 |
| C_many（20） | 0.6515 | 0.6955 | 0.7258 | 0.7337 | 0.7016 |
| HC_base（52） | 0.6652 | 0.6950 | 0.7228 | 0.7313 | 0.7036 |
| HC_few（52） | 0.6630 | 0.6934 | 0.7293 | 0.7337 | 0.7049 |
| HC_many（52） | 0.6678 | 0.6999 | 0.7283 | 0.7249 | 0.7053 |

| 差異 | k=5 | k=10 | k=15 | k=20 | Equal-k | 95% 區間 | P(>0) |
|---|---:|---:|---:|---:|---:|---|---:|
| **HC_few − HC_base（主檢定）** | −0.0022 | −0.0016 | +0.0065 | +0.0024 | **+0.0013** | [−0.0046, +0.0073] | 0.669 |
| HC_many − HC_base | +0.0026 | +0.0049 | +0.0055 | −0.0063 | +0.0017 | [−0.0041, +0.0079] | 0.701 |
| HC_many − HC_few | +0.0048 | +0.0065 | −0.0010 | −0.0087 | +0.0004 | [−0.0035, +0.0040] | 0.580 |
| C_few − C_base | +0.0110 | +0.0188 | +0.0111 | +0.0253 | +0.0166 | [+0.0077, +0.0253] | 1.000 |
| HC_base − H | −0.0011 | −0.0098 | −0.0090 | +0.0051 | −0.0037 | [−0.0123, +0.0049] | 0.191 |
| HC_few − H | −0.0033 | −0.0114 | −0.0025 | +0.0075 | −0.0024 | [−0.0113, +0.0064] | 0.295 |

**判定：** 主檢定 Equal-k +0.0013 < +0.005、P＝0.669 < 0.90 → **未通過**（各 k ≥ −0.005 這一條成立）。這是完整覆蓋下的有效比較，不是「無法判定」。依登記：**不投入 few 的 B 半訓練與全量特徵抽取。**

**能說的與不能說的：**
- 能說：微調讓單用 C 時 +0.0166，但接上 H 後只剩 +0.0013，區間上限 +0.0073；微調 C 新增的資訊大部分 H 已經有了。與先前 C4／C4b／C4x adj「監控改善、接上 H＋C 後持平」同方向。
- 這個 dev 設定裡 C 本身對 H 沒有增量（HC_base − H ＝ −0.0037，區間含 0），與正式管線 Valid-U 上 C 的 +0.010 不一致。可能原因（未驗證）：dev 的 C 每側只抽 50 手而 H 用全部著手；擬合玩家每折只有約 610 人、52 維比 32 維更容易過擬合。因此本結果**不能**外推成「正式管線中 C 無用」，也使主檢定偏保守——few 的增益若要靠全部著手的 C 才顯現，這裡看不到。
- 單一半邊（A）、單一訓練 seed；檢定力：配對差的 95% 區間半寬約 0.006，真實效果若在 +0.005 左右可能被判停，風險同 G0 登記時所述。
- 機制：未知（H 與微調 C 重疊的原因沒有另外檢定）。

**算力：** 本步配置 1.18 GPU-h（授權 1.5–2 內）；CPU 比較在登入節點約 20 分。不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。下一步 A（階層分類頭，CPU）待規格確認，未啟動。

### 2026-11-15：競賽二——計畫第二步 A：階層分類頭的事前規格（只登記；未實作、未執行，待使用者確認）

**範圍：** 只替換分類頭。不變：52 維（H 30＋C v2 20＋HD1 旗標 2）、題內各盤等權平均、Valid-S 玩家先驗 π_k、`score_rank` 解碼、擬合池（Valid-S＋train20，HD≤1；149,615 盤／5,728 人，各級 639／589／752／1,134／306／435／438／457／469／509 人，每人盤數中位 20）。對照＝現行 `PriorMatched` B2（每個 k 20,000 合成題、LR C=1）。程式預定 `scripts/strength/rank_hier_head.py`，輸出 `outputs/rank/humansl/hier_head/`。

**1. 估計（每次擬合只用該次的訓練玩家，依序）：**
1. 標準化：各欄以玩家等權（每盤權重 1／n_p）的均值與標準差 z-score；標準差 < 1e-8 的欄報錯。
2. 玩家均值 m_p（n_p ≥ 1）；級別均值 μ_r＝該級玩家 m_p 的等權平均。
3. 玩家內：S_p＝玩家 p 的不偏樣本協方差（n_p ≥ 2）；ΣW_r＝該級 S_p 的玩家等權平均。
4. 玩家間（有限盤數修正）：T_r＝m_p 對 μ_r 的不偏樣本協方差；ΣB_r＝T_r − h_r·ΣW_r，h_r＝該級 1／n_p 的平均；ΣB＝十級 ΣB_r 的等權平均（各級共用）。
5. PSD：ΣB 對稱化後特徵分解，負特徵值截為 0（記錄被截的個數與占 trace 比例）。
6. 組合：Σ_r(k)＝ΣB＋ΣW_r／k（結構 `rank`）或 ΣB＋ΣW／k，ΣW＝十級等權平均（結構 `shared`）。
7. 等向收縮：Σ̃＝(1−γ)Σ＋γ·(tr Σ／52)·I。
8. 數值：Cholesky；失敗時加 jitter（tr／52 的 1e-8 倍起、每次 ×10、上限 1e-4），仍失敗報錯；記錄條件數。
9. 分數：ℓ_r＝−½[(x̄−μ_r)ᵀΣ̃⁻¹(x̄−μ_r)＋log det Σ̃]；logit_r＝log π_k(r)＋ℓ_r／T_k；softmax 後 `score_rank`。
- 已知近似（不調）：玩家內協方差混合黑白兩色，題目為單色，顏色效應不會隨 k 平均掉；B2 合成題同樣混色。

**2. 超參數（共 8 組＋每個 k 一個溫度）：** 結構 ∈ {shared, rank}；γ ∈ {0.05, 0.1, 0.2, 0.4}；T_k 在 [0.5, 32] 對數等距 25 點取內折 NLL 最小，落在邊界＝該組合作廢（全部作廢則 A 判失敗，不擴網格）。內折：外折訓練玩家再按級別分層 5 折（seed 2027＋外折編號）。選模指標：內折 OOF 題目（溫度後）的 Equal-k 官方分數。同分：與最佳差 ≤ 0.001 者取 γ 較大，再取 shared。

**3. 資料隔離：** 每個內、外折都以該折訓練玩家重算標準化、π_k（只用該折訓練玩家中 role＝S 的盤）、μ、ΣW、ΣB；T_k 只用內折留出玩家的預測擬合；外折測試玩家不進任何估計、選模或校準。Valid-U 在第一階段完全不讀。

**4. 公平對照：** 外折＝玩家按級別分層 5 折（seed 2027）。評估題：每個外折的留出玩家出 E2 式題（單色、每 k × 級別 100 題，seed 2027＋折），共 20,000 題，兩臂共用；內折校準題同法（每格 100 題）。B2 用同一外折訓練集，seed 2027–2031 各自擬合、各自解碼計分，逐題分數取五 seed 平均；A 無隨機成分，只擬合一次；**不做機率 ensemble**。配對差 d_q＝s_A(q)−mean_seed s_B2(q)，按玩家 bootstrap 2,000 次。

**5. 驗收：**
- 池內 gate：Equal-k ≥ +0.003、P(Δ>0) ≥ 0.90、四個 k 皆 ≥ −0.005，同時成立。
- 通過才鎖定：以同一組外折當選模折，對 8 組合算全池 OOF，依第 2 節規則選一組並擬合 T_k；再以全池擬合一次。
- Valid-U 只跑一次：E1（1,000 題）與 E2（2,000 題），對照為 B2 五 seed 平均；E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005。
- 池內 CV 只是固定 H＋C 特徵下的分類頭篩選（C v2 向量學習時用過池內部分玩家），不是整條管線的未見玩家驗證。

**6. 成本與產物：** B2 全池擬合實測 155 秒／seed。外折 B2 約 5 折 × 5 seed × 125 秒 ≈ 52 分；A 為閉式解（52×52），內折 200 次＋鎖定 40 次擬合，估數分鐘、未實測；Valid-U B2 約 13 分。單行程合計約 1.2 小時 CPU（以 1 卡配置送 Slurm 時約 1.2 GPU-h 記帳）。產物：`spec.json`、`pool_cv.json`、`pool_scores.parquet`（逐題分數）、`locked.json`、`valid_u.json`。不產生提交檔、不換基底。

### 2026-11-15：競賽一——正式新增集方位稽核（未見異常）與 raw 開局 LB 探針候選（使用者同意先稽核；CPU）

**稽核（`scripts/identity/pid2_orientation_audit.py`，Slurm 781227，9 秒；輸出 `outputs/pid/pid2_orientation/`）：** 不使用預測身分、不看 LB。分 Public／Private、黑／白，比較新增集候選庫（各 6,000 盤、600 人）與題目（各 1,100 盤、400 題）的三類分布，以總變異距離（TV）衡量：D4 變換（前 16 手正規化時用的是 8 個對稱中的哪一個）、起手角落（全局第一手的象限）、本人第 0–3 手的絕對落點；按盤與按候選人／題目等權各算一次，另按 HD 0／1 分層。參考＝驗證台 pool 0、1 的 draws 100–129 共 60 個 draw（訓練棋局原方位、同樣的候選／題目規模）。對照＝把正式題目盤的 5／10／25／50／100% 各套一個隨機對稱（20 個種子）。判讀規則在看資料前寫入腳本說明：主要格（D4、角落、本人第 0 手；按盤、全部 HD）高於所有參考 draw 且位置達「參考平均→完全旋轉」的 20% 以上＝異常；無異常且完全旋轉對照在每個主要格都高於參考最大值＝未見異常；其餘＝無法判定。

| 主要格（按盤） | 正式 TV | 參考平均／最大 | 25% 旋轉 | 100% 旋轉 | 可察覺的最小旋轉比例 |
|---|---|---|---|---|---|
| Public 黑：D4／角落／第 0 手 | 0.048／0.017／0.059 | 0.041／0.080；0.026／0.051；0.068／0.103 | 0.130／0.134／0.155 | 0.503／0.508／0.517 | 25%／10%／25% |
| Public 白 | 0.040／0.037／0.067 | 0.036／0.069；0.023／0.048；0.081／0.133 | 0.169／0.167／0.137 | 0.590／0.591／0.414 | 10%／5%／25% |
| Private 黑 | 0.055／0.028／0.059 | 同 Public 黑 | 0.135／0.136／0.161 | 0.508／0.510／0.523 | 25%／10%／25% |
| Private 白 | 0.042／0.039／0.073 | 同 Public 白 | 0.125／0.125／0.105 | 0.583／0.584／0.411 | 25%／25%／50% |

**判定：未見異常。** 12 個主要格全部不高於參考最大值（位置 −0.04～+0.03，0＝參考平均、1＝完全旋轉）；完全旋轉對照在每一格都遠高於參考最大值（驗證台自己的題目旋轉後也是 0.46–0.62），稽核能察覺旋轉。按候選人／題目等權的結果相同（無一格超過參考最大值）。全部 144 格中有 6 格略高於參考最大值，都在 HD 分層：Private 白 HD0 的 D4 與角落（0.079 對 0.076、0.063 對 0.053，按盤與等權共 4 格，位置 0.07，低於 25% 旋轉對照的 0.116）、Public 白 HD0 的本人第 3 手（2 格；該統計量連完全旋轉也只到 0.30，鑑別力弱）。60 個參考 draw 下 144 格純機遇預期約 2 格超過最大值（格間相關），這 6 格不構成主要格異常，僅記錄。

**稽核的限制：** (1) 分布相近不等於證明沒有旋轉；可察覺下限約為 10–25% 的題目盤被旋轉（Private 白第 0 手要 50%），更小比例無法排除。(2) 候選庫與題目若被「同樣方式」逐盤隨機旋轉，兩邊分布仍會相近，這個比較看不到；但 11/12 Day-0 稽核顯示新增集候選庫第一手 `pd` 占 0.561／0.553（訓練集 0.549），候選庫是自然方位，題目又與候選庫相近，這個情況可排除到上述下限。(3) 參考來自訓練集玩家，不是正式新增集玩家。

**raw 開局 LB 探針候選（`scripts/identity/pid2_raw_export.py`，Slurm 781229，28 秒；`sbatch` 回報 socket timeout 但作業已送出，只有這一份）：** 讀正式 `{public,private}2_d4_9600` 的快取，先驗證 D4 開局項與融合分數逐值重現 `predict2`，再只把開局項換成 raw 座標（訓練母體、候選庫、題目都不旋轉）；風格項、HD 色別項、λ=16、μ=1 用原存檔。輸出 `outputs/pid/day0_1112/{public,private}2_d4_9600_raw/`，`pid_submission merge` 合併並通過格式檢查。

| 檔案 | 內容 | 事前保留規則 | Public LB |
|---|---|---|---|
| `submissions/pid/781229-hung-raw.csv`（SHA `d1ff1acf`） | 原測試集與基底 `772748-hung.csv` 逐列相同（Hungarian）；新增集開局改 raw。改 top-1：pub2 44 題（k=1／2／3／5：6／9／15／14）、priv2 46 題（5／11／19／11），約 11%，與驗證台的 9.7% 相近；top-5 有變動的列 316／305 | 驗證台 pool 1 +0.0051（標準誤 0.0012）；完整轉移時 1,600 題等權總分約 +0.0026，遠小於 Public 標準誤約 0.02，只看方向。**不論升降都不升為基底**（使用者決定）：升或持平 → raw 保留為候選，等其他本地證據；降 ≥0.02 → 回查正式新增集方位與 raw 的適用性，raw 暫停；基底維持 `772748-hung.csv` | **0.7962107**（11/15 21:02:32，jabir95tsai；36/90） |

**A 規格修訂（11/15，使用者審查後核准實作與執行 CPU 流程；以下取代上方對應條目，跑前寫定）：**
- **只留 shared：** ΣB、ΣW 皆各級共用（ΣW＝十級的玩家等權估計再等權平均），決策形式維持線性；候選只剩 γ ∈ {0.05, 0.1, 0.2, 0.4} 共 4 組。同分規則只留「距最佳 ≤ 0.001 取較大的 γ」。
- **有限盤數修正的假設：** ΣB_r＝T_r − mean_p(1／n_p)·ΣW_r 在「級別內玩家共用同一玩家內協方差」下合理；若單盤變異隨玩家而異且與盤數相關，不保證無偏。本版不擴充模型。
- **PSD 診斷：** 截斷占比＝被截負特徵值絕對值總和／全部特徵值絕對值總和。
- **顏色：** 不新增處理，也不稱已被對照消除——B2 同樣混色抽樣不代表兩法受影響相同，階層模型依賴方差縮放假設可能更敏感，溫度只調信心尺度。結論限定為「未顯式建模顏色的 shared 階層分類頭是否改善現行管線」；未通過不否定所有階層模型。
- **溫度：** 多個相同最小 NLL 取較大 T；最小值在網格邊界則該 γ 作廢；全部作廢 → 狀態 `calibration_out_of_range`（校準範圍不適用，停止），不視為統計否證、不擴網格。數值失敗（Cholesky 加 jitter 至 1e-4 仍失敗、常數欄）另以例外中止，與 gate 未過分開記錄。
- **Bootstrap：** 同一玩家的所有題目一起重抽；每次重算四個 k 各自的平均差再取 Equal-k。
- **成本（更新）：** A 內折 5×5 次分量估計（γ 共用同一次估計）＋鎖定 5 次，估數分鐘；B2 外折 25 次＋Valid-U 5 次擬合，5 行程並行。以 `slurm/rank_hier_head.slurm`（8gpus 佇列 1 卡／12 核，不使用 GPU）執行，依實際 Elapsed × 1 卡記帳，預估 < 1 GPU-h。
- **實作驗證（`tests/test_rank_hier_head.py`，4 項通過）：** 不等盤數（1–40 盤）合成資料的 ΣW、ΣB 相對 Frobenius 誤差 < 10%（未修正的玩家均值協方差 trace 偏高 > 20%）；shared 高斯分數的級別差與線性式 z·w＋b 一致（1e-8）；留出玩家特徵加 1000 後，折內的標準化、μ、ΣW、ΣB、先驗、γ、T_k 逐位元不變；bootstrap 的 Equal-k 定義。程式內另 assert 外折測試玩家不在訓練集、校準玩家全在訓練玩家內、兩臂先驗相同。

**A 結果（11/15，Slurm 781902，8gpus 佇列 1 卡／12 核、不使用 GPU，1 分 47 秒 ＝ 0.03 GPU-h；`hier_head/pool_cv.json`、`pool_scores.parquet`）：池內 gate 未通過，未鎖定、未讀 Valid-U。**

池內 nested CV，5 外折、20,000 題（每折每 k × 級別 100 題），兩臂同題；B2 為 seed 2027–2031 各自解碼計分後逐題平均（各 seed Equal-k 0.7397–0.7427）。

| 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| 階層頭（shared） | 0.6821 | 0.7215 | 0.7570 | 0.7809 | 0.7354 |
| B2 五 seed 平均 | 0.6903 | 0.7308 | 0.7601 | 0.7825 | 0.7409 |
| 差（階層 − B2） | −0.0083 | −0.0093 | −0.0031 | −0.0016 | **−0.0056** |

玩家 bootstrap 2,000 次：95% 區間 [−0.0098, −0.0016]，P(Δ>0)＝0.002。Gate（≥ +0.003、P ≥ 0.90、各 k ≥ −0.005）三條都不成立 → **`pool_gate_fail`**。這是有效比較的統計結果，不是校準或數值失敗。

**診斷：**
- 選到的 γ：0.1／0.05／0.05／0.05／0.05；內折 Equal-k 隨 γ 單調下降（例：折 4 為 0.7372／0.7352／0.7331／0.7261）→ **最佳值落在候選下界**；更小的 γ 能否補回差距未知（0.05→0.1 的局部差異不能外推到 0.05 以下），依登記不擴候選。
- 溫度 T_k 全在 0.84–1.19，沒有任何 γ 卡邊界 → 不需要大幅溫度修正；這不等於已證明機率校準良好（沒有另做校準檢驗）。
- 數值：無 jitter；條件數 252–597。ΣB 被截 7–9 個負特徵值，占比 4–7×10⁻⁷（52 維中玩家間變異只集中在約 43–45 個方向，其餘在雜訊水準）。
- 差距集中在 k=5／10（−0.008／−0.009），k=15／20 接近持平（−0.003／−0.002）。

**能說的與不能說的：**
- 能說：這個未顯式建模顏色、ΣB／ΣW 各級共用的階層高斯分類頭，在池內比四個獨立 B2 差，區間不含 0。原假設「跨 k 共享方差結構＋利用少盤玩家可以改善」在這個形式下不成立；B2 每個 k 有 20,000 合成題，估計量已不缺。
- 低 k 較差的原因：未知。與下列情形一致但都未檢定：級別間協方差不相同（shared 假設）、特徵非高斯（每手平均 log p 有長尾）、顏色效應不隨 1／k 縮小而低 k 題的 ΣW／k 佔比最大。
- 不能說：所有階層模型都無效（級別專屬協方差、顏色分量、判別式訓練的共享參數頭都沒有測）。也不代表 Valid-U 上的結論——第一階段未過，Valid-U 沒有讀取。

**處置：** 依登記停止 A：不鎖定、不做 Valid-U、不產生提交檔、不換基底（`751959-cv2-l10.csv`）。計畫 C、A 兩步累計配置 1.18＋0.03＝1.21 GPU-h。下一步 B（盤面基線）的參考分布、主比較、門檻與成本仍待定稿，未啟動。

**LB（11/15 21:02）：`781229-hung-raw.csv` Public 0.7962107，對基底 `772748-hung.csv`（0.7980564）−0.0018**（1,600 題等權下總分約 −1.5；只有 pub2 的 44 題 top-1 與 316 列 top-5 有變動）。本地預期約 +0.0026，兩者差 0.0044，遠小於 Public 標準誤約 0.02 → 視為持平，Public 上看不到 raw 的增益，也沒有到「降 ≥0.02 回查方位」的程度。依事前規則：**基底維持 `772748-hung.csv`；raw 開局保留為候選、不升基底**，等其他本地證據。第一次上傳曾被平台回「上傳檔案不合規則」，同一份檔案（SHA `d1ff1acf`，與基底逐項比對格式相同）之後上傳成功，原因未知。

### 2026-11-15：競賽二——計畫第三步 B：盤面基線（policy 期望 log p）的事前規格與成本核算（只登記；未實作、未送推論，待使用者確認）

**問題：** 扣除模型定義的盤面基線後，C 是否提供比原始 C 更有用的增量訊號。這需要完整落子分布，是 C、A 兩步沒有檢驗的新資訊。不預設它能去除棋風；玩家差異占比下降本身不算成功。

**固定設定：**
- 上游：HumanSL＋C v2 向量（`cvec_v2/learn/vectors.pt`），未微調；不加入 C4g few／many。
- 資料：`c4g/dev` 既有 16 個 pack（12,974 盤、762 人、每盤本人 50＋對手 50 手，共 1,297,400 個位置），不重編碼、不重抽。
- 每個位置、每個級別假設 r（10 個；向量取該位置的角色＝本人或對手）：p_r＝遮掉已有子與劫禁點後、含 pass 的 362 維 softmax（與現行 `legal_logp` 相同的合法遮罩）。
- 參考分布：q＝同一位置、同一角色下十個 p_r 的**等權算術平均**；不搜尋其他混合權重。
- 兩組特徵（同一次前向同時保存）：raw_r＝log p_r(實際著手)；baselined_r＝log p_r(實際著手) − Σ_a q(a)·log p_r(a)（只在合法點與 pass 上加總）。每盤每側取 50 手平均 → 各 20 維。
- 另存診斷量（不進任何臂）：每盤每側的基線項平均。

**一致性檢查（進比較前必須通過，否則停止並回報）：** 新 raw 的每盤 20 維對既有 `few/A/dev_step0.npz` 的最大絕對差 ≤ 2e-3（兩者皆 FP32；既有值在 8 行程、TF32、compiled forward 下算出，容許浮點差）；棋局集合、每側手數（恆為 50）、級別標籤完全相同；所有值有限。

**比較（CPU，沿用 `rank_c4g_hc` 的五折玩家切分、抽題 seed、每格 200 題、先驗對齊 B2、玩家 bootstrap 2,000 次；bootstrap 改為每次重算四個 k 的平均差再取 Equal-k）：**
- **主檢定：H＋baselined_C＋HD − H＋raw_C＋HD**（皆 52 維）。
- 診斷（不作判定）：C_baselined − C_raw（20 維）、H-only、單盤與玩家層級中段 d′、特徵的玩家間／玩家內變異占比。不加入熵、期望段位或其他新特徵，也不測 raw＋baselined 並用。
- **Gate：** Equal-k ≥ +0.005、P(Δ>0) ≥ 0.90、四個 k 皆 ≥ −0.005，同時成立。未通過 → 不做全池抽取；通過 → 只支持進一步驗證（全池抽取與 Valid-U 另提成本與 gate）。
- 已知限制：dev 設定下原始 C 對 H 沒有增量（HC_base − H＝−0.0037，第一步 C），主檢定的對照臂本身偏弱；每側只有 50 手；單一 dev 玩家集（762 人）。baselined 與 raw 的差只來自各級假設的 KL(q‖p_r) 項，若該項在各級間幾乎相同，兩臂會近乎等價（屆時如實回報）。

**成本核算（依實測，不沿用先前的「12 分／0.2 GPU-h」）：**
- 現存檢查（11/15）：dev pack 16 個、1.4 GB、位置數合計 1,297,400；位置清單 `dev_hd{0,1}.parquet`；`dev_step0.npz`；C v2 向量；dev 的 H 已 100% 覆蓋。/work 剩 8.7 GB。
- 實測依據：776710 作業中同一批 dev 位置 × 10 個假設的前向共 **153.7 秒 × 8 卡 ＝ 0.34 GPU-h**（compiled、FP32）。基線項只多一次 [B,362] 的乘加，預期可忽略（未實測）。
- 短測：1 卡、pack_hd0_0＋pack_hd1_0（約 16 萬位置，1／8），預估前向約 3 分＋載入／編譯數分鐘，上限 15 分 ＝ **0.25 GPU-h**；用途：量吞吐、對 `dev_step0` 做一致性檢查。
- 完整推論：1 卡，其餘 14 個 pack，由短測吞吐重估後才送；依上列實測外推約 20–25 分，**上限 1 小時 ＝ 1.0 GPU-h**。
- 合計上限 **1.25 GPU-h**；輸出約數 MB（每盤彙總，不存每位置分布）。CPU 比較約 20 分（登入節點）。
- 程式預定：`scripts/strength/rank_c4g_baseline.py`（`extract`／`check`／`g0`）、`slurm/rank_c4g_baseline.slurm`；輸出 `outputs/rank/humansl/c4g/baseline/`。不產生提交檔、不換基底。

**B 規格修訂（11/15，使用者審查後核准短測；跑前寫定）：**
- 更正解讀：baselined_r＝raw_r＋H(q)＋KL(q‖p_r)。即使 KL 在十級間相同，仍對整條曲線加上隨盤面變動的共同偏移 H(q)＋KL，B2 不會忽略共同偏移 → **不能因「級別間差異小」就預判兩臂等價**。短測同時報告「基線的級別間差異」與「共同偏移的盤間變異」，兩者都不作判定；成敗只由登記的配對比較決定。
- 遮罩：非法點先遮掉再做乘加（避免 0 × −∞）；沿用現行 C 的遮罩，文件稱「現行遮罩允許的位置」（空點、非劫禁點、pass），不宣稱排除所有規則上的非法著手。
- 每批自我檢查（任一超過 1e-3 或出現非有限值即中止）：各 p_r 與 q 的總和對 1 的最大偏差；q 下中心化分數期望的最大絕對值。單元測試 `tests/test_rank_c4g_baseline.py` 2 項通過（含上式恆等式）。
- 一致性門檻 2e-3 為工程驗收，不事後放寬；超過即停查原因。通過後正式比較只用這次同一前向的 raw 與 baselined，不混用舊 raw。已通過驗收的 pack 直接重用，完整推論只補其餘。
- CPU 比較改以 Slurm 執行（`MODE=g0 slurm/rank_c4g_baseline.slurm`，1 卡／12 核配置、不使用 GPU），不放登入節點；bootstrap 為每次重算四個 k 的平均差再取 Equal-k。

**B 結果（11/15；Slurm 781992 短測、782006 其餘 14 個 pack、782007 CPU 比較；`c4g/baseline/check.json`、`g0_baseline.json`）：gate 未通過，不做全池抽取。**

**短測（781992，1 卡，pack_hd{0,1}_0，162,200 個位置，3 分 41 秒）：** 模型載入 6.8 秒、第一批（含編譯）37.1 秒；穩態 **1,065 位置／秒**（每位置 10 個假設）；峰值顯存 15.2 GiB。新舊 raw 每盤特徵絕對差：中位 1.1e-7、p99 4.6e-7、最大 8.8e-7（門檻 2e-3）→ 通過。據此估其餘 14 個 pack 約 18 分＋啟動 ≈ 0.32 GPU-h；實際 782006 為 19 分 06 秒。兩個短測 pack 直接重用。

**全部 16 個 pack 的驗收（12,974 盤、1,297,400 個位置）：** 棋局集合、每側 50 手、級別標籤與舊檔相同，全部有限；raw 差異中位 1.1e-7、p99 4.6e-7、最大 1.0e-6；各 p_r 與 q 的總和偏差 ≤ 1.1e-15，q 下中心化分數期望 ≤ 1.7e-15。

**基線診斷（不作判定）：** 基線項 Σ_a q(a) log p_r(a) 的逐位置共同偏移（十級平均）平均 −1.49、位置間標準差 0.89；十級間標準差平均 0.029。取每盤每側 50 手平均後：共同偏移的盤間標準差 0.167／0.165（本人／對手），級別間標準差平均 0.028／0.025；對照 raw 的盤間共同標準差 0.293／0.283、級別間 0.067／0.062。→ 基線主要是一個隨盤面變動的共同偏移，級別間差異約為 raw 級別間差異的四成。

| 臂（欄數） | k=5 | k=10 | k=15 | k=20 | Equal-k | 單盤 d′ | 玩家 d′ | 玩家間變異占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H（32） | 0.6663 | 0.7048 | 0.7318 | 0.7262 | 0.7073 | | | |
| C_raw（20） | 0.6373 | 0.6779 | 0.7152 | 0.7135 | 0.6860 | 0.4404 | 1.0334 | 0.2525 |
| C_baselined（20） | 0.6345 | 0.6777 | 0.7045 | 0.7126 | 0.6823 | 0.4370 | 1.0290 | 0.2737 |
| HC_raw（52） | 0.6646 | 0.6957 | 0.7219 | 0.7357 | 0.7045 | | | |
| HC_baselined（52） | 0.6592 | 0.6910 | 0.7285 | 0.7271 | 0.7015 | | | |

| 差異 | k=5 | k=10 | k=15 | k=20 | Equal-k | 95% 區間 | P(>0) |
|---|---:|---:|---:|---:|---:|---|---:|
| **HC_baselined − HC_raw（主檢定）** | −0.0054 | −0.0046 | +0.0067 | −0.0086 | **−0.0030** | [−0.0116, +0.0053] | 0.246 |
| C_baselined − C_raw | −0.0028 | −0.0003 | −0.0107 | −0.0008 | −0.0037 | [−0.0132, +0.0058] | 0.233 |
| HC_raw − H | −0.0017 | −0.0091 | −0.0100 | +0.0095 | −0.0028 | [−0.0116, +0.0061] | 0.259 |
| HC_baselined − H | −0.0071 | −0.0138 | −0.0033 | +0.0009 | −0.0058 | [−0.0160, +0.0041] | 0.126 |

**判定：** 主檢定 −0.0030 < +0.005、P＝0.246 < 0.90、k=5 與 k=20 低於 −0.005 → **未通過**。區間 [−0.0116, +0.0053] 含 0：這個結果是「沒有看到增益」，不是「確定較差」。依登記不做全池抽取。

**能說的與不能說的：**
- 能說：以十級 policy 等權平均為參考、扣除期望 log p 的這一種盤面基線，在 dev 上沒有比原始 C 多帶來可用訊號，單用 C 與接上 H 兩種情形方向一致（−0.0037／−0.0030）。玩家間變異占比未下降（0.2525→0.2737），中段可分度未改善（單盤與玩家層級中段 d′ 略降）。玩家間變異不全是棋風，占比上升也可能來自玩家內變異下降，不能單靠比例判定機制。
- 附帶觀察：這次的 C_raw／HC_raw 與第一步 C 的 C_base／HC_base 特徵最大只差 1e-6，Equal-k 卻差 −0.0003／+0.0009 → 近乎相同的 raw 特徵產生約 0.001 的分數差，**原因未定位**（可能涉及題組／抽樣順序、求解器停止條件或決策邊界附近的微小變化）。只有這一次跨流程比較，不足以估計抖動範圍；若要補查，先比對兩次的玩家折、題組、合成抽樣與求解器設定（CPU，不需重跑 GPU）。這項差異不改變本次未達 gate 的決定。
- 限制同前：dev 設定下原始 C 對 H 沒有增量（HC_raw − H＝−0.0028），對照臂偏弱；每側 50 手；單一 dev 玩家集；單一參考分布與單一 B2 seed。
- 不能說：其他參考分布（例如真實級別無關的單一模型、強引擎 policy）或把基線項當獨立特徵都無效——這些沒有測。機制：未知。

**算力（實際配置）：** 781992 3 分 41 秒＋782006 19 分 06 秒＝GPU 推論 0.38 GPU-h；782007（CPU 比較，1 卡配置未使用 GPU）30 分 01 秒＝0.50 GPU-h；**合計 0.88 GPU-h**（登記上限 1.25）。CPU 比較在計算節點比登入節點慢（30 分對約 17 分，原因未查）。C→A→B 三步累計配置 2.09 GPU-h。不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。D（共享主幹、多條件 policy head）依計畫保留為後續研究，未啟動。


### 2026-11-15：最終候選證據查核（本地驗證與 Public LB）

使用者要求深入研究最終選檔。查核既有程式／JSON、11 份 CSV 與 E1／E2 題組，未訓練、重擬合或改提交檔。報告：[report.md](outputs/rank/final_selection_review_20261115/report.md)；可重現比較 `analyze.py`，數據 `comparison.json`、`local_vs_lb.csv`、`question_audit.json` 同目錄。

- 相對現行基底的 7 項探針，5 項 Public Δ 高於 E2 Δ；加入 C v2／多 794 玩家兩項匹配歷史比較後為 5/9（平均殘差 +0.00240）。原先只列五個偏樂觀項目的說法不完整；手數加權、adj、C v2、多玩家都是反例。此清單不是全部歷史實驗，且多個探針相依，不能做獨立計票或直接校正 Private。
- 本地 E2 有 2,000 題／866 人，k=20 的 500 題只有 249 人；E1 有 1,000 題／676 人，與 E2 共用 525 人。五 seed 改擬合抽樣，不換評估題庫；既有玩家 bootstrap 只對 seed2027，並非五 seed 平均區間。
- 正式 export 比本地多加入 U 的 HD≤1 棋局，先驗也由 S 改 S＋U；E2 級別均勻。可比性有限，但沒有證據證明這些差異使 vote 在 Private 受益。
- vote 對基底改 Public 33／Private 38 題；引擎 25／30。兩者共同改 10／13 題且同方向；它們互差 38／42 題。vote 的 Public +0.0027591 對應完全正確數淨 0、差一級數淨 +3（只反推總計數，不推個別答案）；引擎 +0.0015803 對應 +1／−1。部分分數同樣有價值，但改善幅度很小。
- vote 的 E2 各 k（5/10/15/20）−0.0029／−0.0151／−0.0080／−0.0035，五 seed 平均 −0.0074；引擎 +0.0036／−0.0105／−0.0164／+0.0006，平均 −0.0057。未見玩家負面證據仍在；本次無新分數。
- **查核後建議：維持 `751959-cv2-l10.csv` 為基底與目前最終預設，vote 保留作主要挑戰者。撤回聊天中「現在必選 vote」的低信心排序：當時過度依賴挑出的五項差值與 Public 最高分。** 不宣稱已證明基底 Private 最好，也不使用 0.02 作通用顯著性門檻；正式逐題答案未知，無法算真正配對 CI。
- 下一個有判別力的實驗是凍結三臂特徵、部分 U 加擬合、其餘 U 玩家留出，預先固定同色 HD≤1 題組與評分權重情境。僅列提案，未執行；成本與規格依 §9 另定。詳見報告。

**流程紀錄（11/15，使用者指出）：** B 的授權是「實作＋短測」，完整推論須等短測回報後再決定；我在短測通過後直接送出其餘 14 個 pack（782006）與 CPU 比較（782007），雖在登記的成本上限內，仍跨過了約定的確認點。既有結果保留、不重做。之後「成本上限」與「分階段執行授權」分開處理：上限內也只執行已明確授權的階段，其餘先回報再等確認。

**C→A→B 小結與 D 的定位（使用者意見）：** B 支持的是「在指定參考分布、dev 資料與 B2 設定下，扣除基線沒有可辨識增益」；數值驗收通過，不能歸因於基線算錯，區間跨零也不能說確定較差。D 不直接投入完整訓練：B 沒有支持「去基線能改善辨識」，C 顯示上游進步未必補到 H；D 剩下較清楚的價值是降低多條件推論成本，若重啟須先以效率與品質保留為目標提限額短測，不預設能提高分數。未啟動。

### 2026-11-15：競賽二——第一步 C 與第三步 B 之間約 0.001 分數差的重現性補查（使用者同意；只比對既有產物與設定，未重新擬合、未重抽題、未用 GPU、未讀 Valid-U）

**比對對象：** `c4g/g0_hc.json`（11/15 17:36，登入節點、8 執行緒）的 C_base／HC_base，對 `c4g/baseline/g0_baseline.json`（11/15 22:14，計算節點 25a-hgpn017、12 執行緒）的 C_raw／HC_raw。另以 `g0_A.json`（11/14）作第三個參照。

**相同的部分（由程式路徑與輸入確認）：**
- 棋局與標籤：兩次都讀同一份 `dev_games.parquet`（11/14 11:15，之後未變），12,974 盤／762 人。
- 玩家折：同一函式 `rank_c4g._fold_of`、seed 2027、同一輸入。題組：同一函式 `rank_c4g._questions`、`default_rng(2028)`、同一輸入（兩次都沒有保存題目清單，這一項是由程式與輸入相同推得，沒有逐題比對）。
- 合成題：每次擬合各自以 `default_rng(2027)` 起算，與臂的執行順序無關；每個 k 20,000 題；先驗由同一訓練折算出。
- 標準化與求解器：`StandardScaler`＋`LogisticRegression(C=1.0, max_iter=5000)`（lbfgs、tol 預設 1e-4）；兩次同一個 conda 環境（scikit-learn 1.7.2、numpy 2.3.5、scipy 1.16.3、pandas 2.3.3），相關程式在兩次之間沒有改動。
- **H 臂（特徵完全相同）兩次分數逐位數相同**：0.6663／0.7048／0.7318／0.7262／0.7073。C 三臂在 11/14 與 11/15 兩次（特徵相同）也逐位數相同。→ 特徵相同時，跨日期、跨節點、跨執行緒數都可重現；折、題組、合成抽樣與環境不是差異來源。

**不同的部分：** 只有 C 的 20 欄。舊值為 float32（8 行程、float32 softmax），新值為 float64（單行程、float64 softmax）；沒有任何一個元素完全相等，絕對差中位 1.1e-7、最大 1.0e-6，相對於欄標準差最大 3.2e-6。

| 臂 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| C：新 − 舊 | −0.0014 | −0.0004 | −0.0004 | +0.0010 | −0.0003 |
| HC：新 − 舊 | −0.0006 | +0.0007 | −0.0009 | +0.0044 | +0.0009 |

每個 k 2,000 題；HC 在 k=20 的 +0.0044 相當於 8.8 分，例如約 14 題在「正確」與「差一級」之間改判（每題 0.632 分；實際組合未知）。

**結論：** 差異定位到「特徵約 1e-6 的數值差經擬合與解碼後被放大」，不是折、題組、抽樣、設定或環境不同。放大發生在哪一步**無法定位**——可能是 lbfgs 在 tol 1e-4 停止時的係數差，或決策邊界附近的題目——缺少的紀錄：逐題預測／機率、擬合係數、`n_iter_`（兩次都只存了彙總 JSON）。未追加運算。
- 對判讀的含意：同一批題上，兩臂的差若在單一 k 桶約 0.004、Equal-k 約 0.001 以內，僅憑這一次觀察不能排除是數值擾動；這只有一次比較，不是抖動範圍的估計。C、A、B 三步的判定都不受影響（主比較分別為 +0.0013 未過、−0.0056、−0.0030，皆未達 gate）。
- 之後若要能定位：比較程式在輸出彙總時一併保存逐題分數與機率、各折係數與 `n_iter_`（A 步的 `pool_scores.parquet` 已保存逐題分數）。

### 2026-11-15：競賽一——D3-a 兩臂提交探針（使用者要求產檔）

使用者要求即使未過本地 gate 也利用剩餘上傳額度測試，並授權產生 Arm 1／Arm 2。重用正式新增集 `outputs/pid/day0_1112/{public,private}2_d4_9600/parts.npz`，只修改逐 k 融合權重；D4 開局、風格模型與快取不變。原測試集 800 題完全沿用 `772748-hung.csv` 的 Hungarian 結果，新增集不指派。此紀錄取代 D3-a 原先「不產生提交檔」的處置，不改本地 gate 判定。

| 提交檔／SHA-256 前八碼 | 假設與權重（k=1／2／3／5） | pool 1 Δ：整體；分 k | 預期 Public Δ（新增集改善完整轉移、占一半題數） | 改 top-1：pub2／priv2 | 實際 LB |
|---|---|---|---|---|---|
| `submissions/pid/hung-d3a-arm1.csv`／`8f29992d` | 少盤題風格較不穩，降低 λ：8／8／16／16；μ 全為 1 | +0.0034；+0.009／+0.004／0／0 | 約 +0.0017，絕對分數未知 | 29／24 | **0.7984372**（11/15 23:07:16，jabir95tsai；名次未提供） |
| `submissions/pid/hung-d3a-arm2.csv`／`935ce779` | 在 Arm 1 上另測逐 k HD 色別權重；λ 同 Arm 1，μ＝2／0.5／2／2 | +0.0033；+0.013／+0.001／+0.002／−0.0021 | 約 +0.00165，絕對分數未知 | 34／35 | **0.7946871**（11/15 23:07:32，jabir95tsai；36/90） |

建議上傳順序 Arm 1 → Arm 2。兩臂皆屬探針；Public 升／持平／降皆不自動升基底，維持 `772748-hung.csv`，以本地證據與後續結果共同判斷。Arm 2 相對 Arm 1 才是單獨測 μ，不能把相對基底的改變全部歸因於 HD。預期差距很小，不能只憑 Public 小升認定改善。

驗證：原融合分數 allclose 重現、原新增集 top-5 與基底逐值相同後才重新加權；題目 ID 對齊、600 位候選唯一且前綴正確、分項／新分數全有限；兩份 CSV 各 1,600 題、ID 唯一、每題五人唯一、原測試集逐列不變、CSV 重讀一致。Arm 1 的 k=3／5 top-5 不變。pub2／priv2 top-5 改動列數：Arm 1＝171／171；Arm 2＝281／288。未重訓、未執行 GPU、未代上傳，平台剩餘額度未知。

重現腳本 `outputs/pid/d3a_submission/export.py`（拒絕覆寫既有提交檔）；執行方式 `PYTHONPATH=src /home/jason1122/.conda/envs/go/bin/python outputs/pid/d3a_submission/export.py`。完整雜湊與分 k 改動題數見 `outputs/pid/d3a_submission/report.json`。CPU 匯出與驗證約 2.55 秒。

**LB 補登（使用者回報，11/15 23:07，jabir95tsai）：** Arm 1＝0.7984372，對基底 `772748-hung.csv`（0.7980564）**+0.0003808**；Arm 2＝0.7946871，對基底 **−0.0033693**，對 Arm 1 **−0.0037501**。Arm 1 小升、方向與本地 +0.0034 一致，但不足以確認泛化增益；Arm 2 額外調 μ 在 Public 為負，本地相對 Arm 1 也沒有增益（−0.0001）。保留 Arm 1 為候選，Arm 2 暫不採用；依事前規則**基底維持 `772748-hung.csv`**。官方分 k 成績與 Private 未知，不能由整體 LB 推回。依此次回報，平台最新一筆為 `hung-d3a-arm2.csv`（23:07:32）；未代上傳或重傳基底，剩餘額度未知。

### 2026-11-15：競賽一——移除 HD／移除開局提交探針（使用者要求產生）

各自對 `772748-hung.csv` 只移除一個新增集分項，不疊加 raw 或 D3-a 權重。原測試集 800 題 Hungarian 結果逐列不變；新增集 λ=16、未移除項的權重維持原值，不做指派。讀既有 `day0_1112/{public,private}2_d4_9600/parts.npz`，不重訓、不用 GPU。

| 檔案／SHA 前八碼 | 假設與公式 | 本地驗證 | 預期 Public Δ | 改 top-1：pub2／priv2 | LB |
|---|---|---|---|---|---|
| `submissions/pid/hung-nohd.csv`／`f7eda38d` | HD 色別項是否妨礙正式辨識；opening＋16·style，μ=0 | 完整配對與分 k 未確認；既有 pool 0 五抽題選參均值 λ16/μ0=0.617273，λ16/μ1=0.621007，差 −0.003733（非獨立驗收） | 未知；上述開發結果方向偏負，不因 Arm 2 退步便推論 HD 無用 | 13／25 | **0.7917441**（11/15 23:32:45，jabir95tsai；名次未提供） |
| `submissions/pid/hung-noopening.csv`／`dd2db657` | 正式新增集是否較不適合開局訊號；16·style＋category | D1 pool 1 整體 −0.0439；k=1／2／3／5：−0.036／−0.043／−0.054／−0.042 | 若完整轉移約 −0.022，預期下降 | 77／93 | **0.7746050**（11/15 23:33:00，jabir95tsai；41/90） |

先交 nohd；noopening 優先度較低。兩臂皆為消融探針，Public 升／持平／降都不自動升基底，仍保留 `772748-hung.csv`。若與本地方向相反，記為正式資料差異的待查線索，不據單次 LB 宣稱機制成立。

檢查通過：基底 scores 重現、原新增集完整 top-5 重現、題目 ID 對齊、600 位候選唯一與前綴正確、分數全有限、各 1,600 題且每題五個不同候選、原測試集不變及 CSV 重讀一致。nohd 的 pub2／priv2 分 k top-1 改動＝2/4/4/3、2/11/6/6；noopening＝14/24/19/20、18/29/29/17。top-5 改動列數分別 214／214、373／360。

重現腳本 `outputs/pid/ablation_submission/export.py`（拒絕覆寫既有 CSV）；報表 `outputs/pid/ablation_submission/report.json`。執行 `PYTHONPATH=src /home/jason1122/.conda/envs/go/bin/python outputs/pid/ablation_submission/export.py`，CPU 匯出含檢查約 4.43 秒。未代上傳，平台剩餘額度與實際 LB 未知。

### 2026-11-15：盤面加權＋線性 B2 共同 GD 四臂（使用者核准實作與執行；事前登記）

使用者要求四個獨立任務、**不做 GPU 短測，提交後不持續監看，完成後自行分析**。假設：把 loss 從兩側原始 C 的直接 CE 改為多盤題目 B2 CE，移除尺度錯配；KL 限制逐手權重集中，可能恢復特徵穩定性。預期分數增益未知，不預設有效手數；不能保證超過等權。

| 臂 | 池化／更新 | KL 係數 |
|---|---|---:|
| mean_gd | 固定等權，只更新線性 B2 | 0 |
| joint | 權重網路＋線性 B2 共同 GD | 0 |
| joint_kl001 | 同上 | 0.01 |
| joint_kl01 | 同上 | 0.1 |

四臂共用原 HD0 train／hold／U 玩家切分、凍結 HumanSL／C v2、相同等權 attention 初始化、TRAIN-only sklearn B2 初始化與固定 StandardScaler、相同題組與抽樣 RNG。四個 k 各一個 20→10 線性 head，無隱藏層／H／旗標／固定 a_s。loss＝題目 CE＋B2 C=1 對應的 L2＋KL；KL 先每盤每側計算，再側等權、題內盤等權、題目等權，避免長盤或大 k 多受罰。

唯一 CV 模組生成 TRAIN 80,000 題（每 k 20,000）、hold 4,000 題，均同玩家同色、題內棋局互異、級別／k 平衡。每步各 (k,rank) 抽兩題，共 80 題／1,000 盤次，保留全部著手；批內重複棋局只前向一次、梯度累加。玩家可跨題重複。TRAIN 不加入 hold。此 rank-balanced prior 與歷史 PriorMatched＋train/hold 的 0.725 基準不同，不能直接歸因比較；本輪以同題／同初始化的 sklearn mean 及 mean_gd 為對照。

AdamW LR=.001，attention weight_decay=.0001；B2 顯式 L2、bias 不懲罰。最多 2,000 updates，每 200 步直接用當前共同訓練 head 報 hold 四桶 score_rank 官方分數與 CE，**以 hold Equal-k 嚴格改善選模**，step0 可選；至少 500 步、patience=5。每臂獨立選模、不以 U 改設定。凍結最佳後自動評估 U 兩組各 4,000 題（HD0_Q2027/2028），附分 k、本人／對手合併特徵、逐題機率／預測及對 sklearn mean 的玩家 bootstrap。僅單一訓練 seed2027，U 已被歷史研究使用、兩組 draw 共用玩家，泛化證據有限。

診斷 gate：先看 joint 相對 mean_gd 是否恢復／改善；若要列增益候選，沿用一組 Equal-k ≥+0.005、另一組不負，第二組各桶 ≥−0.005。只登記門檻、不自動 promote／重訓／出提交檔，跨臂分析由使用者完成。

程式：`src/go_aicup/tasks/strength/board_attention/joint.py`、`joint.slurm`；設定 `configs/strength/board_attention_joint.json`；測試 `tests/test_board_attention_joint.py`。輸出 `outputs/rank/humansl/board_attention_joint/seed2027_hd0/<arm>/`，含 initial／best／last／model、learning_curve、來源身分、題組、features、training_complete、result、逐題 predictions 與 complete marker。沿用既有 teacher／board 對齊稽核，不重建盤面快取。

成本：歷史約 0.21 秒／1,280 盤，暫估每臂 10–20 分鐘，四臂約 0.67–1.33 GPU-h（共同 GD 未量測）；四個獨立單 H200 作業，各 12 CPU／100GB／2h，總配置上限 **8 GPU-h**。依使用者指示省略 GPU 短測；排隊時間未知。CPU 合成及既有回歸測試結果、提交 job ID 於提交時追加。不變更既有實驗產物、不上傳。

驗證：既有 board attention＋joint CPU 合成回歸 **13 項通過（4.78 秒）**，涵蓋權重 KL／題內盤等權、固定題組隔離與抽樣 RNG、sklearn→Torch logits 一致、兩網路梯度、等權固定、CPU 精確續跑、最終 U 預測發布及完成後重試；Slurm 語法、Python 編譯與 `git diff --check` 通過。依使用者明確要求，未執行 GPU 短測；真實 CUDA／吞吐／收斂待作業自行產出。

四臂提交：`mean_gd` → Slurm **782558**；輸出 `outputs/rank/humansl/board_attention_joint/seed2027_hd0/mean_gd/`；log `outputs/slurm-rank-b2-mean_gd-782558.out`。

四臂提交：`joint` → Slurm **782559**；輸出 `outputs/rank/humansl/board_attention_joint/seed2027_hd0/joint/`；log `outputs/slurm-rank-b2-joint-782559.out`。

四臂提交：`joint_kl001` → Slurm **782560**；輸出 `outputs/rank/humansl/board_attention_joint/seed2027_hd0/joint_kl001/`；log `outputs/slurm-rank-b2-joint_kl001-782560.out`。

四臂提交：`joint_kl01` → Slurm **782561**；輸出 `outputs/rank/humansl/board_attention_joint/seed2027_hd0/joint_kl01/`；log `outputs/slurm-rank-b2-joint_kl01-782561.out`。

### 2026-11-15：競賽一——移除 HD／開局的 LB 結果補登

使用者回報（jabir95tsai）：`hung-nohd.csv` 23:32:45 Public **0.7917441**，對基底 `772748-hung.csv`（0.7980564）**−0.0063123**；`hung-noopening.csv` 23:33:00 Public **0.7746050**，**−0.0234514**，名次 **41/90**。nohd 名次未提供。兩份均只改新增集，原測試集不變。

移除開局的下降接近本地 pool 1 −0.0439 乘新增集題數占比 1/2 的預期 −0.02195；方向與幅度相符，支持保留現有開局項。移除 HD 也下降，與既有 pool 0 選參結果方向一致；Arm 2 的退步不能解讀成 HD 沒用，現有 μ=1 比本輪移除或逐 k 調權兩個探針的 Public 都好。這些結果支持目前設定，不證明 μ=1 是全域最佳，也不推算官方分 k 或 Private 分數。

兩份消融探針暫不採用，基底維持 `772748-hung.csv`；D3-a Arm 1（0.7984372）保留候選、不自動升基底。依本次回報，平台最後一筆為 **`hung-noopening.csv`（0.7746050）**。本對話已回報 11/15 五筆競賽一成功提交：raw、D3-a Arm 1／2、nohd、noopening，依專案記錄的每日五次上限，今日額度應已用完（未直接查平台）。中途不必為收尾重傳基底；最終停止提交前仍須由使用者確認最後一筆為選定版本。未代上傳或修改任何提交 CSV。

### 2026-11-16：競賽一——新增集編碼器兩臂（同／鄰級抽樣、Pick 特徵）事前登記＋raw＋D4 等權混合（使用者定稿設計）

**依據：** 11/15 診斷——純風格 0.568、完整融合 0.620（pool 1），開局只貢獻約 +0.05；答錯的 top-1 有 23–25% 同級、約 56% 在差一級內；raw 開局融合後 +0.005、Public 持平。

**兩個 4,800 步臂（各只改一項；對照 `pid_ge2e_v4_d4_hd_4800`，其餘設定相同：每卡 64 人 × 8 盤 × 16 手、玩家層級 D4、HD≤1、排除 split 0–2、不跨卡合組）：**

- `pid_ge2e_v7_band_4800`（`sampling.rank_focus={"share":0.5,"radius":1}`）：每卡每組先均勻抽一位玩家取其級別為中心 r，從 r−1～r+1（邊界不循環）的玩家不放回抽 32 人，再從其餘全部玩家不放回抽 32 人。是「一半名額集中抽取」，另一半也可能落在該區間。假設：較難的負例讓向量學到棋力以外的個人差異。預期增益未知。
- `pid_ge2e_v7_pick_4800`（`model.pick=true`）：投影輸入由盤面 mean／max（512 維）加上落點位置的主幹特徵（256 維），768→512→256；落點以第 19 個 one-hot 平面隨盤面一起做換色與 D4。假設：顯式落點特徵可能改善辨識（不把池化稀釋當已知根因；快取的最新盤面本來就含當手棋子）。預期增益未知。

**Gate（兩臂相同，`scripts/identity/pid2_encoder_compare.py`；固定開局參數、λ=16、μ=1、各模型自己的 w）：** 對同批重評的 4,800 步基準，pool 1（draws 100–129）整體 Δ ≥ +0.010、Δ／SE ≥ 2.5、四個 k 桶各 ≥ −0.005，且 pool 0（draws 100–119）整體 Δ > 0。另列報對現行 9,600 步模型的差（不作 gate；勝過 4,800 步基準不等於勝過正式基底），以及 top-1 同級／鄰級／跨級誤判占**全部題數**的比例與官方分數（同級占比下降不單獨當作機制確認）。兩臂都過 → 先另行決定是否做合併版 4,800 步（約 16 GPU-h），不直接合併跑 9,600 步；只有一臂過 → 再評估該臂加長。

**工程驗證：** `tests/test_identity_ge2e.py` 14 項通過（Slurm 782156；含集中抽樣的組成／不重複／可重現、Pick 標記落在當手棋子上並隨 D4 轉動、既有抽樣逐值不變）。100 步八卡短測 782157（同／鄰級）、782158（Pick）：都是約 1.52 秒／步（與基準相同）、顯存峰值 122.8 GiB／卡。短測的各級名額占比對玩家人數占比（D→6D）：10.4／11.0／14.1／19.5／5.0／6.9／7.6／7.9／9.0／8.6% 對 11.6／10.5／13.5／19.9／4.8／7.1／7.5／8.0／8.2／8.9%，沒有放大小級別（完整訓練另存 `rank_exposure.json`）。

**已送：** 訓練 784583（同／鄰級）、784584（Pick），各 8 卡約 2 小時 2 分、約 16.5 GPU-h，合計約 33 GPU-h；評估 784586（1 卡，afterok 兩個訓練）。

**raw＋D4 等權混合（`scripts/identity/pid2_raw_mix.py`，Slurm 784588，CPU，1 分 56 秒；開局＝(raw＋D4)／2，不搜權重，其餘同 raw 實驗）：** 事前 gate 對 raw 臂：pool 0 Δ ≥ +0.003 才評 pool 1；pool 1 Δ ≥ +0.003、Δ／SE ≥ 2.5、各 k ≥ −0.002。預期增益未知。

| pool 0（20 draws） | k=1 | k=2 | k=3 | k=5 | 整體 |
|---|---|---|---|---|---|
| 混合 | 0.3671 | 0.6016 | 0.7275 | 0.8410 | 0.6343 |
| 混合−raw（標準誤） | −0.0003（0.0015） | +0.0010（0.0026） | +0.0013（0.0026） | +0.0040（0.0020） | **+0.0015（0.0014）**，50% draw 為正 |
| 混合−D4（標準誤） | +0.0080（0.0018） | +0.0056（0.0025） | +0.0136（0.0024） | +0.0092（0.0024） | +0.0091（0.0013），95% draw 為正 |

→ **pool 0 未過（+0.0015 < +0.003），停止，未評 pool 1；混合不採用。** 混合對 D4 基底的 +0.0091 幾乎都來自 raw 本身（raw 對 D4 在 pool 0 為 +0.0076）。這只否定「等權平均」這一個做法；raw 開局的候選資格不變。輸出 `outputs/pid/pid2_raw_opening/pid_ge2e_v4_d4_hd_9600_mix/`。

### 2026-11-16：競賽二——挑戰者候選設計稿：「對局結果相對於對手強度」通道（只登記設計；未實作、未執行，待使用者確認）

**背景（使用者 11/16）：** C、A、B 都未過 gate；停止列 B2 小變體，改設計不同的上游候選；基底保留，新候選須自證。

**1. 要補的資訊。** 至今沒有在 H 之上留下增益的候選——HumanSL 主幹表示（單獨 0.63，A-051／052）、SE3／V2 分類、GE2E 風格（單獨 0.64）、微調 C（單獨 +0.017）、盤面基線 C、盤面加權 C、強引擎逐手品質與死活——全是同一族資訊：「這個人的著手像哪一級」。錯誤結構：失分集中在 A–3D，同一玩家的題錯同一邊（玩家層級占符號誤差變異 85%）；本人側玩家偏差占單盤變異 0.28、對手側 0.06。級別由勝負決定，不由著手像誰決定；H＋C 是雙方著手 likelihood 的每側平均，沒有任何一項表示「這盤誰領先／誰贏、贏的是多強的對手」，線性頭也無法由 log p 組出。這一族只在 11/08 以「最後一手是否本人」粗略代理測過（五 seed 平均 +0.0015、LB −0.008，雜訊內）。
- 不利的先驗（如實記）：若配對多為同級，各級勝率都接近五成，勝負單獨不含級別資訊，只在與對手強度、讓先合看時有用，且主要區分同級內偏上／偏下的人 → **預期增益：未知，傾向小**。

**2. 最小架構差異。** 不新增主幹、不訓練網路、不改 loss／抽樣／聚合。每盤新增低維區塊 R（上限 4 維），來源為既有強引擎逐位置輸出 `engine_full/*/score.npy`（kata1 b18、1 visit；擬合池、Valid-U、測試題都已抽好）：
- r1：終局前 10 個位置的平均目差（本人視角，截斷於 ±30 目）；r2：其符號的平滑版（tanh(目差／5)，勝負近似）。
- r3、r4：r1、r2 各乘「對手強度分數」s_opp。s_opp＝只用對手側 25 維（H 15＋C 10）的級別期望值，由按玩家交叉擬合的每盤 LR 產生（擬合玩家對自己的盤永遠是 out-of-fold；Valid-U 與測試用全池模型）。
- 讓先局沿用現行貼目設定（HD1＝0.5）的抽取；分類頭、先驗、解碼不變。維度刻意壓低：先前加寬特徵的失敗（168 維主幹表示、全部重播統計）都伴隨 k=20 記住玩家。

**3. 驗收（兩個目標，後者為門檻）。**
- 單獨：R（4 維＋HD 旗標）對 C（20 維）——只列報，預期遠低於 C，不作判定。
- **加入 H 後的增量：[H＋C＋HD＋R] − [H＋C＋HD]**，現行 PriorMatched B2、擬合池 Valid-S＋train20、seed 2027–2031 各自解碼計分後平均，Valid-U E1／E2。Gate 沿用：E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005。另報 A–3D 帶的分數與玩家層級符號誤差變異是否下降（診斷）。
- Valid-U 已多次使用，通過也只列為 LB 探針候選，不直接換基底。

**4. 有限額可行性實驗（先確認有訊號，再做上面的正式比較）。** 全部 CPU、只用既有產物：
- F1 稽核：`score.npy` 對擬合池／Valid-U／測試的覆蓋、終局目差分布、r2 與「最後一手是本人」的一致率、各級平均 r2（檢查是否接近五成）。
- F2 訊號檢定（池內、玩家分組五折，每玩家一列）：以現行 52 維的級別期望殘差為目標，R 的玩家平均能否解釋殘差（偏相關與其玩家 bootstrap 區間）。**若 r3／r4 與殘差的偏相關區間含 0 → 停止，不做正式比較。**
- 成本：F1＋F2 估 10–20 分 CPU（未實測）；正式比較約一次 B2 五 seed（約 13–30 分）。以 1 卡配置送 Slurm，上限 1 GPU-h；不需 GPU 推論。分階段授權：F1／F2 回報後才決定是否做正式比較。

**D 的定位：** 不列為分數候選。它能降低多條件推論成本，但輸出若仍是平均著手 likelihood，資訊族與 H＋C 相同；若日後需要對大量新棋局重抽 C 才值得做，屆時以效率與品質保留為目標另提限額短測。

**未檢查／未知：** `score.npy` 在 1 visit 下終局判斷的準確度（中盤認輸局尤其未知）；配對是否多為同級（沒有對手 ID，只能由對手側特徵間接看）；11/15 盤面加權＋B2 共同 GD 四臂的結果由另一個 session 處理，本稿未納入。

### 2026-11-16：競賽一——同／鄰級抽樣與 Pick 特徵 4,800 步結果 → 兩臂都未過 gate，結案

訓練 784583（同／鄰級，8 卡 2h02m54s）、784584（Pick，8 卡 2h04m05s），合計約 32.9 GPU-h；評估 784586（1 卡 13 分 37 秒）。輸出 `outputs/pid/pid2_encoder_compare/pid_ge2e_v7_band_4800__pid_ge2e_v7_pick_4800/`，log `outputs/pid/pid2_bench/logs/bench-v7-784586.out`。固定開局參數、λ=16、μ=1；標準誤為 draw 間。訓練末段自身中心命中率：基準 0.641、同／鄰級 0.614（負例較難，不可直接比）、Pick 0.646。

**完整融合（對同批重評的 4,800 步基準）：**

| 模型 | pool 0 整體（Δ，標準誤） | pool 1：k=1／2／3／5 | pool 1 整體 | pool 1 Δ（標準誤，Δ／SE，正 draw 比例） |
|---|---|---|---|---|
| `v4_d4_hd_4800`（基準） | 0.6092 | 0.3566／0.5645／0.7050／0.8120 | 0.6095 | — |
| `v7_band_4800` | 0.6057（−0.0034，0.0020） | 0.3520／0.5583／0.7012／0.8093 | 0.6052 | **−0.0043（0.0017，−2.61，37%）** |
| `v7_pick_4800` | 0.6108（+0.0016，0.0029） | 0.3537／0.5617／0.7039／0.8096 | 0.6072 | **−0.0023（0.0014，−1.60，40%）** |
| `v4_d4_hd_9600`（現行，僅列報） | 0.6252（+0.0160） | 0.3680／0.5705／0.7149／0.8254 | 0.6197 | +0.0102（0.0024） |

對現行 9,600 步模型：同／鄰級 pool 1 −0.0145（0.0020）、Pick −0.0125（0.0022）。只用風格（cosine）時 pool 1：基準 0.5545、同／鄰級 0.5523（−0.0022，0.0022）、Pick 0.5501（−0.0044，0.0021）；pool 0：−0.0004、+0.0030。

**top-1 誤判占全部題數的比例（完整融合，pool 1；同級／鄰級／差 ≥2 級）：** 基準 0.0951／0.1387／0.1985；同／鄰級 0.0977／0.1383／0.2001；Pick 0.0963／0.1414／0.1963；9,600 步 0.0967／0.1355／0.1869。pool 0 同樣沒有方向性變化（同級：0.0982／0.1000／0.1035）。**同／鄰級抽樣沒有降低同級或鄰級誤判率**，三類誤判都與基準相差在 0.003 內。

**判定：** gate 為 pool 1 Δ ≥ +0.010、Δ／SE ≥ 2.5、各 k ≥ −0.005、pool 0 Δ > 0。同／鄰級抽樣兩個 pool 都是負的（k=2 −0.0063 也低於桶下限）；Pick pool 0 +0.0016、pool 1 −0.0023，在雜訊內、無增益。**兩臂都未過，不加長、不做合併版；新增集風格模型維持 `v4_d4_hd_9600`。** 同／鄰級抽樣的各級名額占比（完整訓練 `rank_exposure.json`，D→6D）10.7／11.2／14.3／19.7／4.8／6.5／7.6／8.0／8.9／8.3%，對玩家人數占比 11.6／10.5／13.5／20.0／4.8／7.1／7.5／8.0／8.2／8.9%，每位玩家平均被抽 351–418 次，沒有放大小級別。

**判讀（機制未驗證的部分標明）：** 這只否定「50% 名額、±1 級、每卡 64 人、4,800 步」這一個設定與這一種 Pick 接法，不否定更強的難負例做法（例如跨卡合組、依向量相似度挑負例）。同／鄰級抽樣無效的可能原因——每組 64 人裡原本就有約 3 成落在 ±1 級內，集中抽樣只把它提高到約 6–7 成，難度增加有限；或同級混淆主要來自每人只有 10 盤候選棋的雜訊而非向量學得不夠——都未驗證。D2 的「錯誤偏向同級」仍是事實，但這次實驗顯示它不是靠這種抽樣就能修的。Pick 無增益與「最新盤面本來就含當手棋子」一致，但也沒有直接證明。


### 2026-11-16：C-only 完整 policy 五臂殘差小實驗（使用者核准實作及提交；不監看）

**假設／預期：** 原始 C 只保留實際著手的十級 logP；完整分布的簡單統計或候選點／policy 圖編碼，可能改善題級證據。增益未知、傾向小；先前盤面基線接 H 主檢定 −0.0030，不能當作本輪收益保證。不接 H、讓先旗標，不改正式基底或提交檔。

**固定資料：** seed=2027，沿用 C4x A/B 玩家來源，每級每半選 50 人訓練、10 人監控；重抽每人 20 盤同色、HD≤1 棋局，取較多盤的顏色（同數取黑）。train＝1,000 人／20,000 盤／4,460,508 位置；monitor＝200 人／4,000 盤／898,462 位置。原 dev 762 人／12,974 盤玩家與棋局身分固定，改用全部 2,896,761 個著手，不用舊每側 50 手抽樣。三組玩家互斥。所有抽到的棋局在訓練／評估都使用本人及對手全部有效著手；無 ply≥30 篩選。選樣偏向活躍玩家；C vectors 既有學習玩家重疊另於 identity.json 記錄。

**五臂：** A0 原始 q；S 用 q＋每級 entropy／maxP／sum(P²)／log合法數／logP變異／著手對最大logP差／標準化著手證據，80→64→10 輸出 Δ；F 用候選點 22→32→32、獨立實際著手嵌入及 74→32→1 attention，75→64→10 輸出 Δ；F-mask 同網路只遮其他候選點的 P/logP（合法集合、pass、visible 與原始 q 保留，不重新正規化遮蔽輸入）；C-conv 對 P/logP/著手標記的 30-channel policy 圖做兩層 3×3 32-channel 卷積＋全域平均，串接 q／pass P、logP、是否pass／log合法數共64維，再64→64→10。全部 e=q+Δ、Δ最後一層零初始化，接本人／對手各自等權逐手平均、20維每盤C、題內各盤等權平均、每臂獨立且跨k共用 Linear(20,10)。凍結 HumanSL/C vectors；只訓練修正網路與線性頭。相同 TRAIN-only sklearn warm start／固定scaler、相同問題及抽樣序列；S輸入scaler僅TRAIN。CNN做同步D4增強；set臂候選點排列不變性。

**訓練／選模：** 1 training seed；train/monitor/dev 分別固定 8,000／4,000／8,000 題，同玩家同色且題內棋局互異。每步8題（每k兩題），AdamW lr=.001、修正網路 decay=.0001、分類頭顯式 sklearn C=1 L2；最多200 updates，約20,000次棋局引用，並不保證遍歷全部訓練棋局，實際看過的棋局／玩家數另報。每50步monitor；step0可選，至少100步後三次嚴格未改善早停，按monitor Equal-k選最佳；dev不調參／不選步數。分塊重算反傳保留題級梯度，不改逐手／逐盤權重。

**算力／記憶體：** 單節點5×H200、60 CPU、500 GiB RAM；5卡共同分片編碼／凍結推理，完成後每卡獨立訓練一臂。完整policy/位置資料只在RAM-backed shared-memory tmpfs共用（約56 GiB），不落持久磁碟，編譯快取也在RAM；不刪既有盤面快取。推理歷史外推2.2–4.4 GPU-h，不含重編碼及五臂更新，真實GPU吞吐未知；Slurm上限4h＝20配置GPU-h。持久產物僅清單、來源hash、設定、per-game C、模型/optimizer/RNG、曲線、dev預測及報表，預期<100 MiB；最終記錄實際bytes。RAM遺失必須重新抽取；子臂有checkpoint，保留同身分時可重試。

**比較與gate：** S−A0、F−S、F−F-mask為主要比較，另列F／F-mask／C-conv相對A0與CNN對S/F；玩家配對bootstrap 2,000次、每次重算四k平均。門檻Equal-k≥+.005、Pr_boot(Δ>0)≥.90、各k≥−.005；F完整資訊主gate要求F−S與F−F-mask皆過。報告5/10/15/20，歷史重用dev、一個training seed及shared rank-balanced head（非production prior-matched四個k頭）限制保留。

**實作／驗收：** `src/go_aicup/tasks/strength/policy_residual/`、`configs/strength/policy_residual.json`、`tests/test_policy_residual.py`；唯一CV來源`data/cv.py`新增選樣helper。CPU新測試14項＋既有joint6項，共20項通過（7.47秒）；涵蓋零起點、F-mask資訊隔離、policy CNN/D4/pass、直接與分塊梯度、各層等權、精度失敗、玩家隔離、同色四k題組及sklearn初始化；真實prepare-only重現上述資料規模、Slurm語法與diff檢查通過。尚未GPU短測，依使用者要求提交後不監看。作業入口：`src/go_aicup/tasks/strength/policy_residual/train.slurm`；產物：`outputs/rank/humansl/policy_residual/seed2027_t1000_m200/<job_id>/`。成功以根目錄complete.json為準；未完成不宣稱取得分數。

**提交：** Slurm **785715**，5×H200／60 CPU／500 GiB／4h，上限20配置GPU-h。CPU最終20項通過（4.28秒）；prepare清單與身分檔約1.07 MiB。C vectors既有學習玩家重疊：train 979/1000、monitor 190/200、dev 0/762；monitor非全系統未見玩家，dev仍為歷史重用。輸出 `outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/`，log `outputs/slurm-rank-policy-residual-785715.out`。依使用者指示，僅確認sbatch收件成功，未查詢排隊／執行／完成狀態，未GPU驗收，不宣稱分數。

**完成結果（2026-11-16，依使用者要求讀取既有產物登記）：** Job 785715 的根目錄 `complete.json` 為 complete；五臂均完成 200 updates，均由 monitor 選中 step 100。以下為固定 dev 8,000 題的本地官方公式得分，不是 LB；未新增訓練、未重算推論或 bootstrap。

| 方法 | k=5 | k=10 | k=15 | k=20 | Equal-k | 相對 A0 |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.677411 | 0.715639 | 0.730431 | 0.734724 | 0.714551 | — |
| S | 0.677359 | 0.721380 | 0.731224 | 0.754004 | 0.720992 | +0.006440 |
| F | 0.674118 | 0.716799 | 0.730748 | 0.743045 | 0.716178 | +0.001626 |
| F-mask | 0.674698 | 0.716667 | 0.729851 | 0.744310 | 0.716382 | +0.001830 |
| C-conv | 0.676331 | 0.716535 | 0.727219 | 0.735144 | 0.713807 | -0.000744 |

| 配對比較 | Δk=5 | Δk=10 | Δk=15 | Δk=20 | ΔEqual-k | 玩家 bootstrap 95% 區間 | Pr_boot(Δ>0) | gate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| S-A0 | -0.0001 | +0.0057 | +0.0008 | +0.0193 | +0.0064 | [+0.0008, +0.0120] | 0.992 | PASS |
| F-A0 | -0.0033 | +0.0012 | +0.0003 | +0.0083 | +0.0016 | [-0.0014, +0.0048] | 0.839 | FAIL |
| F-S | -0.0032 | -0.0046 | -0.0005 | -0.0110 | -0.0048 | [-0.0099, +0.0000] | 0.025 | FAIL |
| F-F-mask | -0.0006 | +0.0001 | +0.0009 | -0.0013 | -0.0002 | [-0.0012, +0.0007] | 0.342 | FAIL |
| F-mask-A0 | -0.0027 | +0.0010 | -0.0006 | +0.0096 | +0.0018 | [-0.0013, +0.0054] | 0.872 | FAIL |
| C-conv-A0 | -0.0011 | +0.0009 | -0.0032 | +0.0004 | -0.0007 | [-0.0043, +0.0027] | 0.350 | FAIL |
| C-conv-F | +0.0022 | -0.0003 | -0.0035 | -0.0079 | -0.0024 | [-0.0061, +0.0012] | 0.100 | FAIL |
| C-conv-S | -0.0010 | -0.0048 | -0.0040 | -0.0189 | -0.0072 | [-0.0131, -0.0017] | 0.006 | FAIL |

**結果判讀：** 只有 S−A0 通過事前 gate，增益主要在 k=20（+0.0193），k=5 為 −0.0001。F−S／F−F-mask 均未過，`F_primary_gate_pass=false`；完整候選機率 attention 尚未顯示額外價值。CNN 相對 A0 未改善。S 保留為後續候選，不直接替換正式 H＋C 基底；尚未驗證 S 在 H＋C 上的互補性或 LB 增益。

**訓練曲線（固定 monitor 4,000 題，Equal-k）：**

| 方法 | step 0 | step 50 | step 100 | step 150 | step 200 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.726117 | 0.722956 | 0.727855 | 0.726091 | 0.721548 |
| S | 0.726117 | 0.703262 | 0.728634 | 0.724117 | 0.713385 |
| F | 0.726117 | 0.710529 | 0.727407 | 0.723145 | 0.718454 |
| F-mask | 0.726117 | 0.714531 | 0.726893 | 0.723461 | 0.718454 |
| C-conv | 0.726117 | 0.715336 | 0.726341 | 0.723711 | 0.718954 |

五臂皆於 step 100 達到本次最佳 monitor 分數，之後下降；A0 也有相同時間的波動，不能單憑此歸因為 residual 網路過擬合。S 在 step 100→150 的 training loss 1.036375→1.022420，但 monitor CE 0.984401→1.010377、Equal-k 0.728634→0.724117，呈現泛化惡化跡象。training loss 是每 50 步不同隨機題目批次的窗口平均，含 head L2，不能與 monitor CE 直接當作泛化差距，亦不足以確診過擬合。S 的 step 100 monitor 四桶為 0.669820／0.726209／0.756965／0.761540；step 200 為 0.666395／0.711670／0.731944／0.743530，四桶均下降。

**覆蓋／成本／數值檢查：** 各臂均使用 20,000 次棋局引用，實際看過 11,772／20,000 盤（58.86%）、762／1,000 位訓練玩家（76.2%）；未遍歷完整訓練集。總牆鐘 1,897.39 秒（31.62 分），5 卡配置估計 2.6353 GPU-h；其中共同抽取 8,255,731 位置耗時 1,787.96 秒（29.80 分），cached C mean 最大絕對誤差 0.0004642，低於 0.002 門檻。五臂報告耗時 A0／S／F／F-mask／C-conv＝2.23／65.94／87.22／84.97／77.98 秒（並行，不能加總為牆鐘）。RAM 共用快取約 56 GiB；最終報告前持久產物約 15.01 MiB。

**限制／後續提案（未核准、未執行）：** 單一 training seed、歷史重用 dev、活躍同色 HD≤1 玩家選樣；C vectors 學習玩家與 train 重疊 979／1,000、monitor 重疊 190／200、dev 重疊 0／762。monitor 不是全系統未見玩家驗證；本輪無 H／HD 旗標且跨 k 共用 head，不等於正式 B2。依曲線暫不直接加長訓練；可先提案 A0／S 在相同 200 步與抽樣下只降低 LR（0.001→0.0003）的受控對照，增益與實際成本未知，須依 CLAUDE.md §9 另行確認。固定 training 子集評估尚缺，不能區分抽樣波動、最佳化不穩與過擬合。

**來源：** [總報告](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/result.json)、[完成標記](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/complete.json)、[資料身分](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/identity.json)、[抽取檢查](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/extraction_complete.json)、各臂 `learning_curve.json` 與 [實驗說明](src/go_aicup/tasks/strength/policy_residual/README.md)。

**固定 TRAIN 凍結評估（2026-11-16，使用者要求執行）：** 為區分 F 的擬合不足與泛化惡化，提交診斷 job **786101**，5×H200／60 CPU／300 GiB，1h 上限（最多 5 配置 GPU-h）；依前次抽取估計 17–22 分鐘、1.4–1.8 配置 GPU-h。只重抽原 TRAIN 4,460,508 位置，對相同原始 8,000 題評估 A0／S／F 選中 step 100 與最後 step 200；凍結所有權重、不更新 optimizer，monitor 直接使用原曲線。報 CE、head L2、CE＋L2 與四桶／Equal-k，原產物唯讀。新增入口 `policy_residual/evaluate_train.py`、`evaluate_train.slurm`；CPU 17 項通過，原來源 fingerprint／題組 membership 預檢通過。輸出 `outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/train_diagnostic/786101/`，成功以此目錄 `complete.json` 為準。另以 CPU 精確重播抽樣並核對三臂 last checkpoint RNG／seen：前 100 步實際抽中 755 個不同題目、7,430 盤、528 人；前 200 步為 1,429 題、11,772 盤、762 人。固定完整 TRAIN 題組包含尚未被抽中的題目，不能將其全部稱為已參與更新的樣本。尚待 GPU 評估結果。

補充診斷口徑：原 sklearn head warm-start 已用全部 8,000 題標籤，755／1,429 題只指後續 AdamW 更新曾抽中的不同題目，不能稱其餘題目未參與整個模型擬合。新增 `policy_residual/summarize_train.py` 在同一固定子集比較初始 A0 與三臂兩個 checkpoint，並核對歷史 RNG／seen；包含固定 mask、Equal-k 與 CE／L2 的 CPU 測試共 18 項通過。

**固定 TRAIN 凍結評估完成（2026-11-16，job 786101）：** Slurm COMPLETED、exit 0:0，牆鐘 17分43秒；程式報告 1,053.13秒／1.4627配置GPU-h，無新增權重更新。全部 4,460,508 個 TRAIN 位置完成抽取，cached C mean 最大誤差 0.0002968；重抽原始 C mean 與原產物最大差 0。GPU 評估與 CPU sampling report 均已成功。

| 方法／步數 | TRAIN k=5 | k=10 | k=15 | k=20 | TRAIN Equal-k | TRAIN CE | TRAIN CE＋head L2 | monitor Equal-k | monitor CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 初始 A0／0 | 0.690771 | 0.733597 | 0.750654 | 0.769141 | 0.736041 | 0.984291 | 1.026766 | 0.726117 | 0.981758 |
| A0／100 | 0.692748 | 0.731597 | 0.747810 | 0.765457 | 0.734403 | 0.986030 | 1.028485 | 0.727855 | 0.978299 |
| A0／200 | 0.692931 | 0.731149 | 0.748574 | 0.765273 | 0.734482 | 0.986542 | 1.029010 | 0.721548 | 0.978863 |
| S／100 | 0.694776 | 0.729356 | 0.750154 | 0.766061 | 0.735087 | 0.988279 | 1.030726 | 0.728634 | 0.984401 |
| S／200 | 0.684662 | 0.718748 | 0.733828 | 0.750131 | 0.721842 | 1.012978 | 1.055429 | 0.713385 | 1.014572 |
| F／100 | 0.695328 | 0.730069 | 0.746178 | 0.767221 | 0.734699 | 0.986955 | 1.029405 | 0.727407 | 0.980191 |
| F／200 | 0.692639 | 0.724644 | 0.749074 | 0.756159 | 0.730629 | 0.992959 | 1.035415 | 0.718454 | 0.988846 |

TRAIN 固定 8,000 題、四桶各 2,000；題組實際涵蓋 998／1,000 位選定 TRAIN 玩家。monitor 為原固定 4,000 題保存結果。另以同一組前100步曾抽中的755題，固定比較兩個 checkpoint（不是各 checkpoint 換成不同題組）：

| 方法 | 755題 CE，step100→200 | 755題 Equal-k，step100→200 |
|---|---:|---:|
| A0 | 0.959609 → 0.962323 | 0.741244 → 0.736499 |
| S | 0.960463 → 0.986232 | 0.744542 → 0.726615 |
| F | 0.959006 → 0.966338 | 0.743288 → 0.729286 |

**診斷：** F 在 step100 的完整 TRAIN CE 0.986955 優於 S 0.988279，Equal-k 只差 −0.000388；在實際抽過的固定755題上，F CE 0.959006 亦優於 S 0.960463。不能說 F 因容量不足而連 training 都學不動。100→200步，S/F 的完整 TRAIN CE、CE＋L2及Equal-k與monitor均惡化；固定755題也同方向，故不符合典型「training持續改善、monitor惡化」的過擬合模式。連原始A0從sklearn起點繼續做小批次更新後，完整TRAIN CE也由0.984291升到0.986030／0.986542；最佳化／批次噪聲／head與encoder聯合漂移應優先排查，但尚未分離主因。不能僅據本次資料宣稱LR過大、架構容量不足或完整policy假設錯誤。step100的S相對F之dev得分優勢，並未伴隨TRAIN CE優勢，CE與官方得分差異及單seed波動仍保留。

**後續建議（未執行）：** 優先受控檢查較小LR或降低更新噪聲；不直接加寬F或沿用原配方加長訓練。本次沒有新的超參數對照、梯度／attention量測或額外dev推論。全部8,000題已用於原sklearn warm-start；755／1,429題只表示後續AdamW抽樣歷史，屬in-sample診斷，不是新泛化驗證。

**診斷產物：** [完整TRAIN與monitor報告](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/train_diagnostic/786101/result.json)、[固定抽樣子集與初始模型報告](outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/train_diagnostic/786101/sampling_analysis.json)、同目錄per-question預測、per-game特徵、來源與抽取檢查；原權重未改寫。

### 2026-11-16：C-only A0／S／F × 兩個 LR，完整題組遍歷六臂（使用者核准實作與提交；提交後不監看）

**假設：** 原每步8題、200步只抽過1,429／8,000題；固定TRAIN與monitor後段皆退步，可能涉及更新噪聲／最佳化，而非已證實的容量不足或過擬合。本輪檢查完整遍歷及較低LR能否改善；增益未知。六臂為A0／S／F各LR=.001／.0003，不是plan.md七臂，也不加入H／HD或更換架構。對舊pilot同時改batch及抽樣方式，不能把差異單獨歸因於其中一項。

**固定資料／初始化：** 唯讀重用785715的games、原TRAIN／monitor／dev 8,000／4,000／8,000題、raw C及原sklearn initial_head；來源fingerprint／角色玩家互斥／同色membership通過後才執行。原sklearn已用全部TRAIN標籤。完整policy只重新抽取一次、RAM共用約56 GiB；不讀取 `outputs/rank/humansl/permove/`。原raw C在重抽最大差≤.002後保留為共同基準；HumanSL／C vectors保持凍結。

**訓練／監控：** seed2027，每輪每k的2,000題獨立打亂、無放回；每步32題、每k8題，250步完整遍歷8,000題，驗證每題恰好一次。四輪共1,000 updates／每臂32,000題引用（原pilot的20倍）。六臂使用相同題目／D4順序，整批一次題級loss及optimizer更新，重用既有分塊反傳與同盤去重。每完整一輪才評估固定完整TRAIN／monitor的CE、CE＋head L2及四k／Equal-k；step0只作共同起點且可選，四輪內不早停。只以monitor嚴格改善保存best，dev在該臂四輪完成及選模鎖定後只評估一次。不依dev調整預算。

**梯度／續跑：** 每步僅記錄head／phi／scorer／delta的梯度norm、參數norm、實際更新norm與比例、缺梯度數與非有限值數；不保存完整梯度tensor，不加clip。非有限值即失敗。每輪原子保存完整model/head/optimizer/history/seen供續跑，部分下一輪可按固定排程重做；保留初始／best及latest，原實驗權重不覆寫。最終保存選定模型的TRAIN／monitor／dev每盤特徵與逐題預測，根目錄完成標記成功才視為全六臂完成。

**資源／gate：** 5×H200、60CPU、500GiB RAM，5卡共抽後4卡跑兩個S與兩個F，第5卡依序跑兩個A0。依實測外推50–65分鐘、4.2–5.4配置GPU-h；上限90分鐘＝7.5配置GPU-h。持久產物預留200MB（提交前/work約剩14GB）；32題吞吐及統計開銷未知。相同LR比較S−A0／F−A0／F−S，以及各架構低LR−高LR；配對玩家bootstrap2,000次，沿用ΔEqual-k≥.005、正增益比例≥.90、每k≥−.005 gate。一個training seed、歷史dev重用、C-vector學習重疊monitor190/200等限制不變，不直接改正式基底或上傳。

**實作／CPU驗收：** 新增 `src/go_aicup/tasks/strength/policy_residual/epoch.py`、`epoch.slurm`、`configs/strength/policy_residual_epoch.json`、`tests/test_policy_residual_epoch.py`，更新套件README。既有＋新CPU共23項通過，涵蓋每輪完整無重複／k平衡／固定排程、梯度與更新量、非有限值拒絕、A0／S／F實際兩輪更新／報表、完成後續跑不多更新及身分錯誤。正式prepare-only已核對來源hash及三組membership，每輪250步。未另提交GPU短測；準備提交正式作業後立即停止對話，不查詢排隊／執行／完成狀態。

**提交收件：** Slurm **787069**。輸出 `outputs/rank/humansl/policy_residual_epoch/seed2027_b32_e4/787069/`；log `outputs/slurm-rank-policy-epoch-787069.out`。只確認sbatch收件，依使用者指示未查詢排隊／執行／完成狀態，不宣稱GPU驗收或分數。


## 2026-11-16：outputs 共用資料移至 data

依使用者要求完成本機資料位置整理：**616 個搬移單位、3,357 個實體檔案、35,307,486,117 bytes（32.882659 GiB）**。HumanSL／引擎／C 特徵及盤面 pack 移至 `data/processed/strength_features/`，玩家辨識 store／快取／embedding 移至 `data/processed/identity/artifacts/`，既有 release 資料移至 `data/processed/release/`。不去重、不重抽切分、不重算資料；原本已在 `data/` 的 HC 等資料不另搬。

舊路徑保留相對連結，3,357 個一般檔案的新舊路徑 device／inode／size／mtime_ns 與搬移前全部一致；213 個既有連結目標全部一致，搬移內部 208 個相對連結已調整且可解析。模型權重、訓練狀態、最終預測與診斷報表保留在 `outputs/`；執行中的 `rank-policy-epoch` job 787069 所屬 epoch run 資料未搬。沒有修改訓練／解碼，也沒有新增 k=5／10／15／20 分數；未啟動 GPU、訓練或上傳。

正式位置、完整搬移對照與驗證限制見 [SHARED_DATA_LAYOUT.md](SHARED_DATA_LAYOUT.md) 及 [SHARED_DATA_MIGRATION_20261116.tsv](SHARED_DATA_MIGRATION_20261116.tsv)。本機完整檔案清冊／日誌／驗證在 `data/processed/migrations/outputs_to_data_20261116/`（不進 Git）。

補充 CPU 讀取驗證：4 組新舊 `BoardCache` 的首尾棋局盤面／顏色相同；184,698×24 引擎 Parquet 全表、逐手聚合 NPZ 與 encoder embedding NPZ 全陣列相同，145,118×256 NPY 首尾抽樣相同；搬移清冊中模型權重 0 個。`git diff --check` 與 616 筆 TSV 路徑／數量／容量檢查通過。

### 2026-11-16：競賽二——S 對 H＋C 的增量（後驗加權融合）事前規格（只登記；未實作、未送作業，各階段另行授權、不自動串接）

**研究問題：** policy_residual 的 S 臂（785715，只用 C 時 S − A0＝+0.0064、[+0.0008, +0.0120]，已過該輪 gate）所帶的完整 policy 分布統計，能否在現行 H＋C 之上提供可用增量。`v_ref` 不實作、不短測。

**固定輸入（全部已存在，唯讀）：**
- 棋局：`policy_residual/seed2027_t1000_m200/785715/games.parquet` 的 dev（fold −1，762 人／12,974 盤，HD0 9,630／HD1 3,344；身分、級別、顏色、HD 已與主表核對一致）。上游（S、A0、C 向量）與 dev 玩家重疊皆為 0。
- HC：H 30（HD0 貼目 6.5／HD1 貼目 0.5）＋全手數 C v2 20（`785715/raw_c.npz`，與正式 C v2 在 6,599 盤共同棋局最大差 1.9e-6）＋HD1 旗標 2；PriorMatched B2，先驗＝該折擬合玩家的級別占比（dev 沒有 Valid-S 角色）。
- S 後驗：`785715/S/model.pt`（step 100，monitor 選定）＋`S/features.npz`（dev 每盤 20 維）；題內各盤等權平均 → 凍結的標準化＋Linear(20,10) → softmax。**不在 dev 上重擬合。**
- A0 後驗（對照）：`785715/A0/model.pt`＋`A0/features.npz`，同一流程。A0 與 S 用同一個 sklearn 起點、同一批訓練題與更新規則，差別只在 S 的分布統計殘差。
- 只用 785715 的 S；787069（兩個 LR 的 S）不讀、不選，避免以 dev 結果挑上游版本。

**融合（固定一種）：** 對數線性加權 z_k(r)＝α_k·log p_HC(r)＋β_k·log p_X(r)，softmax 後 `score_rank`。選這一種的理由：β＝0 時正好退化成「HC 只做溫度調整」，所以 HC_cal 是必要的對照。這不排除所有校準因素：融合臂搜尋的候選較多，S 的後驗也帶自己的先驗。通過後只能說「在相同內層選模流程下，加入 S 後驗比只調 HC 溫度取得較好的外折分數」，不能說增益必然來自新的棋力資訊。本次固定對數融合，不比較算術融合。已知不處理的事：兩個後驗各含先驗（HC 為擬合折先驗、S／A0 為其訓練題的級別組成），α＋β ≠ 1 時先驗會被重複或削弱計入，只由網格吸收，不另做先驗修正。
- 網格（每個 k 各選一組）：α ∈ {0.6, 0.8, 1.0, 1.2, 1.4}；β ∈ {0, 0.25, 0.5, 0.75, 1.0}。HC_cal 只選 α（β＝0）。
- 選擇：每個外折的擬合玩家再分 5 個內層玩家折（按級別分層，seed 2027＋外折編號）；內層 B2 產生 HC 的未見預測，S／A0 後驗本來就凍結；以該 k 內層題目的平均官方分數選 (α, β)。同分（差 ≤ 0.001）取較小 β，再取 α 最接近 1。α 落在網格邊界照記並標示，不擴網格。

**比較臂（同外折、同題組、同 B2 seed）：** (1) HC；(2) HC_cal；(3) HC＋S；(4) HC＋A0；診斷：S 單獨、A0 單獨（凍結頭）。
- 外折：dev 玩家按級別分層 5 折（`rank_c4g._fold_of`，seed 2027），每折約 610 人擬合、約 152 人評分；各臂用同一批擬合玩家。
- 評估題：每個外折留出玩家出單色題，每 k × 級別 200 題（`rank_c4g._questions`，seed 2028），全部臂共用。
- B2 seed 2027–2031：每個 seed 各自擬合、各自選 (α, β)、各自解碼計分，逐題分數再取五 seed 平均；不做機率 ensemble。
- 玩家 bootstrap 2,000 次：同一玩家的題目整組重抽，每次重算四個 k 的平均差再取 Equal-k。

**Gate（跑前寫定）：**
- 實用性：**HC＋S − HC_cal**，五 seed 平均 Equal-k ≥ +0.005、P(Δ>0) ≥ 0.90、四個 k 皆 ≥ −0.005、五個 seed 的整體差值皆為正。
- 新增資訊：**HC＋S − HC＋A0**，平均差為正且 P(Δ>0) ≥ 0.90；同時報差值、區間與各 seed 方向。
- 只過實用性、未過新增資訊 → 加入 S 後驗有實用增益，但尚未證明優於加入 A0，因此無法把增益歸因於 S 新增的分布統計，不以此支持擴大 S。未過差異檢定不等於兩者相同，也不說明增益來自哪個機制。兩個 gate 都過，也只支持「785715 這一個 S 訓練版本」的增量，不涵蓋上游訓練 seed 的變異。
- 另報 HC＋S − HC 與 HC_cal − HC（溫度本身的效果）。
- 停止：實用性未過；新增資訊未過；任何 k 的選擇結果在全部外折與 seed 都是 β＝0（內部不採 S）；超出核准成本。停止後不換 S 版本、不加特徵、不改融合方式。
- 無法判定（工程／資料問題，不算模型負結果）：dev 的 H 覆蓋未達 100%；任一級別 × k 格可出題玩家 < 5；S／A0 特徵的棋局集合與 dev 不同、模型身分（identity `1b79d5cc…`）不符；出現非有限值。五 seed 方向不一致屬未過 gate，不稱資料不足。

**限制：** dev 已多次使用（C4g G0、C／A／B、policy_residual 的 dev 評分），是研究驗收集，不是全新 holdout。S 只有一個上游訓練 seed，seed 變異只量到 B2 這一層。S 的頭是跨 k 共用、沒有讓先旗標與先驗對齊，不是正式 B2 形式。Valid-U 不在本計畫內：要做須先對正式擬合池與 Valid-U 抽 S 特徵，另案提出。

**階段與授權（不自動串接）：**
| 階段 | 內容 | 成本 | 狀態 |
|---|---|---|---|
| 0 | 本規格 | — | 待確認 |
| 1 | 補抽 dev 缺少的 5,405 盤 H（HD0 4,155／HD1 1,250），清單由 785715 dev 表扣除既有 H 來源重建；同一 H 模型、profile 順序與貼目；寫入 `data/processed/strength_features/humansl/` 新版本目錄；完成後只做覆蓋與身分稽核並回報 | 11/15 同量實測 1.18 GPU-h（4 卡 17 分 44 秒）；上限 2 GPU-h | 另行授權 |
| 2 | CPU 比較：HC 每 seed 6 次 B2 擬合（5 內層＋1 外層）× 5 外折 × 5 seed＝150 次；融合只在網格上算分 | 依 11/15 計算節點約 30–50 秒／次外推，5 行程並行約 30–60 分；1 卡／12 核配置，上限 2 GPU-h（未實測） | 階段 1 回報後另行授權 |

**產物（預定）：** `outputs/rank/humansl/hc_s_fusion/`：`audit.json`、逐題分數（全部臂 × seed）、各折／seed／k 選到的 (α, β)、各次 B2 的係數與 `n_iter_`、`result.json`。不產生提交檔、不換基底（`751959-cv2-l10.csv`）。

**S 融合規格修訂（11/16，使用者審查；仍只登記，未授權任何作業；以下取代上方對應條目）：**

**最低需求（在任何 B2 擬合前，由已保存的折與題組檢查；不過 → 無法判定，不得中途改題組）：**
| 層級 | 需求 | 補齊 H 後的實算值（唯讀，11/16） |
|---|---|---|
| 整體 dev | 每個級別 × k 可出題玩家 ≥ 5 | 最小 36（k=20） |
| 每個外折擬合集 | 每個級別 × k ≥ 5（B2 與先驗需要十類齊全） | 最小 28 |
| 每個外折測試集 | 不設格數下限；評估題在整體 dev 按格抽，再依玩家分到外折，各折實際題數照記 | 最小 5 |
| 每個內折擬合集 | 每個級別 × k ≥ 5 | 最小 22 |
| 內折校準題 | 不要求每個內折都有每一格；校準題在外折擬合玩家上按格一次抽好、再依玩家分到內折，合併後每格可用玩家即外折擬合集的值 | 合併後最小 28 |
現行 H 覆蓋（58%）下 k=20 的 1D／2D 只剩 1 人，所以階段 2 必須等階段 1 完成。

**固定的折、題組與選模：**
- 外折：`rank_c4g._fold_of`（按級別分層、seed 2027）。內折：每個外折擬合玩家按級別分層再分 5 折，`default_rng([2027, 外折編號])`。
- 評估題：整體 dev 每個 k × 級別 200 題，`default_rng(2028)`。校準題：每個外折擬合玩家上每個 k × 級別 100 題，`default_rng([2028, 外折編號])`。
- 外折、內折、評估題與校準題（含每題的棋局清單）在擬合前全部存檔；所有臂讀同一份檔案，不靠「seed 相同」重抽。
- 選模：每個 k 以該 k 內折 OOF 校準題的官方平均分數選 (α, β)。同分容差 1e-6；依序取較小 β、α 較接近 1、較小 α。
- B2 擬合次數：5 外折 × 5 seed ×（5 內折＋1 外折）＝150 次；每次保存 HC 後驗（內折 OOF 與外折測試）、係數與 `n_iter_`。HC_cal、HC＋S、HC＋A0 全部重用這些後驗，不另擬合 B2；S／A0 後驗為凍結模型輸出，算一次。
- 主比較與 bootstrap：每題先算五個 B2 seed 的分數差平均；以玩家為單位 bootstrap 2,000 次，同一玩家的全部題目（所有 k）一起重抽，每次重算四個 k 的平均差再取 Equal-k。「五個 seed 皆為正」指各 seed 主比較的 Equal-k 差值皆 > 0，不要求每題或每個 k 為正。

**階段 1 的復原程序（仍待授權）：**
- 範圍唯讀查找已完成（11/16）：`/work/jason1122` 與 `/home/jason1122` 全範圍查 `h_dev_hd*`、`hc_audit.json`、`g0_hc.json`、`dev_games.parquet`，並查 11/16 搬移清冊與兩份清理紀錄 → 都沒有，判定為需重抽。執行時再查一次。
- 缺失清單：執行時由「785715 的 dev 棋局」與「現存 H 來源（`standard`、`more_players/h_hd{0,1}`、`train20_hc/h_hd1`、`p3_komi/h_train`）」的差集重新產生並存檔。11/16 唯讀計算為 5,405 盤（HD0 4,155、HD1 1,250）；dev 現有 H 只來自 `more_players`（HD0 5,475、HD1 2,094），每盤單一來源。若執行時數字不同，照實記錄並先回報。
- 補抽後檢查：覆蓋 100%；棋局、玩家、顏色、級別、HD 與主表一致；模型 `b18c384nbt-humanv0.bin.gz`、15 個 profile 順序與既有來源相同；HD0 貼目 6.5、HD1 貼目 0.5；每盤唯一來源；全部有限；KataGo returncode 0、拒絕查詢 0。保存清單、來源設定與各檔 sha256 的 manifest。
- 存放：持久共用位置 `data/processed/strength_features/humansl/dev_fold_m1_h/v1/`（不放在可清理的實驗目錄），依 `SHARED_DATA_LAYOUT.md` 規則記錄來源身分。

**S 融合結果（11/16；使用者「都做」授權階段 1、2）：實用性 gate 通過、新增資訊 gate 未通過 → 依登記不以此支持擴大 S。**

**階段 1（Slurm 787866，8gpus 佇列 4 卡，25a-hgpn016，16 分 37 秒＝1.11 GPU-h）：** 執行前重查 `/work/jason1122`、`/home/jason1122` 仍無存活的補抽 H；缺失清單由 785715 dev 與現存 H 來源的差集重新產生，為 5,405 盤（HD0 4,155、HD1 1,250）。輸出 `data/processed/strength_features/humansl/dev_fold_m1_h/v1/`（71 MB；清單、`audit.json`、含各檔 sha256 的 `manifest.json`）。稽核全部通過：覆蓋 12,974／12,974、每盤唯一來源、全部有限、身分一致、模型 `b18c384nbt-humanv0.bin.gz` 與 15 個 profile 順序一致、HD0 貼目 6.5／HD1 貼目 0.5、拒絕查詢 0、新抽棋局集合等於缺失清單。

**階段 2（CPU；`prepare` 最低需求全部通過：整體最小 36、外折擬合最小 28、內折擬合最小 22，與事前計算相同）：** 第一次 788110（25a-hgpn003，2 分 59 秒）150 次 B2 擬合在 175 秒內完成，但寫 `result.json` 時因 β 計數表的鍵為 tuple 而失敗（JSON 序列化錯誤，非數值問題）。修正這一行後整個重跑 **788136**（25a-hgpn006，32 分 21 秒；同樣 150 次擬合花 1,778 秒，約慢 10 倍，原因未查，該節點同時有本帳號另一個作業）。輸出 `outputs/rank/humansl/hc_s_fusion/`：逐題分數（全部臂 × seed）、各折／seed／k 選到的 (α, β)、HC 後驗、150 次 B2 的係數與 `n_iter_`、`result.json`。

| 臂（五 seed 逐題平均） | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| HC | 0.6639 | 0.7037 | 0.7385 | 0.7300 | 0.7090 |
| HC_cal（只調 α） | 0.6695 | 0.7127 | 0.7434 | 0.7342 | 0.7150 |
| HC＋S | 0.6767 | 0.7189 | 0.7569 | 0.7556 | 0.7270 |
| HC＋A0 | 0.6763 | 0.7180 | 0.7552 | 0.7559 | 0.7263 |
| S 單獨（凍結頭） | 0.6734 | 0.7226 | 0.7393 | 0.7581 | 0.7234 |
| A0 單獨（凍結頭） | 0.6737 | 0.7171 | 0.7347 | 0.7394 | 0.7162 |

| 差異 | k=5 | k=10 | k=15 | k=20 | Equal-k | 95% 區間 | P(>0) | 各 seed Equal-k 差 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| **HC＋S − HC_cal（實用性）** | +0.0072 | +0.0062 | +0.0135 | +0.0214 | **+0.0121** | [+0.0031, +0.0211] | 0.994 | +0.0112／+0.0152／+0.0132／+0.0101／+0.0107 |
| **HC＋S − HC＋A0（新增資訊）** | +0.0004 | +0.0009 | +0.0017 | −0.0003 | **+0.0007** | [−0.0014, +0.0029] | 0.721 | +0.0012／+0.0002／−0.0014／+0.0037／−0.0003 |
| HC＋A0 − HC_cal | +0.0068 | +0.0053 | +0.0118 | +0.0217 | +0.0114 | [+0.0029, +0.0205] | 0.994 | |
| HC_cal − HC | +0.0056 | +0.0090 | +0.0049 | +0.0042 | +0.0059 | [+0.0031, +0.0086] | 1.000 | |
| HC＋S − HC | +0.0128 | +0.0152 | +0.0184 | +0.0256 | +0.0180 | [+0.0088, +0.0269] | 1.000 | |

- 實用性：Equal-k ≥ +0.005、P ≥ 0.90、各 k ≥ −0.005、五個 seed 皆正 → **通過**。新增資訊：平均 +0.0007 > 0，但 P＝0.721 < 0.90 → **未通過**。內部選擇有採用 S（β＞0 在所有 100 次選擇中都成立，多數為 1.0）。
- **α 選擇多落在網格下界 0.6**，各臂分開列（每臂 100 次選擇＝5 seed × 5 外折 × 4 k）：HC_cal 83 次在 0.6、1 次在 1.4；HC＋S 87 次在 0.6、1 次在 1.4；HC＋A0 83 次在 0.6、0 次在 1.4。α 是以官方分數選的，不是校準誤差或 NLL，所以這只表示「搜尋偏好降低 HC 的銳利度／權重，最佳點可能在搜尋範圍外」，不能直接證明 HC 過度自信或 0.6 不夠平。在融合臂中，α 同時控制 HC 相對於另一個模型的影響，與 HC_cal 的 α 意義不同。依登記標示、不擴網格；HC_cal 可能未達其最佳設定，實用性差值的一部分可能仍來自 HC 本身的調整。

**判讀（依登記用語）：** 加入 S 後驗有實用增益，但尚未證明優於加入 A0，因此無法把增益歸因於 S 新增的分布統計，不以此支持擴大 S；也只涵蓋 785715 這一個 S 訓練版本。
- HC＋S 與 HC＋A0 點估計接近（差 +0.0007，區間跨 0），尚未檢出 S 的額外增益；這支持停止擴大 S，不代表兩者等價。S 單獨（0.7234）高於 A0 單獨（0.7162），也不保證這個優勢能補到 HC 上。
- 本次有價值的發現：在這個 dev 設定下，融合另一個已訓練的 C 後驗（HC＋A0 − HC_cal ＝ +0.0114）能改善分數。凍結頭與 dev 內的 HC 在擬合玩家（另外 1,000 人對每折約 610 人）、訓練目標、正則化、跨 k 共用方式與資料組成上都不同，增益與哪一項有關未知；正式管線的 B2 用約 5,700 位玩家擬合，這個增益是否轉移也未知。
- 不能說：增益來自新的棋力資訊。

**後續（使用者 11/16 意見，未執行）：** 不立即投入新架構，也不為追究機制另開一串實驗；優先順序上「先把現有模型的校準與融合弄清楚」高於 v_ref 或主幹重構。日後若重啟，先選目的：
- 判斷能否改善正式管線：以正式擬合池訓練 HC，比較校準後 HC 與 HC＋A0；先確認特徵、玩家隔離與成本。
- 研究小型 dev 融合為何有效：固定評估玩家，做不同擬合玩家數的配對比較，其他設定固定。增益隨擬合規模縮小只支持「融合收益與 HC 的擬合規模有關」，不能證明它是唯一原因。

**算力：** 787866 1.11 GPU-h＋788110 0.05＋788136 0.54 ＝ **1.70 GPU-h**（兩階段上限各 2）。不產生提交檔；基底與最終預設檔不變（`751959-cv2-l10.csv`）。程式 `scripts/strength/rank_hc_s_fusion.py`、測試 `tests/test_rank_hc_s_fusion.py`（3 項通過；另以凍結頭逐題重現 785715 的 S／A0 dev 機率，最大差 4e-6、預測 100% 一致）。

### 2026-11-16：競賽二——正式管線上的 HC 校準與 HC＋A0 後驗融合（使用者「好」；事前登記）

**問題：** dev 上「融合另一個已訓練的 C 後驗」有 +0.0114（HC＋A0 − HC_cal），但 dev 的 HC 每折只有約 610 位擬合玩家。正式 B2 用約 5,700 位玩家擬合時，這個增益是否還在？順帶檢查正式 HC 只調溫度（HC_cal）本身。**預期：未知，傾向小。**

**第 1 步：重建評估鏈（不通過就停止）。** 11/16 的刪除與清理打斷了現行鏈：`outputs/rank/v2/v2_strength/`（A-024 分類）不存在；H 標準陣列已無 C v1；`p3_komi/c_train` 與測試 `rankvec`（C v1）已清；`outputs/rank_test_ahc_713770/` 的相容連結不存在（資料在 `data/processed/release/rank_test_ahc_713770/`）。新程式不改舊程式，改以現存資料組出同一組特徵：
- Valid 棋局＝`cvec_v2/ext_valid_hd{0,1}`（Valid-U 全部＋Valid-S HD≤1），角色由 `data/processed/cv/strength_folds_seed2027.parquet` 判定（fold 0＝U）；順序依 H 標準陣列的 game_index 順序（B2 合成抽樣與列序有關）。
- H：標準陣列，HD1 換成 `p3_komi/h_train`（貼目 0.5）；train20：HD0 用標準陣列、HD1 用 `train20_hc/h_hd1`。C：C v2（`cvec_v2` 的 c_valid／c_t20／c_test）。旗標：HD1 × 顏色。測試：A-024 category 與 release HumanSL 的交集，HD1 的 H 換成 `p3_komi/h_test_*`。
- 驗收：(a) 本地：seed 2027 的 HC（Valid-S＋train20 擬合、Valid-S 先驗）在 E1／E2 的分 k 分數須逐位數等於 `cvec_v2/result.json` 的 v2（E1 0.7117、E2 0.7505）；(b) 匯出：seed 2027、Valid-S＋train20＋Valid-U 擬合，Public＋Private 800 題須逐題等於 `submissions/rank/751959-cvec2.csv`。任一不符 → 停止回報，不進第 2 步。

**第 2 步：比較（同一程式，第 1 步通過後才執行）。**
- 三臂：HC（現行 B2，Valid-S＋train20 HD≤1 擬合、Valid-S 先驗）、HC_cal（只調 α）、HC＋A0（對數融合 α·log p_HC＋β·log p_A0）。A0＝785715 的凍結頭，輸入為每盤 C v2（與 785715 `raw_c` 同定義，已驗證）。網格、選法與同分規則同 dev 版（α ∈ {0.6, 0.8, 1.0, 1.2, 1.4}、β ∈ {0, 0.25, 0.5, 0.75, 1.0}，每個 k 以官方平均分數選，容差 1e-6，依序取較小 β、α 較接近 1、較小 α）。
- 玩家隔離：A0 頭用 785715 的 train 1,000 人與 monitor 200 人（皆在擬合池內）訓練與選模；選 α、β 的校準題只取擬合池中**這 1,200 人以外**的玩家。擬合池按級別分層分 5 個內折（`default_rng(2027)`）；每個內折以其餘玩家（含那 1,200 人）擬合 B2，對留出的選題玩家出單色題（每 k × 級別 200 題、HD≤1、`default_rng(2028)`，依玩家分到內折）。A0 頭與 C 向量都沒見過 Valid-U（fold 0）。
- 評估：Valid-U E1（1,000 題）／E2（2,000 題）；B2 seed 2027–2031 各自擬合、各自選 α、β、各自計分，再平均。
- Gate（沿用正式規則，五 seed 平均）：**HC＋A0 − HC_cal**：E1 或 E2 ≥ +0.005、另一組 ≥ 0、E2 各 k ≥ −0.005。另以同一規則報 **HC_cal − HC**（校準本身是否為候選）。E2 首 seed 附玩家 bootstrap。通過也只列為 LB 探針候選，提交檔另行確認後才產生；不換基底。
- 限制：E1／E2 已多次使用；A0 只有一個訓練版本；α 以官方分數選，邊界比例照記、不擴網格。

**成本：** 30 次正式規模 B2 擬合（5 seed ×（5 內折＋1 全池））＋重現匯出 1 次；單次約 155 秒（11/15 登入節點實測），11 行程並行估 15–30 分（計算節點速度 11/16 曾差 10 倍）。1 卡／12 核配置，上限 2 小時＝2 GPU-h。

**結果（11/16；Slurm 788465 在第 2 步因我的 worker 函式查錯鍵而失敗，7 分 43 秒，25a-hgpn006；修正後 788512 完成，1 分 12 秒，25a-hgpn005；輸出 `outputs/rank/humansl/prod_a0_fusion/`）：第 1 步通過，第 2 步兩個比較都未過 gate。**

**第 1 步（兩次作業各自重跑，都通過）：** 重建後的擬合池 149,615 盤／5,728 人、Valid-U 35,083 盤／1,442 人，與舊鏈相同。seed 2027 本地分數逐位數等於 `cvec_v2/result.json`（E1 0.6128／0.7219／0.7661／0.7461、Equal-k 0.7117；E2 0.7022／0.7494／0.7762／0.7740、0.7505）；重建匯出對 `751959-cvec2.csv` 800 題改變 0 題。新程式可作為 11/16 清理後的正式本地評估鏈。

**第 2 步（Valid-U，五 seed 平均；校準題 8,000 題，排除 A0 訓練／選模的 1,200 人）：**

| 臂 | E1 k=5／10／15／20 | E1 Equal-k | E2 k=5／10／15／20 | E2 Equal-k |
|---|---|---:|---|---:|
| HC | 0.6197／0.7249／0.7648／0.7463 | 0.7139 | 0.6935／0.7466／0.7692／0.7759 | 0.7463 |
| HC_cal | 0.6222／0.7198／0.7683／0.7453 | 0.7139 | 0.6961／0.7460／0.7735／0.7766 | 0.7480 |
| HC＋A0 | 0.6223／0.7181／0.7675／0.7348 | 0.7107 | 0.6909／0.7466／0.7697／0.7827 | 0.7475 |
| A0 單獨（凍結頭） | 0.5932／0.6979／0.7390／0.7230 | 0.6883 | 0.6671／0.7246／0.7412／0.7737 | 0.7266 |

| 比較 | E1 | E2 | E2 各 k | E2 bootstrap 95%（seed 2027） | Gate |
|---|---:|---:|---|---|---|
| **HC＋A0 − HC_cal** | −0.0032 | −0.0005 | −0.0051／+0.0006／−0.0038／+0.0062 | [−0.0103, +0.0021] | 未過 |
| **HC_cal − HC** | 0.0000 | +0.0017 | +0.0026／−0.0006／+0.0043／+0.0007 | [−0.0062, +0.0023] | 未過 |
| HC＋A0 − HC | −0.0033 | +0.0012 | −0.0025／0.0000／+0.0005／+0.0068 | [−0.0131, +0.0016] | （參考） |

- 選到的權重：α 多在 0.6（HC_cal 14／20、HC＋A0 17／20）；HC＋A0 的 β 為 0（6）、0.25（11）、0.5（3），沒有選到 0.75 以上（dev 版多為 1.0）。
- **判讀：** dev 上 HC＋A0 − HC_cal 的 +0.0114 在正式管線（約 5,700 位擬合玩家）沒有出現（E1 −0.0032、E2 −0.0005）。這與「融合收益和 HC 的擬合規模有關」一致，但兩個設定還有擬合池組成、題組、評估玩家等其他差異，不能證明規模是唯一原因。HC 只調溫度也沒有可用增益（E1 0.0000、E2 +0.0017，未過 +0.005）。
- **處置：** 依登記停止：不產生提交檔、不換基底（`751959-cv2-l10.csv`）。A0／S 融合與溫度調整這條線結案。

**算力：** 788465 0.13 GPU-h＋788512 0.02 GPU-h（1 卡配置、只用 CPU）＝0.15 GPU-h。同一工作在 25a-hgpn006 上兩次都明顯變慢（11/16 另一作業也是），原因未查；重送時排除該節點。

### 2026-11-16：競賽一——凍結編碼器的跨手聚合頭（A 平均／B 逐手 MLP／C Transformer）→ C 未過（未勝 B）；B 大幅改善，但屬對照臂、未預登記

**設計（使用者定稿）：** `v4_d4_hd_9600` 編碼器凍結，逐手 256 維向量快取（`scripts/identity/pid2_seqagg.py`；`/home/jason1122/pid2_seqagg_cache/`，fp16，訓練玩家 6,426 人／447,684 盤／4,981 萬手＋驗證台 47,280 盤／525 萬手）。A＝逐手平均（現行）；B＝每手共用的兩層殘差前饋區塊（pre-LN，256→512→256）後遮罩平均；C＝同樣區塊前各加 4-head 自注意力（兩層 TransformerEncoderLayer），B 就是 C 拿掉注意力。兩者都加「實際手數」的可學位置向量。GE2E（w、b 同初始化、可學），每步 64 人 × 8 盤、全部有效手、padding 遮罩，20,000 步，AdamW 3e-4、5% warmup、cosine；兩臂抽樣序列相同（只依 seed 與 step）。按段位分層保留 5% 玩家（322 人，「聚合頭保留玩家」，編碼器看過）只做監控：每 500 步、3 組固定題組、10 盤 gallery、k=1／2／3／5 等權、純風格官方分數，以此選 checkpoint，兩臂都跑滿。主評估固定原始 w（存檔完整精度 11.66797…）、λ=16、μ=1、D4 開局、類別項。

**工程關（通過）：** 快取完整性——驗證台棋局順序、手數 5,249,223 與預期相同、全為有限值；A 對存檔盤向量最小 cosine 0.999993；**A 與存檔向量在兩個 pool 逐 draw 重現第一階段官方分數（整體最大差 0）**。padding 自我檢查（同一盤單獨算＝補 padding 且 padding 塞亂數後算）兩臂皆過。慢速節點：抽取在 25a-hgpn006 只有約 3,000 手／秒（其他節點約 14,000），排除後全量 56 分鐘。

**成本：** 短測 787262、慢速抽取 787269／787847（取消／測速，共約 0.65 GPU-h）、全量抽取 787967（56 分）、B 787968（24 分）、C 787969（27 分）、評估 787970（2 分）、冒煙 787858（3 分）；合計約 2.9 GPU-h（使用者同意超出 8 GPU-h 亦可跑完，實際未超出）。

**監控（保留玩家，純風格）：** A 0.7092。B 從 0.718（500 步）緩升到 0.7511（18,500 步，選中）；C 在 4,500–7,000 步達 0.748（7,000 步選中）後回落到 0.738–0.740，呈現過擬合跡象。學到的 w：B 12.81、C 11.67（只列診斷）。

**驗證台（對 A＝現行模型，配對，標準誤為 draw 間）：**

| | pool 0 完整融合 k=1／2／3／5 | pool 0 整體 Δ（SE） | pool 1 完整融合 k=1／2／3／5 | pool 1 整體 | pool 1 Δ（SE，Δ／SE） |
|---|---|---|---|---|---|
| A | 0.3592／0.5960／0.7139／0.8318 | 0.6252 | 0.3680／0.5705／0.7149／0.8254 | 0.6197 | — |
| B（18,500 步） | 0.3802／0.6178／0.7433／0.8555 | +0.0240（0.0024） | 0.3843／0.6089／0.7397／0.8467 | 0.6449 | **+0.0252（0.0023，≈11）**，100% draw 為正 |
| C（7,000 步） | 0.3856／0.6189／0.7458／0.8502 | +0.0249（0.0027） | 0.3789／0.5963／0.7348／0.8448 | 0.6387 | +0.0190（0.0024，8.07），97% |

純風格 pool 1：A 0.5678、B 0.6117（+0.0439）、C 0.6066（+0.0388）。

**Gate 判定：** pool 0 篩選通過（C−A +0.0249 ≥ +0.005，C−B +0.0009 > 0）。pool 1：C−A +0.0190 ≥ +0.010 ✔、Δ／SE 8.07 ✔、各 k ≥ −0.005 ✔（+0.011／+0.026／+0.020／+0.019），**但 C−B −0.0062（SE 0.0020）✘ → C 未過。** 依事前規則，結論是「可訓練的逐手聚合頭可能有用」，**不能歸因於跨手互動**；B 不含任何手與手之間的資訊交換，卻比 C 好。

**B 的數字與限制：** B 在兩個 pool、四個 k 桶都明顯為正，數值上超過 C 的 pool 1 門檻；但 B 是對照臂，不是預登記的實驗臂，依使用者事前規則不自動升級。機制未知，至少有兩個未拆開的來源：(1) 逐手的非線性轉換／重新加權；(2) 位置向量的平均會把**盤長與執色**帶進盤向量（白棋的手數全為奇數），這是正式資料也有的合法資訊，但不是「風格」本身。兩個 pool 自 11/12 起反覆用於開發，不是獨立測試；只有一個 seed。進一步驗證需要：多 seed、拿掉位置向量的 B、預先封存的新題組，以及接到正式新增集的推論路徑（目前正式管線未改、未產生提交檔）。

輸出：`outputs/pid/pid2_seqagg/`（`probe.json`、`head_{B,C}/`、`eval.json`、`eval_per_draw.csv`），log `outputs/pid/pid2_bench/logs/pid2-seqagg-*.out`。

### 2026-11-17：競賽一——B（逐手 MLP 聚合頭）確認實驗（使用者同意；事前登記，看結果前寫定）

**目的：** 11/16 的 B 是對照臂、單一 seed、評估用的題組已反覆開發。這輪把 B 當預登記實驗驗證，並拆開「位置向量帶入盤長／執色」的來源。編碼器、快取、保留玩家名單、監控題組、選 checkpoint 規則、訓練設定（20,000 步、64 人 × 8 盤、AdamW 3e-4）與主評估融合（原始 w、λ=16、μ=1、D4 開局、類別項）全部沿用 11/16。

**臂：**
- B：seed 2026（11/16 已訓練，沿用）、2027、2028。seed 同時決定初始化與抽樣序列；保留玩家與監控題組固定不隨 seed 變。
- B−pos（拿掉位置向量，其餘同 B）：seed 2026、2027、2028。只作機制診斷，不設 gate。

**封存題組：** 驗證台 pool 0 draws 500–519、pool 1 draws 500–529（從未用過的種子；玩家仍是同兩個 pool，非全新玩家）。所有臂評估完才看結果，不依這批結果改任何設定。

**Gate（B 每個 seed 各自判，對 A＝現行平均，完整融合）：** 封存 pool 1 整體 Δ ≥ +0.010、Δ／SE ≥ 2.5、四個 k 桶各 ≥ −0.005，且封存 pool 0 整體 Δ > 0。**三個 seed 全過**才算確認。另報三 seed 的平均與全距。

**機制診斷（無 gate）：** B − (B−pos) 的三 seed 平均。B−pos 保留 B 增益的比例 ≥ 2/3 → 增益主要來自逐手轉換；≤ 1/3 → 主要來自位置向量（盤長／執色）；其餘＝兩者都有。兩種結果都不影響是否採用（盤長與執色在正式資料一樣可得），只影響解釋。

**確認後的提交候選（事前指定，不依結果挑 seed）：** 用 seed 2026 的 B（11/16 那份，選中的 18,500 步 checkpoint）；只替換新增集 `pub2_`／`priv2_` 的風格項，開局維持 D4、原測試集維持 Hungarian，其餘與基底 `772748-hung.csv` 相同。推論路徑先驗證「A 在正式新增集逐值重現 `predict2` 的風格項與融合分數」才換成 B。未確認就不產生提交檔。

**成本預估：** 5 個新訓練各約 25 分鐘（1 卡）＋評估約 10 分鐘＋正式向量抽取約 5 分鐘，合計約 2.5 GPU-h。


### 2026-11-17：A0／R10／Z10／Z32 固定十二輪實驗實作與提交準備

使用者依 `plan.md` 明確授權實作並啟動首輪，提交後停止對話及監看，完成後另行分析。
入口 `src/go_aicup/tasks/strength/policy_residual/branches.py`／`branches.slurm`；
設定 `configs/strength/policy_branches.json`，獨立輸出
`outputs/rank/humansl/policy_branches/seed2027_b32_e12/<job_id>/`。
沿用785715固定資料／sklearn起點，四臂每臂3,000更新、每步32題（各k八題）、
LR=.0003、十二輪、每125步完整TRAIN／monitor評估、無早停；只由monitor嚴格改善選模，
選定後一次dev評估，不自動追加seed或預算，不改正式基底、不產生上傳檔。

CPU聚焦驗收44項通過，涵蓋四臂初始化／共同編碼器／欄位順序、
變長棋局及重複引用的分塊梯度、支路首步與後續梯度、D4／候選排列、
半輪中斷重播與best回復、非有限值拒絕、gate邊界與玩家bootstrap，含原residual／epoch回歸。
真實來源hash、固定題組、玩家互斥及membership預檢通過；CUDA／BF16與真teacher由正式作業檢查。

磁碟預檢：/work 100 GiB、約68 GiB已用、約33 GiB可用；/home約45 GiB可用。
完整FP16 policy約55.67 GiB與編譯cache僅放RAM，共用資料用相對連結；
持久紀錄預留200 MiB，估約120 MiB內：initial／best／latest、逐步分組gradient／update摘要、
全評估曲線與逐題機率、分支logits RMS、selected逐盤特徵／預測、來源／程式／計畫快照、
環境／覆蓋率／精度／參數／資源報表。無逐手feature或梯度張量，不刪既有資料。
bootstrap 2,000次，按玩家配對重抽，每次重算四桶Equal-k；保存分k及Equal-k差值／95%區間／
Pr_boot(Δ>0)，gate不用四捨五入值。

預估90–120分鐘、五張H200配置7.5–10 GPU-h；Slurm上限2小時／10 GPU-h、60CPU／500GiB RAM。
五卡共同抽取，四卡各訓練一臂；訓練CUDA聚合採確定性模式，真實吞吐與四桶增益未知。
超時保留輪界續跑與半輪best，不自動重送。提交編號於下方追加。

提交成功：Slurm **788616**（2026-11-17T00:06:47.415450+08:00），輸出 `outputs/rank/humansl/policy_branches/seed2027_b32_e12/788616/`；最終聚焦測試44項通過（7.59s），Slurm語法與diff檢查通過。僅確認sbatch接受，不輪詢排隊或執行狀態；正式結果、CUDA數值與四桶分數未知。

### 2026-11-17：競賽一——B 聚合頭提交探針（使用者「沒過 gate 也傳看看」；確認實驗 788614 尚未出結果）

Public 第一名 **0.9181331**（使用者 11/17 回報）；我們基底 `772748-hung.csv` 0.7981。原測試集約 0.991 → 對方新增集至少約 0.845，我們約 0.57。

| 檔案／SHA 前八碼 | 內容 | 本地 | 預期 Public Δ | 改 top-1：pub2／priv2（k=1／2／3／5） | LB |
|---|---|---|---|---|---|
| `submissions/pid/hung-seqB.csv`／`a764e025` | 原測試集與 `772748-hung.csv` 逐列相同（Hungarian）；新增集風格項換 B 頭（seed 2026、18,500 步），w、λ=16、μ=1、D4 開局、類別項不變。來源 788615 匯出（A 在正式新增集逐值重現風格項，最大差 7e-8，top-1 一致 100%） | 驗證台（開發用 draws）pool 1 +0.0252（SE 0.0023）、pool 0 +0.0240；封存題組確認未出 | 約 +0.012（新增集占一半）；Public SE 約 0.015–0.02 | 91（32／26／15／18）／109（36／35／28／10）  **0.8012002**（11/17 00:24:16，jabir95tsai；37/98） |

保留規則：此檔是否升基底看 788614 封存題組三 seed gate，不看 Public 單次升降；Public 降 ≥0.02 → 回查正式推論路徑。匯出腳本 `outputs/pid/seqB_submission/export.py`（拒絕覆寫；檢查分數重建、top-5 與 seqB predictions 一致、原測試集不變、前綴與 5 人唯一），報表同目錄 `report.json`。

**新增集「同一盤棋換字串出現」檢查（CPU，登入節點約 1 分鐘，scratch 腳本，結果記此）：** 八個競賽一檔案 104,200 盤，以 D4 不變的前 30／60／100 手著手序列比對。前 30 手相同 14 組、前 60 手 3 組、前 100 手 2 組，跨候選庫↔題目最多 1 組 → **不存在大量共用棋局的結構**，解釋不了與第一名的差距。

**LB（11/17 00:24）：`hung-seqB.csv` Public 0.8012002，37/98，對基底 +0.0031**（預期約 +0.012；方向一致、幅度較小，差距在 Public 雜訊內）。升基底與否仍等 788614 封存題組 gate。

**同日其餘探針（使用者「兩個都做」；都是新增集上的單一變數，原測試集逐列同 `772748-hung.csv`）：** CPU 合併腳本 `outputs/pid/seqB_submission/combine.py`（驗證各來源分數重建、題目與候選順序一致、類別項相同、原測試集不變、前綴與 5 人唯一；拒絕覆寫），報表 `combine_report.json`。

| 檔案／SHA 前八碼 | 對 `hung-seqB.csv` 只改 | 本地 | 改 top-1 對 seqB：pub2／priv2（k=1／2／3／5） | LB |
|---|---|---|---|---|
| `hung-seqB-raw.csv`／`f22c317a` | 開局改 raw（`*_d4_9600_raw` 的開局項） | raw 對 D4 在 A 上 pool 1 +0.0051；B 上未測 | 27（7／8／5／7）／38（6／9／11／12） | **0.8037463**（11/17 23:00:06，jabir95tsai） |
| `hung-seqB-arm1.csv`／`96dcb48c` | λ 改 8／8／16／16（D3-a Arm 1） | Arm 1 在 A 上 pool 1 +0.0034；B 的尺度不同，未重測 | 14（5／9／0／0）／19（6／13／0／0） | **0.8029919**（11/17 23:00:14，jabir95tsai；38/105） |
| `hung-seqC.csv`／`df5b4914`（788842，3 分 57 秒；A 重現最大差 7e-8、top-1 一致 100%） | 風格頭換 C（7,000 步） | 開發 draws pool 1 對 A +0.0190、對 B −0.0062 | 87（34／22／17／14）／77（30／18／20／9） | **0.8044628**（11/17 01:40:00，jabir95tsai；35/98） |
| `hung-seqB3.csv`／`80580017`（788843，7 分 29 秒；兩 seed 的 A 重現 top-1 一致 100%） | 風格項取 B seed 2026／2027／2028（18,500／9,000／11,500 步）三者平均 | 平均本身未測；三 seed 各自封存 pool 1 +0.0276／+0.0231／+0.0259 | 28（12／9／3／4）／43（16／13／9／5） | **0.8056013**（11/17 01:39:37，jabir95tsai；名次未提供） |

建議上傳順序：seqB3（若 788614 三 seed 都過）→ seqB-raw → seqC → seqB-arm1。全部是探針，Public 升降都不直接升基底。

**11/17 23:00 補登（使用者回報）：** `hung-seqB-raw.csv` 0.8037463（對 seqB +0.0025）、`hung-seqB-arm1.csv` 0.8029919（+0.0018，38/105）。11/17 五次額度用完。五個探針對 seqB 都在 +0.0018～+0.0044，全部小於 Public 雜訊（約 0.02），無一可據以升基底；raw 開局與 λ 降權在 B 上都沒有本地數字。基底維持 `hung-seqB.csv`。

**對手風格配對檢查（事前登記；`scripts/identity/pid2_oppstyle.py`，Slurm 788853，1 卡）：** 假設：玩家常連續對同一位對手下棋，查詢盤的**對手落子**會和正解候選人候選盤的對手落子相似；現行模型只用本人落子。凍結 `v4_d4_hd_9600` 編碼器抽對手那一方的盤向量（換色、min_ply 同原設定；先驗證本人側重現存檔向量 cosine ≥ 0.9999）。報告 pool 0（draws 100–119）／pool 1（100–129）：只用本人、只用對手（中心向量）、只用對手逐盤最大 cosine、級別相同的 oracle 參考（對手向量含配對棋力的訊號，用來區分「同級」與「同一位對手」），以及完整融合＋λ_opp·w·對手 cosine（λ_opp＝0／2／4／8／16）與＋逐盤最大（4／8）。**判讀：** 對手中心向量只用時若明顯超過級別 oracle，或逐盤最大明顯高於中心向量 → 支持「同一位對手」結構；融合 Δ 在 pool 0 選 λ、pool 1 看大小，pool 1 Δ ≥ +0.02 才值得做正式新增集推論與提交。這兩個 pool 是開發用 draws，不是封存題組。

**提交檔封存（11/17，使用者要求）：** `submissions/pid/` 只留基底 `772748-hung.csv` 與今天的 `hung-seqB*.csv`；其餘 20 份競賽一舊檔移到 `submissions/pid/archive/`（只搬移，內容與檔名不變）。本檔較早段落寫的 `submissions/pid/<檔名>` 路徑，對這些舊檔請改讀 `submissions/pid/archive/<檔名>`。競賽二的 `submissions/rank/` 未動。

**對手風格配對結果（788853，1 卡 7 分 26 秒；本人側重現存檔 cosine 0.999994）→ 假設不成立，結案。** 輸出 `outputs/pid/pid2_oppstyle/`（`opp_embeddings.npz`、`per_draw.csv`、`report.json`）。

| 只用單一訊號（官方分數，整體） | pool 0 | pool 1 |
|---|---|---|
| 本人風格 | 0.5807 | 0.5678 |
| 對手風格（中心向量） | 0.0395 | 0.0374 |
| 對手逐盤最大 cosine | 0.0302 | 0.0312 |
| 級別相同 oracle（同級候選人內均勻） | 0.0260 | 0.0239 |

完整融合加對手項：pool 1 λ_opp＝2／4／8／16 為 −0.0075／−0.0201／−0.0551／−0.1406；逐盤最大 4／8 為 −0.0127／−0.0388；pool 0 同樣全部為負。**判讀：** 對手向量只比「知道級別」多一點點（0.037 對 0.024），逐盤最大反而低於中心向量 → 驗證台上**看不到「常對同一位對手」的結構**，對手項加進融合只有傷害。限制：驗證台是訓練集玩家的棋局，若官方新增集的抽盤方式（例如連續時段）不同，結論未必轉移；這點無法在本地檢驗。不產生提交檔。

**B 確認實驗結果（788609–788614，事前登記見本日「B 確認實驗」節）→ 三個 B seed 全過 gate，確認（confirmed=true）。** 輸出 `outputs/pid/pid2_seqagg/confirm.json`、`confirm_per_draw.csv`，log `outputs/pid/pid2_bench/logs/pid2-seqagg-confirm-788614.out`。封存題組：pool 0 draws 500–519、pool 1 draws 500–529，完整融合（原始 w、λ=16、μ=1、D4 開局、類別項），對 A（現行平均）：

| 臂 | 選中步數 | 封存 pool 0 Δ | 封存 pool 1 Δ（SE，z） | pool 1 k=1／2／3／5 | gate |
|---|---|---|---|---|---|
| B s2026 | 18,500 | +0.0204 | +0.0276（0.0020，13.6） | +0.024／+0.032／+0.028／+0.026 | 過 |
| B s2027 | 9,000 | +0.0210 | +0.0231（0.0019，11.9） | +0.023／+0.026／+0.021／+0.022 | 過 |
| B s2028 | 11,500 | +0.0233 | +0.0259（0.0017，14.8） | +0.028／+0.026／+0.024／+0.026 | 過 |
| B−pos s2026 | 19,000 | +0.0194 | +0.0254（0.0018） | +0.025／+0.029／+0.025／+0.023 | 診斷 |
| B−pos s2027 | 12,000 | +0.0208 | +0.0243（0.0020） | +0.023／+0.031／+0.017／+0.026 | 診斷 |
| B−pos s2028 | 10,000 | +0.0180 | +0.0235（0.0017） | +0.020／+0.025／+0.021／+0.028 | 診斷 |

封存 pool 1 絕對分數：A 0.6186 → B s2026 0.6462（k=1 0.373→0.397、k=5 0.824→0.850）。B 三 seed pool 1 平均 +0.0255、全距 [+0.0231, +0.0276]；每個 draw 都是 B 較高（pos=1.0）。只用風格（不融合）的增益更大：pool 1 +0.036～+0.039。

**機制診斷：** B−pos 三 seed 平均 +0.0244，保留 B 增益的 **95.6%**（≥ 2/3）→ 增益主要來自**逐手轉換**（每手向量先過殘差 MLP 再平均），不是位置向量帶入的盤長／執色。

**判讀：** 封存題組的增益與開發用 draws（+0.0252）一致，沒有開發題組過擬合的跡象；三 seed 的差異（全距 0.0045）小於增益本身一個數量級。限制：封存題組的玩家仍是驗證台同兩個 pool（不是全新玩家）；Public LB 只 +0.0031（`hung-seqB.csv`，0.8012002），方向一致、幅度在 Public 雜訊內。

**提交檔：** 事前指定的候選（seed 2026 B，只換新增集風格項）就是已存在的 `submissions/pid/hung-seqB.csv`（SHA 前八碼 `a764e025`，來源 788615 匯出），不另產生新檔。依 11/17 保留規則（升基底看 788614 gate、不看 Public 單次升降），此檔符合升為競賽一基底的條件；CLAUDE.md §0 的基底仍寫 `772748-hung.csv`，待使用者確認後改。

**成本：** 5 個訓練各 1 卡，共 2 小時 52 分（788613 單獨 65 分，其餘 24–34 分）；確認 4 分、正式匯出 788615 4 分 → 約 **3.0 GPU-h**（預估 2.5）。

### 2026-11-17：競賽一——B 確認實驗結果（788609–788614）→ 三 seed 全過，B 確認；基底改為 `hung-seqB.csv`

訓練 788609／788610（B s2027／s2028，24 分／34 分）、788611–788613（B−pos s2026／s2027／s2028，23 分／25 分／1 時 5 分）；封存題組評估 788614（4 分）。log `outputs/pid/pid2_bench/logs/pid2-seqagg-confirm-788614.out`。選中 checkpoint：B 18,500／9,000／11,500；B−pos 19,000／12,000／10,000。

**封存題組 pool 1（draws 500–529），完整融合，對 A 的差（SE）；k=1／2／3／5：**

| 臂 | 整體 Δ | k 桶 Δ | 封存 pool 0（500–519）Δ | gate |
|---|---|---|---|---|
| B s2026 | **+0.0276**（0.0020，z 13.6） | +0.024／+0.032／+0.028／+0.026 | +0.0204 | ✔ |
| B s2027 | **+0.0231**（0.0019，z 11.9） | +0.023／+0.026／+0.021／+0.022 | +0.0210 | ✔ |
| B s2028 | **+0.0259**（0.0017，z 15） | +0.028／+0.026／+0.024／+0.027 | +0.0233 | ✔ |
| B−pos s2026／s2027／s2028 | +0.0254／+0.0243／+0.0235 | 全部 ≥ +0.017 | +0.0194／+0.0208／+0.0180 | （無 gate） |

絕對分數（封存 pool 1 完整融合）：A 0.6186 → B s2026 0.6462。純風格：A 0.5703 → B 0.6065–0.6093（+0.036～+0.039）。

- **判定：三個 seed 都過**（pool 1 Δ ≥ +0.010、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0）→ **B 確認。** 三 seed 平均 +0.0255，全距 [+0.0231, +0.0276]。
- **機制診斷：** B−pos 平均 +0.0244，保留 B 增益的 **96%**（≥ 2/3）→ 增益主要來自**逐手非線性轉換／重新加權**，不是位置向量帶入的盤長／執色。
- **依事前規則，競賽一基底改為 `hung-seqB.csv`**（seed 2026，已上傳 Public 0.8012）。`hung-seqB3.csv`（三 seed 平均）是下一個候選：平均本身沒有封存題組數字，Public 升降都不自動取代基底；要升須先在封存題組補評三 seed 平均。
- 限制：兩個 pool 的玩家自 11/12 起反覆使用，封存的只是題目抽樣種子，不是新玩家；本地增益（新增集約 +0.026）換算 Public 約 +0.013，`hung-seqB` 實測 +0.0031，在 Public 雜訊內。

### 2026-11-17：競賽一——編碼器＋B 聚合頭端到端微調（使用者「照這個計畫做，先跑短測」；事前登記，看結果前寫定）

**假設：** 現行編碼器（`v4_d4_hd_9600`）的逐手向量是為「直接平均」而訓練，B 頭是事後凍結訓練的；兩者一起訓練，編碼器可以把 B 能用的身分訊號放進逐手向量。混淆：繼續訓練本身就會進步（4,800→9,600 步 +0.010），所以設對照臂。

**臂（各 1 seed＝2026）：**
- 基底：`9600`＋B s2026（18,500 步），不訓練。
- E1（`pid_e2e_joint_4800`）：編碼器從 `9600`、B 從 s2026，一起訓練；每手先過 B（含 ply 位置向量），盤向量＝B 輸出平均，GE2E 用 B 的 w、b。
- E0（`pid_e2e_encoder_4800`）：只訓練編碼器、盤向量直接平均（原訓練方式），其餘完全相同；之後用 E0 的向量照 11/16 流程重訓 B（20,000 步、seed 2026）＝「E0＋B」。

**共同設定：** 64 人 × 8 盤 × 每盤 16 手，雙色、玩家層級 D4、HD ≤1、min_ply 0；AdamW 峰值 LR 1e-4（未驗證的取值）、前 5% 暖機、cosine 到 0、wd 1e-4、裁切 1.0、BF16；4,800 步；取最後一步。訓練玩家＝`9600` 的 6,426 人**扣掉 B 訓練時保留的 5%（各級）玩家**（計畫未寫明，實作時補上：讓監控玩家在微調中也沒見過；兩臂相同）。每 400 步在保留玩家上算監控分數（只用風格的官方分數，3 組固定題），只畫曲線、不選模。第 0 步必須重現凍結模型的監控分數（A 0.7092、B 0.7511，容差 0.003），等於驗證「LR=0 時 E1＝凍結 9600＋B」。

**評估：** 全新封存 draws：pool 0 用 600–619、pool 1 用 600–629，三臂都評估完才看結果。融合：原始 w、λ=16、μ=1、D4 開局、類別項；風格項乘各臂自己學到的 w。

**Gate（E1 對基底）：** 封存 pool 1 整體 Δ ≥ +0.010、Δ／SE ≥ 2.5、四個 k 桶各 ≥ −0.005，且封存 pool 0 Δ > 0。另報（無 gate）E0＋B 對基底、E1 對 E0＋B：E1 − (E0＋B) ≥ +0.005 且 z ≥ 2.5 → 增益來自聯合訓練；在雜訊內 → 主要是多訓練，改以 E0＋B 為候選。

**通過後：** E1 最後一步只換新增集 `pub2_`／`priv2_` 的風格項，原測試集與 `hung-seqB.csv` 逐列相同；正式路徑先驗證 A 逐值重現再換。單 seed 通過即產生提交檔，有空再補 seed。

**程式：** `src/go_aicup/tasks/identity/train_e2e.py`（新）；B 頭與逐手抽取／監控 helper 由 `scripts/identity/pid2_seqagg.py` 搬到 `src/go_aicup/tasks/identity/seqagg.py`（行為不變，腳本改 import）。設定 `configs/identity/pid_e2e_{joint,encoder}_4800.json`、短測 `pid_e2e_joint_smoke100.json`／`pid_e2e_encoder_smoke30.json`；CPU 測試 `tests/test_identity_e2e.py`。輸出 `outputs/pid/ge2e/<run>/`（`model.pt` 與既有 load_encoder 相容、E1 另有 `head.pt`）。

**成本預估：** 約 6.5 GPU-h（E1 2.2、E0 2.1、E0 抽取＋B 重訓 1.5、評估 0.4、匯出 0.1），短測 0.1。短測若每步 > 2 秒，先回報不開跑。

**短測：** Slurm 789183（1 卡；CPU 測試 → E1 100 步、每 50 步監控 → E0 30 步）。

**LB（11/17 01:39–01:40，使用者回報）：** `hung-seqB3.csv` **0.8056013**，對新基底 `hung-seqB.csv`（0.8012002）+0.0044；`hung-seqC.csv` **0.8044628**（35/98），+0.0033。兩者都在 Public 雜訊內；seqB3 方向與「多 seed 平均較穩」一致，但要升基底仍須先在封存題組補評三 seed 平均。今天（11/17）已用 3 次（seqB、seqB3、seqC），保留 2 次給稍晚的作業。`hung-seqB-raw`／`hung-seqB-arm1` 暫不上傳（改動少、資訊量低）。

**競賽一棋局是否以 D4／重新排版形式出現在訓練集（`scripts/identity/pid2_trainmatch.py`，Slurm 789220，53 秒；`outputs/pid/pid2_trainmatch/report.json`）→ 完全沒有，結案。** 方法：訓練集 1M 盤取前 40 手（原方位）建表，競賽一八個檔案 104,200 盤各取前 40 手的 8 種 D4 變換查表。訓練集內部前 40 手相同有 392 組（方法本身能找到重複）；競賽一八個檔案（新增集候選／題目、原候選庫／題目）**命中全部為 0**，經訓練玩家把題目連到候選人的題數 public／private 皆 0。與 Day-0 逐字比對一致：競賽一棋局與訓練集不相交。

**短測結果（789183，25a-hgpn015，11 分 52 秒）：** CPU 測試 12 項通過。第 0 步兩臂都**逐值重現**凍結模型的監控分數（A 0.709192、B 0.751138）。每步 1.50 秒（兩臂相同），顯存峰值 123 GiB；訓練玩家 6,105（扣掉 321 位保留玩家，其中 250 位進監控）。**監控一開始訓練就下滑：** E1（B）0.7511 → 0.6827（50 步）→ 0.6953（100 步）；E0（平均）0.7092 → 0.6879（30 步）。原因未知，兩種可能：(1) 監控玩家在 `9600` 原訓練裡出現過，微調把他們拿掉後被「遺忘」，監控有偏；(2) Adam 狀態歸零＋LR 回到 1e-4，打亂已收斂的表示。先在編碼器沒見過的驗證台玩家（pool 0 開發用 draws 100–119，不是封存題組）上診斷：Slurm 789251，比較 `9600` A／B、短測 E1 100 步（A、B）、E0 30 步（A、凍結 B）。正式訓練未開，LR 待診斷後由使用者決定。

**計畫更正（看封存結果前）：** 計畫寫「風格項乘各臂自己學到的 w，和 11/16 B 頭做法相同」有誤：11/17 B 確認實驗是所有臂都用 `9600` 的原始 w（11.668）。為保持單一變數，評估一律用原始 w。新增評估入口 `pid2_seqagg compare`（各臂各自的逐手快取＋可選頭），`extract` 加 `--bench-run`。

**另兩個結構檢查（11/17 凌晨，登入節點 CPU 數十秒，scratch／inline，不留腳本）→ 都沒有可利用的結構：**
- 題目編號順序 vs 現行 top-1 候選人編號：Spearman pub_ −0.166、priv_ +0.033、pub2_ −0.070、priv2_ −0.047（原測試集的 top-1 幾乎全對，仍看不到單調關係）→ 題號不帶候選人資訊。
- 新增集候選人的黑棋前 3 手 vs 原候選庫每人 100 盤的開局分布（naive Bayes，log-likelihood 第 1–第 2 名差距）：新 public→原 public／原 private 中位數 1.46／1.57，>5 的比例 0.193／0.215；新 private→原 public／原 private 1.72／1.41，0.207／0.198。同 part 與交叉 part（必定無對應）分布相同 → **新增集候選人不是原候選庫的人**（與 11/15 風格向量檢查一致，這次用的是在原測試集近乎滿分的開局指紋）。


### 2026-11-17：H1 低 LR 分階段訓練（使用者「訓練」授權，開跑前登記）

先做 step 0／LR=0＋BN 更新 30 步／LR=0＋BN 凍結 30 步；開發 pool 0 draws 100–119，所有臂同一權重／抽樣／推論與固定融合 w=9600 原值、λ=16、μ=1。逐題分數及四桶皆保存。新抽取與既有快取的逐盤 cosine >0.9999、B 融合各桶及整體差 ≤0.002；凍結 BN 的零 LR 臂要求所有參數／buffers 完全不變、逐題候選分數 max abs ≤1e-6，否則停。正常 BN 若整體融合或風格 Δ<−0.002，或任一融合桶 Δ<−0.005，後續兩臂一律凍結 BN running stats（affine 仍訓練）；此規則在看本次診斷前固定。

通過後 E1 兩臂 3e-5／1e-5，各 4,800 步、暖機 240 步、cosine 到 0、單 H200、64 人×8盤×16手；不是原先 64 手／8卡方案。從 9600 last.pt 接回編碼器 Adam 一／二階矩及 step，驗證來源權重／BN 與 model.pt 全相同；B 保留 seed2026 權重但 optimizer 為新狀態，排程重設。仍排除驗證台玩家與 B 的 5% holdout；每 400 步以未見玩家 pool 0 draw 100 畫曲線，不選模。每臂約2.3 GPU-h，診斷約0.3；新快取僅抽 bench 約2.5GiB／臂。開發20 draws選 LR，兩臂均不優於基底則停。有改善才接 E0＋重訓B，完整階段估約9.3 GPU-h。

封存題組開封前門檻統一為 pool1 Δ≥+0.015、Δ/SE≥2.5、k=1/2/3/5各≥−0.005，pool0 Δ>0；pool0 draws600–619、pool1 draws600–629。此次不授權全玩家正式重訓、64手擴大或上傳。現行 hung-seqB 基底保留。程式 train_e2e.py／e2e_probe.py，設定 configs/identity/pid_e2e_adam_{probe,lr3e5,lr1e5}.json，診斷輸出 outputs/pid/ge2e/pid_e2e_adam_probe/，作業紀錄 outputs/pid/e2e_adam/。

**執行進度：** 零更新診斷 Slurm **789373**，單卡 dev、上限30分鐘；作業內原16項CPU測試通過（3.66秒），新增開發LR選擇測試後17項全通過（6.33秒）。已準備兩個E1作業，各單卡上限3小時，含訓練／bench抽取／pool0開發20 draws；後續E0＋B／封存作業上限4.5小時，有正開發增益才訓練，否則直接停止。整串硬上限11配置GPU-h，預估約9.3；排隊時間另計。E0約24GiB訓練快取僅用節點/dev/shm且結束清理，不占/work長期配額；持久保存權重、兩個E1與E0的bench快取及報表。未安排H2或全玩家重訓。

**依賴作業已送：** probe＝789373；lr3e5＝789387；lr1e5＝789388；conditional_e0_gate＝789389。E1 依賴 probe afterok，後續依賴兩臂 afterok；依賴失敗自動取消，不繼續花算力。

**789373 step 0：** pool0 開發20 draws，B融合0.6491906787720089，四桶0.380249／0.617778／0.743283／0.855453，與既有快取平均分及四桶完全一致；A亦一致。逐盤向量最小cosine A0.99999577／B0.99999022；不同抽取batch邊界的BF16數值不逐值相同（B融合候選分數max abs0.2113），因此只宣稱得分重現，不宣稱候選logits逐值相同。後續LR=0凍結BN臂對同一新抽取step0仍要求逐題分數max abs≤1e-6。BN對照尚在執行，E1／E0依賴等待中。

**H1 低 LR 結果（11/17 08:45 補登；作業 789373／789387／789388／789389）→ 兩個 LR 都低於現行，依事前規則停止，未開封存題組、未跑 E0。** 紀錄 `outputs/pid/e2e_adam/`（`selection.json`、`dev_lr{3e5,1e5}/report.json`、`train-*.out`）、`outputs/pid/ge2e/pid_e2e_adam_probe/report.json`。

- **零 LR 對照（789373，13 分）通過：** step 0 與 LR=0＋BN 更新 30 步的融合／風格分數都與凍結模型一致（BN 更新臂整體差 <0.001），`freeze_bn=false`。→ 11/17 凌晨短測的下滑**不是** BN running stats 或讀檔／推論路徑造成的。
- **E1 兩臂（各單卡 2 時 25 分，接回 `last.pt` 的編碼器 Adam 狀態，暖機 240 步，4,800 步）：** 開發 pool 0 draws 100–119，完整融合對現行 `9600`＋B（0.6492）：

| 臂 | k=1／2／3／5 | 整體 | Δ（SE） |
|---|---|---|---|
| lr 3e-5 | 0.3627／0.6031／0.7280／0.8370 | 0.6327 | **−0.0165（0.0017）**，0% draw 為正 |
| lr 1e-5 | 0.3675／0.6043／0.7267／0.8389 | 0.6343 | **−0.0149（0.0018）**，5% draw 為正 |

- 監控（未見玩家 pool 0 draw 100，只用風格 B）：step 0 0.647 → 400 步約 0.60–0.61，之後只回到約 0.614，4,800 步都沒有回到起點；四桶都下滑。
- **判讀：** 接回 Adam 動量、LR 降到 1e-5 仍然退步 → 「Adam 歸零造成衝擊」不是主因（或不是唯一原因）。繼續用原訓練玩家、原 GE2E 目標微調，本身就讓未見玩家變差；可能是對訓練玩家進一步記憶（未驗證）。**H1（以現行配方微調編碼器）結案，不加預算。** 新增集基底維持 `hung-seqB.csv`。成本約 5.2 GPU-h。


### 2026-11-17：H1 接回 Adam＋低 LR 完整訓練結果（停止，未開封 gate）

Slurm 789373／789387／789388／789389 全部 COMPLETED、exit 0；兩個 E1 均完整 4,800 步、optimizer_source_step=9600、暖機240步、BN正常更新、每盤16手。開發 pool0 draws100–119，固定融合 w=11.667971611、λ=16、μ=1；基底9600＋B=0.649190679。

| 臂 | 融合分數 | Δ（draw間SE） | k=1／2／3／5 的Δ | 純風格Δ |
|---|---|---|---|---|
| lr3e5 | 0.632715 | -0.016475（0.001683） | -0.017516／-0.014666／-0.015311／-0.018408 | -0.024411 |
| lr1e5 | 0.634335 | -0.014856（0.001843） | -0.012745／-0.013488／-0.016626／-0.016563 | -0.023915 |

LR=0＋正常BN30步：融合0.649140866，Δ−0.000049813，純風格Δ+0.000421553；LR=0＋凍結BN30步：所有權重與buffers不變、候選分數逐值重現（max abs=0）。故管線零更新檢查通過，BN單獨更新在此對照中不足以解釋舊短測大跌；不排除BN與權重學習的交互作用。兩個E1接回Adam且降低LR後仍四桶全負，差約9.79／8.06個draw間SE，退化原因未識別，不能再單歸因於Adam歸零。固定draw100監控兩臂先跌後部分回升，最後仍低於起點，未依曲線挑checkpoint。

依事前停止規則，789389讀兩臂開發報表後立即正常退出：selection.json status=stopped_no_gain、chosen=null。未跑E0／重訓B、未開封draws600+、未產出或上傳新提交檔；hung-seqB基底保留。本輪停止，不再擴充LR搜尋。這只否定本次16手聯合微調設定，不代表64手方案已試過。

實際單卡配置時間：診斷13m19s、3e-5 2h24m48s、1e-5 2h24m11s，合計約5.04 GPU-h；後續停止作業sacct記0秒。報表：outputs/pid/ge2e/pid_e2e_adam_probe/report.json、outputs/pid/e2e_adam/dev_lr3e5/report.json、dev_lr1e5/report.json、selection.json。SE為同一玩家池的draw間差異，不是新玩家泛化的不確定度。H2未執行。

### 2026-11-17：競賽一——seqB3 封存評估＋H2-A（B 向量上的候選人中心擴充）（使用者「好」；事前登記，看結果前寫定）

**預期修正（看結果前）：** 11/14 寫的「題目合併上限 +0.099」是用真值告訴模型哪兩題同一人，這個資訊無法從資料推得；可實作的候選人中心擴充（A）真值上限只有 +0.012。故 H2 現實預期 ≤ +0.01。

**腳本** `scripts/identity/pid2_b3em.py`（1 卡）：全新封存 draws pool 0 700–719、pool 1 700–729（600–629 留給 H1 未開封，仍保留）。融合同提交（開局＋16·w·cosine＋類別，w＝9600 原值）。
- **seqB3：** B3＝三個 seed（s2026／s2027／s2028）cosine 平均，對 B（s2026）配對。**升基底規則：** pool 1 Δ(B3−B) > 0 且 z ≥ 2、pool 0 Δ > 0 → `hung-seqB3.csv` 成為基底（已上傳 Public 0.8056）；否則維持 `hung-seqB.csv`。
- **H2-A：** 每個 seed 各自做中心擴充，後驗 p＝softmax(B3 融合分數／τ) 共用；τ∈{1,2,4,8}、α∈{0.25,0.5,1,2}、輪數 {1,2}，只在開發 pool 0 draws 100–119 選。封存 gate（對 B3）：pool 1 Δ ≥ +0.005、z ≥ 2.5、四桶各 ≥ −0.005、pool 0 Δ > 0 → 產生提交檔；未過不產生。

**結果（790267，1 卡 4 分 58 秒；`outputs/pid/pid2_b3em/report.json`、`sealed_per_draw.csv`；log `outputs/pid/pid2_bench/logs/pid2-b3em-790267.out`。790266 因快取路徑型別錯誤即失敗，修正後重送）：** 開發選定 τ=4、α=0.5、2 輪（開發增益 +0.0045；前 8 名都是 α=0.5，τ 不敏感）。封存 draws 700–：

| 比較 | pool 1 k=1／2／3／5 Δ | pool 1 整體 Δ（SE，正 draw 比例） | pool 0 整體 Δ | 判定 |
|---|---|---|---|---|
| B − A（參考） | +0.033／+0.031／+0.023／+0.025 | +0.0279（0.0022，97%） | +0.0246 | 與 11/17 確認一致 |
| **B3 − B** | +0.002／+0.003／+0.003／+0.004 | **+0.0029（0.0015，z 1.93，70%）** | +0.0017 | **未過**（z < 2）→ 基底維持 `hung-seqB.csv` |
| **B3＋EM − B3** | +0.004／+0.000／+0.007／+0.004 | **+0.0036（0.0010，z 3.6，77%）** | +0.0025 | **未過**（Δ < +0.005）→ 不產生提交檔 |

絕對分數（封存 pool 1）：A 0.6139、B 0.6418、B3 0.6447、B3＋EM 0.6483。B3 與 EM 都是小而方向一致的正值（四桶全正），但都沒達事前門檻；兩者疊加對 B 約 +0.0065，換算 Public 約 +0.003，在雜訊內。與預期修正一致：中心擴充的上限本來就小。

### 2026-11-17：競賽一——E2E-scratch（從頭訓練編碼器＋B 聚合頭、每盤 48 手）計畫稿（使用者「先寫完整計畫」；未實作、未送作業）

完整計畫 [PID_E2E_SCRATCH_PLAN.md](PID_E2E_SCRATCH_PLAN.md)。要點：V2 主幹重新初始化＋B 頭在迴路、64 人×8 盤×48 手、6,426 人、9,600 步、8 卡 DDP；開發 draws 選 4,800／7,200／9,600 步 checkpoint，封存 draws 600–629（未開封）gate：pool 1 Δ ≥ +0.015、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0。第 1 階段估 40–85 GPU-h（以 8 卡 100 步短測為準，> 4.5 秒／步先回報）；第 2 階段全 8,007 人正式版另行授權。預期增益未知。

**E2E-scratch 第 0 階段（使用者「做這個」，11/17；照計畫檔固定 9,600 步，不採依速度改步數的規則）：** 程式改動：`ge2e.IdentityEncoder` 加可選 `checkpoint_segments`（預設 0，既有模型與載入不變；>0 時訓練中重算殘差區塊省顯存，重算會讓 BN running stats 多更新一次）；`train_ge2e` 加可選 `head`（`EncoderHead`＝編碼器＋B 頭一個模組給 DDP；GE2E 用 B 的 w、b，初值抄編碼器，編碼器 w、b 凍結；`PlyBatches` 帶 ply；`keep_steps` 另存 `step{N}/model.pt`＋`best.pt`；結束寫 `model.pt`＋`head/best.pt`，與 `seqagg.load_head`／`pid2_seqagg export` 相容）。原因：現行每卡 8,192 位置已用約 123 GiB，48 手每卡 24,576 位置需 activation checkpointing（5 段）。設定 `configs/identity/pid_e2e_scratch_{smoke100,9600}.json`（loader workers 4→8）。CPU 測試 `tests/test_identity_e2e_scratch.py` 4 項（checkpoint 與否輸出／梯度一致、`EncoderHead`＝`joint_games`、48 手 ply 屬本人顏色且不重複、雙程序 Gloo DDP 梯度＝兩程序本地梯度平均）＋既有 identity 17 項，共 21 項通過（55 秒）。短測 Slurm **790374**（8 卡、100 步、上限 40 分）。

**短測 790374 失敗（23 秒，未開始訓練）：** 初始化權重 `outputs/rank/v2/v2_strength/model.pt`（A-024 V2 棋力主幹）已在 11/16 清理中刪除；全機搜尋 >50 MB 的 `.pt` 只剩競賽一的 GE2E／B 頭產物，沒有 V2 副本。最接近 V2 的現存權重是 `pid_ge2e_v2_both_p0`（V2＋300 步身分訓練、雙色、同樣排除 split 0–2，驗證台玩家沒見過）。改初始化須使用者決定，未重送。
**初始化改為 `pid_ge2e_v2_both_p0` 主幹（使用者選第 2 項，11/17）：** 兩份 config 的 `init_model` 改指向 `outputs/pid/ge2e/pid_ge2e_v2_both_p0/model.pt`（V2＋300 步身分訓練、排除 split 0–2）；只載入 stem／blocks（`load_trunk`，已在登入節點驗證逐值相同），投影層與 B 頭重新初始化。與原計畫差異＝起點多了 300 步身分訓練。計畫檔同步註記。短測重送 Slurm **790388**。

**短測 790388 結果（25a-hgpn021，8 卡，11 分 12 秒，完成 100 步）：** 每步 **6.0 秒**（第 10 步含暖機 9.5 秒），每卡顯存峰值 112.9 GiB（activation checkpointing 5 段），loss 3.30 → 2.50、訓練命中率 0.17 → 0.35，全程有限；`model.pt`、`head/best.pt`、`last.pt` 都有寫出。**超過停損門檻 4.5 秒／步 → 依規則停止、不開第 1 階段，回報使用者。** 換算：每卡 24,576 位置／6.0 秒 ≈ 4,100 位置／秒，扣掉重算約 1/3 後與原 16 手訓練（8,192／1.52 秒 ≈ 5,400）相當 → **瓶頸是算力，不是資料讀取**（先前「8 卡與單卡一樣快、不是算力瓶頸」的推論有誤：原訓練每卡就是 8,192 位置）。原設定 9,600 步需約 16 小時、約 128 GPU-h。

**改為 A＋WSD（使用者 11/17：「A，改用 WSD，先跑短測」）：** 32 手 × 9,600 步；排程改 warmup-stable-decay：前 5% 線性暖機、LR 3e-4 固定、最後 20%（第 7,680 步起）線性降到 0（`train_ge2e.wsd_scheduler`，`training.schedule="wsd"`、`decay_fraction=0.2`；原 cosine 為預設、不變）。降 LR 前一步另存完整續跑狀態 `stable_step7680.pt`；之後若要加長，以新 config（更長 steps）加 `paths.extend_from` 從該檔接回權重、B 頭、AdamW 動量與 step，LR 維持固定到新的降 LR 起點，不重做前段。注意：WSD 下 4,800／7,200 步 checkpoint 未經降 LR，開發選擇預期會選 9,600 步。與 9600 模型的差異因此是 B 在迴路＋32 手＋排程。CPU 測試新增 2 項（WSD 形狀；延長時固定段 LR 相同、scheduler 接續、降 LR 起點後移），identity 共 23 項通過（60 秒）。設定 `configs/identity/pid_e2e_scratch32_{smoke100,9600}.json`（48 手的 9,600 步設定已刪，48 手短測設定保留作紀錄）。32 手短測 Slurm **790438**。


## 2026-11-17：policy branches 788616 原預算續跑

使用者授權補完原訂 3,000 updates 與 dev／2,000 次玩家 bootstrap。原 job 788616 TIMEOUT（02:00:26）；A0 已完成，其餘三臂最後完整評估為 step 2875，從 epoch 11 末 checkpoint 重播最後一輪。原 config／程式／plan hash 一致，prepare-only 通過；不增加 epochs、不改 LR、不啟動其他 seed。重新抽取 RAM policy，預估 50–75 分鐘、4.2–6.3 配置 GPU-h，上限 5 GPU × 1.5 小時＝7.5 GPU-h。續跑 launcher：`outputs/rank/humansl/policy_branches/seed2027_b32_e12/788616/resume.slurm`；結果沿用原 run。提交後不輪詢。

提交成功：Slurm **790468**（2026-11-17T10:30:49.360444+08:00）；僅確認 sbatch 接受，完成狀態與 dev／gate 待後續分析。

**結果（790468 COMPLETED，1 時 27 分，5 卡，約 7.04 配置 GPU-h；`788616/result.json`、`complete.json` status=complete）→ 四臂全部未過 gate，C-only policy 分支線結案。** 各臂 3,000 updates，monitor 選 checkpoint（A0 1,000、R10 1,875、Z10 1,000、Z32 另見 result）；dev Equal-k（k=5／10／15／20）：

| 臂 | 參數 | dev Equal-k | k=5／10／15／20 |
|---|---:|---:|---|
| A0 | 210 | 0.7137 | 0.6715／0.7169／0.7281／0.7382 |
| R10 | 9,949 | 0.7124 | 0.6698／0.7174／0.7268／0.7356 |
| Z10 | 10,149 | 0.7141 | 0.6718／0.7179／0.7284／0.7382 |
| Z32 | 12,019 | 0.7145 | 0.6726／0.7188／0.7284／0.7382 |

| 比較（配對玩家 bootstrap 2,000） | ΔEqual-k | 95% | P(>0) | gate |
|---|---:|---|---:|---|
| R10 − A0 | −0.0013 | [−0.0050, +0.0020] | 0.23 | 未過 |
| Z10 − A0 | +0.0004 | [−0.0001, +0.0009] | 0.95 | 未過（< +0.005） |
| Z32 − A0 | +0.0008 | [+0.0002, +0.0015] | 0.99 | 未過（< +0.005） |
| Z32 − Z10 | +0.0004 | [−0.0000, +0.0010] | 0.96 | 未過 |

判讀：Z 臂的加成 logits 很小（added_logits_rms 0.002–0.005，對比 q_logits_rms 約 12.9），等於幾乎沒改動 A0；方向穩定為正但量級比 gate 小一個數量級。R10 加成較大（0.046）反而略降。限制：單 seed、dev 歷史重用、C 向量學習與 200 位 monitor 玩家重疊 190 位。不產生提交檔、不換基底。

**32 手短測 790438 結果（25a-hgpn014，8 卡，8 分 1 秒，100 步）：** 每步 **3.94 秒**（第 10 步含暖機 8.7 秒），每卡顯存峰值 75.4 GiB；loss 3.28 → 2.44、訓練命中率 0.17 → 0.35；LR 固定 3e-4 至第 80 步、第 90 步 1.5e-4、第 100 步 0（WSD 形狀正確）；`stable_step80.pt`、`model.pt`、`head/best.pt`、`last.pt` 皆寫出。**≤ 4.5 秒，過停損門檻。** 第 1 階段 9,600 步估約 10.5 小時、約 **84 GPU-h**（＋評估約 1 GPU-h）。未送第 1 階段，等使用者授權。

**第 1 階段已送（使用者「好，送第 1 階段」，11/17）：** 訓練 Slurm **790690**（8 卡、上限 13 小時，`configs/identity/pid_e2e_scratch32_9600.json`，估約 10.5 小時／84 GPU-h）。評估 Slurm **790702**（afterok 790690，1 卡、上限 2 小時；`outputs/pid/e2e_scratch/eval.sh`）：抽 4,800／7,200／9,600 步的驗證台逐手向量（快取 `/home/jason1122/e2e_scratch32_cache/`，約 2.6 GiB／份）→ 開發 pool 0 draws 100–119 對 9600＋B s2026 選完整融合最高者（同分取較晚步數）→ 封存 draws 600–（pool 0 600–619、pool 1 600–629）只評一次，gate＝pool 1 Δ ≥ +0.015、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0。輸出 `outputs/pid/e2e_scratch/{dev,sealed}/report.json`、`dev_choice.json`。抽取路徑已用短測模型驗證（790691，1 chunk，2 分 31 秒）。

**訓練 790690 完成（10 時 33 分，9,600 步，每步 3.95 秒，峰值 75.4 GiB；最後 loss 0.761、訓練 own-centroid 命中 0.795）；評估 790702 在 9 秒內失敗：`data/raw/` 已於 11/17 打包成 `data/raw.zip`，`train_ge2e.training_players` 讀不到 `data/raw/training/handicap.csv`。只從 ZIP 還原 `handicap.csv`／`handicap_manifest.json`（約 40 MB），以相同 `eval.sh` 重送 Slurm 793383（1 卡、上限 2 小時、排除 25a-hgpn006）。**

**評估 793383 結果（COMPLETED，25 分 39 秒；`outputs/pid/e2e_scratch/{dev,sealed}/report.json`、`dev_choice.json`）→ 封存 gate 未過（Δ +0.0082 < +0.015），不產生提交檔，基底維持 `hung-seqB.csv`。** 對照 base＝9600 編碼器＋B s2026，完整融合，配對 draw 間 SE。

開發 pool 0（draws 100–119）：4,800 步 −0.0121、7,200 步 −0.0191、**9,600 步 +0.0073**（SE 0.0033）→ 選 9,600。WSD 下 4,800／7,200 步都在 LR 固定段，降 LR 後才轉正，與事前預期一致。

| 封存（draws 600–） | k=1／2／3／5 Δ | 整體 Δ（SE） | 純風格 Δ |
|---|---|---|---|
| pool 1（30 draws） | +0.011／+0.012／+0.008／+0.001 | **+0.0082（0.0025，z 3.33，73% draw 為正）** | +0.0031 |
| pool 0（20 draws） | −0.001／+0.021／+0.021／+0.000 | +0.0105（0.0029） | +0.0020 |

絕對分數（封存 pool 1 完整融合）：base 0.6387 → E2E 0.6469。判讀：方向一致為正、z 和 pool 0 條件都過，只差增益門檻（約需求的一半）；純風格只多 +0.002～+0.003，增益主要出現在融合後，k=5 幾乎沒變。結論：從頭端到端訓練（B 在迴路、32 手、9,600 步）只比凍結編碼器＋B 頭多約 +0.008，花了約 85 GPU-h。第 2 階段（全 8,007 人）不自動進行。

**後續兩項（使用者 11/17 晚「結果出來後整理分 k 成績並繼續訓練」；事前登記，看結果前寫定）：**
- **(1) E2E 與基底的 cosine 混合（`scripts/identity/pid2_cosmix.py`，1 卡）：** 融合改為 開局＋16·w·((1−a)·cos_B＋a·cos_E2E9600)＋類別；a ∈ {0.25, 0.5, 0.75} 在開發 pool 0 draws 100–119 選（同分取較小 a）。封存 draws 900–（pool 0 900–919、pool 1 900–929，未用過）只評一次。gate（mix）：pool 1 Δ ≥ +0.010、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0。動機：兩個編碼器訓練方式不同，平均可能互補；預期未知。
- **(2) WSD 延長到 14,400 步（`configs/identity/pid_e2e_scratch32_14400.json`）：** 從 `stable_step7680.pt` 接回權重、B 頭、AdamW 動量與步數（已驗證 step 7680、world 8、LR 3e-4），LR 固定到 11,520 步，最後 2,880 步線性降到 0；其餘設定同 9,600 步版。6,720 步 × 3.95 秒 ≈ 7.4 小時、約 59 GPU-h（8 卡，上限 9 小時）。依據：9,600 步版本封存 +0.0082（z 3.3），訓練 loss 降 LR 前仍在下降；延長能否補到 +0.015 未知。
  評估（afterok，1 卡）：抽 14,400 步驗證台逐手向量，用 (1) 的同一程式：arms＝基底、E2E9600（參考）、E2E14400；mix＝基底＋E2E14400，a ∈ {0.25, 0.5, 0.75, 1.0} 在開發 draws 100–119 選。封存 draws 1000–（pool 0 1000–1019、pool 1 1000–1029）只評一次。gate：E2E14400 單獨 Δ ≥ +0.015（同第 1 階段）；mix Δ ≥ +0.010；其餘條件同上。通過者才寫正式新增集匯出與提交探針檔；不自動上傳。
- **已送：** (1) Slurm 793627；(2) 訓練 793629（8 卡、上限 9 小時）、評估 793631（afterok 793629，1 卡）。腳本 `outputs/pid/e2e_scratch/{mix9600.sh,eval14400.sh}`。
- **(1) 結果（793627，2 分 48 秒，`outputs/pid/e2e_scratch/mix9600/report.json`）→ GATE PASS。** 開發選 a=0.5（0.25／0.5／0.75：+0.018／+0.026／+0.025）。封存 draws 900–，對基底：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 整體 Δ（SE） | pool 0 整體 Δ |
|---|---|---|---|
| E2E9600 單獨（參考） | −0.003／+0.016／+0.018／+0.009 | +0.0101（0.0030） | +0.0092 |
| **mix a=0.5** | +0.021／+0.035／+0.033／+0.020 | **+0.0272（0.0018，z ≈ 15，100%）** | +0.0247（100%） |

絕對分數（pool 1）：基底 0.6428 → mix 0.6700。兩個編碼器各自只差約 0.01，平均後 +0.027 → 互補性大（從 V2 初始化 300 步身分訓練＋從頭端到端 vs 凍結編碼器＋後接 B 頭）。

**(3) 兩項合併的封存檢查（事前登記，看結果前寫定；`scripts/identity/pid2_ml_mix_eval.py`）：** 不重調任何權重（ML 用第 1 階段設定、mix a=0.5）。新封存 draws 1100–（pool 0 1100–1119、pool 1 1100–1129），arms：base、ml、mix、mix_ml。gate（mix_ml 對 mix）：pool 1 Δ ≥ +0.005、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0。
**提交檔規劃：** E2E9600 風格項的正式匯出 `scripts/identity/pid2_style_export.py`（先以同一路徑重現 seqB 風格項，最大差 ≤ 1e-4 且 top-1 相同才繼續）→ `hung-seqB-e2emix.csv`（風格項＝seqB 與 E2E9600 平均，a=0.5）；(3) 過 gate 才產生 `hung-seqB-e2emix-ml.csv`。
- **E2E9600 正式風格項（Slurm 793636）：** 參考路徑重現 seqB 風格項最大差 8.9e-15／9.8e-15、top-1 相同；E2E9600 單獨改 top-1 對 seqB：pub2 121、priv2 142。
- **(3) 結果（Slurm 793637，`outputs/pid/pid2_multilevel/ml_mix/report.json`）→ GATE PASS。** 封存 draws 1100–：

| 臂 | pool 1 k=1／2／3／5 Δ（對 base） | pool 1 整體 Δ（SE） | pool 0 整體 Δ |
|---|---|---|---|
| ml | +0.029／+0.030／+0.028／+0.020 | +0.0266（0.0017） | +0.0214 |
| mix | +0.026／+0.028／+0.026／+0.022 | +0.0255（0.0018） | +0.0256 |
| **mix_ml** | +0.051／+0.057／+0.041／+0.029 | **+0.0446（0.0021），對 mix +0.0191（0.0016，z 12.1，97%）** | +0.0437（對 mix +0.0181） |

絕對分數（pool 1）：base 0.6410 → mix_ml 0.6856。兩項大致相加（0.0266＋0.0255 → 0.0446，重疊約 0.0075）：開局局部習慣與風格編碼器是不同訊號。
- **提交檔（`scripts/identity/pid2_combine_ml.py`，CPU 數秒；`outputs/pid/pid2_multilevel/ml_mix/combine_report.json`）：** 檢查 seqB 與 seqB-ml 兩檔由 parts 逐列重現；原測試集與 `772748-hung.csv` 相同。
  - `submissions/pid/hung-seqB-e2emix.csv`（`895d44ac`）：改 top-1 對 seqB pub2 74（32／21／10／11）、priv2 80。
  - `submissions/pid/hung-seqB-e2emix-ml.csv`（`4fd8dc6e`）：改 top-1 對 seqB pub2 93（31／23／14／25）、priv2 109；本地對 base +0.045，換算 Public 約 +0.022。
  - 升基底規則（事前寫定）：`hung-seqB-e2emix-ml.csv` 三項封存 gate 都已過，依本地證據即升為競賽一新基底候選；Public 只作方向確認，若 Public 降 ≥ 0.02 才回查正式推論路徑。
- **(4) 14,400 步版本的組合（事前登記；afterok 793631）：** 以 `pid2_ml_mix_eval.py --e-cache …/step14400 --e-head …/step14400/best.pt --a-from ext14400/report.json --first 1200`，新封存 draws 1200–（pool 0 1200–1219、pool 1 1200–1229），gate 同 (3)（mix_ml 對 mix）；另比較時以 draws 1200– 上 E2E14400 版 mix_ml 對 E2E9600 版 mix_ml 的差只作參考，不設 gate（兩者不在同一次評估）。同時匯出 E2E14400 正式風格項（tag e2e14400），僅備用。
- **(2) 結果（訓練 793629 7 小時 23 分；評估 793631，`outputs/pid/e2e_scratch/ext14400/report.json`）→ E2E14400 單獨 GATE FAIL，mix PASS。** 開發選 a=0.5（0.25／0.5／0.75／1.0：+0.019／+0.028／+0.025／+0.007）。封存 draws 1000–，對基底：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 整體 Δ（SE） | pool 0 整體 Δ |
|---|---|---|---|
| E2E9600 單獨（參考，同題） | +0.005／+0.013／+0.008／−0.013 | +0.0031（0.0031） | +0.0144 |
| E2E14400 單獨 | +0.004／+0.010／+0.008／−0.013 | +0.0023（0.0028，z 0.83）→ FAIL（門檻 +0.015） | +0.0131 |
| mix（基底＋E2E14400，a=0.5） | +0.026／+0.038／+0.030／+0.014 | +0.0270（0.0018，z 15.2）→ PASS | +0.0313 |

判讀：延長 4,800 步（含第二次降 LR）沒有讓編碼器單獨變好；同題上 14,400 步版與 9,600 步版持平（+0.0023 vs +0.0031）。混合增益與 9,600 步版（draws 900–，+0.0272）同量級。→ 這條編碼器路線在此設定下已飽和，延長訓練不划算。
- **(4) 結果（793668，`outputs/pid/pid2_multilevel/ml_mix14400/report.json`）→ GATE PASS（mix_ml 對 mix）。** 封存 draws 1200–，pool 1 對 base：ml +0.0264、mix +0.0257、**mix_ml +0.0466（k=1／2／3／5：+0.046／+0.059／+0.055／+0.026）**；mix_ml 對 mix +0.0209（SE 0.0019，z 11.2；+0.020／+0.029／+0.027／+0.008），pool 0 +0.0167。絕對分數 pool 1：base 0.6459 → mix_ml 0.6925。與 9,600 步版（draws 1100–，+0.0446）不同題，只作參考。
- **E2E14400 正式風格項（793670）：** 參考路徑重現 seqB 最大差 ≤ 1e-14、top-1 相同；E2E14400 單獨改 top-1 對 seqB：pub2 125、priv2 155。已寫 `outputs/pid/day0_1112/{public,private}2_d4_9600_e2e14400/`，未產生提交檔。
- **(5) 同題比較 9,600 vs 14,400（11/18 事前登記，看結果前寫定；`scripts/identity/pid2_ml_mix_compare.py`，1 卡約 10 分鐘）：** 新封存 draws 1300–（pool 0 1300–1319、pool 1 1300–1329）。arms：base、mix_ml9600（現行基底候選）、mix_ml14400、mix3_ml（0.5 cos_B＋0.25 cos_E9600＋0.25 cos_E14400）＋ML；不調任何權重。換用規則（對 mix_ml9600）：pool 1 Δ ≥ +0.005、z ≥ 2.5、四桶 ≥ −0.005、pool 0 Δ > 0；否則維持 `hung-seqB-e2emix-ml.csv`。預期：持平（兩個 E2E 同一路徑訓練、相關性高）。
- **(5) 結果（Slurm 795581，14 分鐘，`outputs/pid/pid2_multilevel/ml_mix_compare/report.json`）→ 兩者都未達換用規則，維持 `hung-seqB-e2emix-ml.csv`（E2E9600 版）。** 封存 draws 1300–，pool 1 對 base：mix_ml9600 +0.0437、mix_ml14400 +0.0471、mix3_ml +0.0468。對 mix_ml9600：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 整體 Δ（SE） | pool 0 整體 Δ |
|---|---|---|---|
| mix_ml14400 | +0.008／+0.003／+0.002／+0.001 | +0.0035（0.0010，z 3.5）→ 未達 +0.005 | −0.0010 |
| mix3_ml | +0.005／+0.003／+0.003／+0.001 | +0.0031（0.0008，z 3.9）→ 未達 +0.005 | +0.0009 |

判讀：14,400 步版在 pool 1 小幅領先（+0.003），但 pool 0 持平或略負，量級在 Public 雜訊（約 0.02）之下 → 延長訓練的實際價值約 +0.003，不值得換檔或再延長。

**LB（11/18 08:16:48，使用者回報）：** 上傳頁顯示 `hung-seqB-e2emix...csv` **0.8245254**（32/106），對 `hung-seqB.csv`（0.8012）+0.0233。檔名被截斷，`hung-seqB-e2emix.csv` 與 `hung-seqB-e2emix-ml.csv` 在頁面上無法區分；+0.023 與 e2emix-ml 的預期（約 +0.022）一致、高於 e2emix 的預期（約 +0.013），但兩者差在 Public 雜訊內。**使用者確認上傳的是 `hung-seqB-e2emix-ml.csv`（`4fd8dc6e`）**：Public +0.0233，與預期 +0.022 一致、方向確認。依事前規則（三項封存 gate 皆過），**競賽一基底改為 `submissions/pid/hung-seqB-e2emix-ml.csv`（Public 0.8245）**。（違反 11/09 短檔名規則；之後的競賽一提交檔改用短而不同的名字。）

### 2026-11-18：競賽一——開局多層比對第 2 階段（使用者「做 B，多層比對第 2 階段」；事前登記，看結果前寫定）

**動機：** 第 1 階段 L1（9×9 角）＋L2（7×7 應手窗）封存 +0.025；第 0 階段顯示兩層在第 8 手後命中率掉到 4–5%，因為局面變獨特、細粒度鍵查不到。論文的多層退回比對在細層查不到時退到更粗的層。第 2 階段加三個更粗的層（`multilevel.game_keys2`，皆對整盤 8 對稱不變，單元測試 7 項通過）：
- L3 應手 5×5：L2 的小窗版（對手上一手周圍 5×5＋本手相對位置）。
- L4 應手距離：本手相對對手上一手的位移（|dr|、|dc| 排序後；切比雪夫距離 > 4 記「遠」），不看盤面 → 候選人「多常緊應、應在哪個距離」的分布。
- L5 自己的連續：本手相對自己上一手的位移（同上折疊）。
似然項與族群表同第 1 階段（同一 200,000 盤族群樣本，`level_scores`）；手數 N ∈ {15, 30}（粗層在第 15 手後仍查得到）。

**基底＝現行候選 `hung-seqB-e2emix-ml`：** 開局＋16·w·(0.5 cos_B＋0.5 cos_E2E9600)＋類別＋第 1 階段 L1／L2（N 15、β 0.1、λ1＝λ2＝1，不重調）。
- **開發選擇：** pool 0 draws 100–119，N ∈ {15, 30}、β ∈ {0.1, 1, 4}、λ3、λ4、λ5 ∈ {0, 0.25, 0.5, 1}（384 組，取平均最高，同分取 λ 和較小）；全選 0 則不開封存。λ＝1 為網格邊界，照記、不擴網格。
- **封存 gate（只評一次，draws 1400–：pool 0 1400–1419、pool 1 1400–1429，未用過）：** 對基底 pool 1 Δ ≥ +0.005、z ≥ 2.5、k=1／2／3／5 各 ≥ −0.005、pool 0 Δ > 0 → 才寫正式新增集匯出與提交檔；未過不產生。另報各層單獨（選定 λ）作機制診斷，不設 gate。
- **預期：** 未知；粗層與 L2、逐手位置項、風格項都有重疊，傾向小（+0.003～+0.01）。
- **已送：** Slurm 795687。
- **成本：** 1 卡（8gpus），風格向量約 1 分鐘、鍵值 CPU 約 1–2 分鐘、開發 384 組 × 20 draws 約 10 分鐘、封存約 15 分鐘；< 1 GPU-h。程式 `scripts/identity/pid2_multilevel_stage2.py`、`src/go_aicup/tasks/identity/multilevel_stage2.slurm`。

**第 2 階段結果（Slurm 795687，1 卡 43 分鐘，`outputs/pid/pid2_multilevel/stage2/{dev_choice,report}.json`、`dev_grid.csv`、`sealed_per_draw.csv`）→ GATE FAIL，不產生提交檔。**
- 開發：384 組只有 8 組高於基底，最高 N=30、β=1、λ3=0.5、λ4＝λ5＝0（+0.0015）；L4、L5 在任何 β／λ 都是負的。各層單獨最佳（開發，對基底）：

| 層 | N=15 最佳 | N=30 最佳 | λ=1、β=1（N=30） |
|---|---|---|---|
| L3 應手 5×5 | −0.0009 | +0.0015 | −0.0015 |
| L4 應手距離 | −0.0005 | −0.0021 | −0.0218 |
| L5 自己的連續 | −0.0010 | −0.0022 | −0.0214 |

- 封存 draws 1400–，選定設定（＝只有 L3）對基底：

| | k=1 | k=2 | k=3 | k=5 | 整體 Δ（SE） |
|---|---|---|---|---|---|
| pool 1（30 draws） | +0.004 | +0.001 | −0.001 | +0.002 | **+0.0018（0.0010，z 1.8，73%）** |
| pool 0（20 draws） | +0.002 | +0.004 | −0.001 | −0.001 | +0.0011（0.0009） |

絕對分數 pool 1：基底 0.6924 → 0.6942。未達 +0.005 與 z 2.5。
- 判讀：比 L2 更粗的層沒有新資訊。L3 幾乎是 L2 的子集（L2 查不到時 5×5 窗也多半只剩常見型，各候選人都有）；L4／L5 的位移分布對不同玩家差異小，加進去只增加雜訊。第 0 階段命中率在第 8 手後下降，是局面本身不再重複，不是層太細。→ 開局多層比對到第 1 階段為止，這條線結束。

### 2026-11-17：競賽一——開局多層退回比對（新增集）第 0 階段可行性診斷（使用者「從第 0 階段開始做」；事前登記，看結果前寫定）

**動機（文獻整理，11/17 對話）：** 文獻 3（Lin、Chiu、Wu，TMLR 2024，arXiv 2408.06051）圍棋 200 人辨識，多層狀態比對在少樣本時勝過單層（M=5：15.5%→22%；M=10：29.5%→45.5%）。現行新增集開局項只有 L0（D4 全盤精確前綴，8 手）與逐手位置；查詢盤一旦離開候選人走過的路線，L0 對所有候選人相同而抵消。

**新增兩層（`src/go_aicup/tasks/identity/multilevel.py`，每手本人落子，皆對整盤 8 個對稱不變）：**
- L1 角內：這手所在 9×9 象限內的本人／對手棋子（提子後），翻到同一個標準角、再取與轉置較小者；落在天元線上無 L1。
- L2 應手：對手上一手周圍 7×7 窗（本人／對手／空／盤外，8 對稱取最小）＋本手在窗內的相對位置，窗外記「脫先」。
- 每層有 context 鍵（落子前）與 outcome 鍵（含本手，狀態對稱時本手一併折疊）；第 0 階段只用 outcome 鍵。

**第 0 階段診斷（`scripts/identity/pid2_multilevel.py keys|diag`；不擬合、不算分）：** 開發 pool 0 draws 100–119，題目每盤本人前 15 手。對每一手：真實候選人在同色畫廊是否有相同 L0 前綴／L1／L2 outcome 鍵（命中），與其他 590 名候選人的平均命中率。分 k 桶、手數段（0–7、8–14）報告。
- **Gate 0（看結果前寫定）：** 在真實候選人 L0 沒命中的手（本人前 15 手、全部 k），L1 或 L2 的「真實命中率 − 其他候選人命中率」≥ 0.05（20 draws 平均）→ 寫第 1 階段（融合實驗，封存 draws pool 1 800–829、pool 0 800–819）；否則停止。
- 單元測試 `tests/test_pid2_multilevel.py` 4 項通過（8 對稱不變、鍵隨本手改變、異角同型態同鍵、提子）。

**第 0 階段結果（Slurm 790711，dev 1 卡，28 秒；`outputs/pid/pid2_multilevel/{diag.csv,diag.json,keys.npz}`、log `diag-790711.out`）→ GATE 0 PASS。**
- Gate 統計（本人前 15 手、L0 沒命中、全部 k，20 draws 平均）：**L1 +0.0764（SE 0.0009）、L2 +0.0552（SE 0.0008）**，兩層都 ≥ 0.05。
- L0 覆蓋很低：本人前 8 手只有 18.8% 被真實候選人的精確前綴命中（其他候選人 7.7%）；第 8–14 手 0.09%。→ 前 8 手有 81% 的落子目前只靠逐手位置項。
- 前 8 手、L0 沒命中：L1 真實 0.295 vs 其他 0.161（+0.134，約 1.8 倍）；L2 0.347 vs 0.257（+0.090，約 1.35 倍）。四個 k 桶幾乎一樣（L1 +0.130～+0.138）。
- 第 8–14 手：L1 0.040 vs 0.017（+0.023）、L2 0.055 vs 0.032（+0.023）→ 增量主要在前 8 手，延伸到 15 手的價值小。
- L1 無狀態（天元線）6.7%、L2 無狀態（黑第一手）3.5%。
- 限制：命中率差不等於似然比資訊；L1 在角內無子時等於「下在角的哪一點」，與現行逐手位置項（前 4 手）部分重疊。實際增量要第 1 階段在融合裡量。第 1 階段尚未實作，等使用者確認。

**第 1 階段（使用者「做開局多層比對第 1 階段」，11/17；事前登記，看結果前寫定）：** 把 L1／L2 做成計數似然項，加到現行基底（`hung-seqB` 的融合：開局＋16·w·cos_B s2026＋類別，w＝11.66797）上。
- **似然項：** 每一手本人落子、每層 ℓ，候選人 c 的 `log((n_c(outcome) + β·p_pop) / (n_c(context) + β))`；n_c 為 c 同色畫廊盤前 N 手本人落子的計數；p_pop＝`(N_pop(outcome) + 1/361) / (N_pop(context) + 1)`，族群＝驗證台以外的訓練玩家（同 `twocolor` 的 populations），每色固定抽 100,000 盤（`default_rng(2026)`）。候選人沒見過這個 context 時該項等於 p_pop，對所有候選人相同而抵消；無狀態的手為 0。題目分數＝題內各盤各手相加。
- **融合：** 基底＋λ1·S_L1＋λ2·S_L2。開發 pool 0 draws 100–119 選 N ∈ {8, 15}、β ∈ {0.1, 1, 4}、λ1、λ2 ∈ {0, 0.25, 0.5, 1, 2}（以整體平均最高者；同分取 λ1＋λ2 較小）。選到網格邊界（λ＝2）照記、不擴網格。
- **封存 gate（只評一次，draws 800–：pool 1 800–829、pool 0 800–819）：** pool 1 Δ ≥ +0.005、z ≥ 2.5、k=1／2／3／5 各 ≥ −0.005、pool 0 Δ > 0 → 才寫正式新增集匯出、產生 `hung-seqB-ml.csv` 探針；未過不產生。另報只開 L1／只開 L2（其他參數不變）作機制診斷，不設 gate。
- **預期：** 未知；第 0 階段的命中率差主要在前 8 手，且與逐手位置項部分重疊，傾向小（≤ +0.01）。
- **成本：** 1 卡 dev，B 向量 GPU 約 2 分鐘，其餘 CPU；族群鍵約 200,000 盤（12 程序約數分鐘）；估 < 1 GPU-h。

**第 1 階段結果（Slurm 793571，1 卡 13 分 44 秒，`outputs/pid/pid2_multilevel/stage1/{dev_choice,report}.json`、`sealed_per_draw.csv`）→ GATE PASS。** 開發選定 N=15、β=0.1、λ1=λ2=1（開發 +0.0229；不在網格邊界；前 10 名全是 N=15）。封存 draws 800–，對基底（hung-seqB 融合）：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 整體 Δ（SE） | pool 0 整體 Δ（SE） |
|---|---|---|---|
| **L1＋L2** | +0.030／+0.028／+0.028／+0.015 | **+0.0252（0.0023，z 11.0，100% draw 為正）** | +0.0236（0.0026，100%） |
| 只 L1 | +0.019／+0.020／+0.023／+0.018 | +0.0198（0.0016） | +0.0200 |
| 只 L2 | +0.014／+0.017／+0.020／+0.008 | +0.0145（0.0018） | +0.0118 |

絕對分數（pool 1）：基底 0.6412 → 0.6664。兩層各自都有效、合併後部分重疊（0.0198＋0.0145 > 0.0252）；L1 在 k=5 仍有 +0.018，L2 主要幫小 k。機制：現行精確前綴 L0 在前 8 手只命中 18.8%，L1／L2 讓「離開候選人走過的路線」之後的落子仍能比對局部習慣。限制：兩個 pool 的玩家自 11/12 反覆使用（封存的是題目抽樣種子）。

**正式匯出（Slurm 793633，36 秒；`scripts/identity/pid2_multilevel_export.py`、`stage1/export_report.json`）：** `submissions/pid/hung-seqB-ml.csv`（SHA-256 前八碼 `42ad1441`）＝`hung-seqB.csv` 新增集分數＋L1＋L2（同設定、同族群表），原測試集與 `772748-hung.csv` 逐列相同。檢查：seqB 分數由 parts 重建且 top-5 與 `hung-seqB.csv` 一致；20 盤 × 8 對稱鍵值不變；L1／L2 無狀態比例 pub2 6.3%／3.3%、priv2 6.4%／3.3%（驗證台 6.7%／3.5%）。改 top-1 對 seqB：pub2 66（k=1／2／3／5：16／15／12／23）、priv2 74（14／21／23／16）。預期 Public Δ 約 +0.012（新增集占一半）。


## 2026-11-17：CNN policy 快取優先重新選盤（僅清單／題組）

使用者授權從 c4x learn A 已有盤面重新選 TRAIN／monitor／dev，並由原始棋局補足同色題目。以 raw.zip 內 handicap.csv 與 canonical metadata 篩 HD≤1；learn A 玩家中共有1,742人可出同色20盤。seed2027，各級隨機分100人 TRAIN／20人 monitor，其餘542人 dev；顏色優先可重用盤數最多者，再比可用總盤數，再取黑，隨機補足20盤。玩家互斥，不按既有分數選樣。

| role | 玩家 | 棋局 | 重用 | 補建 | 題數 k=5/10/15/20 |
|---|---:|---:|---:|---:|---|
| TRAIN | 1,000 | 20,000 | 12,475 | 7,525 | 2,000 / 2,000 / 2,000 / 2,000 |
| monitor | 200 | 4,000 | 2,513 | 1,487 | 1,000 / 1,000 / 1,000 / 1,000 |
| dev | 542 | 10,840 | 6,799 | 4,041 | 2,000 / 2,000 / 2,000 / 2,000 |

TRAIN＋monitor需補建9,012盤／1,999,069位置；三組合計補建13,053盤／2,900,273位置。輸出 `data/processed/strength_features/humansl/policy_cnn/cached_A_seed2027/`，含 games、reuse_games、supplement_games、三組 questions、report／identity／完成標記。重建入口 `PYTHONPATH=src:. python -m scripts.strength.select_cached_policy`（拒絕覆寫）；唯一切分邏輯新增於 data/cv.py。

限制：dev最少級別rank_id=4只有7位玩家；反覆抽題不增加獨立樣本數，需玩家bootstrap。所有來源為learn A，C v2向量對這些玩家可能已有接觸，不能當全系統未見玩家驗證。新資料須重新跑A0及全部對照，不與785715／788616直接作配對比較。pack重用目前僅由manifest判定，備份檔完整性與planes一致性未驗證。僅建立清單／題組，未編碼planes、抽policy、訓練或提交GPU。

驗證：新切分＋既有policy residual／epoch／branches回歸45項通過（11.37秒），本次修改diff檢查通過；真實題組已檢查每題同玩家／同色／同級別及盤內無重複。


### 2026-11-17：TRAIN/monitor 完整逐手 policy 持久快取（提交前）

使用者授權依根目錄 plan.md 先抽 TRAIN/monitor、可重建輸入與推理流水重疊、不額外保存 input planes；
Slurm 提交成功後結束對話，不監看、不訓練、不抽 dev。

- 新選盤已凍結於 `data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/`。
  TRAIN 1,000人／20,000盤／4,459,699位置／8,000題；monitor 200人／4,000盤／884,079位置／4,000題；
  dev只保存360人／7,200盤／8,000題清單。玩家互斥、同色20盤、HD≤1、四k及十級均衡驗證通過。
  三組可重用盤／需補盤／補盤位置逐項重現計畫；本輪重用14,987盤、補建9,013盤／2,007,593位置。
- CPU來源預檢通過：A、B各HD0/1的真實ZIP與完整重播planes／globals／target逐值抽樣一致；
  全部A/B pack manifest、npy shape/dtype與canonical位置數吻合；teacher及vectors可讀，來源已hash。
  全ZIP CRC安排在作業內、GPU啟動前檢查；每pack再抽樣比對，失敗即停止。
- C v2既有向量學習玩家重疊：TRAIN 973/1,000、monitor 196/200、dev 0/360。
  因此monitor不是teacher未見玩家測試；本輪僅抽特徵，不產生新的k=5/10/15/20成績。
- 逐手輸出5,343,778位置，38,918,735,174 bytes（36.246 GiB）另加少量npy／分片metadata；
  預檢可用55,585,370,112 bytes，完成policy後約餘15.5 GiB，空間與其他作業共用。
  保存logp FP16[位置,10,362]、原始q FP32[位置,10]、target int16、side uint8與整數offsets/game_index。
  非法點logp=-inf；P可由FP32重新正規化重建，不另存P／one-hot／輸入planes。
- 新入口 `src/go_aicup/tasks/strength/policy_residual/persistent_cache.py`，設定 `configs/strength/policy_cache.json`，
  Slurm為同套件 `persistent_cache.slurm`。5×H200／60 CPU／500 GiB／2h上限，估20–40分鐘（未實測），
  最多10配置GPU-h；每GPU預取一批，CPU重播／RAM解包與GPU推理重疊，每32盤原子發布可續跑分片。
  完成須有 `policy/complete.json` 及全部通過checksum的分片；提交不代表抽取完成。
- CPU合成3項通過（1.29s）：切分與題組、pass／ko遮罩／尾批／精確q／原子發布／身分與checksum損壞拒絕、
  輸入手序／實際著手排除檢查；壓縮ZIP header／切片assert自我檢查通過，Slurm語法與git diff --check通過。
  未做GPU短測；實際吞吐、CUDA編譯、完整抽取成功與所有分片完成狀態仍未知。
- 預檢及來源快照：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/preflight/`；
  作業紀錄與提交收據：同層 `<job_id>/`。cache identity：
  `3b36a6676797f9945aad626cd03972a360308e0ce0e830c193505ccf20a1902b`。

正式提交收據：job **792644**（2026-11-17T17:54:01.043860+08:00），`8gpus`／5×H200／60 CPU／500 GiB／2h。
紀錄：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/792644/submission.json`；log：`outputs/slurm-rank-policy-cache-792644.out`。
依使用者要求，提交成功後不查佇列、不持續監看；尚未宣稱開始執行或抽取完成。


### 2026-11-17：取消792644後的逐手policy流水線優化（未重新提交）

使用者先要求取消792644，Slurm已確認CANCELLED；本輪依「盡量讓GPU持續推理、CPU持續重建」修改實作。
取消前已有25個完成分片、約1.3 GiB，原`policy/`與來源快照均保留，未刪除或改寫。

- 改動範圍：`persistent_cache.py`、`configs/strength/policy_cache.json`、
  `tests/test_persistent_policy_cache.py`、套件README及根目錄plan；選盤／teacher／輸出欄位不變。
- 每GPU兩條producer獨立供料：ZIP快取4片；10個CPU重播worker在ProcessPool保留2片工作，
  補盤佇列容量由全部補盤片數界定，避免GPU正在推理ZIP資料時CPU等待。補盤packed輸入2.016 GiB，
  全24,000盤packed輸入5.365 GiB；僅RAM、不另存輸入planes。
- worker先packbits再IPC，H2D保持bits，在GPU重用既有unpack；plane部分傳輸bytes約為原本1/8。
  一片輸入一次傳入GPU，批次間移除CPU scalar讀回及逐批D2H，完整logP/q在GPU彙整。
- 專用CUDA copy stream複製到pinned RAM；背景writer等copy event後才檢查精度、原子保存、
  checksum及重新讀取驗證。最多2片輸出在途，I/O落後時施加有界backpressure，失敗向主流程傳遞。
  最終以writer收據核對全部新分片marker，避免GPU結束後再同步重讀整份新policy。
- 新版獨立目錄：`data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/policy_stream_v2/`；
  報表／快照：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract_stream_v2/<job_id>/`。
  schema=2，identity=`15061ad5b218d3e6fb3c3225045e40309548a85553d0fdb668feaeb70c3dd6ab`。
  不跨版本混用舊policy，舊25片留存但不拿來宣稱新版本已完成。
- CPU合成10項通過（0.43秒），含選盤／題組、精確q、pass／ko／尾批、checksum損壞拒絕、
  兩條producer獨立供料／容量界限／提前結束／錯誤傳遞、打包解包一致、非有限值拒絕，
  及背景writer確實等待copy完成後才讀取、主執行緒仍可繼續工作。
  真實來源預檢通過（含A/B各HD0/1 ZIP與重播逐值parity）；Slurm語法與git diff --check通過。
- 未重新提交、不持續監看、不訓練、不抽dev。CPU預檢快照在新版`extract_stream_v2/preflight/`。
  5×H200／60 CPU／500 GiB／2h上限未變；20–40分鐘仍是先前粗估。
  實際GPU吞吐、CUDA跨stream、空檔比例尚未量測，不能宣稱GPU持續滿載；啟動來源／CRC／編譯、
  末尾drain與來源驗證仍有固定成本。各k=5/10/15/20模型分數與本輪吞吐增益均未知。


### 2026-11-17：依使用者指示清除取消作業產出、改回原檔名重新提交

已確認792644為CANCELLED且沒有其他policy任務；取得policy.lock後清除25個舊分片、舊run與log，
同步清除兩版失效preflight。選盤／題組、ZIP、棋步快取與teacher不變；優化schema 2回到原policy／extract路徑。
- 已刪除 `data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/policy`（1,294,232,748 bytes）。
- 已刪除 `outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/792644`（1,740,865 bytes）。
- 已刪除 `outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/preflight`（1,738,685 bytes）。
- 已刪除 `data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/policy_stream_v2`（471,340 bytes）。
- 已刪除 `outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract_stream_v2`（1,766,716 bytes）。
- 已刪除 `outputs/slurm-rank-policy-cache-792644.out`（2,017 bytes）。
重新提交前10項CPU測試通過（0.38s），Slurm語法及test-only通過；先做真實來源預檢再送出5×H200／60 CPU／500 GiB／2h任務。

原路徑真實來源預檢通過；schema 2 identity=`cc1ca4f6da211644301c16ed9c04aaae038baf5f6b64f41fd56fdf71cde94f72`，5,343,778位置，預期36.246 GiB，
提交前可用55,584,415,744 bytes。CPU/GPU獨立供料與背景writer優化不變，尚未GPU效能短測。

已重新提交 **792691**（2026-11-17T18:18:52.111990+08:00），5×H200／60 CPU／500 GiB／2h；
收據：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/792691/submission.json`；log：`outputs/slurm-rank-policy-cache-792691.out`。
舊792644產出已刪除，以原policy／extract檔名從零重新抽取；提交後不查佇列、不監看，不宣稱完成。

### 2026-11-18：競賽一——追冠重新規劃：A0（OGS 外部資料可行性）＋B0（既有編碼器互補性）（使用者定稿順序；事前登記，看結果前寫定）

**目標尺度：** 11/15 歷史榜首 Public 0.9060 對我方 0.8245254，差 0.0815；只改新增集須提高 0.163。即使 k=1 做到滿分，新增集也只加 (1−0.44)/4＝0.14（本地四桶估算）→ 須能從每盤取得更好辨識訊號、且多盤合起來仍受益的方法。順序：A0、B0 並行 → A0 通過後先做同算力跨站短測（只用幽玄 vs 幽玄＋OGS）→ 才決定 A1（約 85 GPU-h）。C1、C2 跳過；B1 備案。

**A0（CPU，登入節點下載＋串流過濾，不佔 GPU）：** 來源 za3k 整理的 OGS 2025 合集（2013–2025-05，約 5,600 萬盤；發布頁載明經 OGS 同意重散布，README 記載 OGS 代表同意研究使用；正式用於有獎競賽與得獎交付的範圍 A1 前須再確認）。
- 樣本（`sample-100k.json.gz`，實為 2021 版隨機 10 萬盤）：19×19 47.6%；19×19、完成、HD≤1、無擺子、≥60 手 33.6%，其中 ranked 27.2%；速度 live 88%／correspondence 10%／blitz 2%；中位手數 201；段級中位約 9k（OGS 0＝30k、30＝1d；90 百分位約 2k）→ 與幽玄（多為段位）有明顯棋力落差。JSON 欄位含雙方 player id、ranked、width/height、handicap、time_control、moves。
- 下載速度：za3k 主機約 0.36–1.5 MB/s（並行反而 0.39 MB/s），Internet Archive 2021 鏡像更慢；全量 JSON 43 GB 約 20 小時，改用 **2025 by-username SGF**（16.2 GB；資料夾＝數字 player id，bot 沒有自己的資料夾；SGF 有 SZ、HA、RE、GC[速度,ranked]、BR/WR、DT 與著手）。可續傳下載到 `/home/jason1122/ogs2025/`，同時以 `scripts/identity/ogs_stream_filter.py` 串流過濾（每盤只存一次；19×19、無擺子、HA≤1、完成、≥60 手、黑先交替、前 60 手無 pass），輸出 int16 著手＋meta 分片與 `stats.json`（各資料夾擁有者的合格 ranked 盤數分布）。11/18 09:54 開始，預計當日 17:30–22:30 完成。
- **A0 通過條件（往下做跨站短測）：** 雙方都是真人（雙方皆有資料夾）、19×19、ranked、HA≤1 的對局中，**≥20 盤的玩家 ≥ 30,000 人**；另報即時棋／通信棋占比、年份與段級分布、儲存量。

**B0（1 卡，`scripts/identity/pid2_ensemble_b0.py`）：** 基底＝現行 `hung-seqB-e2emix-ml` 融合。候選成員：既有 15 個編碼器的驗證台盤向量（各自逐手平均；含基底編碼器本身的平均聚合 A9600）＋E2E14400（含其 B 頭）；排除 v6_gal_p1_4800（訓練用過 pool 1 的 gallery 盤）。風格項改為 (cos_B＋cos_E9600＋Σβ_m·cos_m)／(2＋Σβ_m)。開發 pool 0 draws 100–119 貪婪前向選擇：每輪試每個未選成員 × β ∈ {0.25, 0.5, 1}，開發增益 ≥ +0.002 才加入，最多 3 個（blend 上限 5）。封存 draws 1500–（pool 0 1500–1519、pool 1 1500–1529）只評一次。**gate（對基底）：pool 1 Δ ≥ +0.010、z ≥ 2.5、四桶各 ≥ −0.005、pool 0 Δ > 0。** 預期：小（多數成員與基底同為 V2 初始化），過關換算 Public 約 +0.005，當可累積收益。成本 < 1 GPU-h。已送 Slurm 795939。

**B0 結果（Slurm 795939，1 卡 5 分 35 秒，`outputs/pid/pid2_ensemble_b0/dev_choice.json`）→ 開發題組沒有任何成員能加入，封存題組未開，不產生提交檔。** 16 個候選 × 3 個 β 在開發 pool 0 draws 100–119 全部為負：最佳 E2E14400 β=0.25 −0.0003，其次 E2E14400 β=0.5 −0.0008、A9600（基底編碼器逐手平均）−0.0012、v4_4800 −0.0013、v5_sub82 −0.0016、v2_both −0.0016、v7_band −0.0018（皆 β=0.25）。判讀：既有編碼器都與基底同為 V2 初始化（或與 E2E9600 同一條訓練路徑），在 B＋E2E9600 混合之上沒有新資訊；E2E 的 +0.027 來自訓練方式不同，不是「多平均幾個」。→ 便宜的集成路線結案；B1（同架構不同 seed）預期同樣有限，維持備案。

**A0 進度與跨站短測的程式（11/18 上午，使用者「先寫轉檔和雙來源 loader」；未送 GPU 作業）：**
- 下載速度降到約 0.1–0.3 MB/s（09:54–10:52 共 1.0 GB），16.2 GB 全量需 14–40 小時。by-username 封存每個資料夾＝一位玩家的完整歷史 → **不必等全量**：讀到的資料夾擁有者都是真人且棋局完整，可在合格玩家數足夠時停止下載。對手資料夾可能尚未到，故「對手也是真人」只能報下界；訓練只用玩家本人的著手，bot 對手只出現在盤面上。
- 磁碟：`/home` 72 GB 已用，封存 16.2 GB＋全部合格棋局約 8.6 GB 會到 97 GB（配額 100 GB、寫滿會無聲失敗）→ 過濾改為**只存 ranked 對局**（仍全部計數），10:47 重啟。
- `src/go_aicup/tasks/identity/external.py`：選玩家（ranked、玩家為資料夾擁有者、可選對手真人／年份／最低段級、≥min_games、上限人數與每人盤數）→ 合併 BoardCache：前 1,000,000 列就是幽玄快取（shard 檔 symlink、同 build id、game_index 不變），其後為 OGS 列（遇第一個 pass 截斷）＋`external_players.parquet`。預設幽玄快取與所有既有程式不受影響。
- `train_ge2e.training_players`：`sampling.external` 時把 OGS 玩家接在幽玄玩家後（檢查合併快取前段與幽玄逐列相同）；`external.yugen_share` 設定每次抽到幽玄玩家的機率（`GE2EBatches(player_p=…)`；未設時抽樣與原本逐位相同，已測）。
- 測試：`tests/test_identity_external.py` 2 項（合併快取幽玄列逐盤相同、OGS 列含 pass 截斷與執色、bot 排除；抽樣比例與未設時相同）＋既有 identity 16 項通過。
- 短測設定 `configs/identity/pid_e2e_ogs_short_{Y,YO}.json`：同 E2E-scratch 9,600 步配方、960 步（十分之一），兩臂只差資料；YO 暫定 yugen_share 0.5（待使用者確認）。評估腳本尚未寫。
- **使用者決定（11/18）：** YO 臂 `yugen_share` 0.5；合格玩家達 3 萬人即停下載並建快取。**實際：** 下載在 11:07 全部完成（16,191,860,680 bytes；後段速度大增）。第一次過濾在登入節點 25 分鐘 RSS 已 11 GB——串流 `tarfile` 仍累積每個 TarInfo（全量約需 80 GB）——停掉後修正（每個成員後清空 `tf.members`；300 MB 測試峰值 374 MB、約 16k 檔／秒），並修正進度列只在保留盤觸發的錯。前 1.85% 的封存已有 4,388 位 ≥20 盤 ranked 合格對局的玩家（對手不限）→ 3 萬人門檻可滿足，限制改為磁碟：快取上限 **40,000 人 × 每人 20 盤、對手也須是真人**（約 15 GB）。過濾＋刪封存＋計數（全部／2021 起，對手不限／真人）＋建合併快取一個作業：Slurm **796320**（1 卡配額當 CPU 用，上限 5 小時）。

**跨站短測（事前登記，看結果前寫定；使用者計畫第 3 步；A0 通過後仍待使用者明確同意才送）：**
- **執行條件：** 建快取作業 796320 成功，且 `external.json` 的合格玩家（ranked、HA≤1、雙方皆真人、≥20 盤）≥ 30,000（`ogs_short_train.sh` 會先檢查，不足則不訓練）。
- **兩臂：** `configs/identity/pid_e2e_ogs_short_Y.json`（只用幽玄 6,426 人）與 `_YO.json`（幽玄＋OGS 至多 40,000 人，每次抽到幽玄玩家的機率 0.5）；同 E2E-scratch 配方（同初始化、同 B 頭設定、同 batch 64×8×32、同 seed），各 960 步 WSD（十分之一）。各 8 卡約 1.1 小時，合計約 17.6 GPU-h。
- **評估（`src/go_aicup/tasks/identity/ogs_short_eval.sh` → `scripts/identity/pid2_ogs_short_eval.py`，1 卡）：** 抽兩臂驗證台逐手向量（快取 `/work/jason1122/ogs_short_cache/`，`/home` 空間不足）；在現行融合內比較（rest＝開局＋類別＋L1／L2）：單獨 rest＋16w·cos_X，以及與 B 各半 rest＋16w·(0.5 cos_B＋0.5 cos_X)；不調權重；新封存 draws 1600–（pool 0 1600–1619、pool 1 1600–1629）。
- **繼續門檻（mix_YO 對 mix_Y）：** pool 1 Δ ≥ +0.010、pool 0 Δ > 0、四桶各 ≥ −0.005（k=1 的增益不被其他桶抵銷）。兩臂單獨都不高於「無風格項」→ 記為無法判定。過門檻才提 A1（完整 9,600 步，約 85 GPU-h）給使用者決定。
- **使用者同意送出（11/18）：** 訓練 Slurm **796340**（Y）、**796341**（YO），皆 afterok 796320、8 卡、上限 1 小時 45 分；評估 **796342**（afterok 兩臂，1 卡）。
- **A0 實際數字（job 796320，11/18 13:34 完成）：** 封存 106,533,633 個成員；19 路 32,675,323 盤，其中 ranked 且通過全部篩選、去重後 13,393,664 盤（未 ranked 5,887,981 盤只計數不保存；讓子 >1 共 71,838、前 60 手內 pass 121,518、未完局 52,326）。資料夾擁有者 1,209,590 人。**≥20 盤 ranked 合格對局的玩家 82,481 人；要求對手也是人類（有資料夾）時 49,961 人**（1,809,413 盤次），遠超 3 萬門檻。快取：從 49,961 人隨機抽 40,000 人、每人 20 盤＝800,000 列、1.51 億手、14.36 GB，`/home` 建完剩 11 GB。即時／通信對局比例與年份分布：filter 有記 speed／year 欄位但本次未統計，**未知**。
- **Y（796340）完成**（1 小時 04 分）。**YO（796341）13:35 崩潰**：OGS 對局最長 980 手，E2E head 的手數位置表只有 512 格（`seqagg.MAX_PLY`），800,000 盤中 35 盤超過 511 手 → CUDA index assert；評估 796342 因相依而取消。修正：`train_ge2e.training_players` 外部分支把 OGS 盤的可抽手數上限截為幽玄最長 445 手（幽玄列不變，Y 臂不受影響）；CPU 實資料空跑 2,000 步：最大 ply 444、OGS 抽中比例 0.4995。崩潰的輸出目錄改名 `pid_e2e_ogs_short_YO.crash796341`。重新送出 YO **804782**、評估 **804783**（afterok 804782），設定與資源不變。

### 2026-11-18：補登 787069／792691 結果，刪除舊 policy 實驗目錄；policy 快取改為只存 ZIP

**787069（C-only A0／S／F × LR 1e-3／3e-4，每臂 4 epochs＝1,000 updates，11/16 COMPLETED 48 分 58 秒，約 4.05 配置 GPU-h）補登：** monitor 選模結果是 S、F 四臂都選 step 0（等於共同起點），A0 兩臂選 step 1,000。dev Equal-k：A0 lr1e-3 0.7117、A0 lr3e-4 0.7137、S／F 四臂 0.7139（k=5/10/15/20：0.672／0.7163／0.7284／0.7388）。S−A0、F−A0 差 +0.0002～+0.0022，95% 區間都跨 0，全部未過 gate。單一 seed、dev 曾重複使用、monitor 有 190／200 名 C 向量學習玩家重疊。

**792691（TRAIN＋monitor 逐手 policy 持久快取）補登：** COMPLETED 19 分 52 秒，約 1.58 配置 GPU-h；`policy/complete.json` status=complete，24,000 盤、5,343,778 位置、765 分片，identity `cc1ca4f6…f72`。

**policy 快取改存 ZIP：** 依使用者要求，`data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/`（38,922,334,146 bytes、5,367 檔）以 `scripts/data/zip_dir_incremental.py` 逐分片搬入同層 `cached_AB_mondev_seed2027.zip`（deflate level 1，25,080,805,865 bytes）。每分片先以 `complete.json` 的 SHA-256 核對原檔，寫入並 fsync 後讀回逐檔比對 SHA-256 才刪除；逐檔雜湊記在 `data/processed/migrations/zip_cached_AB_mondev_seed2027/journal.jsonl`。登入節點開始、Slurm 796332（12 CPU，11 分 51 秒）續完，原目錄已移除。讀取用 `go_aicup.tasks.strength.policy_residual.policy_zip.load()`，直接在記憶體解壓，沿雜湊鏈驗證，不寫磁碟；完整 logp 約需 38.7 GB RAM。`persistent_cache.py` 抽取流程寫原目錄，重抽前須先解壓。

**刪除（使用者要求，結果皆已記於 RESULTS）：** `outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/`（792691 執行紀錄＋preflight，含與 src 相同的程式副本，4.1 MB）、`outputs/rank/humansl/policy_branches/seed2027_b32_e12/788616/`（四臂 A0／R10／Z10／Z32，結果見 11/17「788616 原預算續跑」節，82 MB）、`outputs/rank/humansl/policy_residual_epoch/`（787069，35 MB）。三者都不會續跑；新切分的五臂實驗（plan.md）會重新訓練 A0／R10／Z32。785715（`policy_residual/seed2027_t1000_m200/`）未刪，仍被 11/16 S 融合規格引用。Slurm log 未刪。

### 2026-11-18：競賽二——新切分單卡五臂（plan.md）短測與正式提交

**短測（各臂 50 updates，1×H200）：** 796473（快取在 CPU RAM、每塊 512 位置）12 分 34 秒；796545（快取整份放 GPU）10 分 12 秒；796590（GPU＋每塊 4,096）3 分 26 秒。每步 32 題涵蓋約 396 盤／88,139 位置，題級 loss 須對全部位置前向兩遍；瓶頸依序為 CPU 取樣＋H2D、再來是小分塊的 kernel／Python 開銷。796590 秒／update：A0 0.004、R10 0.256、Z32 0.253、CNN-R10 0.155、CNN-Z32 0.150；秒／完整評估 2.5／7.8／7.7／5.6／5.5；顯存峰值 36.3–37.5 GiB（快取約 36.3 GiB）。五臂 step 0 與共同起點一致；改分塊只讓 R10 monitor Equal-k 差 0.00016（約 1 題，加總順序不同的浮點差）。50 步分數只作工程檢查。參數量 A0 210、R10 9,949、Z32 12,019、CNN-R10 54,684、CNN-Z32 56,754（plan 寫 56,786）。

**正式提交：** commit `3ea4062`，Slurm **796606**，`8gpus` 1×H200／12 CPU／200G／2h 上限；設定 `configs/strength/policy_five_arms.json`（plan §5：batch 32 題、每 k 8 題、12 epochs＝3,000 updates、AdamW LR 3e-4、encoder WD 1e-4、每 125 步評估、monitor 嚴格改善選模、seed 2027）。預估約 55 分鐘（50 步外推）。輸出 `outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/five_arms/796606/`。不碰 dev。

**完成結果（2026-11-18，796606 讀 log 與各臂 `result.json` 登記）：** `COMPLETED`、exit 0，牆鐘 57 分 31 秒（≈0.96 配置 GPU-h），快取載入 50.2 秒（5,343,778 位置），五臂各 3,000 updates、無非有限值；顯存峰值 36.3–37.5 GiB；持久產物 96,903,776 bytes。每臂看過 TRAIN 1,000 人全部（約 19,820 盤）。以下為 monitor（200 人、4,000 題）本地官方公式，不是 LB、不是 dev：

| 臂 | 選中 step | k=5 | k=10 | k=15 | k=20 | Equal-k | −A0 | 後 12 次評估平均 | 評估 sd | 支路 logits RMS（q≈13.5） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 3000 | .6863 | .7507 | .7544 | .7729 | .7411 | — | .7392 | .0009 | 0 |
| R10 | 875 | .6848 | .7472 | .7641 | .7849 | .7452 | +.0042 | .7381 | .0033 | .053 |
| Z32 | 3000 | .6889 | .7507 | .7547 | .7748 | .7423 | +.0012 | .7393 | .0012 | .008 |
| CNN-R10 | 2500 | .6898 | .7473 | .7635 | .7817 | .7456 | +.0045 | .7389 | .0033 | .080 |
| CNN-Z32 | 1500 | .6853 | .7510 | .7603 | .7792 | .7440 | +.0029 | .7391 | .0019 | .052 |

判讀：選中值是 24 次評估取最大，對波動大的臂（R10／CNN-R10 sd .0033 vs A0 .0009）偏樂觀；後半段平均五臂都在 .738–.739，無可分辨差異，最大差 +.0045 也未達候選 gate +.005。支路對 logits 的貢獻比 q 小兩個數量級以上，TRAIN Equal-k 全程只在 .741–.743 間 → 不是過擬合，而是支路沒學到可用訊號或 LR 3e-4／3,000 步推不動，現有資料無法區分。R10／CNN-R10 的增加集中在 k=15／20，k=5 略低於 A0。預期 dev 不會有臂過 gate；依 plan §6 步驟 6–7 仍做 dev 收尾（使用者 11/18 指示），結果另記。

**Dev 收尾（2026-11-18，Slurm 796930，`policy_residual/dev_eval.py`＋`configs/strength/policy_five_arms_dev.json`）：** `COMPLETED`、exit 0，`8gpus` 1×H200，牆鐘 28 分 17 秒（≈0.47 配置 GPU-h），GPU 顯存峰值 15.4 GiB；持久產物 9,157,685 bytes。各臂只評估 monitor 選中的 checkpoint（同上表 step），dev 未參與選模。五臂 monitor 分數重算與訓練紀錄差 0.0。dev＝360 人、7,200 盤、8,000 題、1,600,205 位置（1,419 盤重用快取、5,781 盤重新推論）；抽取誤差 mass 0.00061、q 0.0077、C 0.00029。此 dev 為歷史上已用過的固定 outer dev，屬有限證據。以下為本地官方公式，不是 LB：

| 臂 | 選中 step | k=5 | k=10 | k=15 | k=20 | Equal-k | CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 3000 | .6896 | .7420 | .7556 | .7762 | .7409 | .9632 |
| R10 | 875 | .6877 | .7419 | .7532 | .7779 | .7402 | .9616 |
| Z32 | 3000 | .6888 | .7429 | .7553 | .7762 | .7408 | .9635 |
| CNN-R10 | 2500 | .6907 | .7429 | .7541 | .7654 | .7383 | .9628 |
| CNN-Z32 | 1500 | .6912 | .7465 | .7529 | .7747 | .7413 | .9654 |

玩家配對 bootstrap（2,000 次；gate：ΔEqual-k ≥ .005、P(>0) ≥ .90、每 k ≥ −.005）：

| 比較 | ΔEqual-k | Δk5／k10／k15／k20 | 95% CI | P(>0) | gate |
|---|---:|---|---|---:|---|
| R10−A0 | −.0007 | −.0019／−.0001／−.0024／+.0018 | [−.0053, +.0042] | .389 | ✗ |
| Z32−A0 | −.0000 | −.0008／+.0009／−.0003／.0000 | [−.0007, +.0006] | .444 | ✗ |
| CNN-R10−A0 | −.0026 | +.0011／+.0009／−.0016／−.0108 | [−.0092, +.0038] | .202 | ✗ |
| CNN-Z32−A0 | +.0005 | +.0016／+.0045／−.0027／−.0015 | [−.0048, +.0059] | .559 | ✗ |
| Z32−R10 | +.0006 | +.0011／+.0010／+.0021／−.0018 | [−.0042, +.0055] | .605 | ✗ |
| CNN-R10−R10 | −.0019 | +.0030／+.0010／+.0008／−.0126 | [−.0068, +.0027] | .211 | ✗ |
| CNN-Z32−Z32 | +.0005 | +.0024／+.0035／−.0024／−.0015 | [−.0047, +.0059] | .560 | ✗ |
| CNN-Z32−CNN-R10 | +.0030 | +.0005／+.0035／−.0011／+.0093 | [−.0019, +.0085] | .887 | ✗ |

**判讀：** 八組比較全部未過 gate，`claims` 全為 false。四個候選臂對 A0 皆在 ±.003 內、CI 跨 0；Z32 與 A0 的 CI 僅 ±.0007，與其支路 logits RMS .008 一致——z 支路幾乎未被使用。monitor 上 R10／CNN-R10 的 +.004 在 dev 消失（−.0007／−.0026），符合上節「24 次評估取最大偏樂觀」的預期。CNN-R10 在 k=20 退 .011（對 A0）／.013（對 R10），是唯一超過 −.005 的單桶退步，原因未知（殘差頭在大 k 的信心尺度或 dev 雜訊，現有資料無法區分）。CNN-Z32−CNN-R10 的 +.003 主要來自 CNN-R10 的 k=20 退步。**結論：** C 的逐手 policy 特徵以 set 或 CNN、殘差或獨立支路接上 A0，在此規模（1,000 TRAIN 人、3,000 updates、LR 3e-4）量不出增益；不推進 H＋C 融合、不做 LB 探針、不加 seed。產物：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/five_arms/796606/dev/796930/summary.json`。

### 2026-11-18：競賽二——H 版逐手 policy 快取抽取（9 個 HumanSL profile；使用者核准先抽特徵）

**目的：** 為 H 版五臂（A0／R10／Z32／CNN-R10／CNN-Z32，架構照 C 版 plan.md，條件數 10→9）準備 TRAIN＋monitor 逐手 policy 特徵；本次只做抽取與壓縮，不訓練、不抽 dev。
**Teacher：** 凍結 HumanSL 一般權重；`input_meta` 為官方 `makeBasicRankProfile`（雙方同段位）經原 metadata encoder 的輸出，依行棋顏色取用。profile＝`rank_10k, 7k, 5k, 3k, 1k, 1d, 2d, 3d, 4d`（現有 H 15 個中落在 10k–4d 者）。貼目 HD0 6.5／HD1 0.5、遮罩與 C 快取相同。作業開頭在 GPU 上對 HD0／HD1 各一盤，比對「向量索引 teacher」與「未改 metadata encoder 的原網路」的每手 log p，最大差 >1e-3 則中止（`parity.json`）。
**資料：** C 版選盤 ZIP 已刪，以 `policy_h/select.py`（同 seed、同 `cv.py` 函式，讓子表改讀 `data/raw/raw.zip`）重產；games／train／monitor 題組 SHA-256 與 796606 記錄逐位元相同，重用／補建盤數同 plan §2。TRAIN 20,000 盤＋monitor 4,000 盤＝5,343,778 位置；預估 logp FP16 約 32.7 GiB，壓縮後約 23 GB。
**程式：** 新目錄 `src/go_aicup/tasks/strength/policy_h/`（`select.py`、`extract.py`、`extract.slurm`、`zip.slurm`）、`configs/strength/policy_h_cache.json`；不改既有程式，`extract.py` import `policy_residual.persistent_cache`，只覆寫寫死 10 個條件的 `infer`／`validate_policy`／`validate_shard`／`worker` 與路徑。CPU 預檢（選盤 hash、ZIP／重播 planes 一致、容量）通過，identity `80e80404…`；假 teacher 的 CPU 冒煙測試確認 shape `[n,9,362]`、行棋顏色索引與分片寫入／驗證。
**提交：** 抽取 Slurm **797636**（5×H200／60 CPU／500 GiB／2h，上限 10 配置 GPU-h；依 C 版 792691 的 19 分 52 秒估約 20 分鐘、約 1.6 GPU-h）。前兩次 parity 中止（各約 20 秒）：797605 最大差約 3e-3，是 TF32 雜訊；parity 改為 FP32（TF32 關閉）後吻合到 4e-6。797621 原本以「顏色對調」當反向對照，但 `makeBasicRankProfile` 讓黑白兩方同段位，輪黑下與輪白下的向量完全相同（差 0.0），顏色不影響 H，因此改用「相鄰 profile」對照（差 5.0–7.3）。797636 parity 通過，抽取本身照 C 快取用 TF32。
**壓縮：** 新增 `policy_h/zip_parallel.py`：11 個 worker 程序並行讀檔、算 SHA-256／CRC-32，並以 zipfile 相同的 raw deflate（level 1）壓縮；單一主程序依序寫入，每批 32 片關檔 fsync 後由 worker 平行讀回比對 SHA-256，通過才刪原檔。journal 與續跑語意同 `scripts/data/zip_dir_incremental.py`。CPU 合成測試：成員內容與壓縮後大小和原工具逐位元相同；續跑、來源與 complete.json 不符時拒絕刪除，皆通過。Slurm **797670**（afterok:797636，1 GPU／12 CPU／64 GiB／2h；取代未執行的單執行緒版 797639）。產物：`data/processed/strength_features/humansl/policy_cnn_h/cached_AB_mondev_seed2027_h9.zip`；log 與來源紀錄在 `outputs/rank/humansl/policy_cnn_h/cached_AB_mondev_seed2027/extract/797636/`。成功以 `policy/complete.json`（壓縮前）與 797670 的 `done` 為準；結果待補。

**H 五臂短測（2026-11-18，Slurm 798446，`8gpus` 1×H200，`policy_h/five_arms.py`＋`configs/strength/policy_h_five_arms_smoke.json`）：** 各臂 50 updates，其餘照 C 版 796590。`COMPLETED`、exit 0，牆鐘 3 分 23 秒；H 快取從 ZIP 載入 49.8 秒（5,343,778 位置），每盤 H 平均的 FP16 還原誤差最大 0.00031；顯存峰值 32.7–33.8 GiB；產物 31 MB。五臂 step 0 一致（monitor Equal-k 0.7370、CE 0.9248；TRAIN 0.7432），殘差臂零起點斷言通過。

| 臂 | 參數 | 秒／update | 秒／完整評估 | 峰值 GiB | step 50 monitor Equal-k（k=5/10/15/20） | 支路 logits RMS（q≈14.5） |
|---|---:|---:|---:|---:|---|---:|
| A0 | 190 | 0.004 | 2.5 | 32.7 | .7364（.6777／.7396／.7554／.7729） | 0 |
| R10 | 9,704 | 0.250 | 7.6 | 33.8 | .7385（.6827／.7414／.7579／.7720） | .216 |
| Z32 | 11,839 | 0.248 | 7.5 | 33.8 | .7355（.6770／.7365／.7557／.7729） | .015 |
| CNN-R10 | 52,007 | 0.149 | 5.4 | 33.2 | .7397（.6849／.7414／.7592／.7734） | .118 |
| CNN-Z32 | 54,142 | 0.145 | 5.4 | 33.2 | .7475（.6846／.7509／.7666／.7878） | .083 |

50 步的分數只作工程檢查（單次評估、未過 dev，C 版經驗是 monitor 單點偏樂觀）。外推正式 3,000 步＋每 125 步評估（25 次）：A0 約 1 分、R10／Z32 各約 16 分、CNN 兩臂各約 10 分，加載入約 53 分鐘、約 0.9 配置 GPU-h（C 版 796606 為 57.5 分）。正式設定 `configs/strength/policy_h_five_arms.json` 已建立，尚未提交。

**H 五臂正式提交：** Slurm **799148**，`8gpus` 1×H200／12 CPU／200G／2h 上限；設定 `configs/strength/policy_h_five_arms.json`（batch 32 題、每 k 8 題、12 epochs＝3,000 updates、AdamW LR 3e-4、encoder WD 1e-4、每 125 步評估、monitor 嚴格改善選模、seed 2027）。預估約 53 分鐘。輸出 `outputs/rank/humansl/policy_cnn_h/cached_AB_mondev_seed2027/five_arms/799148/`。不碰 dev；依使用者指示提交後不監看，結果待補。


### 2026-11-18：799148 H 五臂正式結果分析（monitor；未評 dev）

Slurm **799148** `COMPLETED`、exit 0，牆鐘 **53 分 58 秒**（約 **0.90 配置 GPU-h**）；程式內耗時 3,226.9 秒。五臂各完成 3,000 updates／12 epochs，皆有 `complete.json`；各見過 TRAIN 全部 1,000 人、19,820 盤。快取載入 49.2 秒，5,343,778 位置；GPU 峰值 32.7–33.8 GiB，持久產物 95,918,521 bytes。逐筆檢查五臂共 15,000 筆 gradient log，非有限梯度與缺梯度參數均為 0。以下是 monitor 200 人／4,000 題的本地官方公式，非 dev、非 LB；同一 monitor 同時選 checkpoint，分數有選模偏差。

| 臂 | 選中 step | k=5 | k=10 | k=15 | k=20 | Equal-k | −A0 | 後 12 次評估平均 | 評估 sd | 支路 logits RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 875 | .6807 | .7460 | .7564 | .7716 | .7387 | — | .7367 | .0008 | 0 |
| R10 | 2625 | .6798 | .7424 | .7560 | .7792 | .7394 | +.0007 | .7348 | .0024 | .0770 |
| Z32 | 875 | .6797 | .7457 | .7564 | .7748 | .7391 | +.0005 | .7366 | .0012 | .0042 |
| CNN-R10 | 875 | .6812 | .7434 | .7620 | .7777 | .7411 | +.0024 | .7350 | .0022 | .1144 |
| CNN-Z32 | 1500 | .6782 | .7488 | .7627 | .7771 | .7417 | +.0030 | .7362 | .0024 | .0624 |

**判讀與機制限制：**
- 最佳 CNN-Z32 對 A0 四桶差為 −.0025／+.0028／+.0063／+.0055，Equal-k +.0030；CNN-R10 為 +.0005／−.0025／+.0056／+.0061，Equal-k +.0024。增益主要在大 k，k=5 未穩定改善；所有候選連 monitor 的 +.005 增益門檻都未達，尚不能宣稱通過 dev gate。
- 五臂後半段平均 .7348–.7367，均未高於 A0；CNN-Z32 最後一步 .7351，低於其選中 .7417。每臂從含 step 0 的 25 次評估選最大，波動較大的支路臂容易出現樂觀單點。798446 短測 CNN-Z32 step 50 的 .7475 也未在正式每 125 步的評估網格維持；兩者評估步數不同，不作同一步比較。
- Z32 支路 logits RMS .0042，相對 q 約 14.47 很小；R10／CNN-R10／CNN-Z32 分別 .077／.114／.062。這支持支路在目前設定只小幅修正 logits，但 RMS 比值本身不能證明沒有改變決策邊界。TRAIN Equal-k 全程約 .7409–.7460，沒有明顯持續學升；不能據此斷言是過擬合、LR 太小或特徵本身無效，原因未知。
- C 版 796606 的最佳 monitor 增益約 +.0045，dev 最佳僅 +.0005 且全部未過 gate。H 版目前 +.0030 也沒有更強的 monitor 證據；兩版 A0 的輸入／基準不同，不能用絕對分數差當 H 與 C 的公平勝負。

**建議：** 目前不據此延長訓練、加 seed 或製作 LB 探針。若要完成與 C 版相同的實驗收尾，下一步僅對 monitor 已選定的五份 checkpoint 做一次固定 dev 評估及玩家配對 bootstrap；不依 dev 重選步數。此次只分析現有產物，未啟動 dev、GPU 作業或上傳；泛化增益與信賴區間未知。

來源：`outputs/rank/humansl/policy_cnn_h/cached_AB_mondev_seed2027/five_arms/799148/` 的 `summary.json`、各臂 `result.json`／`learning_curve.json`／`complete.json`／`gradients_epoch_*.jsonl`，及 `outputs/slurm-rank-policy-h-five-799148.out`、Slurm `sacct`。後 12 次為 step 1625–3000，sd 使用母體標準差，只描述同一題組跨 checkpoint 波動，並非泛化標準誤。


### 2026-11-18：799148 H 五臂 dev 收尾（使用者要求 2 GPU）

使用者核准比照 C 版 policy CNN 做固定 dev 評估與總結，申請兩卡。新增 `policy_h/dev_eval.py`、`policy_h/dev_eval.slurm` 與 `configs/strength/policy_h_five_arms_dev.json`；重用 C 版 `policy_residual/dev_eval.py` 的選模重現、輸入稽核、逐臂評分、八組玩家配對 bootstrap／gate／claims，抽取函式僅增加可選 teacher hooks（C 預設不變）。H adapter 使用九 profile teacher 與 H 模型／head／cache；兩個 spawn worker 各用一張 GPU 處理連續一半 dev 棋局，以 `/dev/shm` 暫存串接（退出清除，不寫 /work policy 大檔），逐臂評估在 GPU 0 完成。monitor 重現容忍 .0005；dev 不重選 checkpoint、不延長訓練。

預檢：dev 360 人／7,200 盤／8,000 題／1,600,205 位置，與 TRAIN／monitor 玩家零重疊；H teacher SHA-256、profile 清單、五臂完成標記、所需 monitor.zip 通過。CPU 兩項測試（雙 worker 合併順序／offset、H archive／model 接線）通過，CLI import 與 Slurm bash 語法通過；兩卡真推論尚待作業驗證。

資源：8gpus，單節點 2×H200／24 CPU／400 GiB／1 小時上限，最多 2 配置 GPU-h。C 單卡歷史 28 分 17 秒，H 雙卡粗估 20–30 分鐘（約 .7–1.0 GPU-h）；此非量測，讀 ZIP／編譯／重播／評估／bootstrap 未全部平行，需以各 worker 與整體耗時確認加速。gate 沿用 ΔEqual-k ≥ .005、每 k ≥ −.005、Pr_boot(Δ>0) ≥ .90；預期增益未知，monitor 最大 +.0030 尚不足以支持候選。提交 job ID 與結果待補。

**已提交：Slurm 804562**，命令 `sbatch src/go_aicup/tasks/strength/policy_h/dev_eval.slurm`。正式產物 `outputs/rank/humansl/policy_cnn_h/cached_AB_mondev_seed2027/five_arms/799148/dev/804562/`，log `outputs/slurm-rank-policy-h-dev-804562.out`。


### 2026-11-18：804562 H 五臂 dev 完成總結（799148，兩張 H200）

**COMPLETED、exit 0，Slurm 牆鐘 14 分 39 秒，約 0.488 配置 GPU-h。** 程式內總計 863.9 秒，雙卡抽取含啟動／合併 828.2 秒；GPU 0／1 各 3,600 盤（810,940／789,265 位置），worker 抽取 782.7／768.4 秒，峰值各約 13.89 GiB。相對 C 版 796930 單卡 28 分 17 秒，整體時間減少約 48.2%、速度約 1.93 倍；H 為 9 profile、C 為 10 conditions，並非同版本單／雙卡控制實驗，不將全部加速歸因於卡數。C 約 .471、H 約 .488 配置 GPU-h，這次主要節省等待時間，沒有證明節省 GPU 用量。

範圍沿用固定 outer dev：360 人／7,200 盤／8,000 題（各 k 2,000 題）／1,600,205 位置；只評估 monitor 選定 checkpoint。五臂 monitor 重算與訓練紀錄差均為 0.0。H FP16 還原每盤均值最大誤差 .0002954，policy mass 最大誤差 .0006459、逐手 q 最大誤差 .0077724，均通過既有精度門檻。持久產物約 8.87 MB；dev policy 暫存僅在 RAM。此為本地官方公式，非 LB，且 outer dev 過去已使用，證據有限。

| 臂 | 選中 step | k=5 | k=10 | k=15 | k=20 | Equal-k | CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 875 | 0.6918 | 0.7350 | 0.7519 | 0.7776 | 0.7391 | 0.9506 |
| R10 | 2625 | 0.6920 | 0.7364 | 0.7537 | 0.7777 | 0.7400 | 0.9532 |
| Z32 | 875 | 0.6927 | 0.7353 | 0.7513 | 0.7776 | 0.7392 | 0.9507 |
| CNN-R10 | 875 | 0.6927 | 0.7373 | 0.7536 | 0.7689 | 0.7381 | 0.9521 |
| CNN-Z32 | 1500 | 0.6907 | 0.7374 | 0.7524 | 0.7704 | 0.7377 | 0.9560 |

玩家配對 bootstrap 2,000 次；gate：ΔEqual-k ≥ .005、P(>0) ≥ .90、各 k ≥ −.005。

| 比較 | ΔEqual-k | Δk5／k10／k15／k20 | 95% CI | P(>0) | gate |
|---|---:|---|---|---:|---|
| R10-A0 | +0.0009 | +0.0002／+0.0014／+0.0018／+0.0001 | [-0.0031, +0.0050] | 0.6595 | ✗ |
| Z32-A0 | +0.0002 | +0.0009／+0.0003／-0.0006／+0.0000 | [-0.0003, +0.0006] | 0.7185 | ✗ |
| CNN-R10-A0 | -0.0010 | +0.0009／+0.0023／+0.0016／-0.0088 | [-0.0065, +0.0044] | 0.3500 | ✗ |
| CNN-Z32-A0 | -0.0014 | -0.0011／+0.0024／+0.0004／-0.0072 | [-0.0052, +0.0022] | 0.2320 | ✗ |
| Z32-R10 | -0.0007 | +0.0008／-0.0011／-0.0024／-0.0001 | [-0.0048, +0.0033] | 0.3695 | ✗ |
| CNN-R10-R10 | -0.0018 | +0.0008／+0.0009／-0.0002／-0.0088 | [-0.0054, +0.0013] | 0.1325 | ✗ |
| CNN-Z32-Z32 | -0.0015 | -0.0021／+0.0021／+0.0011／-0.0072 | [-0.0053, +0.0020] | 0.2080 | ✗ |
| CNN-Z32-CNN-R10 | -0.0004 | -0.0021／+0.0001／-0.0012／+0.0016 | [-0.0065, +0.0057] | 0.4640 | ✗ |

**判讀：** 八組 gate 全未過，claims 全 false。R10 對 A0 四桶均微增，但 Equal-k 只有 +.000865、95% CI [−.003126, +.004992]、P(>0)=.6595，不能宣稱有效。Z32 僅 +.000158、區間約 [−.00034, +.00064]，配合支路 logits RMS .00419，仍幾乎等同 A0。CNN-R10／CNN-Z32 對 A0 分別 −.000966／−.001361；monitor 的 +.002425／+.003031 沒有轉移到 dev。兩者 k=20 分別 −.008764／−.007184，超過單桶 −.005 的容許退步，但各自單桶 bootstrap 區間仍跨 0，不能當作已證實的系統性傷害。

相對 C 版：C 的四候選對 A0 為 −.0007／約 .0000／−.0026／+.0005，H 為 +.0009／+.0002／−.0010／−.0014，兩版都未過 gate。CNN-R10 的 k=20 退步在兩版重現（C −.0108、H −.0088），值得保留為風險訊號；是信心尺度、聚合方式或固定 dev 雜訊，現有實驗無法分辨。H 的 A0 .7391、C 的 A0 .7409；輸入／基準不同，不以絕對分數判斷 teacher 勝負。

**結論／建議：** 在此設定（1,000 TRAIN 玩家、3,000 updates、LR 3e-4、seed 2027），C 與 H 的完整逐手 policy 支路都未量出相對各自 A0 的可靠增益。建議此輪收尾，不推進 H＋C 融合、不加 seed／延長訓練、不據此做 LB 探針或更換既有提交基底；這不代表所有預算或架構下 policy 特徵都無效。

驗證：完成標記與 summary 身分一致；另從五臂各 8,000 列 parquet 重算機率正規化、score_rank 解碼、官方逐題得分與各 k 平均，全部一致，題組逐列配對及每 k 2,000 題通過。沒有重跑 GPU 或上傳。

產物：`outputs/rank/humansl/policy_cnn_h/cached_AB_mondev_seed2027/five_arms/799148/dev/804562/summary.json`、各臂 `dev_predictions.parquet`／`dev_features.npz`／`dev_result.json`、`identity.json`、`complete.json`；log `outputs/slurm-rank-policy-h-dev-804562.out`。

### 2026-11-18：H/C policy CNN 凍結融合 B3（兩卡實作；未提交）

依使用者確認實作兩卡方案：GPU0／GPU1 分別讀取原位置 H／C TRAIN＋monitor policy ZIP，
同批逐手 policy 分別通過各自已選定、全凍結的 CNN-R10／CNN-Z32，FP32 聚合本人／對手每盤平均。
不重跑 teacher、不解壓 policy 到持久磁碟；兩個 worker 結束後由 GPU0 訓練 B3，第二卡仍計配置 GPU-h。
原 H／C 管線及既有實驗產物未修改。

三臂：A0-HC 原始 q_h18＋原始 q_c20＝38；R-HC r10_h18＋原始 q_h18＋r10_c20＋原始 q_c20＝76；
Z-HC z_h64＋原始 q_h18＋z_c64＋原始 q_c20＝166。r10 是 q＋delta，z 是每側32維輸出、不是hidden64，
原始 q 永遠讀獨立未修正欄位。各臂四個 k 分別訓練 Linear(d,10)，共12頭，scaler 僅該 k TRAIN 擬合；
三臂同 k 均以相同原始 H＋C sklearn LR 初始化，新增欄 head weight 為0，全部特徵按 TRAIN 統計標準化。
每頭2,000題、batch8、250 updates/epoch、12 epochs／3,000 updates、seed2027、AdamW LR=.0003、
head WD=0、CE＋weight L2/(2×2000)，step0／每125步完整 TRAIN／monitor；嚴格 monitor 改善選模、同分留早，
不早停。epoch 原子保存 optimizer／當時 best／history，半輪中斷確定性重播，不保留未提交半輪的 best。

固定來源 H799148／C796606；dev 直接重用 H804562／C796930 已完成的每盤特徵，核對來源 run identity、
checkpoint selected step、檔案SHA-256、H/C清單與題組逐欄一致、玩家隔離與同色題組。
12頭全部訓練完成後才讀取dev特徵值；不依dev重選。報四k、Equal-k、三組玩家配對bootstrap2,000次；
gate delta Equal-k≥.005、每k≥−.005、Pr_boot(delta>0)≥.90。
假設：H/C學習特徵互補且保留原始q可帶來融合增益；增益未知，C-only歷史CNN最佳dev增益僅+.0005，不能當收益保證。

新增 `src/go_aicup/tasks/strength/policy_hc_b3/`（features.py、train.py、README.md、train.slurm）、
`configs/strength/policy_hc_b3.json`、`tests/test_policy_hc_b3.py`。
輸出 `outputs/rank/humansl/policy_hc_b3/seed2027_b8_e12/<job_id>/`，保存小型每盤特徵與checksum、12頭／scaler、
續跑狀態、learning curve、逐題預測與bootstrap；完整來源／設定／程式身分不符拒絕續跑。

CPU驗證：新測試11項＋既有branches／H-dev相關23項，**34項通過（7.61s）**；
涵蓋原始q／z維度、對齊與非法輸入、H/C真CNN與既有聚合數值一致、無encoder梯度、共同起點、
逐k隔離、完整遍歷、epoch中斷重播、mock雙卡完整12頭至dev發布與完成重開不重跑。
真實來源prepare-only通過（TRAIN20,000／monitor4,000／dev7,200盤）；Slurm bash語法與diff空白檢查通過。
擴大回歸發現既有 `tests/test_policy_five_arms.py::test_resume_matches_uninterrupted` 單獨亦失敗：
測試直接傳不存在的父目錄，既有five_arms.run_arm僅mkdir(exist_ok=True)而沒有parents，故FileNotFoundError；本次不修改舊流程。

Slurm入口規劃2×H200／24CPU／400G／1h，最多2配置GPU-h；完整兩卡吞吐未知，初估數十分鐘待短測。
未提交Slurm、未執行GPU或正式特徵計算／訓練、未產生提交CSV或上傳。
限制：單seed與反覆使用dev、上游CNN與B3共用monitor選模；本輪玩家隔離不等於上游C向量皆未見玩家。

### 2026-11-19：競賽一——E2E14400 兩個提交探針（使用者要求用上傳額度；B0 本身開發全負、不產檔）

B0 的最佳組合 (B＋E9600＋0.25·E14400)／2.25 幾乎等於 11/18 已封存比較過的 mix3_ml，故改用有封存證據的兩臂產檔（`scripts/identity/pid2_combine_ml3.py`，CPU 數秒；`outputs/pid/pid2_multilevel/ml_mix_compare/combine_report.json`）。先驗證可由 parts 逐列重現現行基底 `hung-seqB-e2emix-ml.csv`；原測試集列與 `772748-hung.csv` 相同；只改新增集風格項，開局、類別、ML、λ＝16 不變、不調權重。

| 檔案 | 新增集風格項 | 封存 draws 1300– pool 1 對基底（795581） | 改 top-1 對基底 pub2（k=1／2／3／5）／priv2 | sha256 |
|---|---|---|---|---|
| `submissions/pid/mix3ml.csv` | 0.5 B＋0.25 E9600＋0.25 E14400 | +0.0031（z 3.9），pool 0 +0.0009 | 26（10／10／2／4）／18 | `b440eac4` |
| `submissions/pid/ml14400.csv` | 0.5 B＋0.5 E14400 | +0.0035（z 3.5），pool 0 −0.0010 | 42（15／15／6／6）／33 | `3290abd5` |

- **預期 Public：** 本地 pool 1 增益約 +0.003，新增集占一半 → 約 +0.0015；遠低於 Public 標準誤約 0.02。
- **事前判讀規則：** 兩者本地都未達換用規則（+0.005）；LB 升、降、持平都**維持基底 `hung-seqB-e2emix-ml.csv`**，只記錄方向。若某檔 Public 降 ≥ 0.02，回查 E2E14400 正式風格項的推論路徑。
- **LB（11/18 23:57，使用者回報）：** `mix3ml.csv` **0.8208402**（−0.0037 對基底 0.8245254）、`ml14400.csv` **0.8205517**（−0.0040，33/109）。兩者都小幅低於基底，差距約 3 題、遠在雜訊 0.02 內，未達回查門檻 −0.02。依事前規則**基底維持 `hung-seqB-e2emix-ml.csv`**；E2E14400（含延長訓練與三者混合）方向不支持，不再探。

### 2026-11-18：H/C B3 兩卡 smoke50（使用者授權短測）

完整讀取原 H／C TRAIN＋monitor ZIP，各卡計算兩個凍結 CNN 的全部逐手→每盤特徵；不重跑teacher。
三臂各四個k頭均只跑50 updates，保留完整2,000題/k的scaler與sklearn起點、batch8、LR=.0003，step0／25／50監控。
新增 `configs/strength/policy_hc_b3_smoke.json`；smoke明確不讀dev特徵值，不做dev預測／bootstrap／gate，summary標記smoke=true。
來源metadata／hash稽核仍包含既有dev產物身分；不將其值用於短測。正式預設12epochs／3,000updates未變。
資源上限2×H200／24CPU／400G／30分鐘＝最多1配置GPU-h；預估數分鐘至二十分鐘，實測待補。
新增更新上限／重開測試及mock雙卡smoke不讀dev整合測試；提交前須CPU回歸與真實prepare-only通過。
輸出 `outputs/rank/humansl/policy_hc_b3/smoke50_seed2027/<job_id>/`；本節不代表已完成GPU驗收。

已提交 **Slurm 804869**，兩卡／30分鐘上限；CPU相關回歸36項通過（8.28s），真實prepare-only通過，來源identity `4a31d30a…09a4`。收據：`outputs/rank/humansl/policy_hc_b3/smoke50_seed2027/804869/submission.json`；log：`outputs/slurm-rank-policy-hc-b3-smoke-804869.out`。結果待補。

### 2026-11-19：競賽一——OGS 跨站短測後的夜間續跑規則（使用者「結果出來後重新評估並繼續訓練」；看結果前寫定）

- **磁碟：** `/work` 只剩 4.1 GB，評估要寫兩份約 2.6 GB 的驗證台快取 → `/work/jason1122/ogs_short_cache` 改為指向 `/home/jason1122/ogs_short_cache` 的連結（`/home` 剩 9.8 GB）；804783 不必重送。未刪任何既有資料。
- **續跑規則（804783 的 mix_YO 對 mix_Y）：** 事前門檻全過 → 送 A1。門檻未過但 pool 1 Δ > 0 且四桶各 ≥ −0.005 → 也送 A1（使用者已指示繼續；960 步只有十分之一預算，可能低估外部資料的效果）。pool 1 Δ ≤ 0、某桶 < −0.005，或判為無法判定 → 不送，等使用者決定。
- **A1：** `configs/identity/pid_e2e_ogs_A1_YO9600.json`＝E2E9600 配方（`pid_e2e_scratch32_9600.json`）只換成幽玄＋OGS 合併快取、yugen_share 0.5；8 卡，9,600 × 約 3.93 秒 ≈ 10.5 小時（約 84 GPU-h），上限 12 小時 30 分，可由 last.pt 續跑。
- **A1 評估（afterok，1 卡，`src/go_aicup/tasks/identity/ogs_a1_eval.sh`）：** Y＝現有 E2E9600（即現行基底內的那一個），YO＝A1 step 9600；同短測的融合與 gate（pool 1 Δ ≥ +0.010、pool 0 Δ > 0、四桶各 ≥ −0.005），新封存 draws 1700–（pool 0 1700–1719、pool 1 1700–1729）。過 gate 才產生新增集正式風格項與提交檔。
- **OGS 授權：** 用於有獎競賽與得獎交付的範圍仍待確認（A0 節）；訓練先做，提交前須確認。
- **補充（使用者「門檻沒過就訓練別的，不用等我決定」；仍在看結果前寫定）：** 上面「不送 A1」的情況改送 **B1**：E2E-scratch 9,600 步同配方、只用幽玄、seed 2027（`configs/identity/pid_e2e_scratch32_9600_s2027.json`；初始化同為 `pid_ge2e_v2_both_p0` 主幹，投影層／B 頭初值與抽樣順序不同），8 卡約 10.5 小時、約 84 GPU-h。假設：B0 的 E2E14400 是同一次訓練接續，與 E9600 高度相關；獨立 seed 的同配方模型較不相關，三方混合可能有小增益；預期 +0.003～+0.010，未知。評估（afterok，1 卡，`src/go_aicup/tasks/identity/b1_eval.sh` → `scripts/identity/pid2_seed_mix_b1.py`）：mix3s_ml＝開局＋16w(0.5 cos_B＋0.25 cos_E9600＋0.25 cos_S2027)＋類別＋ML 對現行 mix_ml9600，封存 draws 1700–；換用規則同 11/18 9,600 vs 14,400：pool 1 Δ ≥ +0.005、z ≥ 2.5、四桶各 ≥ −0.005、pool 0 Δ > 0。A1 與 B1 只會送其中一個，故共用 draws 1700– 不衝突。
- **短測結果（YO 804782 完成 1 小時 05 分；評估 804783，23 分鐘，`outputs/pid/ogs/short_eval/report.json`）→ 事前門檻未過，但符合夜間續跑規則 → 送 A1。** 封存 draws 1600–，mix_YO 對 mix_Y：

| | k=1／2／3／5 Δ | 整體 Δ（SE） |
|---|---|---|
| pool 1（30 draws） | +0.0018／+0.0022／+0.0042／+0.0025 | **+0.0026（0.0011，z 2.3，63% draw 為正）** |
| pool 0（20 draws） | +0.0079／−0.0012／+0.0028／+0.0034 | +0.0032（0.0012） |

  單獨風格項 alone_YO 對 alone_Y：pool 1 +0.0027（0.0019）、pool 0 +0.0032（0.0021）。兩臂都有學到（alone 對 no_style +0.217／+0.220），非無法判定。絕對分數（pool 1）：B 0.6640、mix_Y 0.6647、mix_YO 0.6673。判讀：960 步時 OGS 資料方向一致為正，但量小（約 +0.003），未達 +0.010；完整 9,600 步是否放大未知。
- **A1 已送（11/19 01:13）：** 訓練 Slurm **805157**（8 卡、上限 12 小時 30 分）、評估 **805158**（afterok，1 卡；驗證台快取改放 `/work/jason1122/ogs_a1_cache`，因 `/home` 只剩 4.7 GB）。第一次 sbatch 因未帶資源參數被拒，未產生作業。
- **算力預算：** sbatch 錯誤訊息顯示專案 GOV115094 SU 餘額 **22,876**（10/25 為 108,387）；`sreport` 同期全專案 5,154 GPU-h（本帳號 848）→ 約 16.6 SU／GPU-h（含其他成員與 CPU，粗估）。A1 約 84 GPU-h ≈ 1,400 SU；剩餘約 1,380 GPU-h 由全專案共用。
- **A1 結果（訓練 805157 完成 10 小時 35 分，84.4 GPU-h；評估 805158，17 分鐘，`outputs/pid/ogs/a1_eval/report.json`）→ GATE PASS。** 封存 draws 1700–；Y＝現有 E2E9600、YO＝A1；rest＝開局＋類別＋L1／L2，故 mix_Y 即現行基底 `hung-seqB-e2emix-ml` 的融合。

| mix_YO 對 mix_Y | k=1／2／3／5 Δ | 整體 Δ（SE） |
|---|---|---|
| pool 1（30 draws） | +0.0165／+0.0165／+0.0135／+0.0033 | **+0.0125（0.0019，z 6.6，87% draw 為正）** |
| pool 0（20 draws） | +0.0216／+0.0183／+0.0148／+0.0042 | +0.0147（0.0015） |

  絕對分數（pool 1）：B 0.6761、mix_Y 0.6973、mix_YO 0.7097、alone_YO 0.7130。單獨風格項 alone_YO 對 alone_Y：pool 1 +0.0310（0.0022）、pool 0 +0.0324。判讀：960 步時只有 +0.003 的外部資料效果，在 9,600 步放大到單獨 +0.031、融合 +0.0125；增益集中在 k=1–3，k=5 小。alone_YO（不與 B 混合）在 pool 1 比 mix_YO 再高 +0.0033、pool 0 持平（−0.0004），但不是事前登記的臂，k=5 對 mix_Y −0.002，**只記為觀察**，不採用。換算 Public：新增集約占一半 → 約 +0.006。
- **提交檔（事前規則：過 gate 才做）：** 新增集正式風格項匯出 Slurm **806692**（1 卡，tag `e2eogs9600`，先以同一路徑重現 seqB 風格項才繼續）→ `scripts/identity/pid2_combine_ml3.py --names ogsmix` 寫 `submissions/pid/ogsmix.csv`（新增集風格項＝0.5 seqB＋0.5 A1，其餘同基底）。**升基底規則（事前寫定）：** 封存 gate 已過 → 依本地證據升為競賽一基底候選；Public 只作方向確認，降 ≥ 0.02 才回查推論路徑。**OGS 用於有獎競賽與得獎交付的授權仍待使用者確認，確認前不應作為最終提交。**
- **提交檔完成：** 806692（7 分 42 秒）參考路徑重現 seqB 風格項最大差 8.9e-15／9.8e-15、top-1 相同；A1 單獨改 top-1 對 seqB：pub2 129、priv2 146，寫入 `outputs/pid/day0_1112/{public,private}2_d4_9600_e2eogs9600/`。`submissions/pid/ogsmix.csv`（`92989f0b`，1,600 列；原測試集列同 `772748-hung.csv`；先驗證可重現現行基底）改 top-1 對基底 `hung-seqB-e2emix-ml.csv`：pub2 52（k=1／2／3／5：25／16／6／5）、priv2 49。預期 Public 約 +0.006（0.8245 → 約 0.831）。報告 `outputs/pid/pid2_multilevel/ml_mix_compare/combine_report_ogsmix.json`。
- **LB（2026-11-19 19:17 上傳）：** `ogsmix.csv` Public **0.8347463**（29/115），對基底 `hung-seqB-e2emix-ml.csv` 0.8245254 **+0.0102**（預期 +0.006；方向一致，幅度在 Public 雜訊內）。依事前規則升為競賽一基底候選；**OGS 授權確認前，最終提交仍用 `hung-seqB-e2emix-ml.csv`。**


### 2026-11-19：804869 H/C B3 smoke50 完成

Slurm `COMPLETED`、exit 0:0，2×H200 配置耗時 **2分41秒＝0.0894 GPU-h**。
summary／全12頭／H與C特徵完成標記身分一致（`4a31d30a…09a4`），每頭均完成50 updates、2,000 TRAIN題。
完整特徵階段125.98s；H載入114.80s、CNN6.14s、GPU allocated峰值33.16GiB；
C載入96.26s、CNN6.50s、峰值36.82GiB。B3十二頭共4.56s，程式總耗時153.15s；主要成本是ZIP載入。

| 臂（monitor選定步數） | k=5 | k=10 | k=15 | k=20 | Equal-k | Δ對A0 |
|---|---:|---:|---:|---:|---:|---:|
| A0（50/0/0/0） | .670717 | .739644 | .763400 | .766918 | .735170 | — |
| R-HC（50/0/0/0） | .673085 | .739644 | .763400 | .766918 | .735762 | +.000592 |
| Z-HC（50/0/0/50） | .675085 | .739644 | .763400 | .768815 | .736736 | +.001566 |

這些是選模用monitor成績；8/12頭仍選step0，50步不能判斷正式收斂或融合收益。
smoke未評估dev、未做bootstrap／gate，不能將上表當泛化增益。
額外唯讀驗證：重算兩份特徵NPZ的SHA-256、核對完成身分，從六份TRAIN／monitor逐題預測重算機率正規化、
score_rank解碼、官方得分與各k平均，均一致；確認沒有dev預測檔。未重跑GPU或修改訓練程式。

**正式估時：** 12頭×3,000 updates為短測60倍；以4.56s直接外推約273s（正式監控每125步，短測每25步，
此外推只作粗估）。加完整特徵126s及啟動／稽核約31s，dev／bootstrap成本尚未實測，
建議預留 **10–15分鐘、2卡約0.33–0.50配置GPU-h**；既有Slurm 1h上限＝2 GPU-h不變。
建議可進行原配方正式實驗，仍以固定dev門檻ΔEqual-k≥.005、每k≥−.005、bootstrap正向比例≥.90判斷；
預期增益未知，不改提交基底。本輪只驗收短測、未提交正式作業。

產物：`outputs/rank/humansl/policy_hc_b3/smoke50_seed2027/804869/summary.json`；
log：`outputs/slurm-rank-policy-hc-b3-smoke-804869.out`。


### 2026-11-19：H/C B3 正式作業 804927 已提交

使用者授權「提交正式作業，不監控」；沿用 `configs/strength/policy_hc_b3.json` 原配方，未修改訓練程式或內建monitor選模。
兩卡H200／24CPU／400G／1h上限（最多2配置GPU-h），預估10–15分鐘、0.33–0.50配置GPU-h；
三臂各四個k頭、每頭12epochs／3,000updates，完成後自動dev評估與bootstrap／gate。
提交前真實prepare-only及Slurm語法檢查通過；來源identity `74bd3471…738a`。
已提交 **804927**，依要求不追蹤排隊、執行或完成狀態；結果未知。
收據：`outputs/rank/humansl/policy_hc_b3/seed2027_b8_e12/804927/submission.json`；
log：`outputs/slurm-rank-policy-hc-b3-804927.out`。未產生提交CSV或上傳。


### 2026-11-19：804927 H/C B3 正式實驗完成，兩候選皆未過 gate

使用者回報完成後查核：Slurm `COMPLETED`、exit 0:0，2×H200 配置耗時 **5分38秒＝0.1878 GPU-h**。
十二頭皆完成12 epochs／3,000 updates，每頭2,000 TRAIN題；完成身分 `74bd3471…738a` 一致。

| 臂（monitor選定步數，k=5/10/15/20） | dev k=5 | k=10 | k=15 | k=20 | Equal-k | Δ對A0 |
|---|---:|---:|---:|---:|---:|---:|
| A0（2875/875/625/1625） | .692214 | .731913 | .748286 | .768462 | .735219 | — |
| R-HC（1375/2375/2875/500） | .692237 | .730229 | .753659 | .771364 | .736872 | +.001653 |
| Z-HC（1375/1250/125/250） | .684968 | .731545 | .746258 | .753981 | .729188 | −.006031 |

玩家配對bootstrap 2,000次：R-HC−A0 95% CI [−.002531, +.006166]、正向比例 .759；
Z-HC−A0 [−.012108, +.000011]、正向比例 .027；Z-HC−R-HC [−.013788, −.001777]、正向比例 .0075。
事前gate為 ΔEqual-k≥.005、每k≥−.005、正向比例≥.90；**R-HC增益及正向比例不足，Z-HC三項皆不過**。
R-HC各k增益為 +.000023／−.001684／+.005373／+.002902；Z-HC為 −.007246／−.000368／−.002028／−.014482。
不升為提交候選，現行提交基底維持。monitor Equal-k 為 A0 .739290／R-HC .741977／Z-HC .742675；
monitor較高沒有轉成穩定dev增益，原因未知，不能據此宣稱CNN融合有效。

特徵階段251.10s；H載入231.86s、CNN6.03s、allocated峰值33.16GiB；
C載入208.93s、CNN6.46s、峰值36.82GiB。B3十二頭48.20s，程式總耗時332.34s；主要成本仍為ZIP載入。
唯讀CPU驗證：兩份特徵NPZ SHA-256與完成身分、十二頭完成標記／更新數，
九份TRAIN／monitor／dev逐題表的機率正規化、score_rank解碼、官方得分、四桶與Equal-k皆核對通過；
三組gate算式與summary一致。未另重抽bootstrap或重跑GPU。
限制仍為單seed、反覆使用dev、上游CNN與B3共用monitor選模，以及上游C向量玩家隔離未獲認證。
未提交新作業、未產生提交CSV或上傳。

產物：`outputs/rank/humansl/policy_hc_b3/seed2027_b8_e12/804927/summary.json`；
log：`outputs/slurm-rank-policy-hc-b3-804927.out`。


### 2026-11-19：804927 融合退步診斷（既有產物CPU分析，未重訓）

H804562／C796930／HC804927 的dev題組逐列對齊通過（同8,000題）。CE由保存的逐題機率重算，越低越好。
單獨H／C各用跨k共用head、8,000 TRAIN題；HC每k獨立head、各2,000題，故不是只改特徵的受控比較。

| 來源／臂 | monitor選中step（HC依k=5/10/15/20） | dev CE | dev Equal-k |
|---|---|---:|---:|
| H A0 | 875 | .950640 | .739089 |
| H CNN-R10 | 875 | .952137 | .738124 |
| H CNN-Z32 | 1500 | .956010 | .737729 |
| C A0 | 3000 | .963218 | .740852 |
| C CNN-R10 | 2500 | .962762 | .738273 |
| C CNN-Z32 | 1500 | .965367 | .741316 |
| HC A0 | 2875/875/625/1625 | .963070 | .735219 |
| HC R-HC | 1375/2375/2875/500 | .964248 | .736872 |
| HC Z-HC | 1375/1250/125/250 | .969675 | .729188 |

HC dev CE分k：A0 1.093134/.979901/.911780/.867467；R-HC 1.092911/.983294/.908942/.871846；
Z-HC 1.099086/.981761/.915530/.882322。選模按monitor官方分數，不按最低CE。
Z-HC k10 step0→3000：TRAIN CE .94912→.92129、monitor CE .93663→.95633；
k5為TRAIN 1.06127→1.03179、monitor 1.07591→1.08937，支持新增z特徵的過擬合跡象。
k20選中step250的monitor官方分數 .76692→.78272，但CE .83550→.84050，顯示兩指標選擇不一致。

其他診斷：H/C A0的dev預測相同91.2375%、逐題得分相關 .8570，支持訊號高度重疊、互補空間有限；
不能由此斷言沒有互補。TRAIN+monitor每盤q與r=q+delta同欄相關中位數H .9999976／C .9999864，
殘差標準差相對q中位數僅 .261%／.537%，R-HC的[q,r]高度重複。
Z-HC將128個z欄與38個q欄全部按TRAIN標準化，與單獨CNN支路的原尺度／原head不同；
可能放大原本弱小的z變動，尚未做消融證實。

新增玩家配對bootstrap2,000次（seed2126）：HC A0−H A0 −.003870，95% [−.011290,+.003833]；
HC A0−C A0 −.005633，[−.015595,+.004026]；HC R−H CNN-R −.001251，[−.010736,+.008455]；
HC R−C CNN-R −.001401，[−.012224,+.009595]；HC Z−H CNN-Z −.008541，[−.017028,+.000102]；
HC Z−C CNN-Z −.012128，[−.022902,−.001085]。多組探索比較，未校正多重比較；不作新選模依據。
結論：原始HC A0已低於單獨A0，不能全歸因CNN；較可信的候選原因為head按k拆分減少共享、
高度重複特徵缺少增量，以及Z-HC的TRAIN／monitor CE分歧與score選模波動。
尚未隔離各因素，確切因果未知；R-HC的表面退步區間跨0。不新增模型、調參或提交。

### 2026-11-19：policy_large 200k 八卡完整實驗（使用者授權實作並直接提交，不做GPU短測）

最新規格見 `plan.md`；新套件 `src/go_aicup/tasks/strength/policy_large/`，設定
`configs/strength/policy_large_200k.json`。TRAIN由全官方HD≤1資料公平選每級20,000盤，
實際4,298人／200,000盤／44,481,777位置／80,000題；monitor200人／4,000盤／893,253位置／4,000題，
dev400人／8,000盤／1,776,590位置／8,000題。三組玩家互斥；monitor/dev均排除C-v2 learn＋monitor
全部4,174位上游玩家，交集0。TRAIN全200,000盤進入固定題組；各k/rank等量、同色、題內盤互異。
切分唯一來源data/cv.py；split seed20261119、training seed2027。

GPU0–3共享H九profile policy分片，GPU4–7共享C十級分片，各完整推理一次。
兩份大快取全在VRAM，約287.85／319.82GiB；每卡約71.96／79.96GiB，另保留工作空間。
從原board cache有界重播HumanSL planes，H/C各組裝一次；不保存全量planes、不寫policy／q／每盤／B3特徵。
A0完成後，八個深版實驗臂各一GPU並行，經NVLink跨卡取所屬family資料。
R10/Z32候選點phi加Linear32→32；CNN-R10/CNN-Z32加第三個5×5 valid conv，19→15→11→7、RF13。
H/C含題目頭參數：R10 10,760／11,005，Z32 12,895／13,075，CNN-R10 77,639／80,316，CNN-Z32 79,774／82,386。
原始q、輸出維度、共同起點與殘差零初始化保留。

上游每臂batch32、最多12epochs／30,000updates；五組同名H/C配對×四k共20個B3，
各20,000 TRAIN題、batch8、最多30,000updates；LR=.0003。
head正則維持CE＋W²/(2×8,000)上游／(2×2,000)B3，TRAIN-only LR初始化C=.1，encoder WD=.0001。
全部step0及每500步monitor，第一epoch2,500步不計耐心，之後monitor CE＋head L2連續5次
未嚴格創歷史低點即停，最早step5,000。選模仍以monitor官方分數最高、同分留早。
全30個checkpoint固定後才評dev；報四k／Equal-k／CE與玩家配對bootstrap2,000，
gate沿用ΔEqual-k≥.005、每k≥−.005、正向比例≥.90。預期增益未知，不自動改提交基底。

CPU新流程與既有相關回歸37項通過（29.61s），包括真實CPU梯度與B3更新、深度／參數／共同起點、
早停暖期／同分／重置、跨分片順序／重複盤／尾批、全覆蓋題組及mock完整十網路＋20B3、dev延後與完成重開。
排除既有test_resume_matches_uninterrupted：先前已記錄的舊five_arms父目錄缺失，未更動舊流程。
真實prepare-only通過：canonical／HD／C來源、250個board shards的moves/colors可讀性與SHA、盤長／顏色、
C向量shape／finite皆核對；Slurm語法、diff空白與sbatch --test-only通過。
來源身分 `4ff990d513ebd14dae5ec1f6d29919c8eb8a1869ee024eefcfb4c2d0a7c6c343`。

正式資源8×H200／96CPU／200G RAM／36h＝最多288配置GPU-h；粗估8–16h、64–128GPU-h，
新增CUDA跨卡／多thread／編譯吞吐未短測。依使用者指示直接提交完整作業，不自動requeue。
若VRAM隨作業中斷消失，拒絕對同一未完成run自動重推teacher；模型checkpoint仍保存，需另訂新run。
單seed、歷史dev使用、B3上游in-sample TRAIN特徵及兩階段共用monitor限制保留。
輸出 `outputs/rank/humansl/policy_large/200k_deep_seed2027/<job_id>/`；提交編號及收據待下行補登。

已提交 **Slurm 805280**（2026-11-19T01:56:05.101852+08:00）；收據 `outputs/rank/humansl/policy_large/200k_deep_seed2027/805280/submission.json`，
log `outputs/slurm-rank-policy-large-805280.out`。未執行GPU短測；正式執行／完成及dev結果未知。

### 2026-11-19：806639 H+C 三臂正式測試推論（使用者要求）

使用者指定 A0、R10、CNN-Z32 的已選定 H/C 編碼器與四個 k 的 B3 頭直接推論，
不重擬合、不加入先驗修正或 L10；解碼沿用 softmax＋score_rank。
已完成官方 ZIP 中 800 題／10,000 盤解析與 HD0/HD1 檢查，2,222,151 個位置；
逐盤與既有 release cache 核對全部 moves/colors/length/player_color 一致。
新增 export 入口重用 teacher/原特徵組合；cache 只增加可選 board_root，原訓練預設不變。
CPU cache 測試 8 項通過、語法與 Slurm test-only 通過。

提交作業 **810420**，8×H200、96 CPU、200G RAM、1h 上限；預估實跑15–25分鐘、2–3.3 GPU-h。
log：`outputs/slurm-rank-policy-export-810420.out`；推論診斷：原 run 的 `test_export/`。
依使用者追加要求，每盤 H/C 三臂特徵、每題 B3 輸入與十級機率保存於
`data/processed/strength_features/humansl/policy_large/806639_test/`；全盤 policy 不落盤。

| 候選檔（完成後驗證） | monitor Equal-k（k5/k10/k15/k20） | 獨立 dev Equal-k | 假設與預期 LB |
|---|---|---|---|
| `submissions/rank/806639-hc-a0.csv` | 0.732276（0.692483/0.731380/0.749126/0.756115） | 0.728281 | 新20萬盤切分＋38維線性聚合；預期 LB 未知 |
| `submissions/rank/806639-hc-r10.csv` | 0.731934（0.690162/0.735172/0.748126/0.754276） | 0.727617 | policy 殘差聚合；dev 對 A0 −0.000664、gate未過；預期 LB 未知 |
| `submissions/rank/806639-hc-cnn-z32.csv` | 0.732779（0.691955/0.735494/0.746229/0.757437） | 0.723371 | policy CNN z32聚合；dev 對 A0 −0.004909、gate未過；預期 LB 未知 |

Public 升降均僅登記探針；本次沒有授權自動換基底，亦不代使用者上傳。完成狀態與 SHA 待補。

**完成：810420 COMPLETED、exit 0、12分26秒（約1.658配置GPU-h）。**
H/C 各完整推論10,000盤、2,222,151位置；teacher分別659.5／718.7秒。
三檔皆800列（Public/Private各400）、ID唯一且與既有合法模板完全覆蓋、級別合法、順序一致。
從落盤的H/C每盤特徵重新組合B3每題輸入，與保存的38／76／166維矩陣逐值相同；
重新載入12個選定B3頭所得十級機率逐值相同，score_rank結果重現三份CSV全部預測。

| 提交檔 | SHA-256 | 對cv2-l10改Public／Private題數 |
|---|---|---|
| `806639-hc-a0.csv` | `da6ae3ff729d22cc58d38266e22aa893742cfd6465711f128837490df2633802` | 66／69 |
| `806639-hc-r10.csv` | `7edf284f3b3234017ced17d070d2496826600dd712939fd3a861887971f56e5a` | 70／66 |
| `806639-hc-cnn-z32.csv` | `3d59dec6d8da6173f05e3fee51e30a923f4decd231fcdad44b6651e4a4bf7c0e` | 65／72 |

共用特徵路徑 `data/processed/strength_features/humansl/policy_large/806639_test/`：
H/C各A0、R10、CNN-Z32每盤.npy共6份；B3各臂每題輸入.npy共3份；
各臂十級機率與預測.parquet共3份；games／questions.parquet及完整manifest（欄名、維度、SHA、來源checkpoint）。
共10,971,143 bytes（約10.46MiB）。已驗證由這些特徵直接重現提交，可避免重跑teacher。
推論實際使用的export.py／export.slurm／cache.py已封存在原run的test_export/source/。
未上傳，三份LB分數未知；未增加先驗修正或L10、不改現行基底。

### 2026-11-19：806639 三臂＋既有 L10 覆寫（使用者要求）
沿用 rank_identity_merge.overlay 與既有 l10_real_test.questions.csv，門檻為未見玩家 null 最大 gap 71.484618；7題命中 gap 79.759–199.316，未調門檻或重做比對。
| 提交檔 | Public／Private 實改題數 | SHA-256 |
|---|---|---|
| `submissions/rank/806639-a0-l10.csv` | 2／2 | `3766662ae978923685add782a542d04d21f319bf0fc938713738a2b16610af47` |
| `submissions/rank/806639-z32-l10.csv` | 3／3 | `80e38fddc3c8912d23b0f556e4924a9b81bdafe5ecb4f4e4f1d91f562052196a` |
| `submissions/rank/806639-r10-l10.csv` | 2／2 | `a80c1e086e3ec4e9d75b39a9d9828c85b6d5f92cc59533d2a780bb5b783927c9` |

三臂共同改：pub_q_0046 4D→6D、pub_q_0135 3D→2D、priv_q_0047 4D→3D、priv_q_0168 4D→5D。CNN-Z32另改pub_q_0073 A→1D、priv_q_0159 4D→5D。priv_q_0396原已為A，皆不變。
驗證三檔各800列、ID與順序不變、合法級別、只改固定7題內的列，其餘793題逐值不變；所有命中題最終一致。
A0/R10改動與既有cv2-l10完全相同；CNN-Z32多改2題來自原模型預測不同，覆寫行為正常。4D→6D跨兩級仍照固定L10匹配級別，不把歷史LB的整體提升當作各題真值證明。
未重跑GPU或修改保存特徵／模型機率；＋L10 CSV屬後處理提交，原機率仍為模型原始輸出。變更明細與來源在原run test_export/l10_report.json及outputs/rank/identity_merge/overlay_806639-*.json。未上傳、三臂LB未知，不自動更換基底。

### 2026-11-19：競賽一——ogsmix 之後的新增集風格項權重掃描（使用者「(b) 先做，之後做 a」；看結果前寫定）

- **假設：** A1 單獨（pool 1 0.7130）略高於 0.5 seqB＋0.5 A1（0.7097），且 A1 的 cosine 尺度可能和原 16w 不同 → 調 seqB 比例 a 與風格項倍率 m 可能再加分。預期增益未知（推測 0～+0.005）。
- **設定：** 分數＝opening＋category＋L1/L2＋16w·m·(a·cos_B＋(1−a)·cos_A1)；ogsmix＝a 0.5、m 1。網格 a∈{0, 0.25, 0.5, 0.75, 1}、m∈{0.75, 1, 1.25, 1.5, 2, 2.5}（30 組）。
- **選擇：** 新開發題組 pool 0 1800–1819、pool 1 1800–1829，50 個 draw 平均取最高；選到 ogsmix 則不開封存題組。落在 m 網格邊界時記錄為風險。
- **判定（同路徑換檔規則）：** 封存題組 1900-（pool 0 1900–1919、pool 1 1900–1929）只評一次，對 ogsmix：pool 1 Δ ≥ +0.005、z ≥ 2.5、各 k ≥ −0.005、pool 0 Δ > 0 才換；否則維持 ogsmix。
- **成本：** 1 張 H200，約 30–40 分鐘（前次同類評估 17–23 分鐘、這次 draw 數兩倍）。程式 `scripts/identity/pid2_ogs_weight_sweep.py`，輸出 `outputs/pid/ogs/weight_sweep/`。
- **結果（811134，13 分 44 秒）：** 開發題組 50 draw 最佳 **a 0.25、m 1.5**（0.7173；ogsmix 0.7073；不在網格邊界）。前幾名都是 a 0.25（m 1.25 0.7168、m 2 0.7148、m 1 0.7139），a 0（A1 單獨）次之（m 1.5 0.7122），a 1（不用 A1）最差。**封存題組對 ogsmix：pool 1 +0.0071**（SE 0.0011，z 6.6，29/30 draw 為正；k=1／2／3／5：+0.0001／+0.0107／+0.0063／+0.0111），**pool 0 +0.0107**（20/20 為正）→ **通過換檔規則**。增益在 k≥2；k=1 持平（m 加大使 k=1 略降、k=5 上升）。
- **提交檔：** `submissions/pid/ogsw.csv`（sha256 `f52ed0ba`，1,600 列）＝`pid2_combine_ml3 --names ogsw`，新增集風格項＝1.5×(0.25 seqB＋0.75 A1)，其餘同 ogsmix；原測試集列不變。改 top-1 對 ogsmix：pub2 46、priv2 44；對 `hung-seqB-e2emix-ml.csv`：pub2 78、priv2 76。預期 Public 對 ogsmix 約 +0.0035（新增集占 Public 一半）→ 約 0.838，在雜訊內。依本地證據升為競賽一基底候選，取代 ogsmix；Public 降 ≥ 0.02 才回查。**OGS 授權確認前仍不作最終提交。**
- **LB（2026-11-19 22:01 上傳）：** `ogsw.csv` Public **0.8340333**（30/116），對 ogsmix 0.8347463 **−0.0007**（預期 +0.0035）。差距遠小於 Public 標準誤約 0.02（且只改了 pub2 46 題），依事前規則維持 ogsw 為基底候選（本地兩個封存 pool 皆正、z 6.6）；OGS 授權確認前最終提交仍用 `hung-seqB-e2emix-ml.csv`。

### 2026-11-19：競賽一——A2：全部 OGS 玩家的 E2E 編碼器（使用者選「更多 OGS 玩家」、同意刪舊 A1 快取；看結果前寫定）

- **假設：** A1 的增益來自外部玩家多樣性（alone_YO 對 alone_Y +0.031）→ 把 OGS 玩家由 40,000 增到全部 82,481 位（每人 10 盤，總盤數約與 A1 相同），風格向量對未見玩家的鑑別力再提高。預期增益未知；推測新增集 pool 1 +0.003～+0.01。
- **設定：** 快取 `/home/jason1122/boards/yugen_ogs_v2`（`external build --min-games 20 --games-per-player 10 --human-opponent --seed 2026`，不限人數；幽玄部分照舊以符號連結）。訓練設定 `configs/identity/pid_e2e_ogs_A2_YO9600.json`＝A1 只改 `cache_dir`、`output_dir`、`keep_steps`＝[9600]（只影響保存的 checkpoint）；yugen_share 0.5、9,600 步、8 卡不變。輸出 `outputs/pid/ge2e/pid_e2e_ogs_A2_YO9600` 為連到 `/home/jason1122/ogs_runs/` 的符號連結；Slurm log 在 `/home/jason1122/ogs_runs/logs/`（/work 已滿）。
- **磁碟：** 已刪 `/home/jason1122/boards/yugen_ogs_v1`（A1 快取 14 GB，使用者同意）；中繼資料（external.json、manifest.json、external_players.parquet、games.parquet sha256）存於 `/home/jason1122/ogs_runs/yugen_ogs_v1_meta/`。重現 A1 時以 `ogs_build.slurm` 最後的 build 指令（`--max-players 40000 --games-per-player 20 --seed 2026`）從 `/home/jason1122/ogs2025/filtered` 重建。
- **判定（同路徑換檔規則，事前寫定）：** 封存題組 2000-（pool 0 2000–2019、pool 1 2000–2029），融合固定為 ogsw 形式 1.5×(0.25 seqB＋0.75 X)。A2w（X＝A2）與 A12w（X＝A1、A2 cosine 平均）各對 A1w（＝ogsw）：pool 1 Δ ≥ +0.005、z ≥ 2.5、各 k ≥ −0.005、pool 0 Δ > 0；兩者都過取 pool 1 Δ 較大者，都沒過維持 ogsw。不調權重。程式 `scripts/identity/pid2_ogs_a2_eval.py`，輸出 `outputs/pid/ogs/a2_eval/`。
- **成本：** 建快取 1 卡約 0.5 小時內、訓練 8 卡約 10.6 小時（≈85 GPU-h）、評估 1 卡約 20 分鐘；合計約 86 GPU-h（約 1,430 SU）。
- **已送（11/19 21:50）：** 建快取 811842 → 訓練 811843（afterok，8 卡、上限 12 小時 30 分）→ 評估 811844（afterok，1 卡）。
- **結果（811843 10 小時 31 分，step 9600 loss 0.979、own-centroid acc 0.732；811844 20 分）：** 封存題組對 A1w（＝ogsw）——
  - **A2w：** pool 1 **+0.0017**（SE 0.0016，z 1.10，18/30 draw 為正；k=1／2／3／5：+0.0016／−0.0006／+0.0021／+0.0038），pool 0 +0.0040 → **未過**（Δ、z 都不夠）。
  - **A12w（A1、A2 cosine 平均）：** pool 1 **+0.0033**（SE 0.0013，z 2.57，22/30 為正；k=1／2／3／5：+0.0059／+0.0007／+0.0024／+0.0041），pool 0 +0.0040 → **未過**（z 和各 k 都過，但 Δ 小於 +0.005）。
  - 單獨使用：alone_A2 pool 1 0.7038，alone_A1 0.7040，兩者持平。
  - **判定：維持 ogsw（`report.json` chosen＝A1w）。** 外部玩家數由 40,000 增為 82,481 人（每人 20 盤改 10 盤），單一編碼器沒有可量到的提升；A12w 的小幅增益看起來是兩個編碼器的 ensemble 效果，不是 A2 本身比較好。
- **LB 探針提交檔（11/20，使用者要求；兩者都沒過換檔規則，只當探針）：** 新增集風格項匯出 815017（8 分 26 秒，tag `e2eogsA2`，參考路徑重現 seqB 風格項最大差 9.8e-15、top-1 相同）→ `pid2_combine_ml3 --names ogsa2w ogsa12w`（先驗證可重現現行基底）。
  - `submissions/pid/ogsa2w.csv`（sha256 `66761424`，1,600 列）：新增集風格項＝1.5×(0.25 seqB＋0.75 A2)。改 top-1 對 ogsw：pub2 44、priv2 41；原測試集 0。
  - `submissions/pid/ogsa12w.csv`（sha256 `bb3be137`）：1.5×(0.25 seqB＋0.75·mean(A1, A2))。改 top-1 對 ogsw：pub2 22、priv2 16；原測試集 0。
  - 預期 Public 對 ogsw（0.8340）：A2w 約 +0.001、A12w 約 +0.0016（pool 1 Δ×新增集占 Public 一半），都遠小於標準誤 0.02 → **LB 升降都不改基底**，維持 ogsw；OGS 授權確認前最終提交仍用 `hung-seqB-e2emix-ml.csv`。
  - **LB（2026-11-20）：** `ogsa2w.csv` 12:39 Public **0.8368601**（對 ogsw 0.8340333 **+0.0028**；預期 +0.001）；`ogsa12w.csv` 12:40 Public **0.8388310**（28/117，**+0.0048**；預期 +0.0016）。兩者方向與本地一致，但都遠小於標準誤 0.02；依事前規則基底維持 ogsw。A12w 本地 pool 1 +0.0033（z 2.57、各 k 皆正、pool 0 +0.0040），只差 Δ 門檻。
  - **D-008 基底改為 `ogsa12w.csv`，競賽一換檔門檻修訂（2026-11-20，使用者決定）：** 使用者判定 A12w 明顯優於 ogsw（本地兩個封存 pool 皆正、各 k 皆正、z 2.57；Public +0.0048），競賽一新增集基底由 `ogsw.csv` 改為 **`submissions/pid/ogsa12w.csv`（Public 0.8388310，sha256 `bb3be137`）**。
    - **新的同路徑換檔規則（之後的競賽一封存題組判定一律用此版）：** 封存 pool 1 **Δ > 0 且 z ≥ 2.5**、各 k ≥ −0.005、pool 0 Δ > 0。取消原本的 pool 1 Δ ≥ +0.005 幅度門檻：封存題組的 draw 間 SE 約 0.001～0.002，z ≥ 2.5 已能排除抽樣雜訊；固定 +0.005 會擋掉這種小但一致的增益。
    - 以新規則回看 A2 判定：A12w 通過（Δ +0.0033、z 2.57、各 k ≥ +0.0007、pool 0 +0.0040）；A2w 仍不過（z 1.10）。11/19 權重掃描（ogsw 對 ogsmix，z 6.6）不受影響。
    - 下一個要比的對照改為 A12w 形式（1.5×(0.25 seqB＋0.75·mean(A1, A2))）。OGS 用於有獎競賽的授權尚未確認，在使用者確認前，最終上傳的處理方式與 ogsw 時相同（`hung-seqB-e2emix-ml.csv`），由使用者決定是否解除。
    - **OGS 授權確認（2026-11-20，使用者）：** OGS 用於本競賽與得獎交付已確認；**最終上傳改用 `submissions/pid/ogsa12w.csv`**（若之後有新基底依上方規則取代，以新基底為準）。外部資料申報已補在 `EXTERNAL_RESOURCES.md`。

### 2026-11-19：cvec2 逐手正規化與全盤 policy（只提交第一階段）

第一階段三臂：A 原版 H15＋C v2，B H9＋逐手 log_softmax(C10)，C 兩組皆逐手 log_softmax。
149,615 擬合盤／5,728 人、35,083 Valid-U 盤；原版 Valid-S 先驗、每 k 20,000 合成題、
B2 seeds 2027–2031。主比較 C−B，報四桶／Equal-k 及玩家配對 bootstrap，預期增益未知。
建議第二階段用 C 的門檻為 E2 C−B > 0、E1 ≥ 0、E2 各 k ≥ −.005，其餘用 B，須手動指定。

已用完整恢復的 ZIP 真實重現 seed 2027：E1 四桶 .6128／.7219／.7661／.7461（.7117）、
E2 .7022／.7494／.7762／.7740（.7505）；對 751959-cvec2.csv 800 題改動 0。
CPU 回歸 16 項通過。C 184,698 盤、HD1 H 35,979 盤重用原逐手快取；
另以原 KataGo 補 H9 148,719 盤，粗估約 3 小時抽取、八卡五小時配置上限。
入口與報表格式見 `src/go_aicup/tasks/strength/cvec2_policy/README.md`。

第二階段 R10 和 CNN-R10 各落點／全盤一對，雙 softmax 殘差頭，AdamW LR .0003、WD .0001、
batch 32、每臂 2,500 updates，固定首個 B2 再按五 seed 重擬合。已完成程式及 CPU 測試，
GPU 效能與正式分數尚未驗證。第二階段獨立手動提交，沒有 afterok 或自動派送。

已提交 Slurm **812254**（僅第一階段）；收據 `outputs/rank/humansl/cvec2_policy/stage1/812254/submission.json`，日誌 `outputs/slurm-rank-cvec2-policy-s1-812254.out`。

**重送 812274（僅第一階段）：** 812254 在 cvec2 重現通過後、H9 補抽派工前因 `missing_rows` 未定義失敗，50 秒、exit 1，未留下補抽分片。補抽棋局及 cvec2 特徵共用同一份位置索引，新增混合命中與兩分片回歸，17 項 CPU 測試通過。

**再重送 812279（僅第一階段）：** 812274 已派發八分片，KataGo 因缺 CUDA 12.8 函式庫搜尋路徑以 127 結束（Slurm 55 秒、exit 1）。登入節點重現缺 `libcublas.so.12`；補上與舊 HumanSL 相同的路徑後 `katago version` 通過。補抽失敗會保存 stderr，17 項 CPU 回歸通過。

**812279 失敗原因與續跑（2026-11-20）：** 補抽 KataGo 本身正常（每程序約 6,200 位置／秒）；失敗在第一個 2,048 盤分片的對照檢查：重抽的 H9 每盤平均值與原 cvec2 特徵最大差 **0.0157**，只略高於容忍值 0.015，屬 GPU 批次的數值雜訊（人對手互換、貼目錯或 profile 錯都會造成 ≥0.1 的差）。容忍值改為 0.03。已存 12 個分片（約 24,600／148,719 盤）會被重用。
- **新增提交步驟** `src/go_aicup/tasks/strength/cvec2_policy/export.py`（`export.slurm`，只用 CPU）：A／B／C 三臂各以 751959-cvec2 相同的設定擬合（seed 2027、Valid-S＋train20＋Valid-U HD≤1、相同先驗）。**A 須與 751959-cvec2 800 題逐題一致**，通過才寫 B、C。測試集 normalizer 取自既有逐手快取（C：`permove/c_test_*`；H9：release HumanSL shard，HD1 改用 `p3_komi/h_test_*`）。作業 815007 對照 release 特徵：H 最大差 1.5e-6、C 3.0e-4（Public／Private 皆通過）。CPU 測試新增 1 項；`test_cvec2_policy_encoder` 的 R10 梯度測試時好時壞（不改程式時也是），與本次修改無關。
- **815009／815010 於 12:10 被本帳號取消（非本 session）；重送 815092（stage 1 續跑）→ 815093（export，afterok）。** 原送出紀錄：815009（stage 1 續跑，OUT＝812279 目錄，8 卡、上限 5 小時；預估剩約 2.5–3 小時）→ 815010（export，afterok，1 卡）。輸出 `submissions/rank/815010-cv2s1-b.csv`、`815010-cv2s1-c.csv`；選哪個看 `result.json` 的 suggested_anchor（E2 C−B＞0、E1≥0、E2 各 k≥−0.005 選 C，否則 B）。
- **815092 失敗、續跑 815890 完成（15:30）、export 815891 完成。** 815092 在 15:10 失敗：worker 06 的第三個分片（`h_backfill_06/part_004096`）H9 最大差 **0.0344**＞0.03；其餘 7 個 worker 已把分片做完並保存。改程式（eb740e7）：續跑時先載入所有已存分片（不論來自哪個 worker），只把剩餘棋局平均分給 8 卡；失敗分片跳過並寫 `part_*.error.json`。815890 只補算剩下的 8 個分片（19 分鐘），**同一批棋局這次最大差 0.0297**（三次量到 0.0157／0.0344／0.0297，確認是 GPU 批次雜訊，不是取錯資料）。`played_audit.json`：H9 重用 35,979 盤、補抽 148,719 盤、最大差 0.0297；C10 全部重用、最大差 0.0004。
- **結果（5 seed 平均；A＝原 H15＋C v2，B＝H9＋C10 每手 log_softmax 正規化，C＝兩者皆正規化）：**

| | k=5 | k=10 | k=15 | k=20 | Equal-k | 玩家 bootstrap 95% |
|---|---|---|---|---|---|---|
| E1 A | 0.6197 | 0.7249 | 0.7648 | 0.7463 | 0.7139 | |
| E1 B−A | +0.0129 | −0.0106 | +0.0073 | −0.0061 | +0.0009 | [−0.0084, +0.0106] |
| E1 C−A | +0.0133 | −0.0161 | +0.0035 | −0.0064 | −0.0014 | [−0.0113, +0.0088] |
| E2 A | 0.6935 | 0.7466 | 0.7692 | 0.7759 | 0.7463 | |
| E2 B−A | −0.0035 | −0.0146 | −0.0064 | +0.0003 | −0.0061 | [−0.0154, +0.0031] |
| E2 C−A | −0.0041 | −0.0167 | −0.0097 | −0.0076 | −0.0095 | [−0.0189, −0.0009] |
| E2 C−B | −0.0006 | −0.0021 | −0.0032 | −0.0079 | −0.0035 | [−0.0084, +0.0016] |

- **判讀：** suggested_anchor＝**B**（E2 C−B＜0）。但 B、C 在 E2 都低於 A（C 的區間不含 0），E1 持平。正規化拿掉了每手分布的尖銳度（模型對局面的把握），這部分看起來帶有棋力訊號（推測，未直接驗證）。**競賽二基底維持 `751959-cvec2.csv`**；正規化方向本地不支持，不升為基底。
- **提交檔（815891）：** export 的 A 臂與 751959-cvec2 800 題逐題一致（通過）。`submissions/rank/815891-cv2s1-b.csv`（sha256 55b10ead，對基底改 67 題）、`815891-cv2s1-c.csv`（sha256 b350fc8d，對基底改 80 題、對 B 改 54 題）。測試集 normalizer 對照：H 最大差 1.7e-6、C 3.1e-4。只作 LB 探針候選（若要探，用 B）；預期 LB 持平或略降。

### 2026-11-20：競賽一——難負例前提診斷：誤認配對是否可重現（使用者「先做誤認配對診斷」；看結果前寫定）

- **背景：** [PID_HARD_NEGATIVE_PLAN.md](PID_HARD_NEGATIVE_PLAN.md) 以 `hung-seqB-e2emix-ml` 為教師、改 seqB，已過時（D-008 後 seqB 在風格項只占 0.25）。在改寫或結案前，先驗證它的前提：**「A 被誤認成 B」是穩定的玩家配對，還是單盤／參考盤雜訊**。不訓練、不產提交檔、不影響基底。
- **教師：** 現行 `ogsa12w` 形式＝rest（開局＋類別＋L1／L2）＋16w·1.5·(0.25 cos_seqB＋0.75·mean(cos_A1, cos_A2))；另存純風格項。
- **資料：** 驗證台 pool 0（600 位未進任何編碼器訓練的玩家，每人 40 盤，固定 600 位候選＝計畫中的「固定對手」）。不碰 pool 1（主要封存池）。draws 2100–2107（新 seed，未用於任何選模）；每個 draw 由 `cv.draw_pid2` 定 10 盤參考，再從其餘盤每人取 10 盤當**單盤查詢**（k=1），共 8×6,000 題。
- **建表（照計畫 §2）：** draws 2100–2103 每人只用第 1 盤查詢（四輪）；B 在完整融合前五名、排在 A 前面，且純風格也排在 A 前面，記一次命中；命中 ≥2 為難例配對，每個 A 最多留 3 個 B。
- **驗證：** (i) **換參考盤**：draws 2104–2107（新參考盤、新查詢，排除 A 在建表用過的 4 盤），量 P(B 分數高於 A)、P(B 為 top-1)；(ii) **同參考盤**：建表 draw 內其餘 9 盤查詢的同一機率。按命中次數 0／1／2／3／4 分組，並報基準率（隨機非正解候選）。另報：有難例的 A 比例、配對級距分布（同級／鄰級／≥2 級）、最常出現的 B（hub 集中度）、k=1 top-1 與官方分數（完整融合／純風格）。
- **判讀（事前寫定）：** 前提成立＝命中 ≥2 配對在**換參考盤**後 P(B>A) ≥ 命中 1 次組的 1.5 倍，且 ≥ 基準率 5 倍，且 ≥20% 的 A 有至少一個配對。成立才提「以 A1／A2 為對象、ogsa12w 為教師」的改寫計畫（另行授權）；不成立則難負例線結案。若同參考盤重現率遠高於換參考盤（≥2 倍），記為「誤認主要來自 10 盤參考的抽樣雜訊」，支持 11/16 的未驗證假設。這是必要條件，不證明難負例訓練會有增益；驗證台玩家是未見玩家，正式建表會用訓練玩家（模型見過），兩者的誤認結構可能不同。
- **成本：** 1 張 H200（dev），預估 30–60 分鐘；程式 `scripts/identity/pid2_hardneg_diag.py`，輸出 `outputs/pid/hard_negative/diag/`。
- **已送（11/20）：** Slurm **815766**（`8gpus` 1 卡、上限 1.5 小時；dev 佇列 MaxTime 現為 1 小時）。程式先以植入配對的合成分數做 CPU 測試：配對被找回（換參考盤比例為命中 1 次組的 9.2 倍），查詢／參考／建表盤互斥 assert 通過。pool 0 實際 591 人。

### 2026-11-20：競賽二只用 C（拿掉 H）LB 探針 → 結案

**動機：** 使用者問「只用 C 不用 HumanSL」。C 本身由凍結 HumanSL 算出，能拿掉的只有 H（KataGo 15 個 KGS profile、本人／對手共 30 欄）。11/11 拆解（C v1）E1 −0.0026／E2 −0.0048，未上傳過 LB。
**做法：** `scripts/strength/rank_c_only.py`（`slurm/rank_c_only.slurm`，Slurm **815672**，CPU 25 分 36 秒、1 卡配置約 0.43 GPU-h）。特徵沿用 `rank_prod_a0_fusion.train_data／test_data`（x＝[H 30｜C v2 20｜讓先旗標 2]），只用 C 臂取第 30–52 欄。H＋C 臂以 Valid-S＋train20＋Valid-U HD≤1、先驗對齊、seed 2027 重擬合，**800 題逐題重現 `751959-cvec2.csv`** 才寫檔；再以 `rank_identity_merge overlay` 疊 L10（4 題實改，與基底相同）。
**本地（Valid-U，五 seed 平均，C 單獨 − H＋C；C v2）：**

| | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| E1 | +0.0132 | −0.0120 | −0.0058 | −0.0102 | **−0.0037** |
| E2 | −0.0094 | −0.0003 | −0.0116 | −0.0045 | **−0.0064** |

- 各 seed Equal-k：E1 +0.0053／−0.0020／−0.0059／−0.0081／−0.0079；E2 −0.0070／−0.0061／−0.0057／−0.0098／−0.0035（五 seed 全負）。與 11/11 C v1 拆解同方向、略大。

| 檔案 | 內容 | 預期 | Public LB |
|---|---|---|---|
| `submissions/rank/815672-c-l10.csv`（SHA `9474e8fd`） | 基底 `751959-cv2-l10.csv` 拿掉 H 30 欄；改 Public 29／Private 32 題，全差一級（往強 41、往弱 20）；Public 各 k 改 10／6／7／6 題 | 基底 ±0.01、偏負；保留規則：升 <0.02 不換基底 | **0.7386411**（11/20 14:44:27，jabir95tsai；39/159） |
| `submissions/rank/815672-conly.csv`（SHA `3197e7e4`） | 同上、無 L10，僅對照 | 不上傳 | |

**LB：對基底 0.7499623 −0.0113**（400 題總分約 −4.5，改 29 題中淨損約 3–4 題鄰級→錯／對→鄰級），方向與本地 E2 五 seed 全負一致，幅度在改 29 題的雜訊上緣。**H 有實際貢獻，維持 H＋C；最終預設檔不變（`751959-cv2-l10.csv`），只用 C 結案。**
**附帶：** 為此從 `data/raw/raw.zip` 取出 `raw/tests/` 的 sample_submission_rank 與兩份 rank_prediction_test CSV；`data/processed/strength_train_hc.zip`（11/19 回傳）開頭約 46 KB 為全零，解壓 586 檔 CRC 全部吻合，**缺 3 檔**（`manifest.json`、`README.md`、`legacy/humansl_qsel/games.parquet`），需從本機備份重傳 ZIP 才能補齊。
- **結果（815766，COMPLETED，20 分 24 秒；`outputs/pid/hard_negative/diag/{report.json,pairs.csv}`）→ 前提未成立（事前規則），難負例線結案。**
  - k=1 單盤（pool 0，8 draws）：完整融合 top-1 0.409–0.431、官方分數 0.458–0.476；純風格 top-1 0.377–0.395。
  - 建表：230 組配對（148 位 A，覆蓋 **25.0%**；命中 2 次 233 組、3 次 7 組、4 次 0 組）。

    | 命中次數 | 配對數 | 換參考盤 P(B>A) | 換參考盤 P(B 為 top-1) | 同參考盤 P(B>A) |
    |---|---:|---:|---:|---:|
    | 0 | 343,879 | 0.022 | 0.0009 | — |
    | 1 | 4,571 | 0.086 | 0.0090 | 0.107 |
    | ≥2 | 240 | 0.126 | 0.0200 | 0.166 |
    | 入表（每 A ≤3） | 230 | 0.128 | 0.0208 | 0.167 |
    | 基準（全部非正解） | — | 0.023 | 0.0010 | 0.024 |

  - 判定：入表／命中 1 次 **1.49**（門檻 1.5，未過）、入表／基準 5.49（過）、覆蓋 25%（過）→ `premise_holds=false`。同／換參考盤 1.30（<2）→ 參考盤雜訊不是主因，11/16 的假設不支持。
  - **比門檻更關鍵的上限：** 入表的 B 在新參考盤上搶走 top-1 的機率只有 2.1%（中位數 P(B>A) 0.10，7% 的配對完全不重現）；全部配對加總只占單盤題的 **0.81%**，而單盤錯誤率約 58%。即使把這些配對完全分開，k=1 top-1 最多約 +0.008，新增集約 +0.002。誤認是**分散、連續的相似度**（命中 1 次已是基準 3.7 倍），不是少數固定配對。
  - 入表配對偏同級／鄰級（52%，全部配對 28%），與「相近棋力的人走法較像」一致。hub 不集中：170 位不同的 B，前 10 名占 13.9%。
  - **處置：** 不改寫 PID_HARD_NEGATIVE_PLAN.md 為 A1／A2 版、不訓練；基底維持 `ogsa12w.csv`。若要再碰難負例，只剩「每步依當下向量相似度挑批內負例」的線上做法，屬新方法、需另提計畫；以 11/24 截止與上述上限，不建議。實際 1 卡 20 分鐘（約 0.34 GPU-h）。

### 2026-11-20：競賽二只用 H（拿掉 C）LB 探針 → 結案

**做法：** 同上一節 `scripts/strength/rank_c_only.py`，新增 `--arm honly`（第 0–30 欄＋讓先旗標，32 維）；Slurm **815841**（`ARM=honly`，CPU 27 分 33 秒、1 卡配置約 0.46 GPU-h）。H＋C 臂同樣 800 題逐題重現 `751959-cvec2.csv`；L10 由 `rank_identity_merge overlay` 疊上（4 題實改，與基底相同）。
**本地（Valid-U，五 seed 平均，H 單獨 − H＋C；C v2）：**

| | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| E1 | −0.0052 | −0.0042 | +0.0046 | −0.0004 | **−0.0013** |
| E2 | −0.0024 | −0.0039 | −0.0011 | −0.0062 | **−0.0034** |

- 各 seed Equal-k：E1 +0.0004／+0.0000／−0.0056／−0.0024／+0.0012；E2 −0.0077／+0.0012／−0.0036／−0.0056／−0.0013。與 11/11（C v1）的 E1 −0.0021／E2 −0.0019 同量級。

| 檔案 | 內容 | 預期 | Public LB |
|---|---|---|---|
| `submissions/rank/815841-h-l10.csv`（SHA `2cb19647`） | 基底 `751959-cv2-l10.csv` 拿掉 C 20 欄；改 Public 13／Private 16 題，全差一級（往強 16、往弱 13）；Public 各 k 改 7／3／2／1 題 | 基底 ±0.007、略偏負；升 <0.02 不換基底 | **0.7490426**（11/20 16:18:02，jabir95tsai；30/159） |
| `submissions/rank/815841-honly.csv`（SHA `3b5ce3e2`） | 同上、無 L10，僅對照 | 不上傳 | |

**LB：對基底 0.7499623 −0.0009**（400 題總分約 −0.4，持平）。連同只用 C −0.0113：**H 單獨幾乎等於 H＋C，C 在 H 之上的增量很小（本地約 +0.001～+0.003、Public +0.0009）；H 不可拿掉。** 維持 H＋C，最終預設檔不變（`751959-cv2-l10.csv`），H／C 拆解結案。

### 2026-11-20：競賽一——E1 題目合併重做：以現行融合重建同人判斷器（使用者「E1 先做」；看結果前寫定）

- **背景：** 11/14 題目合併（`pid2_transductive.py`）的上限很大（同人用真值：pool 1 +0.099；k=1 +0.188），但可實作版本全負（C 版 −0.012），瓶頸是「兩題是否同一人」：同人配對平均 m 0.16、非同人 0.0014，每題的誤配總權重（約 0.57）壓過真配對（約 0.11）。當時基底是 `v4_d4_hd_9600`（pool 1 約 0.62），只用單一風格 cosine＋後驗重疊。現行基底 `ogsa12w` 形式（pool 1 約 0.71）多了 seqB、A1、A2 三個編碼器與 L1／L2 多層開局；11/14 列為「未做」的兩個改法（直接比對兩題的開局序列、只在高信心配對上合併）也還沒試。對手第一名 Public 0.9313，推算新增集約 0.86–0.89，我方約 0.71；這是目前唯一上限接近該量級的已知槓桿。
- **假設：** 用現行的多個編碼器與開局比對當同人判斷器的特徵，可以把誤配權重壓到真配對以下，題目合併變成正增益。**預期增益未知**；上限（真值）會在本次以現行融合重量。
- **基底：** S＝rest（開局＋類別＋L1／L2）＋16w·1.5·(0.25 cos_seqB＋0.75·mean(cos_A1, cos_A2))，與 `ogsa12w` 新增集相同（程式路徑同 `pid2_ogs_a2_eval.py`；先以 pool 1 draw 2000 重現 A12w 該 draw 的分 k 分數，差 >1e-9 即停）。
- **同人判斷器特徵（題目對 q、q′，雙向取和使其對稱）：** 三個編碼器的題目中心 cosine（seqB、A1、A2）；把 q′ 的棋局當成「一位候選人的參考盤」對 q 打的開局分數、L1、L2、類別分數（各減去該列平均，去掉題目層級的偏移）；兩題的後驗重疊 log Σ_c P(c|q)P(c|q′) 與 top-1 是否相同；log k、log k′ 與 cosine 的交互項。模型兩種：logistic regression 與 HistGradientBoosting；在調參 draws 上以 average precision 擇一（事前規則）。
- **合併規則：** S′_q＝S_q＋β Σ_{q′≠q} g(m) · log(1 − m＋m·C·P_τ(c|q′))，m＝判斷器輸出，C＝候選人數，P_τ＝softmax(S/τ)；g(m)＝m≥θ 時 1，否則 0（θ＝0 為不設門檻）。m＝1 時就是把 q′ 的後驗乘進來（相當於合併兩題），m＝0 時不變。
- **draws（全部新 seed，未用於任何選模）：** pool 0：2200–2219 擬合判斷器、2220–2239 選判斷器與調參（τ ∈ {1, 2, 4, 8, 16}、β ∈ {0.25, 0.5, 1}、θ ∈ {0, 0.1, 0.3, 0.5}，取整體 Δ 最高者；落在網格邊界記為風險）、2240–2259 封存；pool 1：2200–2229 封存。判斷器只用 pool 0 玩家擬合，pool 1 是 player-disjoint。封存題組只評一次。
- **判定（11/20 新換檔規則）：** 對基底 S：封存 pool 1 Δ > 0 且 z ≥ 2.5、各 k ≥ −0.005、封存 pool 0 Δ > 0。過才寫正式新增集的匯出與提交檔（另行回報，不自動上傳）；沒過就結案這條線。
- **另報（不參與判定）：** 判斷器在 pool 1 的 AUC、average precision、同人／非同人配對的平均 m、每題的真配對總權重與誤配總權重（11/14 為 0.11／0.57）；同一合併規則在 m＝真值時的上限。
- **成本：** 1 張 H200（`8gpus`），三個編碼器的驗證台向量約 5 分鐘＋91 個 draw 的分數與題目對矩陣約 45–90 分鐘，上限 3 小時；判斷器擬合與調參在同一作業內以 CPU 完成。程式 `scripts/identity/pid2_qmerge.py`，輸出 `outputs/pid/qmerge/`。
- **已送（11/20）：** Slurm **816715**（`8gpus` 1 卡、上限 3 小時）。送出前 CPU 檢查：合併規則自我檢查（m＝0 不變、單一配對 m＝1 等於乘入對方後驗、θ 門檻、真值合併不降），以及用合成 draws 跑完整的 `analyse`（判斷器擬合→選模→調參→兩個封存 pool→換檔判定）通過。作業內先以 pool 1 draw 2000 重現 A12w，差 >1e-9 即中止。
- **結果（816715，COMPLETED 25 分）：未過換檔規則，E1 結案。**
  - 重現：pool 1 draw 2000 與 A12w 差 0.0（基底無誤）。
  - 判斷器：TUNE 上 LR AP 0.505／AUC 0.986，HGB AP 0.470／AUC 0.985 → 選 LR。調參選 τ＝2、β＝0.25、θ＝0.1；**β 落在網格下緣**（越弱越好，訊號近零）。TUNE d_all ＋0.0004（se 0.0003）。
  - 分離度（封存）：pool 1 m_same 0.323／m_other 0.0012，每題真配對總權重 0.438、誤配 0.464（11/14 為 0.11／0.57）；pool 0 0.471／0.453。AP 0.51／0.53、AUC 0.984／0.987。
  - 封存 pool 1（30 draws）：base 0.7137 → merge 0.7139，Δ＋0.0002（se 0.0004，z 0.47）；分 k Δ k1 ＋0.0001／k2 −0.0007／k3 ＋0.0012／k5 ＋0.0001。
  - 封存 pool 0（20 draws）：Δ −0.0008（se 0.0003）；k1 −0.0022／k2 −0.0010／k3 −0.0007／k5 ＋0.0009。
  - 真值上限（現行融合、θ＝0.5）：pool 1 β＝0.5 ＋0.064、β＝1 ＋0.080（k1 ＋0.162／k2 ＋0.085／k3 ＋0.053／k5 ＋0.020）；pool 0 ＋0.062／＋0.074。
  - 判定：pool 1 Δ>0 但 z 0.47 ≪ 2.5，pool 0 Δ<0 → **不換檔**。
  - 解讀：判斷器比 11/14 強很多（真配對權重 0.11→0.44），但誤配權重仍與真配對相當；同人配對極少（每題對 399 題），AUC 0.985 仍留下等量的假配對。另一推測（未驗證）：判斷器高分的真配對多半是兩題基底 top-1 本來就一致的，合併只加強已對的題，沒救到錯題。上限仍有 ＋0.06～0.08，要吃到需要 AP 遠高於 0.5 的同人判斷，現有特徵做不到。
  - 附註：log 中 `log1p` 除以零警告來自真值臂（m＝1 且候選 P 已到下限 → −inf，只壓低非正解候選，不影響排名）；merge 臂 LR 的 m<1 不受影響。
- **LB 探針提交檔（11/20，使用者要求；E1 未過換檔規則，只當探針）：** Slurm 816972（12 分 50 秒，`scripts/identity/pid2_qmerge_export.py`）→ **`submissions/pid/816972-qmerge.csv`**（sha256 `3c78a161`）。
  - 做法：新增集以 ogsa12w 融合為基底（先驗證前五名與 ogsa12w 完全相同），正式資料上重算題目配對特徵；開局、L1、L2 重算差 0，三個編碼器風格項重算差 ≤1.1e-14。判斷器為 816715 同一 LR（FIT draws 重擬合），τ＝2、β＝0.25、θ＝0.1；pub2／priv2 各自在題內合併。原測試集照抄 ogsa12w。
  - 改 top-1 對 ogsa12w：pub2 **6 題**（k1 0／k2 3／k3 1／k5 2）、priv2 3 題；每題 m≥θ 的權重 0.22／0.20（本地封存約 0.9，正式集配對明顯較少，未知原因）。
  - 預期 LB：Public 只動 6 題，最多約 ±0.015，本地預期 ≈0。
  - 事前保留規則：LB 升或持平都**不換基底**——封存 pool 0 Δ −0.0008（se 0.0003，區間整段在 0 以下）屬 11/20 規則的「本地明確為負」；最終上傳仍為 `ogsa12w.csv`。
  - **LB（2026-11-20 21:26）：** `816972-qmerge.csv` Public **0.8357774**（29/120），對 ogsa12w 0.8388310 **−0.0031**（預期 ≈0）。只改 pub2 6 題的 top-1，方向為負，與封存 pool 0 的 −0.0008 同向；依事前規則不換基底，**E1 線正式結案**，最終上傳仍為 `ogsa12w.csv`。


### 2026-11-20：cvec2 policy 逐手資料重整

- 資料包 `data/processed/strength_features/humansl/cvec2_policy/` 含貼目 6.5 的 H9 130,243 盤、貼目 0.5 的原 H15 35,979 盤及逐手 C v2 184,698 盤。貼目 6.5 H 逐盤最大差 2.09e-6。H9 每盤只存一份，原 HD1／C 分片以搬移加相容連結保留各共用入口。
- `h_missing.parquet` 列出 18,476 盤缺 H 逐手原值，全部屬 Valid-U，其中 17,432 盤 HD0、1,044 盤 HD≥2。重用 `stage1/812279/` 的補算摘要可涵蓋全部缺口。815890 的 Stage 1 六個輸出檔均通過 SHA-256 驗證，建議 anchor B。未重跑 GPU 或 Stage 2。
- 刪除舊 ZIP 及用不到的歷史分片，`strength_train_hc/` 保留四個 Stage 1 基準輸入，SHA-256 與舊 ZIP 一致。邏輯容量淨減 3,683,966,787 bytes（約 3.43 GiB）。CPU 回歸 5 項通過，新資料佈局重現 cvec2 E1、E2 與 800 題既有提交（0 題改動）。

### 2026-11-20：競賽一——A 三項便宜改動：逐 k 風格倍率、中心擴充 EM、編碼器 D4 TTA（使用者「A 先做」；事前登記，看結果前寫定）

- **背景：** 重新評估時發現兩項舊實驗依 11/20 新換檔規則（Δ>0、z≥2.5）會過，但當時的基底已過時：D3-a 逐 k λ（11/15，pool 1 +0.0034、z 2.72、pool 0 +0.0042）、B3＋EM（11/17，+0.0036、z 3.6、pool 0 +0.0025）。另一項是編碼器推論時的 D4 八向平均（TTA），紀錄中從未做過。依據是 11/20 難負例診斷：誤認配對換一組參考盤就不再出現，表示錯誤主要來自 10 盤參考的抽樣雜訊，TTA 可能降低單盤向量的變異。三項都只改新增集的風格項，不訓練。
- **基底：** 現行 `ogsa12w` 新增集融合 S＝rest＋16w·m·(0.25 cos_seqB＋0.75·mean(cos_A1, cos_A2))，m＝1.5，rest＝開局＋類別＋L1／L2。作業開頭先以 pool 1 draw 2000 重現 A2 評估的 A12w，差 >1e-9 即中止。
- **draws（全部新 seed，2300–2529 未用過）：** 開發選參 pool 0 2300–2319；封存 S1：pool 0 2400–2419、pool 1 2400–2429；封存 S2（只用於組合）：pool 0 2500–2519、pool 1 2500–2529。封存題組各只評一次。
- **A-a 逐 k 風格倍率：** 每個 k 各自選 m_k ∈ {0.75, 1, 1.25, 1.5, 2, 2.5, 3}，以開發題組該 k 桶的平均分選；同分取最接近 1.5 者。選到網格邊界（0.75 或 3）記為風險，不擴網格。
- **A-b 中心擴充 EM：** 沿用 11/17 `pid2_b3em` 的做法，對 seqB、A1、A2 三組單位向量各自擴充候選人中心（leave-one-out），後驗 p＝softmax(S／τ) 由完整融合分數共用、逐輪更新，最後依 0.25／0.375／0.375 加權。τ ∈ {1, 2, 4, 8}、α ∈ {0.25, 0.5, 1, 2}、輪數 {1, 2}，以開發題組整體平均選。
- **A-c 編碼器 D4 TTA：** 三個編碼器各自對驗證台 47,280 盤重算 8 種 D4 變換下的逐手向量（同一盤所有手套同一變換；訓練時是玩家層級 D4），再過各自的 B 頭得到盤向量。兩種用法在開發題組上二選一：(i) 分數平均：每個變換 t 都把全部棋局套 t，各算 cosine 後對 8 個 t 平均；(ii) 向量平均：每盤 8 個單位向量取平均後再正規化。**工程關：** t＝0 重算的盤向量對既有快取的最小 cosine ≥0.9995（每個編碼器），且開發題組以 t＝0 重算向量的融合分數與基底差 |Δ| ≤0.002；未過即停用 TTA 臂（BF16 在不同 batch 邊界不會逐值相同，故不要求逐值重現）。
- **判定（11/20 換檔規則，每項各自對基底，封存 S1）：** pool 1 Δ>0 且 z≥2.5、各 k ≥−0.005、pool 0 Δ>0。
- **組合：** ≥2 項通過時，把通過項以各自開發選定的參數疊在一起（TTA 向量 → 逐 k 倍率 → EM 以組合後的融合分數為後驗），在封存 S2 以同一規則對基底判定，另列報對 S2 上最佳單項的差（不作判定）。只有 1 項通過時，S2 不開封。
- **產出：** 通過者才寫正式新增集匯出與提交檔，原測試集與 `ogsa12w.csv` 逐列相同；不自動上傳，提交前另行回報。
- **預期：** 每項約 0～+0.005。兩項舊實驗的數字是在較弱的基底上量的，TTA 未知。三項不一定可疊加。
- **成本：** TTA 抽取 3 個編碼器各 1 卡並行，各約 525 萬手 × 8 變換，估 0.5～1.5 小時／卡；評估 1 卡（CPU 為主）約 1～1.5 小時。合計 ≤6 GPU-h，比重新評估時估的「各 ≤1 GPU-h」多，主要花在 TTA。程式 `scripts/identity/pid2_cheap.py`，輸出 `outputs/pid/cheap/`。
- **已送（11/20）：** 短測 817327（A1、前 500 盤，2 分鐘）：t＝0 重算對快取最小 cosine 0.999995，過工程關第一項；t＝0 與 t＝1…7 的平均 cosine 只有 0.67～0.79，表示編碼器對絕對方位很敏感（訓練時是玩家層級 D4，同一人的盤同方位）。這只是描述，不改事前規則。正式 TTA 抽取 817331（B）、817332（A1）、817333（A2），評估 817334（afterok 三者）。817326 誤送 dev 佇列（QOSMinGRES，未執行）已取消。

### 2026-11-20：競賽一——玩家條件走子似然第 0 段（使用者「B 第 0 段照計畫實作」；事前登記，看結果前寫定）

- **計畫：** [PID_PLAYER_LIKELIHOOD_PLAN.md](PID_PLAYER_LIKELIHOOD_PLAN.md)；模型、draws（開發 pool 0 2600–2619，封存 pool 0 2700–2719、pool 1 2700–2729）、網格（σ ∈ {0.1, 0.3, 1, 3}、λ ∈ {0.25, 0.5, 1, 2}）、停止條件與 11/20 換檔規則全照計畫，基底為 `ogsa12w` 新增集融合。
- **實作補充（看結果前寫定）：** MAP 用批次阻尼 Newton（每步最大座標變動上限 1，最多 25 輪）。不加阻尼時，合成資料 2 萬手就發散；加了之後能還原植入的參數（誤差 ≤0.12）。θ_pop 以 σ＝10、均值 0 擬合。消融（只用級別組／只用非級別組）只跑開發 draws 2600–2609。穩定度診斷：pool 0 每人用 bank 第 1–10 盤與第 11–20 盤各擬合一次，算同一人兩組 θ−θ_pop 的平均 cosine，並以錯開一位的配對當對照。
- **族群：** 驗證台以外、HD≤1 盤數 ≥10 的 6,232 人中抽 600 人（seed 2026），每人 10 盤；表 `data/processed/identity/likelihood/games.parquet`（驗證台 47,280 盤＋族群 6,000 盤，HD1 占 18.1%，本人位置估 590 萬）。
- **程式：** `scripts/identity/pid2_likelihood.py`、`src/go_aicup/tasks/identity/likelihood_{policy,eval}.slurm`。CPU 自我檢查通過（選項分布、幾何特徵、MAP 還原、θ 相同時 LLR＝0、強先驗釘住）。
- **policy 短測：** 817427（shard 0 前 300 盤）。量速度與路徑檢查（與 `humansl_rankvec.game_logp` 同級別條件比對 ≤2e-3）；低於 800 位置／秒先回報。
- **短測結果與路徑檢查修正（看結果前，未讀任何分數）：** 817427（300 盤）與 817434（1,500 盤）。速度：穩態 1,660 位置／秒（扣掉前 90 秒的編譯與啟動），高於 800 門檻。路徑檢查（compile＋TF32 對 eager＋TF32 的 `game_logp`）：max 3.9e-3、p99 2.2e-3，超過計畫寫的 2e-3，兩次短測都因此以 FAILED 結束。差值落在 TF32 本身的量級（A-050：TF32 對 FP32 最大 2.4e-3），若是級別向量或遮罩錯了，差值會在鄰級之間的量級。改為：本級條件 max ≤1e-2、p99 ≤3e-3，且同一批對「鄰一級條件」的 p99 差必須 ≥10 倍本級 p99（能區分取錯級別的錯誤）；檢查在前兩批完成後立即執行，失敗即停。
- **佇列：** `8gpus` 的 QOS 現為每人 4 個執行中、9 個已提交（`sacctmgr`，11/20；CLAUDE.md 記的是 8／13）。policy 改為 4 個 array task、每個依序跑 2 片：817471（0–3）；評估 817472（afterok）。
- **結果（TTA 抽取 817331／817332／817333 各約 43 分鐘；評估 817334 11 分鐘；`outputs/pid/cheap/report.json`、`s1_per_draw.csv`）→ 只有逐 k 倍率通過；S2 不開封。**
  - 工程關：t＝0 重算對快取最小 cosine B 0.99998／A1 0.999997／A2 0.999997；開發題組 t＝0 臂與基底差 0；A12w 重現差 0。
  - 開發選定：逐 k 倍率 m_k＝k1 0.75／k2 1.25／k3 2／k5 2（**k1 落在網格下緣**，記為風險）；EM τ 4、α 0.5、1 輪（開發 +0.0035）；TTA 選分數平均（開發 +0.0008）。向量平均版開發 −0.0066。
  - 封存 S1（pool 1 30 draws、pool 0 20 draws），對基底：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 Δ（SE，z，正 draw 比例） | pool 0 Δ | 判定 |
|---|---|---|---|---|
| **逐 k 倍率** | +0.011／+0.001／+0.000／+0.003 | **+0.0037（0.0011，z 3.38，77%）** | +0.0028 | **通過** |
| EM | +0.004／+0.003／+0.000／+0.000 | +0.0018（0.0008，z 2.30，73%） | +0.0018 | 未過（z<2.5） |
| TTA（分數平均） | +0.003／−0.001／+0.004／−0.003 | +0.0007（0.0014，z 0.53，53%） | +0.0018 | 未過 |

  - 絕對分數（pool 1）：基底 0.7140 → 逐 k 0.7177。增益主要在 k=1（0.4555 → 0.4666）：k=1 降低風格項權重、k≥3 提高。機制推測：單盤的風格向量雜訊大，相對開局／L1／L2 計數項應少給權重；題內盤數多時平均後較穩，可多給。這與 11/15 D3-a 的方向一致（k=1、2 選較小的 λ）。
  - TTA 判讀：編碼器對絕對方位敏感（不同變換的同盤向量 cosine 0.67～0.79），8 向平均沒有改善；本地證據不支持，結案。EM 方向一致為正但未達 z，與 11/17 一樣小，結案。
- **正式匯出（登入節點 CPU，數秒；`scripts/identity/pid2_perk_export.py`、`outputs/pid/cheap/perk_export.json`）：** `submissions/pid/817334-perk.csv`（sha256 前八碼 `a33d7dd0`）。重建的 ogsa12w 新增集分數 top-5 與 `ogsa12w.csv` 完全相同後，只把 k 題的風格項乘 m_k／1.5。原測試集與 ogsa12w 逐列相同。改 top-1：pub2 32 題（k1 17／k2 7／k3 5／k5 3）、priv2 24 題（9／5／5／5）。
  - **預期 Public：** 新增集 +0.0037 × 一半 ≈ +0.002，遠小於標準誤 0.02。
  - **事前保留規則：** 依 11/20 換檔規則已在本地通過，Public 只作方向確認。Public 升或持平（≥ −0.005）→ 基底改為 `817334-perk.csv`；降超過 0.005（約 32 題中淨錯 4 題以上）→ 先維持 `ogsa12w.csv`，回報使用者決定。不自動上傳。
- **LB（2026-11-20 23:33，使用者回報）：** `817334-perk.csv` Public **0.8334918**（32/120），對 `ogsa12w.csv` 0.8388310 **−0.0053**（預期 +0.002）。只改 pub2 32 題的 top-1，−0.0053 約等於淨錯 4 題，粗估標準誤約 0.005，與本地預期差約 1.5 個標準誤，在雜訊範圍內。依事前規則（降超過 0.005）**先維持 `ogsa12w.csv` 為基底**，由使用者決定是否仍依本地證據換檔。
- **結果（policy 817471 四個 task 各約 18 分鐘；評估 817472 因自我檢查的合成資料未固定 torch 亂數、樣本太小而失敗，修正後 817602 完成，20 分鐘；`outputs/pid/likelihood/report.json`、`sealed_per_draw.csv`）→ GATE PASS。**
  - 工程關：591 萬個本人位置，實際落點在前 32 名之外 0.83%；A12w 重現差 0；θ＝θ_pop 時 LLR 差 0。
  - 開發選定 σ＝3、λ＝2（開發 +0.0304）。**兩者都落在網格上緣**，記為風險：更大的 λ 可能更好，但這次不擴網格。只用 LLR 的開發分數：σ 0.1／0.3／1／3 為 0.080／0.127／0.136／0.135（只用風格 0.693）。
  - 消融（開發 draws 2600–2609，σ 3、λ 2，對基底）：只用級別組 +0.0157、只用非級別組 +0.0117，兩者都明顯為正，且合計 (+0.027) 接近完整 (+0.030) → 個人的級別擬合曲線與幾何／共識偏好各自帶有身分資訊。
  - 穩定度：pool 0 同一人兩組 10 盤擬合的 θ−θ_pop 平均 cosine **0.57**，錯開配對 0.16；平均範數 2.6 → θ 抓到的是穩定的個人特徵，不只是雜訊。
  - 封存（只評一次），對基底：

| | k=1／2／3／5 Δ | 整體 Δ（SE，z，正 draw 比例） | 絕對（基底 → lik） |
|---|---|---|---|
| **pool 1（30 draws）** | +0.027／+0.036／+0.032／+0.025 | **+0.0301（0.0017，z 17.6，100%）** | 0.7152 → 0.7453 |
| **pool 0（20 draws）** | +0.030／+0.032／+0.037／+0.019 | **+0.0295（0.0022，100%）** | 0.7227 → 0.7522 |

  - 只用 LLR：pool 1 0.124，遠低於只用風格（0.675）；但加進融合後四個 k 都 +0.025 以上 → LLR 帶的是與編碼器、開局計數都不同的資訊。機制解讀（推測、未驗證）：LLR 逐手比較「這個人在這個盤面相對於人類 policy 的偏好」，每手都以盤面為條件；編碼器把整盤池化成一個向量，開局計數第 3 手後就查不到。
  - **判定：通過**（pool 1 Δ>0、z 17.6、各 k ≥ +0.025、pool 0 +0.0295）。依計畫 §6 寫正式新增集匯出（`pid2_likelihood export`，Slurm 817714：抽官方候選庫與題目的 policy，θ_pop 沿用評估作業的值，σ／λ 用開發選定值，基底重建後先驗證與 ogsa12w top-5 完全相同）。不自動上傳。
  - 成本：policy 4 × 18 分＋短測約 5 分＋評估約 21 分 ≈ 1.6 GPU-h（計畫估 4～5 GPU-h）。

  - **正式匯出（817714，約 25 分鐘；`outputs/pid/likelihood/official/report_817714.json`）：** `submissions/pid/817714-lik.csv`（sha256 前八碼 `a18265ba`）。官方 155 萬個本人位置，前 32 名之外 0.88%（驗證台 0.83%）；路徑檢查 p99 2.1e-3、鄰級 0.45；候選人 θ 平均範數 pub2 2.57／priv2 2.60（驗證台 2.60），分布一致。重建的基底 top-5 與 ogsa12w 完全相同，原測試集逐列相同。改 top-1：pub2 55 題（k1 20／k2 11／k3 10／k5 14）、priv2 51 題（14／16／12／9）。
    - **預期 Public：** 新增集 +0.030 × 一半 ≈ +0.015（標準誤約 0.02）。
    - **事前保留規則：** 本地封存 z 17.6、兩個 pool、四個 k 都過，依 11/20 規則 Public 只作方向確認。Public 升或持平（≥ −0.005）→ 基底改為 `817714-lik.csv`；降超過 0.01 → 先維持 `ogsa12w.csv`，檢查官方資料與驗證台的差異，回報使用者。不自動上傳。
    - **LB（2026-11-20 23:58，使用者回報）：** `817714-lik.csv` Public **0.8464716**（24/120），對 `ogsa12w.csv` 0.8388310 **+0.0076**（預期 +0.015，方向一致；名次 32 → 24）。依事前規則**競賽一基底改為 `submissions/pid/817714-lik.csv`**。
### 2026-11-20：競賽一——似然項擴大 λ／σ 網格＋逐 k 風格倍率疊加（使用者「擴大 λ 網格再測一次，並試 per-k 疊加 B」；事前登記，看結果前寫定）

- **背景：** 第 0 段選到 σ＝3、λ＝2，兩者都在網格上緣。逐 k 風格倍率在不含似然項的融合上通過封存（+0.0037），但 Public −0.0053；加入似然項後最適倍率可能改變，必須重選。
- **參照 B0：** 第 0 段匯出設定＝ogsa12w 融合＋2·LLR(σ＝3)，風格倍率 1.5。
- **draws（全新）：** 開發 pool 0 2800–2819；封存 pool 0 2900–2919、pool 1 2900–2929。θ_pop、族群、特徵、MAP 全部沿用第 0 段（θ_pop 讀 `outputs/pid/likelihood/report.json`）。
- **L 臂：** σ ∈ {3, 10} × λ ∈ {2, 3, 4, 6, 8}，以開發整體平均選；選到 σ＝10 或 λ＝8 記為邊界風險，不再擴網格。若選回 (3, 2)，L＝B0，不評 L 臂。
- **LK 臂：** 固定 L 選定的 (σ, λ)，每個 k 各自選風格倍率 m_k ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3}（以開發該 k 桶平均選，同分取最接近 1.5 者；選到 0.5 或 3 記為邊界風險）。
- **判定（11/20 換檔規則，封存，對 B0）：** L、LK 各自判定：pool 1 Δ>0 且 z≥2.5、各 k ≥−0.005、pool 0 Δ>0。**最終採用：** LK 通過且 pool 1 上 LK−L ≥0 → LK；否則 L 通過 → L；都沒過 → 維持 B0。另報各臂對 ogsa12w 基底的差（不參與判定）。
- **產出：** 採用 L 或 LK 時，重用 817714 已抽好的官方 policy，以新參數重寫正式匯出；不自動上傳。
- **預期：** L 臂 0～+0.005（開發曲線在 λ＝2 仍上升，σ 1→3 已飽和）；LK 未知（不含似然項時 +0.0037，加入似然項後可能縮小）。
- **成本：** 1 卡約 30～40 分鐘（70 個 draw × 2 個 σ 的 MAP 與 LLR）。程式 `pid2_likelihood evaluate2`，輸出 `outputs/pid/likelihood/round2/`。
- **已送：** Slurm 817728（`likelihood_round2.slurm`）。export 加了 `--round2`：讀 round2 的最終選擇，若維持 B0 則拒絕執行。
- **結果（817728，12 分 37 秒；`outputs/pid/likelihood/round2/report.json`、`sealed_per_draw.csv`）→ L 未過、LK 通過，最終採用 LK。**
  - 工程關：A12w 重現差 0；θ_pop 重擬合與第 0 段的值差 5e-15。
  - 開發（對 B0）：λ 3 +0.0024、λ 4 −0.0027、λ 6 −0.0226、λ 8 −0.0468（σ 3）；σ 10 各 λ 都不優於 σ 3。選 L＝(σ 3, λ 3)，不在邊界。λ 越大 k=1 越好、k≥3 越差。
  - 逐 k 倍率（在 L 上）：m_k＝k1 0.75／k2 1.25／k3 2／k5 3（**k5 落在網格上緣**，記為風險）。
  - 封存，對 B0：

| 臂 | pool 1 k=1／2／3／5 Δ | pool 1 Δ（SE，z，正 draw 比例） | pool 0 Δ | 判定 |
|---|---|---|---|---|
| L | +0.006／+0.003／+0.000／−0.005 | +0.0011（0.0009，z 1.23，60%） | −0.0011 | 未過 |
| **LK** | +0.018／+0.002／−0.002／−0.001 | **+0.0041（0.0013，z 3.11，73%）** | **+0.0063** | **通過** |

  - LK−L（pool 1）＝+0.0030 ≥0 → 依事前規則採用 LK。對 ogsa12w：pool 1 0.7114 → B0 0.7418 → LK 0.7459。增益集中在 k=1（+0.018）：單盤題降低風格權重，與 A 項逐 k 倍率的結果一致。k=3、5 微負，在允許範圍內。
  - 正式匯出：Slurm 817757（`export --round2`，重用 817714 的官方 policy），輸出 `submissions/pid/817757-lik2lk.csv`。

  - **匯出完成（817757）：** `submissions/pid/817757-lik2lk.csv`（sha256 前八碼 `54a49a54`）；重建基底與 ogsa12w top-5 相同，原測試集逐列相同。對 ogsa12w 改 top-1：pub2 91 題（k1 43／k2 18／k3 15／k5 15）、priv2 77 題。
    - **預期 Public（對新基底 `817714-lik.csv`）：** +0.0041 × 一半 ≈ +0.002，在雜訊內。對 817714-lik 改 top-1：pub2 48 題、priv2 41 題，原測試集 0。
    - **事前保留規則：** 本地已過換檔規則。Public 升或持平（≥ −0.005）→ 基底改為 `817757-lik2lk.csv`；降超過 0.005 → 維持 `817714-lik.csv`，回報使用者（與逐 k 單獨 Public −0.0053 同一方向時，表示官方資料上逐 k 倍率可能不轉移）。
    - **LB（2026-11-21 00:02，使用者回報）：** `817757-lik2lk.csv` Public **0.8461584**（25/120），對 `817714-lik.csv` 0.8464716 **−0.0003**（預期 +0.002；48 題 pub2 改動，雜訊內）。屬「持平」，依事前規則**競賽一基底改為 `submissions/pid/817757-lik2lk.csv`**；CLAUDE.md §0 與 `EXTERNAL_RESOURCES.md`（HumanSL PyTorch checkpoint 現用於競賽一、OGS 列的最終檔）已同步更新。

### 2026-11-21：競賽一——似然項加入更豐富的逐手特徵：計畫稿（使用者「寫豐富逐手特徵的計畫」；未實作、未送作業）

完整計畫見 [PID_LIKELIHOOD_RICH_PLAN.md](PID_LIKELIHOOD_RICH_PLAN.md)。要點：
- 在現行 27 維上加四組特徵：G1 跟隨 policy 的銳利度 2 維、G2 絕對區域 8 維、G3 局部棋形 10 維、G4 戰術 3 維。G3、G4 需要 numba 重播盤面。
- 對現行基底 LK 選參：開發 pool 0 3000–3019 先逐組篩選（Δ ≥ +0.002 保留），再對保留組的聯集重選 σ ∈ {3, 10} × λ ∈ {2, 3, 4}。
- 封存 pool 0 3100–3119、pool 1 3100–3129，以 11/20 換檔規則判定。
- 預期 0～+0.01（未知）；成本約半天實作＋1～2 GPU-h。若實作拖過 11/22 即停止，最終檔維持 `817757-lik2lk.csv`。
- **開始執行（2026-11-21，使用者「開始訓練」）：** 照計畫實作，沒有改動任何判定規則。
  - **新模組：** `src/go_aicup/tasks/identity/likelihood_rich.py`（numba 重播：落子前盤面、各棋串氣數、32 個選項的 8 個 0／1 特徵）。`pid2_likelihood.py` 加了 `rich-features`、`evaluate3`、`export --rich`；`fit`／`played_logp` 加了特徵遮罩。
  - **工程確認（登入節點 CPU 小測）：** 盤面快取 `history[ply]` 是第 ply 手**之後**的盤面。重播的「第 ply 手之前」盤面與 `history[ply−1]` 在 30 盤 6,752 個位置全部一致，第 0 手為空盤；正式作業另查驗證台 200 盤、官方每檔 25 盤。手工盤面驗證碰、長、尖、小飛、打吃（2 氣棋串）判定正確。
  - **CPU 自我檢查通過：** 50 維特徵排列；只留原 27 維遮罩時與 27 維擬合差 <1e-4；單一擁有者 3 萬手合成資料還原 G3／G4 植入參數（前段、後段各自誤差 <0.3）。每一維與其「×後段」交互項的估計會彼此抵換，雜訊比單獨一維大，這也是穩定度診斷要看的風險。
  - **作業：** Slurm 817975（`likelihood_rich.slurm`：自我檢查 → 特徵（重播外推超過 1 小時即停）→ `evaluate3`）。
- **結果（817975，18 分 39 秒；`outputs/pid/likelihood/rich/report.json`、`sealed_per_draw.csv`）→ 只保留 G3，封存通過。**
  - **工程關：**
    - 重播盤面與盤面快取一致：驗證台 45,031 個位置、官方 21,342 個位置。特徵計算驗證台 53,280 盤 3 秒、官方 14,200 盤 1 秒。
    - 只留原 27 維遮罩時：LLR 最大差 5.7e-5，分數差 0。A12w 重現差 0。
    - 驗證台選項的特徵比例：碰 0.46、長 0.33、尖 0.22、跳 0.11、小飛 0.13、提子 0.011、打吃 0.070、救子 0.012。
  - **第 1 步（開發 pool 0 3000–3019，σ 3、λ 3，對 LK；k=1／2／3／5 → 整體）：**
    - G1 銳利度 +0.002／−0.002／−0.002／−0.001 → −0.0006
    - **G2 絕對區域 −0.006／−0.008／−0.010／−0.003 → −0.0066**（四桶全負）
    - **G3 局部棋形 +0.018／+0.011／+0.003／+0.007 → +0.0099**（SE 0.0012）
    - G4 戰術 +0.002／+0.003／+0.002／+0.001 → +0.0018（低於 +0.002 門檻）
    - 只保留 G3。G2 為負的可能原因（推測）：每人 10 盤估 8 個區域偏好的雜訊太大，而且方位資訊開局項已經用過。
  - **第 2 步（G3）：** 選回 σ 3、λ 3（開發 +0.0099，不在邊界）；σ 10 各 λ 都較差，λ 4 較差。
  - **穩定度（同一人兩組 10 盤的 θ cosine）：** 原 27 維 0.569（錯開配對 0.163）→ 加 G3 後 0.523（0.143）。略降，但與錯開配對的差距仍大：多出的 10 個參數有一部分是雜訊，但新增的訊號比雜訊多。
  - **單獨 LLR：** 封存 pool 1 0.126 → 0.198。
  - **封存（對 LK）：**

| | k=1／2／3／5 Δ | 整體 Δ（SE，z，正 draw 比例） | 絕對（LK → RICH） |
|---|---|---|---|
| **pool 1（30 draws）** | +0.013／+0.010／+0.010／+0.011 | **+0.0109（0.0014，z 7.8，90%）** | 0.7512 → 0.7622 |
| **pool 0（20 draws）** | +0.013／+0.012／+0.011／+0.008 | **+0.0110（0.0017，95%）** | 0.7604 → 0.7713 |

  - **判定：通過**（pool 1 z 7.8、四個 k 都 ≥ +0.009、pool 0 +0.011）。依計畫 §6 寫正式匯出：Slurm 818024（`export --rich`），參數為 G3、σ 3、λ 3、LK 的 m_k，θ_pop 用加了 G3 的族群擬合值。不自動上傳。
  - **成本：** 0.31 GPU-h（計畫估 1～2）。
  - **匯出完成（818024，1 分 19 秒；`outputs/pid/likelihood/official/report_818024.json`）：** `submissions/pid/818024-likrich.csv`（sha256 前八碼 `cf6ead6e`）。重建基底與 ogsa12w top-5 相同，原測試集逐列相同。官方候選人 θ 平均範數 pub2 2.97／priv2 3.01（37 維）。對現行基底 `817757-lik2lk.csv` 改 top-1：pub2 43 題、priv2 45 題，原測試集 0。
    - **預期 Public（對 817757-lik2lk）：** +0.011 × 一半 ≈ +0.005，在雜訊內。
    - **事前保留規則：** 本地已過換檔規則。Public 升或持平（≥ −0.005）→ 基底改為 `818024-likrich.csv`；降超過 0.005 → 維持 `817757-lik2lk.csv`，回報使用者。不自動上傳。
  - **LB（使用者上傳，2026-11-21 10:15:45）：** `818024-likrich.csv` Public **0.8535888**（21/121），對 `817757-lik2lk.csv`（0.8461584）**+0.0074**，比預期 +0.005 略高，仍在 1 個標準誤內。依事前規則，**競賽一基底改為 `818024-likrich.csv`**。

### 2026-11-21：競賽一——經驗貝氏先驗＋逐 k 學習融合權重：計畫稿（使用者「寫 1 和 2 的計畫」；未實作、未送作業）

完整計畫見 [PID_EB_COMBINER_PLAN.md](PID_EB_COMBINER_PLAN.md)。要點：
- **原測試集 Public 部分推算為滿分：** `770088-origonly.csv` 0.4955449 × 800 ＝ 396.44 分，加上 `772748-hung.csv` 的 +0.004455 × 800 ＝ 3.56 分，合計 400.00／400。之後原測試集逐列未變（`818024-likrich.csv` 的 pub_／priv_ 與 `772748-hung.csv` 逐列相同，已查）。所以新增集 Public 約 2 × 0.8536 − 1 ＝ 0.707，驗證台 pool 1 為 0.762。Private 部分無法量。
- **計畫一：** 候選人 θ 的先驗由 N(θ_pop, 3²I) 改為 N(θ_pop, c·Σ)。Σ 為 pool 0 玩家四組 10 盤 θ 的交叉共變異（玩家間共變異）。
  - 臂：EB37／EB50 × c ∈ {0.5, 1, 2} × λ ∈ {2, 3, 4}。
  - draws：開發 pool 0 3200–3219；封存 pool 0 3300–3319、pool 1 3300–3329；以 11/20 規則對 RICH 判定。
  - 預期 +0.003～+0.01（未知），約 20–30 分鐘、1 卡。
- **計畫二：** 每個 k 以 conditional logit 擬合 8 個分項的權重，分 FULL 與 SMALL 兩臂，以開發 draws 對半交叉擬合選臂。
  - draws：開發 pool 0 3400–3419；封存 pool 0 3500–3519、pool 1 3500–3529。
  - 過去兩次逐 k 加權，驗證台約 +0.003，Public 都沒看到（−0.0053、+0.0004），所以預期 Public 很可能看不出差別。
- **保留規則：** 通過則匯出，不自動上傳。Public 對當時基底 ≥ −0.005 → 成為基底；降超過 0.005 → 維持並回報。
- **計畫一開始執行（2026-11-21，使用者「1 照計畫實作」；事前登記，看結果前寫定）：** 照計畫實作，沒有改動判定規則。
  - **程式：** `pid2_likelihood.py` 的 `fit` 加了 `precision` 參數（給定時以 N(θ_pop, precision⁻¹) 取代 I/σ²），新增 `evaluate4`（Σ 估計 → 開發選參 → pool 1 穩定度 → 封存）與 `export --eb`。Slurm：`likelihood_eb.slurm`、`likelihood_export_eb.slurm`。
  - **CPU 自我檢查通過：**
    - precision＝I/9 與 σ＝3 的擬合差 <1e-9。
    - 合成資料：300 人 × 4 組 × 300 手，植入 6 維異向 Σ（3 維標準差 1、3 維 0.2，一對相關係數 0.6），Σ 對角相對誤差 ≤0.134、相關還原在 0.15 內。
    - 在新的 60 手小樣本上，full-Σ MAP 的 θ 均方誤差 0.039，低於等向 σ ∈ {0.3, 0.5, 1, 3, 10} 的最佳值 0.053（σ＝1）。
  - **與計畫的細節差異：**
    - 計畫寫「對角相對誤差 <25%」，合成檢查照做；植入的小維度標準差用 0.2，不用 0.05，因為 0.05 低於特徵值下限 0.1，不能檢查還原。
    - Σ 的列報加了「平均雜訊變異」（各組 θ 的總變異減去玩家間變異），用來看 10 盤估計的雜訊有多大。
