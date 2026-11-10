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
- `ahc-p3-noA-t20-allp_733179_rank.csv`（SHA `bd14b596…`）：Valid-S＋train20＋Valid-U；對 noA-allp 改 40／49 題。**LB 11/09 16:57:42（jabir95tsai）Public 0.7404805（15/103），對 noA-allp +0.0018**（40 題不同，淨多約 0.7 分，雜訊範圍）；目前 Public 最高分。
- `ahc-p3-noA-t20_733179_rank.csv`（SHA `695140f6…`）：Valid-S＋train20；對 noA-allp 改 66／59 題（對 noA 76／80）。
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
