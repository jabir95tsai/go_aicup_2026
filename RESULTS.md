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
| SE3 十級 CE，十五位置 | 第 45,000 步 early stopping，發布第 30,000 步最佳監控模型並完成全量 validation 評估 | **Equal-k 0.623873**（固定 10,000 題，game-disjoint）；單盤 0.415296，模型對應與來源見文末補登 |
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
| SE3 十級 CE 三頭 ResNet | 18 blocks／256 channels；strength 十級 CE＋policy CE＋teacher value MSE；AdamW、每盤抽十五位置 | 盤內平均 log_softmax→題內各盤等權平均→argmax；不使用 centroid／score_rank | best 第 30,000 步，全量 validation Equal-k 0.623873；game-disjoint，見文末補登 |

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

使用者要求檢查「論文同級抽盤、競賽同玩家組題」是否造成差距。新增 `scripts/se_paper_eval.py`，只重用三個模型既有逐盤分數；N=1..100，每級每 N 有放回抽樣 500 次，各級等權、位置等權聚合、nearest-centroid，seed=2026。模型間共用抽樣。另從 validation 每級固定抽 100 candidate／900 query，棋局互斥、玩家可重疊；candidate 另建 centroid。沒有 GPU 訓練或新推論。

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

一鍵入口：`scripts/run_release.sh <raw-dir> <out-dir>`。`<part>` 為 public／private；檔案寫在指定輸出目錄。以下為管線輸出約定，正式提交格式仍待 11/04 確認；由使用者手動上傳。

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

## 2026-11-04：SE3 十五位置正式訓練與最佳模型評估（既有產物補登）

run：`outputs/rank/reproduce_paper_3/se3_seed2026_15pos/`（Slurm job 692203）。本節補登後續完成結果；前述「尚無正式模型分數」與十五位置「未知」為當時狀態。

**完成與選模：** [training_summary.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/training_summary.json) 記錄 completed、45,000 updates、stop_reason=patience；最佳固定監控模型為第 **30,000 步**，5,000 題、每盤十五位置的監控 Equal-k 為 **0.6228357894**。第 35k／40k／45k 三次監控嚴格較低而早停。summary 記錄 15,980.56 秒（約 4.44 小時）／35.5124 GPU-h，為 trainer session 計時，不等於完整 Slurm 配置與最終評估成本。

**`evaluation/` 使用哪個模型：** [best.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/best.json) 指向 `best/best_00030000.pt`；完訓將其發布為根目錄 [model.pt](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/model.pt)，並加入停止步數等完成資訊。全量評估讀取此 `model.pt`，使用的是第 30,000 步最佳權重。已重新計算 `model.pt` SHA-256=`74c65097c53d85897c8e4239e910415eec1b8f3b474945b1d3357925a95a8cd0`，與 [model.complete.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/model.complete.json) 及 [evaluation/manifest.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/evaluation/manifest.json) 一致。

**完整驗證：** validation 200,000 盤，使用指定玩家且 ply≥30 的全部有效位置，固定 10,000 題、每桶 2,500 題。盤內平均 log_softmax，題內各盤等權平均後 argmax；不使用 centroid／score_rank。分數來源為 [evaluation/result.json](outputs/rank/reproduce_paper_3/se3_seed2026_15pos/evaluation/result.json)：

| 評估範圍 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---:|---:|---:|---:|---:|
| 第 30,000 步固定監控，5,000 題／每盤十五位置 | 0.571475 | 0.625332 | 0.629166 | 0.665369 | 0.622836 |
| 同模型完整 validation 全有效位置，10,000 題 | 0.574675 | 0.624762 | 0.628528 | 0.667525 | **0.623873** |

全量 Equal-k 未四捨五入值為 **0.6238725773478886**；單盤官方公式分數為 **0.41529552023865274**（200,000 盤，診斷）。四桶均超過 baseline 0.4631；沿用 SE 的 game-disjoint 切分與固定題組，玩家可重疊，且同一 validation 用於選模，不能視為獨立 holdout 或新玩家泛化成績，也不能與 HumanSL 主線的 Valid-U 直接排名。SE3 同時改模型、loss、取樣、optimizer 與訓練長度，無法將差異歸因於單項。

本次只讀取既有產物並補登，未重新訓練、推論或上傳；以上為本地依官方公式計算的分數，平台 LB 未知。
