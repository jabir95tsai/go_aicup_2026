# cvec2 policy H15 八臂實驗實作計畫

四個逐手殘差臂統一以 **原版 A 的 raw H15＋raw C10＋兩個 HD1 旗標**和 A 的 B2 為起點，逐手模型使用全部 **H15＋C10** 的 25 個 profiles。加權臂繼承同一套 raw 特徵及完訓的全盤 CNN 母臂。一次八卡作業完成這五個模型臂，加上原版 A 和兩個 CPU 正規化對照，主表共八臂。

研究問題是 C10 正規化本身的效果、加上 H15 正規化的影響、在原版 A 上學習逐手殘差的增益、同架構使用完整全盤 policy 的增益，以及盤內加權有沒有額外幫助。全部採原版 cvec2 玩家切分、Valid-S 先驗及 seed 2027 抽題，完訓後只評 E2。Public／Private 本輪做到 CPU input-plane 快取。擴大玩家池、正式模型推論和提交 CSV 屬第 10 節後續方案。

## 1 資料與 E2 基準

重用 `outputs/rank/humansl/cvec2_policy/stage1/812279/`。驗證 `complete.json` 的六個 SHA-256、`result.json` 的 cvec2 parity、`games.parquet` 的盤序及 `E2.parquet` 的題序。A 以原始 52 維特徵、原 cvec2 CPU B2 重現既有 seed 2027 的 2,000 題 E2 分數，逐題和 `question_scores.parquet` 對照。四桶為 **0.7022／0.7494／0.7762／0.7740**，Equal-k **0.7505**。

| 棋局範圍 | 盤數 | 手數 | 本輪用途 |
| --- | ---: | ---: | --- |
| 原擬合池，Valid-S＋train20，HD≤1 | 149,615 | 33,363,661 | 5,728 位玩家，建 planes、teacher 推論、訓練及保存擬合特徵。 |
| E2 實際引用的 Valid-U 棋局 | 15,460 | 3,445,528 | 建 planes、teacher 推論、完訓後逐盤推論與 E2 評分。 |
| 其餘 Valid-U HD≤1 | 18,196 | 4,057,717 | 建 planes，供後續擴池。 |
| Public／Private | 10,000 | 2,222,151 | 建 planes，保存 800 題與棋局的對照。 |
| 全部壓縮輸入 | **193,271** | **43,089,057** | 一份 input-plane 快取共用。 |
| 八卡 teacher 使用的棋局 | **165,075** | **36,809,189** | 原擬合池＋E2。 |

原擬合玩家在既有 `strength_folds_seed2027.parquet` 的 fold 1–4，Valid-U 在原 fold 0，兩者玩家互斥。E2 固定 2,000 題，每 k 500 題、每 k 每級 50 題，題內同玩家同色、盤互異。訓練與初始 B2 只讀原擬合池。各輸入集以資料集和 `game_index` 定址，正式測試保留 Public／Private 與原題 ID。HD0 用日本規則貼目 6.5，HD1 用 0.5。

來源和既有實驗見 [套件 README](README.md)、[CLAUDE.md](../../../../../CLAUDE.md) 與 [RESULTS.md](../../../../../RESULTS.md)。本文件定義新的 A 全臂流程。現有提交入口與歷史單 anchor 流程可供重現。

## 2 原版 cvec2 抽題與 CPU 分類頭

沿用 `stage1/812279/games.parquet` 的原順序，取 `role != "U"` 作擬合池。用 `role == "S"` 的 Valid-S 棋局呼叫 `player_prior`，得到各 k 擁有至少 k 盤的玩家棋力分布。以原始 52 維完成 A 的 CPU parity，四個聯合臂各複製 A 的 B2 scaler、W、b。兩個正規化對照依第 6 節的 offsets 各自擬合。所有 CPU B2 都依 k 各自擬合 scaler 與 `LogisticRegression(C=1)`。 [原版 B2](../../../../../scripts/strength/rank_more_players.py)

1. 呼叫 `PriorMatched(prior).synthetic_rows(fit, fit.rank_id.to_numpy(), 2027)`。每 k 20,000 題，共 80,000 題。依先驗抽棋力，再從該級別且盤數足夠的玩家均勻抽一人，從該玩家全部擬合盤不放回抽 k 盤。保留原 cvec2 允許題內混色的規則。
2. 依 `k=5,10,15,20` 逐桶將 `[k, 題數]`、級別向量和題內 `game_index` 矩陣轉為 little-endian `int64`，按列序串接。已驗證的 SHA-256 為 `cd336f09ec163c517d391de6fda5d89fbd0b036bc12c8c3fba3f0621c4a52d2f`。不符時核對角色、盤序、先驗、seed 與套件版本。
3. 重用 `epoch_batches` 產生一輪 **2,500 步**。每步 32 題、每 k 8 題。五個模型臂共用這份批次，三個 CPU 基準共用相同題目抽樣來源及先驗。

## 3 壓縮輸入與完整 H15／C10 teacher

沿用或建立 `data/processed/strength_features/humansl/cvec2_policy/input_planes/` 的版本目錄。193,271 盤整盤重播一次，分片直接 `zstd -1` 壓縮發布。每手保留 `uint8[993]` packed planes、`float32[19]` 全域輸入、`int16` 落點和 `uint8` 相對執色（本人 0、對手 1）。保存 input config、規則、貼目、盤序、offsets、棋局與題目對照及 checksum。Public／Private 沿用 [`policy_large/export.py`](../policy_large/export.py) 的盤與題目準備。

- 八卡只讀 **165,075 盤、36,809,189 手**的原擬合＋E2。保留四張 H 卡、四張 C 卡與現有對齊的棋局分片。H teacher 使用原版 `humansl.PROFILES` 的 15 個 profiles 和實際輪到下棋的顏色，C teacher 使用原版 C v2 的十級向量和本人／對手角色。
- 顯存快取保存 **H15＋C10 全盤 FP16 raw log policy**，每個 profile 362 點含 pass。`Shard.q` 保存合法點 FP32 log-softmax 的實際落點原值，另存 target、side 和 offsets。H15 依日本規則排除自殺手，C10 沿用原遮罩。檢查兩者的盤序、手序、側別和 H15 遮罩為 C10 的子集，聯合模型以 H15 遮罩選取候選點。逐盤 raw q 均值對原版 A 的差若非有限值或任一族平均差 **> .005**，即停止。最大差 **> .05** 的棋局按族記在 `teacher_fit_parity.json` 和 `teacher_parity.json` 的 `game_indices_over_max_tolerance`。 [teacher 快取](../policy_large/cache.py) · [H/C 快取對齊](full_cache.py)
- 全量 policy 快取平均每張 H 卡約 **93.6 GiB**、每張 C 卡約 **62.4 GiB**。納入第 5 節的盤面圖後，H 卡快取約 **94.4 GiB／卡**，以實際分片及裝置容量預檢。舊 `cache_fraction=0.75` 加 16 GiB 工作空間不足以容納 H15。A 版配置先用 `cache_fraction=0.85`、保留現有 **至少 16 GiB** 的檢查，H15 的推論微批先用 **128 個位置**、C10 沿用 256。八卡短測要驗證真實峰值、跨卡搬運以及每臂 step 速度，通過後才定正式配置。A 版設置透過本輪配置傳入，舊 H9 路徑保有各自配置。
- 全盤 log policy 與逐手 q 留在該八卡作業顯存。按第 6 節在 teacher 前向時歸約並原子發布小型 offsets。縮小 F/B 位置分塊可降低工作峰值，完整顯存預檢仍需通過。A 版使用新的 run 身分，原有 input planes 可依 manifest 驗證後重用。

## 4 八臂架構與共同起點

| 臂 | 特徵與模型 |
| --- | --- |
| `A` | 原版 raw H15、raw C10＋2 個 HD1 旗標，52 維，已有 E2 逐題分數。 |
| `A/C10-logsoftmax` | raw H15、逐手跨 profile log-softmax C10＋2 個旗標，52 維，CPU 擬合 B2。用於對照 C10 正規化。 |
| `A/H15-C10-logsoftmax` | H15 與 C10 分別逐手 log-softmax＋2 個旗標，52 維，CPU 擬合 B2。 |
| `A/R10-C`、`A/R10-D` | 以原版 A 為起點，用 raw H15＋raw C10 的 25 維逐手輸入。`-C` 用落點資訊及合法遮罩，`-D` 用全盤 policy。R10 候選點 MLP 為 `52→32→32→32`、attention scorer 為 `89→32→1`，隱藏層 `90→64`。 |
| `A/CNN-R10-C`、`A/CNN-R10-D` | 以原版 A 為起點，使用同一組 25 維 raw 輸入與落點／全盤對照。棋盤輸入為機率、log 機率及落點標記各 25 張，合計 **75 channels**。三層 5×5 Conv 各輸出 32 channels，盤面 `19→15→11→7`，全域平均池化後串接落點／pass／合法點資訊，經 `109→64`。 |
| `A/CNN-R10-D-weighted` | 以 A 的 52 維特徵為底，凍結完訓的 `A/CNN-R10-D` 與其 B2，只訓練第 5 節的 29-channel 權重器。 |

每手的 `q` 取 H15 與 C10 各 profile 對實際落點（含 pass）的 raw log 機率。全盤 `-D` 讀取同一種 raw log policy，實際落點以 `Shard.q` 的 FP32 原值回填，非法點由合法遮罩處理。`-C` 將同一手的落點值填入合法候選點，保留落點標記與遮罩。跨 profile log-softmax 的 C10 特徵用於第 6 節的 CPU 對照。

四個聯合臂最後各接 `64→15` H 與 `64→10` C 頭。H 沿用 `log_softmax(q_H + shift_H) − log_softmax(q_H)` 殘差，C 的殘差直接取 `shift_C`，加到 raw C10。本人／對手各平均 25 維殘差，加到 A 的對應 50 欄，兩個 HD1 旗標直接帶入。每題各盤等權，對應 k 的 B2 輸出 10 個棋力 logits。兩頭零初始化，step 0 的特徵與 logits 須重現 A。

- 四臂各 2,500 步，使用題級十級 CE 聯合更新小模型和 B2。固定 A 的 scaler，B2 的 W、b 可訓練。沿用原 B2 `C=1` 的顯式 L2，四個 k 平均 `sum(W_k**2) / (2 × 20,000)`，作用於完整的 B2 係數矩陣，截距自由擬合。AdamW LR `3e-4`、小模型 weight decay `1e-4`、B2 parameter group weight decay 0。
- 每 500 步保存最近 checkpoint，取第 2,500 步。核對 B2 與小模型有梯度、50 維的欄序及本人／對手方向、分塊反傳與直接 autograd 一致。原版 H9 相關入口使用現有預設，H15 版將 profiles、殘差和 B2 維度接到 A 路徑。
- 所有正式 B2 都讀完整 52 維。另依 `np.r_[H_COLUMNS, 30:52]` 保存原先要求的 40 維相容投影。

## 5 29-channel 多手聚合權重器

凍結 `A/CNN-R10-D` 完訓母臂及其經原版 B2 L2 訓練的 52 維 B2。權重器每手輸入 `[29,19,19]`，圖層依下表排列。

| Channel（從 1 起算） | 內容 |
| --- | --- |
| 1–25 | 母臂的 15 張 H15 與 10 張 C10 policy 機率圖，各 profile 使用原始盤面 policy。 |
| 26 | **落子前**的當前行棋方棋子，從 HumanSL binary plane 索引 1 取得。 |
| 27 | **落子前**的對方棋子，從 HumanSL binary plane 索引 2 取得。棋子視角依當手輪到誰下棋。 |
| 28 | 由同一手的 `target` 生成的實際落點 one-hot。若為 pass（`target == 361`），這張 19×19 圖全為 0。 |
| 29 | 每格填當前手數除以當盤雙方總手數，手數從 1 起算。 |

兩張棋子圖直接取自既有壓縮 input planes。H teacher 解包時把它們以 91 bytes／手重新打包，依原位置順序留在 H 顯存分片，供加權臂讀取。落點圖按 batch 從既有 `target` 生成。這兩張棋子圖約多占 **3.12 GiB** 顯存，平均每張 H 卡約 **0.78 GiB**。顯存預檢須把它們列入，H 卡的快取合計約 **94.4 GiB／卡**，另保留第 3 節工作空間。

兩層 3×3 Conv＋ReLU（29→16→16，stride 1、padding 0），接全域平均池化、`Dense(16→16)＋ReLU` 和 `Dense(16→1)`。最後一層零初始化。以 AdamW `3e-4`、weight decay `1e-4` 只更新權重器 2,500 步。

本人／對手各自在整盤該側的手上對 logit 做 softmax。同一手的 scalar 權重作用於其全部 25 維。`q_t` 使用 raw H15＋raw C10，`delta_t` 為凍結母臂的 25 維修正。將 `sum_t(w_t × (q_t + delta_t)) − mean_t(q_t)` 加到 A 的對應欄。均勻時重現母臂的等權殘差。讓先旗標直接帶入，題內各盤等權。

權重器 loss 為題級 CE 加 `0.01 × mean(n × sum(w²) − 1)`。母臂 B2 的 L2 在此階段為常數。集中度正則對 batch 不重複棋局的各側等權。以 FP32 跨分塊歸約，softmax 和集中度正則的分母納入整盤該側所有有效手。保存本人／對手有效手數比例 `1 / (n × sum(w²))` 的中位數及第 5 百分位，並綁定母臂 checkpoint SHA-256。

## 6 共用顯存 q 的兩個正規化基準

從同一輪顯存的完整 H15 與 C10 `Shard.q` 各自算 `offset = mean_t(logsumexp_profile(q_t))`，每盤本人、對手各一個 offset。從原版 A 對應組別的 raw 平均 log 機率減去該組 offset，沿用原先逐手正規化定義。GPU 產生的各 profile raw q 均值用於 parity。只持久化兩組每盤各側 offset（合計約 **2.5 MiB**）、來源、profile 順序及 raw 均值誤差。 [正規化與側別聚合](played.py)

- `A/C10-logsoftmax` 只套用 C10 offset。C10 offset 落盤即可在 CPU 擬合此 52 維 B2。有排程收益時可和仍在執行的 H teacher 重疊。四個聯合臂以 CPU 前置的 A B2 與對齊完成的擬合盤 H/C 快取啟動。
- `A/H15-C10-logsoftmax` 套用 H15 和同一份 C10 offset，在 CPU 以相同題目重擬合 B2。和 `A/C10-logsoftmax` 的差值量 H15 正規化增加了什麼，和原版 A 比較兩者合計影響。
- 確認所有 165,075 盤的兩側 offsets 完整且有限，H 與 C 分開做 profile normalizer。以小批次 CPU 直接 `log_softmax` 核對逐手歸約。新的 H teacher 已做完整 15 profiles，本對照只需重用 q，額外工作是逐盤歸約及 CPU 擬合。

## 7 排程 成本與續跑

CPU 前置建立或驗證 input planes、固定抽題、擬合 A B2 並驗證原 A parity。GPU 按擬合盤與 E2 分開平衡手數，維持原盤序，各自分配四個對齊的 H／C 常駐分片。四張 H 卡先算擬合盤。C 卡各自完成 C10 擬合盤及 E2 後，保留原 C10 快取，從未領取的 H15 擬合盤範圍取整盤工作，將 policy、q、落點、側別及棋子圖寫回原 H 分片。各範圍只處理一次，發布前同步跨卡寫入並檢查覆蓋。

擬合盤對齊並通過 raw parity 後，四張 C 卡以 A B2 和原固定批次開訓，四張 H 卡同時算剩餘的 E2。全盤母臂完成即排加權臂。兩個 split 皆對齊並通過 raw parity 後，空出的 C 卡抽已完訓臂的擬合盤和 E2 特徵。GPU 產物核對完成即釋放八卡，最終 B2 對照、E2 評分與 bootstrap 由 CPU 完成。

| 段落 | 成本估計 |
| --- | --- |
| CPU 建 193,271 盤 input planes | 約 1.5–2 小時，假設接近歷史 96 CPU／88 workers。可重用符合資料身分的既有分片。 |
| H15／C10 teacher | H15 相對舊 H9 每手多 6 個 profile，額外約 **2.21 億 profile-位置**，H 側前向規模 **+66.7%**。按舊吞吐估 H 側約 **5.1 小時**、C 側約 **3.3 小時**，兩邊重疊。 |
| 模型更新 | 四個聯合臂和一個加權臂各 2,500 步，共 **12,500 updates**。 |
| 八卡正式段 | 粗估 **6–9 小時、48–72 GPU-h**，初步上限 12 小時，另計 CPU 打包和評分。 |

八卡短測量完整 H15 的顯存峰值、微批與編譯啟動，並核對 C 卡協助的棋局、跨卡寫入、首個 optimizer step 與 H15 E2 的重疊時間，以及特徵抽取的就緒順序。短測固定保留少量 H15 擬合盤給 C 卡以覆蓋接手路徑，正式作業依未領取的範圍動態分派。H teacher 成本是對歷史用量外推，H15 的實際吞吐未建立。落點臂可用 compact q、合法遮罩與索引降低跨卡搬運，必須和全讀路徑做前向與梯度 parity。 [歷史成本](../../../../../RESULTS.md)

每 500 步存最近 optimizer、模型權重、B2 scaler／W／b、抽題身分、步數和 loss。加權臂綁定母臂 checkpoint。重跑按新 A 版身分重用已完成的模型、特徵和 offsets；逐手訓練及補特徵使用 input planes 重建顯存 policy。`run all`、`all.slurm` 沿用 CPU 前置、GPU 段、CPU 評分分工。主要調整點為 `all_data.py`、`all_run.py`、`joint.py`、`all_report.py`、`full_cache.py`、`encoder.py` 及 H teacher 的 profile 介接，歷史入口走既有 profile 設置。

## 8 E2 主表 門檻與落盤產物

八臂用固定的 2,000 題 E2、seed 2027、`score_rank` 解碼，報逐題分數、四 k、Equal-k、玩家配對 bootstrap 95% 區間。A 重用既有逐題分數，七個新結果保存十級機率、預測及分數。報 `A/C10-logsoftmax − A`、雙正規化減 `A/C10-logsoftmax` 和雙正規化減 A。各模型臂對照 A 與 `A/C10-logsoftmax`，另報同架構的全盤減落點及加權臂減母臂。

- 進入後續擴池候選統一須對 A **Equal-k ≥ +0.005**、各 k **≥ −0.005**。模型臂還須高於 CPU 對照 `A/C10-logsoftmax`。全盤臂須高於同架構落點臂。加權臂須比母臂至少 **+0.005**、各 k **≥ −0.005**，且本人與對手的有效手數比例第 5 百分位各 **≥ 0.50**。
- 依未四捨五入的 Equal-k 選通過門檻者，同分先取額外參數較少者，再依臂名排序。保存每臂門檻與差值。

| 產物 | 內容 |
| --- | --- |
| 共用輸入 | 193,271 盤壓縮 input planes、棋局／題目索引及 manifest，抽樣估 1.37 GiB。 |
| CPU 前置與 offsets | 原版 A B2、Valid-S 先驗、題組／批次／SHA-256、兩組 offsets、完整 H15／C10 parity、兩個正規化 B2 與設定。 |
| 原擬合盤特徵 | 八臂各 `149615 × 52` FP32，約 **237 MiB**。另存八臂 `149615 × 40` 投影，約 **183 MiB**。 |
| E2 特徵和結果 | 七個新結果各 `15460 × 52` FP32，約 **21.5 MiB**。保存八臂逐題分數、報表及七組新機率。 |
| 模型與續跑 | 五個訓練臂的模型、B2、權重器或母臂身分、checkpoint、config、效能及完成指紋。 |

輸出使用 `outputs/rank/humansl/cvec2_policy/all/<run_id>/` 的新 A run。壓縮 planes 與逐盤特徵約 **1.8 GiB**，預留 2–3 GiB，模型和 checkpoint 另計。正式測試產物止於 CPU 快取與題目索引。

## 9 實作驗收順序

1. CPU 預檢原玩家 fold、列順序、固定抽題 SHA-256、A 的 2,000 題 parity、輸入與 GPU 棋局覆蓋。
2. 壓縮 pack／unpack、斷點重用、HD0/HD1 貼目、profile 順序、正式測試索引與合法著手遮罩。確認五個模型臂取用的 raw H15／C10 與原 A 對齊，核對落點、pass 及非法點。另用明確樣本驗證 CPU 對照的 C10 跨 profile 正規化。擾動非落點 policy 時，落點臂輸出須一致。
3. CPU 合成測試覆蓋 25 個 profiles、50 維聚合、52 維 B2、40 維投影、四臂 step 0 重現 A 的 52 維特徵與 B2 logits、C10 直接加上 `shift_C`、B2 聯合梯度及 C=1 的 L2、分塊梯度與續跑。驗證 29-channel 加權臂在均勻權重時重現完訓母臂。用已知黑白交替盤面驗證兩張棋子圖的行棋方視角、落子前快照、棋盤方向、實際落點 one-hot 與 pass 全零標記。驗證兩個正規化基準共用完全相同的 C10 欄。
4. 八卡短測涵蓋全規模同等每卡 cache 保留量的可行性、H15 連續微批、raw parity、四臂並行、母臂至加權臂、擬合盤／E2 特徵和 CPU 基準評分，再更新正式時數。
5. 正式驗收核對八臂擬合盤特徵、七個新結果的 E2 特徵、B2、八臂 2,000 題 E2 結果與門檻摘要。來源雜湊及產物檢查通過後發布完成標記。

## 10 後續方案 擴大玩家數與正式送件

本輪實作以第 9 節為交付範圍。以下利用已保存資料，待勝出臂及後續作業範圍確認後另行執行。

1. 鎖定勝出臂、模型與擴池前的 E2。依既有 cvec2 release 方式補入 Valid-U HD≤1，從 149,615 盤／5,728 人加 33,656 盤／1,433 人成 **183,271 盤／7,161 人**。其中 15,460 盤已有 E2 特徵，另 18,196 盤已有 input planes 可供後續 teacher 和特徵抽取。
2. 先凍結勝出逐手模型及權重器，重用本輪的完整 52 維特徵，只補新增棋局的相同 52 欄並在 CPU 擬合 B2。保留原勝出 B2 供對照。模型臂以 raw H15／C10 抽特徵，正規化對照沿用各自的 profile offsets。必要時再抽完整 H15＋C10 policy。擴池先驗依既有 Valid-S＋Valid-U HD≤1 玩家分布計算。 [原版 release B2](export.py)
3. 要聯合續訓逐手模型時，從勝出 checkpoint 起跑，混合原有與新增玩家題目，以較小 LR 更新。用已存 planes 重建所需 teacher policy，並用新權重重抽逐盤特徵。要量擴池增益，先留出獨立玩家。Valid-U 納入正式擬合後，沿用保存的 E2 作為擴池前結果。
4. 終版模型鎖定後，對已存 10,000 盤的 Public／Private input planes 做正式模型推論，保存每盤 52 維、800 題十級機率和 `score_rank` 結果。之後只更新 B2 時可在 CPU 重建 800 題提交 CSV。依專案規約登記候選與預期得分，由使用者上傳。
5. fold −1 的 794 位玩家另案比較。先前舊模型加入該 24,747 盤並對齊先驗後，E2 差值為 −0.0007。 [歷史擴池紀錄](../../../../../RESULTS.md)
