# HumanSL 固定主幹＋雙分支等級分類＋十層殘差修正＋B2 實驗計畫

更新：2026-11-09（台灣時間）。狀態：使用者已授權並完成本版程式、三階段入口與 Slurm；CPU／Gloo 工程驗證見第 13 節。尚未建立官方實驗資料、啟動 teacher／GPU 訓練或提交作業。

本目錄專供此實驗後續實作。本文整合本次討論及深入審查後的確認事項；第 12 節區分已選定流程與仍需量測的訓練數值，不將舊 SE3 設定直接視為本版已核准設定。專案共同規範以根目錄 [CLAUDE.md](../../../../../CLAUDE.md) 為準，歷史分數以 [RESULTS.md](../../../../../RESULTS.md) 為準。

## 1. 目的、對照與假設

以固定 HumanSL 的盤面表示，訓練兩個預測「指定玩家本人級別」的分類分支；再加入十層 ResNet 學習分類 logits 的殘差，最後凍結上游，以題目層級線性 B2 融合雙方訊號。

- 本人分支：使用本人落子，預測本人級別。
- 對手分支：使用對手落子，預測本人級別；不預測對手本身的 rank。
- 十級順序固定為 `D,C,B,A,1D,2D,3D,4D,5D,6D`，編碼 `0..9`。
- 假設一：共用可訓練的特徵層，讓 value 輔助目標能影響分類表示。
- 假設二：十層、256-channel 殘差模型能補足固定 HumanSL 表示的級別判別能力；policy／value 作為輔助任務。
- 假設三：本人／對手分開聚合後，以四個 k 各自的線性分類器學習題目層級融合。
- 預期 Equal-k 與各 k 增益：未知；不可將 loss 下降視為泛化增益。

既有提交對照為 H＋C＋B2、P1/P2/P3、noA、Valid-S＋Valid-U 擬合的 `ahc-p3-noA-allp_731429_rank.csv`，紀錄 Public LB 為 **0.7386411**。其 B2 已使用 U 標籤，沒有獨立 U 分數，不能拿來作本地未見玩家的泛化對照。

本實驗主模型以新二十維特徵取代舊 H／C 特徵，不串接 A-024、舊 H 30 維或 C 20 維；H＋C 的既有系統作為獨立對照。下文 P1/P2/P3 指沿用其資料語意修正，不代表仍抽取舊 H／C。

## 2. 棋局篩選、有效落子與抽樣

### 2.1 已確認的有效集合

所有階段先排除整盤 `HD >= 2`，僅使用 HD0／HD1。

```text
own = (ply >= 30) AND (move_color == target_player_color)
opp = (ply >= 30) AND (move_color != target_player_color)
```

- `ply` 從 0 起算，門檻是雙方合計第 31 手起，不是本人第 31 手起。
- 每個樣本使用該手落子前盤面與真實歷史；前 30 手仍須重播，不能直接丟掉。
- 從門檻到棋譜最後均可取樣，不另按著手品質或終盤階段篩除。
- 依實際落子顏色及指定玩家顏色分支；HD0／1 不對調黑白。
- 訓練唯一棋局鍵為 `(rank, game_id)`；`player_id` 僅用於切分與組題，不作特徵。
- HD 缺失、無法重播或顏色不一致須明確報錯，不默認 HD0。

### 2.2 每方十五手

| 用途 | 取樣方式 |
|---|---|
| 第一、二階段訓練 | 每盤本人、對手各無放回隨機抽 15 手，共 30 個位置 |
| 訓練中定期監控 | 每盤固定本人 15 手＋對手 15 手，跨次監控重用 |
| 最終完整驗證 | 雙方全部有效位置 |
| 第三階段 B2 特徵建立 | 雙方全部有效位置 |
| 正式測試推論 | 雙方全部有效位置，不隨機抽樣 |

訓練棋局再次被抽到時，重新抽位置；同次同方不重複，不要求每輪遍歷同盤全部位置。抽樣、D4 與續跑均由 seed／已完成更新步數重建，不以 worker 預取進度計算。若任一方不足 15 個位置，資料準備先報錯並列出棋局，不靜默有放回補抽。完整聚合亦不允許任一分支為空。

採各級獨立 shuffle／輪次／游標，不要求十級棋局數相等，也不截短到最少級別盤數。每組仍為十級各一盤、每盤雙方各 15 手；某級用完整輪才只重洗該級，其他級別繼續自己的游標，不定義「十級同時結束」的 epoch。

- 每個 optimizer update 的同一級內不得重複棋局。跨輪補足 batch 若遇當次已取棋局，延後到下一 update，保留在待取佇列、不刪盤；每級至少需有 global groups_per_update 盤，否則準備階段報錯。
- 所有 GPU 使用同一份全域抽盤次序，分配完整十級組，不各自重新抽。checkpoint 記錄已消費的各級輪次／游標／待取佇列及 RNG，worker 預取不得推進 durable state；續跑需驗證後續抽樣與未中斷完全相同。
- 不直接沿用舊 SE3 sampler：其建構式要求各級盤數相等且整除組數，與本版 HD／玩家篩選不相容。
- 訓練 D4 同步變換棋盤、歷史及實際著手座標／policy target；value 不因 D4 改值。監控與推論不做隨機增強。

### 2.3 抽樣參考的限制

舊 SE3 每盤抽 15 個本人位置，best 第 30,000 步，全量 validation 的 k=5/10/15/20 分別為 **0.574675／0.624762／0.628528／0.667525**，Equal-k **0.623873**。該結果使用 game-disjoint、玩家可重疊，不能視為本版未見玩家分數。

20／30 手版本只有 100 步短測，沒有完整受控的抽樣數比較；採 15 手是沿用已有正式運行經驗，不是宣稱每方 15 手已被證明最佳。本版每盤共 30 個位置，位置數為舊 SE3 的兩倍，時間／顯存不能直接照抄。

## 3. HumanSL 條件與輸入

| 項目 | 本版規格 |
|---|---|
| 網路 | `b18c384nbt-humanv0.ckpt` 一般權重，不使用 SWA |
| 主幹輸出 | `trunkfinal`，最後 norm／activation 後的 `[B,384,19,19]` |
| 條件 | 固定 `rank_1d`，依當手落子方建立 metadata row |
| 規則／貼目 | 日本規則；HD0＝6.5、HD1＝0.5 |
| 模式 | HumanSL 全部參數凍結，始終 `eval()`；保留真實歷史 |
| 輸入編碼 | KataGo 自己的 spatial／global／metadata 編碼，不將 SE3 的 18-plane 輸入直接送給 HumanSL |

固定 metadata 沿用 `features/humansl_torch.py:profile_metadata`：雙方人類、雙方 rank_1d、game ratedness unknown、20 分鐘＋30 秒五次讀秒、日期 2020-03-01、來源 KGS。這些是外部模型的固定條件，不是真實幽玄玩家資料。不得把真實標籤 y 或 `e[y,side]` 餵進分類主幹。

既有實驗供參考：H 使用 15 個固定 profiles；A-051/052 表示同時使用 rank_1d 與 unknown；C 用十級假設的本人／對手條件向量；C4/C4b 延續 C 的條件並微調 policy 路徑。本版不把這些條件全部展開，不宣稱 rank_1d 已證明勝過 unknown。

## 4. 第一階段：共用投影＋兩個等級頭＋value

```text
落子前 KataGo 輸入
  → 固定 HumanSL → F[384,19,19]
  → 可訓練 Conv1x1(384→256) → BN → ReLU → U[256,19,19]
      ├─ 全盤平均(U) + U[實際落子點] → 本人或對手分類頭 → 10 logits
      └─ SE 型 scalar head + tanh → 1 value
```

- 共用層先採單一空間投影，保留 19×19 供後續十層 ResNet 使用。這是第一版選擇，非已量測的最佳架構。
- `+` 在圖中分類讀出表示串接：全盤平均 256 維與實際著手點 256 維，共 512 維；兩個 `Linear(512,10)` 分別處理本人／對手位置。
- 每手僅選其對應等級分支計 loss；兩個分支標籤均為指定玩家 y。
- value 頭重用 SE 的 `scalar_head(256)` 與 tanh；它只讀落子前 U。
- 實際著手只在分類分支作 gather，不修改 F／U，不送進共用投影、value 或後續 policy 路徑。
- 若要支援 pass，分類 gather 需另定沒有盤上座標時的表示；目前官方訓練資料無 pass，不可把 action=361 當作 19×19 內的位置直接索引。

```text
L_rank = (mean_game(mean_own(CE(z_own,y)))
        + mean_game(mean_opp(CE(z_opp,y)))) / 2
L1 = 2 * L_rank + L_value
```

`L_value` 為逐位置 teacher MSE，先各方／各盤平均後等權合併。CE 接收原始 logits；loss 與 DDP 全域平均採 FP32。value 與分類均回傳梯度到可訓練共用投影，HumanSL 不更新。分類與 value 若只共用固定特徵、沒有可訓練共用層，不符合本版設計。

## 5. 第二階段：十層、256-channel 殘差修正

第一階段選定模型後全部凍結，包括共用投影、分類頭、value 頭與 BN running statistics。

```text
固定第一階段 U[256,19,19]
  → 可訓練 10 × SE ResidualBlock(256) → V[256,19,19]
      ├─ 全盤平均(V) 串接 V[實際落子點]
      │    → 本人／對手 Linear(512,10) → Δz
      ├─ SE3 policy head → 362 logits
      └─ SE scalar head + tanh → 1 value

z_final,side = z_stage1,side + Δz_side
```

- 重用 `reproduce_paper/model.py:ResidualBlock` 的卷積 bias、BN、相加後 ReLU 語意。
- 十層從新初始化；不載入舊 SE3 的 18-block 權重，也不另建立 384-channel residual blocks。
- 兩個殘差分類頭的權重及 bias 初始化為零，使初始融合 logits 等於第一階段。
- policy 沿用 `Conv1x1(256→2) → BN → ReLU → Flatten(722) → Linear(722→362)`。
- policy／value 都只讀落子前 V，不能讀著手 gather、著手 one-hot 或落子後盤面；不同任務共用表示，分支後的讀出不同。
- policy 監督是當手真實著手；本人／對手位置都使用。value 監督維持當手落子方視角。

```text
L2 = 3 * L_rank(z_final,y) + L_policy + L_value
p_t = softmax(z_final,side,t)
```

policy CE 與 value MSE 同樣採各方／各盤等權平均。分類相加發生在 logits，不把兩個 softmax 直接相加當成未正規化機率。第二階段新增 blocks／三頭更新，第一階段維持凍結與 eval。

兩階段 loss 的計算單位均為「先每盤每方 15 手平均，再兩方等權，再全域棋局平均」。令每盤 loss 為 ell_g、一次 update 全域共有 N 盤：單卡目標為 sum_g(ell_g)/N；W 卡 DDP 每個 microbatch 的 local loss sum 乘 W/N 再 backward，由 DDP 的 W 卡平均得到相同全域梯度。梯度累積只在完整 update 後 optimizer／scheduler step 一次，不對不同大小 microbatch 的平均 loss 再做不加權平均。

本版維持第二階段讀取固定 U，不新增直接讀 F 的平行主幹。384→256 投影可能捨棄第二階段需要的資訊，列為限制；僅在第一階段有訊號、第二階段無增益時，才提出「第二階段從 F 自行投影」的後續單項實驗，不自動加入本版。

## 6. Teacher value

- 沿用既有逐有效落子前盤面的強 KataGo teacher 路線，1 visit；不把 HumanSL 自己的 value 當作真實勝負。
- 以黑方勝率 p_B 轉成當手落子方視角：黑方 `2*p_B-1`，白方 `-(2*p_B-1)`，範圍 [-1,1]。
- teacher 同樣使用日本規則、HD0 貼 6.5／HD1 貼 0.5；標籤鍵包含棋局、ply、規則／貼目與模型身分。
- 舊 `reproduce_paper/value_v1/train/value.npy` 僅含舊 split 的本人有效位置、原貼目 6.5；不能因長度相同就套到新 split、對手位置或 HD1。
- 對手位置需要自己的落子前 target，不能直接複製、取負或挪用相鄰本人手次的值。
- 舊 HD0 本人 target 僅在能證明棋局／ply／teacher 設定完全對齊時重用；舊匯出只剩 value.npy，來源證明不足時重算。
- 兩階段共用同一套已驗證標籤；缺失或非有限 target 直接報錯，不補零。

## 7. 逐手、單盤與多盤聚合

### 7.1 單盤二十維

對棋局 g，各方有效集合分別為 O_g、R_g：

```text
a_g = mean_{t in O_g}(p_t)  # 10 維，預測本人 rank
b_g = mean_{t in R_g}(p_t)  # 10 維，預測本人 rank
x_g = concat(a_g,b_g)      # 20 維，固定本人在前、對手在後
```

每手 logits 先 softmax，再對機率做算術平均；不用舊 C 的 likelihood 或舊 SE3 的平均 log-prob。兩個十維區塊各自總和為 1，整個二十維總和為 2，不再做二十類 softmax。

### 7.2 題目平均與 P2

```text
x_q = mean_{g in question}(x_g)  # k 盤等權，仍為 20 維
h_g = [HD==1 and 本人黑, HD==1 and 本人白]
h_q = mean_{g in question}(h_g)
f_q = concat(x_q,h_q)           # 保留 P2：22 維
```

不將所有棋局的手混在一起平均，避免長盤權重較高。模型本身始終輸出二十維；完整保留 P2 時 B2 才附加兩個讓先比例，成為二十二維。主流程不增加其他特徵；若要比較嚴格 `20→10`，列為拿掉 P2 的單項消融，不與主模型混稱。

## 8. 第三階段：題目層級線性 B2

1. 固定前兩階段與所有 BN statistics，以全部有效位置計算每盤 x_g。
2. 合成題必須同玩家、同顏色，題內棋局互異；k∈{5,10,15,20}。
3. 先均勻抽一位至少有一種顏色符合 k 盤資格的玩家，再均勻抽該玩家符合資格的顏色，最後無放回抽 k 盤；不直接均勻抽 (player,color) 對，避免雙色皆合格玩家獲得兩倍權重。不強制十級均勻先驗。
4. 四個 k 分別在擬合題的 f_q 上擬合 StandardScaler 與單一線性分類器。
5. 主流程為 `StandardScaler → Linear(22,10)`；沿用既有 multinomial LogisticRegression 路線，C=1、max_iter=5000、每 k 20,000 題作為實作起點。這是單一線性層，沒有隱藏層。
6. 訓練目標為十級 CE（含既有 L2）；推論 `P_q=softmax(W_k*scaled(f_q)+b_k)`，再 `metrics.score_rank(P_q)`。
7. scaler、分類器及任何設定選擇只使用該次擬合資料。題目抽樣 seed 固定保存，不用 test 或評估玩家選特徵／正則化。

組題直接重用 `SE_H_C_B2.data.sample_questions`：`balanced=False,count=20000` 已實作先玩家、再合格顏色、再抽 k 盤，不重寫同色 sampler。M 監控題以 `balanced=True,count=125` 產生每個 k/rank cell 125 題。只需把 B2 擬合接到這份固定題目表；舊 `RankModel.fit(method='B2')` 會忽略傳入 questions、另按 player_id 抽盤，因此不能直接呼叫原入口。不得將第一、二階段 training loss 當作第三階段的選模指標。

prepare 必須輸出每個 k／rank 的合格玩家數、玩家顏色組數及合成題使用頻次，十級缺任一級就報錯。既有 S/U 特徵棋局、HD0/1、同色組題的唯讀稽核如下，未加入 train20，也不是新切分後已發布的題庫：

| 擬合來源 | k=5 合格玩家 | k=10 | k=15 | k=20 |
|---|---:|---:|---:|---:|
| 原 Valid-S | 2,087 | 1,214 | 757 | 490 |
| 原 Valid-S＋U | 3,458 | 2,391 | 1,172 | 641 |

20,000 道合成題不等於 20,000 位獨立玩家。四個 k 使用同一套固定抽題規則，所有比較臂重用相同題目與 seed=2027；區間估計維持玩家層級配對 bootstrap。

## 9. 玩家隔離、監控與最終擬合

### 9.1 已選定：四折玩家 OOF，不把所有 S/U 玩家從上游永久排除

2026-11-09 唯讀稽核：原 S/U 已抽特徵棋局經 HD0/1 篩選後，S 為 62,867 盤／4,885 人，U 為 25,519 盤／1,433 人。若將兩者全部排除於上游，並保留原 fold−1，上游原 folds 1..4 只剩 **854 人，1D 僅 3 人**。因此不採此保留集方案，改用四折玩家 OOF。

選四折是配合現有開發 fold 1..4，讓每折約使用 75% 的 D 玩家訓練，相較兩折縮小與全 D 重訓的資料量差異；不宣稱四折必然提升分數，新增訓練與特徵抽取成本需完整計帳。

切分／組題仍只由 `data/cv.py` 產生，保存獨立 manifest，不覆寫既有 split。原始玩家 fold 身分採既有 seed=2027、在完整原資料上產生的版本；HD 篩選不重新分配舊玩家 fold。模型初始化／位置抽樣 seed=2026，題目／B2 抽樣 seed=2027，均另記於 config。

| 集合 | 定義 | 允許用途 |
|---|---|---|
| R | 原 fold−1 玩家 | 全程保留，這版不使用 |
| U | 原 fold 0 玩家 | 本版設定選定後的一次最後評估；不參與開發梯度、早停或 B2 擬合 |
| M | 原 folds 1..4 的 HD0/1 合格玩家中，依 rank／盤量分層固定抽 10%，seed=2027 | 上游監控、開發 B2 候選選擇與進階段 gate；不參與開發梯度或 B2 擬合 |
| D | 原 folds 1..4 中除 M 外的合格玩家 | 上游開發訓練與四折 OOF |
| B_dev | 原 Valid-S HD0/1 棋局與 D 玩家交集 | 使用 OOF 特徵擬合開發 B2；scaler 亦僅在此擬合 |

- M 的每個非空分層取 round(10%)；小分層先合併同 rank 盤量層，避免產生無法組題的碎片。實作 prepare 輸出實際名單、各級盤數與每 k 資格，不以文字比例替代資格驗證。
- 每位玩家黑白全部棋局綁同一角色，D/M/U/R 互斥；同一棋局不得跨角色。舊 Valid-S 對新模型不再自動代表 seen，僅保留為 B2 棋局來源名稱。
- D 內四折沿用原 fold 1/2/3/4 身分。對當折 H_j，以 T_j=D−H_j 訓練第一、二階段，M 作共同開發監控；H_j 的標籤不得用於兩階段訓練、早停或 fold 內調參。
- 第二階段 fold j 只能載入同一 fold j 的第一階段權重；不得載入曾訓練 H_j 的全 D 第一階段模型。H_j 的完整位置預測即該玩家的 OOF 特徵。
- 另訓練一套使用全 D 的上游模型，以 M 監控，作為開發單模型推論來源。四折模型用於產生 B2 擬合特徵，不在正式推論時作 ensemble。
- U 已在其他專案實驗中被查看；本版承諾的是不以它反覆調整本版設定，不宣稱它是全專案從未接觸的新 holdout。

### 9.2 固定監控、候選選模與開發順序

監控讀出明訂如下，不需要在每次監控重擬合 B2：

```text
P_monitor,q = (mean_game(a_g) + mean_game(b_g)) / 2
prediction = score_rank(P_monitor,q)
```

- M 固定 5,000 題，k=5/10/15/20 各 1,250 題、每個 k/rank cell 125 題，題內同玩家同色、不重複棋局；每盤固定每方 15 手，不隨監控重抽。資格不足就 prepare 報錯，不改成混色組題。
- 每 5,000 個 optimizer updates 監控，另在步數預算最後一步監控。100 步工程短測在最後一步另做一次監控，不據此判斷棋力品質。
- 以 Equal-k 嚴格增加才更新 best；相等保留較早步數。連續三次未增加（含相等）早停；最大更新數另由短測預算固定，不讓早停變成無上限訓練。
- 每次均記本人單分支、對手單分支及固定等權融合的分 k 分數／分類 CE，方便發現某分支退化；只有固定融合的 Equal-k 控制上游早停。
- 每個訓練 run 只保留監控最佳與最後一步兩份候選，以及必要 latest 續跑狀態；不按 training loss 選權重。

開發順序固定為：

1. 第一階段四個 OOF runs＋一個全 D run。建立兩組候選：各 run 的 best，以及各 run 的 final；每組均用全部有效位置產生 B_dev OOF 特徵、擬合自己的四個 B2，再用對應全 D 模型在 M 的全部有效位置評分。
2. 只在 M 比較 best/final 兩組完整 B2 管線，以 Equal-k 高者選擇；同分保留 best 組。這是開發選擇，不能把 M 報成獨立泛化分數。
3. 第一階段是否安排第二階段，依 M 上的配對 gate 與已回報成本決定；不讀 U。第二階段對每個 run 固定第一階段所選同組候選，執行四個 OOF runs＋一個全 D run。
4. 第二階段同樣只比較 best/final 兩組完整 B2 管線，並在 M 與第一階段、對照比較。記錄選中的全 D 第一階段步數 s1 與第二階段步數 s2，供最終固定步數重訓使用。
5. 完成全部開發選擇後，鎖定設定與候選身分，再於 U 的固定題庫一次評估已預先列出的各臂。不得看 U 後再改投影、LR、抽樣數或候選選擇；若另開新版本，須承認 U 已被用於開發。

完整 B2 評估一律用全部有效位置，與每方 15 手的輕量監控分開列報。每個 OOF 與全 D run 都保留相同型別的 best/final 候選；不得用全 D 模型替代 OOF fold 模型產生 B_dev 特徵。

### 9.3 最終提交版：重新交叉擬合＋全資料單模型推論

確認開發設定後，最終上游玩家池 P 採原 folds 0..4 的全部 HD0/1 合格玩家，包含原 M 與 U，仍排除 R。B2 最終棋局來源維持原計畫的 Valid-S＋Valid-U HD0/1，不自行加入 train20 或全部 training 棋局。

1. 在 P 上由唯一 CV 模組建立新的四折玩家分層映射（rank／盤量分層、seed=2027），保存 release 專用身分；不能直接把五個原 fold 當成四折。
2. 固定開發已選的 LR／schedule 與步數 s1/s2；每折只用其他三折玩家從頭訓練第一階段，再凍結該折第一階段訓練第二階段。release OOF 不 early stop，不用當折標籤或 U 作監控選模。
3. 每位 S/U 擬合玩家只使用對應未見過他的 release OOF 模型特徵，全部有效位置聚合，擬合四個最終 B2/scaler。舊開發 OOF 特徵不直接混入。
4. 再以全 P、同一固定步數 s1/s2 從頭訓練單一上游模型。Public／Private 均使用這一套全 P 模型＋上述四個 B2；不平均 OOF 模型、不增加測試期 ensemble。
5. release 的 U 已參與上游／B2 擬合，因此沒有獨立 U 分數；保留第 9.2 節鎖定版本的 U 結果作歷史驗證，不將兩版混稱同一模型。

OOF 模型只用約 75% 的池內玩家，全資料模型用 100%，信心尺度可能不同。開發時在共同 M 上比較 fold 模型與全 D 模型的分支機率熵、NLL 與完整 B2 分 k 成績，並保存差異。若完整管線在 M 上未過 gate，不以 test 擬合溫度補救；任何改用 ensemble 或校準的方案另立實驗。本版每輪完整兩階段最多四組 OOF＋一組全資料模型，共十個階段訓練 runs；開發與 release 兩輪都要列入成本，不能只估兩個 runs。

### 9.4 比較與 gate

在相同、未見玩家題組依序比較：

| 臂 | 用途 |
|---|---|
| H＋C＋P2＋B2（noA、HD0/1、正確貼目） | B2 只用 B_dev 擬合，M 作開發比較、U 作鎖定後評估，不能直接使用已擬合 U 的提交版 |
| 第一階段二十維＋P2＋B2 | 先驗證共用投影與分類是否具有實用訊號 |
| 第一＋二階段二十維＋P2＋B2 | 判斷十層殘差／輔助任務的增量 |

對照 C 向量同樣在 D 內做四折交叉擬合：B_dev 用對應 T_j 學得的向量產生特徵，另以全 D 學一份 C 用於 M/U，全部排除 M/U/R 標籤，避免對照 B2 使用 C 的訓練內特徵。這五份小型條件向量的學習及重抽特徵亦列入對照成本。H/C 維持原本的全盤 likelihood 特徵，新模型使用 ply>=30，這是整體方案比較，不能把與對照的差異只歸因於 residual blocks。既有 LB 0.7386411 僅作正式提交對照，不與本地分數直接相減。

- 每臂報 k=5/10/15/20 官方分數及四桶等權 Equal-k，並保存逐題預測。
- 既定開發 gate：Equal-k 相對配對對照至少 **+0.005**，各 k 差值均 **≥−0.005**；另報玩家層級配對 bootstrap 95% 區間，不冒充獨立題目抽樣。
- 門檻用於開發優先順序，不自動決定正式上傳；未過保留產物，依專案 D-003 不直接刪除方向。
- 第一階段在 M 完成開發評估，再依訊號及短測成本安排第二階段；U 不作此 gate。第二階段增量與既有 H＋C 對照差值分開報告，不承諾新架構一定勝過既有 H＋C。

## 10. 測試資料集推論與提交

1. Public／Private 共用第 9.3 節全 P 固定步數重訓的單一兩階段模型及四個最終 B2；依官方題目 color 定義本人，不使用四折 OOF ensemble。
2. 重播每盤全部棋譜，前 30 手只建立狀態；在 ply>=30 的雙方全部位置推論。HD0／1、規則、貼目及 rank_1d 條件與訓練一致。
3. 第一、二階段 logits 相加後分別 softmax，盤內本人／對手分開平均成二十維，再題內各盤等權平均。
4. 加 P2 兩個題內旗標，以對應 k 的 scaler／B2 計十級機率並 score_rank。
5. 所有模組 eval、無梯度、不做隨機位置抽樣或隨機 D4；policy／value 的輸出不參與分類，推論可省略這些輔助頭。
6. batching 只影響資源使用，不改變取樣範圍／聚合權重。不同題目重用棋局特徵時，需同時符合指定顏色、HD／貼目、模型與輸入版本。
7. 目前棋力 Public／Private 已稽核全為 HD0／1；若遇不支援的 HD、任一分支無有效位置或特徵缺失，停止並報錯，不靜默刪盤、補零或改 k。
8. 不在測試集擬合 scaler、B2、溫度或級別先驗；不強制每級題數相同。
9. 匯出 Public＋Private 合併的 800 列 `question_id,pred_rank`，rank 為官方級別字串、順序對齊 sample submission。候選與驗證結果記錄於 RESULTS.md；使用者手動上傳。

## 11. 實作範圍、算力與驗收

本目錄已承接設定／資料準備、模型／loss、三階段訓練、特徵聚合、評估及提交程式；重用與新增範圍如下，實作／未執行事項見第 13 節及 [README.md](README.md)。

重用以函式為單位，先直接 import 現有純函式／工具，不複製整個舊實驗，也不先建立新的通用框架。相對模組路徑以 `src/go_aicup/` 為基準：

| 功能 | 現有入口 | 本版採用方式 |
|---|---|---|
| 棋譜讀取／SGF 解析 | `data.store.GameStore.game`、`data.preprocess.load_store`、`data.sgf.parse` | 直接使用既有 memmap 與 parser；訓練不重轉 SGF；新 store 與既有 game_index 要明確對齊 |
| 舊棋局 metadata／HD | `data/processed/boards/strength_v1/games.parquet`、`data/raw/training/handicap.csv` | 唯讀重用，依 `(rank,game_id)` join 並驗證；不因雙分支重建全量 board cache |
| 玩家 fold | `data.cv.make_rank_folds`、`player_table`、`assert_player_disjoint` | 原 seed=2027 五折／holdout 身分直接重用；M 與 release 四折規則在唯一 CV 模組最小增補 |
| HumanSL 與棋盤規則 | KataGo 原有 `load_model`、`GameState`、`Features`；`features.humansl_torch.profile_metadata`、`_katago` | 模型、規則、歷史、征子等特徵不重寫；僅新增按選定 ply／HD 貼目的串流編碼迴圈與 trunkfinal 讀取 |
| 網路元件 | `reproduce_paper.model.ResidualBlock`、`scalar_head` | 直接用於十層／value；新共用投影、雙分支讀出與殘差 logits 組裝留在本實驗 |
| D4／value 視角 | `shared.dataset.d4_id`、`apply_d4`；`reproduce_paper.dataset.transform_policy`、`soft_value_target` | 直接使用相同幾何與視角轉換，不另寫第二套 |
| Teacher 引擎協定 | `reproduce_paper.label_value.Engine`、`make_query`、`parse_root` | 直接使用逐位置 query、timeout／錯誤處理及視角驗證；新標籤索引／來源保存與每盤 komi 由本版串接 |
| 同色組題／驗證 | `SE_H_C_B2.data.sample_questions`、`check_games`、`check_questions`、`match_metadata` | 直接使用；fit 每 k 20,000 題，監控每 k/rank 125 題；切分後再呼叫，不沿用該模組的舊 make_pools |
| 題目平均／B2 | `tasks.strength.humansl.question_mean`、`GameLR`、`RankModel.probabilities`；已安裝的 sklearn scaler／LR | 題目平均及 scaler＋LR 可直接用；新增極小的「接收固定題目表」擬合接點，不重寫分類器或最佳化器 |
| 官方計分／區間 | `metrics.score_rank`、`rank_report`；`SE_H_C_B2.evaluate.report`、`official_rows`、`paired_bootstrap` | 直接使用分 k 官方分數與玩家配對 bootstrap；不要照搬該模組的 a/se3 舊 evaluate 主流程 |
| DDP／AMP／seed | `shared.runtime.setup_distributed`、`autocast_context`、`on_rank_zero`、`model_state_hash`；`shared.training.seed_process`、`configure_torch`、`rng_state`、`restore_rng`、`environment`、`log_event` | 按現有 config 欄位呼叫；新訓練迴圈只負責階段凍結、實際 loss、microbatch 累積與新 sampler 狀態 |
| 原子存檔／分片 | `artifacts.file_hash`、`object_hash`、`write_json`；`SE_H_C_B2.common.writer_lock`、`atomic_table`、`publish_table`、`load_table`、`assert_frozen` | 直接使用；本版自己組合 stage／fold／上游身分，不使用綁定 AH/C/SE3 路徑的 source_identity |
| Checkpoint 底層 | `reproduce_paper_3.train.atomic_payload`；`shared.checkpoint.resolve_checkpoint` | 沿用重載驗證、原子發布及 hash 檢查；payload 的 state 用可比較的純 Python 值保存游標／佇列，RNG 另存；階段載入與 best/final/latest 選擇由本版定義 |
| 正式題組／CSV | `SE_H_C_B2.predict.release_questions`、`export` | 題目映射及 sample 順序／機率／rank／checksum 驗證沿用；export 只需相容增加預測檔名參數，不把新結果偽裝成 AH+C 或建立另一套 CSV 驗證 |

### 重用界線與最小新增範圍

- `features.humansl_rankvec.load_frozen` 會把 metadata encoder 換成 identity 並改 TF32 設定，僅適用 C 向量對照，不能拿來抽本版固定 rank_1d 的一般主幹；本版直接用原 KataGo loader 與既有 profile_metadata。
- `humansl_torch.encode_game` 是「全盤逐手編碼並 stack」；重用其 GameState／Features 與步進方式，新增選定 ply 的有界輸出，不複製或改寫圍棋規則。`PaperDataset`／`pre_action_inputs` 只允許本人位置且產生 SE 18 planes，不能直接當成雙方 KataGo dataset。
- `label_value.make_query` 已接受任意 analyzeTurns／komi，可直接分析雙方所需位置；但 `source_data` 固定舊 split_seed=2026、export 完成後會刪來源 sidecar，故不使用其舊 label/export CLI 作本版流程。
- `RankModel._fit_synthetic` 的舊抽題不保證同色；用已存在的 sample_questions 取代抽題入口。C=1／max_iter=5000 時 `GameLR` 可直接在題目矩陣擬合；若沿用 RankModel 保存與 probabilities，僅補接既有 per_k scaler／model，不另建 B2 模型類別。
- 不直接複製 `reproduce_paper_3.train.train`：它綁定八卡、舊資料 manifest／shape、單次整批前向、舊聚合與舊 checkpoint 格式。`validation_transition` 以 `>=` 更新 best 並重設 patience，本版同分不更新，需明確實作嚴格改善邏輯。
- `shared.checkpoint.save_checkpoint` 的 state 要求 epoch／next_batch，且保留策略為 latest-only 或最低 training loss；本版各級獨立游標及監控 best/final 不套用此保留策略。重用 atomic_payload／resolve_checkpoint 即可，不為相容舊欄位捏造全域 epoch。
- 正式預測沿用既有 store／cache 與 release_questions 映射；`prepare_release`、`aggregate_frames` 舊入口綁定 SE3/H/C 模型與 fallback，不沿用。新輸出使用完整的機率／rank／模型身分，再交給相容擴充後的既有 export；不偽造 AH+C fallback 或 humansl_games 欄位來通過舊腳本。

必須新增的程式僅為五部分：(1) 兩階段模型／loss 組裝；(2) 雙方位置索引與選定 ply 串流 dataset；(3) 各級獨立 sampler 與四折／階段工作串接；(4) 帶梯度累積、嚴格早停與本版狀態的訓練迴圈；(5) 既有組題／LR／分片／匯出工具之間的特徵與身分接線。舊 H/C 對照沿用其 policy likelihood／向量學習數學，僅依新 fold 與貼目設定安排工作，不另寫一套 likelihood。

既有回歸測試以 `test_se_h_c_b2.py`（組題／bootstrap／發布）、`test_reproduce_paper_value.py`（teacher／視角）、`test_reproduce_paper_data.py`（D4）、`test_checkpoint_retention.py`、`test_hd1_features.py`、`test_release_pipeline.py` 為基礎。保留其原語意，只針對新增接點、不同早停規則與雙方／OOF／累積梯度補測，不複製整套測試，也不把舊測試通過視為新實驗已驗收。

- 輸出預計隔離於 `outputs/rank/humansl_residual_b2/<run_id>/`；資料索引／teacher 放獨立 processed 目錄，不覆寫舊索引、模型或主線提交。
- 每個 run 保存設定、來源模型／資料身分、玩家切分、監控集合、抽樣 seed、環境、loss、階段模型與評估；checkpoint 須可續跑，最終模型驗證成功前不刪續跑資料。
- 支援目標維持 RTX 3060 Ti 8GB 開發與 Nano4 單節點最多 8×H200；實際 batch／microbatch 依量測決定，不承諾相同 batch 可在兩者運作。
- H200 短測起始目標為 8 卡、global 128 組：每卡每個 microbatch 保留一個完整十級組（10 盤×雙方×15 手＝300 位置），累積 16 次後更新一次；每 update 全域 38,400 位置、每卡 4,800 位置。這是資源驗證起點，不以卡數直接推估速度。3060 Ti 僅用縮小的獨立開發設定，不能把不同 microbatch 的訓練視為同一正式 run。
- 可訓練 BN 採 SyncBN；固定 HumanSL／第一階段模組始終 eval，不更新統計。梯度累積不合併 BN 統計：每次前向的統計範圍為當次跨卡 microbatch。正式 microbatch 在短測通過後固定保存；變更大小需新 config，不只當作不影響模型的效能開關。
- 先做 **100 optimizer steps** 的短測，分別量測兩階段的 steps/s、有效位置/s、顯存、CPU 編碼／I/O、監控時間與 checkpoint；另量測 teacher、全有效位置特徵抽取及 CPU B2 成本。
- 完整訓練、teacher、最終全量推論與 GPU-hours 目前均未知。短測後將開發最多五組上游模型、release 五組上游模型、對照 C 交叉擬合／重抽、teacher、所有全位置特徵抽取、CPU B2 與監控逐項估算；依 CLAUDE.md §9 回報長作業預算，不只估一次兩階段模型。

### 11.1 輸入與輸出保存策略

- 唯讀稽核的 HD0/1 為 963,579 盤、185,406,374 個 ply>=30 雙方位置。僅 22-plane bit-packed spatial inputs 每位置 993 bytes，全部約 184.1 GB，尚未含 global inputs，超過現有 100 GB 配額；float32 value 數值約 0.742 GB。
- 第一版不建立全量或持久化輸入磁碟快取，也不保存全量浮點 HumanSL 主幹；每百萬 FP16 主幹位置約 277 GB。磁碟只增加緊湊棋局／ply 索引、具來源身分的 teacher、每盤二十維、計數、完成標記及模型產物。
- CPU worker 從原有棋譜／盤面快取重播歷史，只在本次選定位置計算 KataGo feature planes；不可直接沿用會對全盤每手算 features 並全部 stack 的 encode_game。完整推論按棋局／有界位置批次處理，不把整個資料集堆入 RAM。
- DataLoader prefetch_factor=1、workers 起始每 GPU 4 個，均記錄 config；每個 worker 最多準備其受限預取 batch，不建立無上限字典快取。訓練 batch 回傳緊湊輸入，到 GPU 前向時才展開所需 dtype，避免多 worker 複製浮點主幹。
- teacher 只處理當前階段確實需要且缺失的訓練位置；完整監控、B2 與測試特徵不因無 value target 失敗。release 新增訓練玩家的 teacher 在允許使用 U 後才納入，並可重用完全同身分的開發標籤。
- 特徵按模型／階段／fold／輸入版本及棋局顏色記錄身分，分片完成後原子發布；建立下一階段前先估剩餘空間，不自動刪除舊實驗產物。短測記錄 CPU 重播手數/s、有效位置/s、GPU 等待、RAM／磁碟峰值，瓶頸明確後才另議輸入快取，不預先新增快取系統。

必要工程驗收限於實際實作：HD／ply 邊界與雙方互斥、每方 15 手無重複及續跑一致、凍結參數與 BN 不變、分類／value 對共用層有梯度、實際著手不流入 policy／value、零殘差初始化一致、teacher 身分與視角、盤／題等權及兩組機率和、同色玩家隔離、scaler 無洩漏、分片批次一致與 CSV 完整性。先 CPU 合成測試，再按變更範圍做 GPU／DDP 短測，不把工程測試當成分數驗證。

## 12. 確認後選擇與量測界線

| 項目 | 目前狀態 |
|---|---|
| 兩組分類標籤、共用投影、階段凍結、256 channels／10 blocks | 維持第 4／5 節；第二階段讀 U，不新增讀 F 分支 |
| 有效落子、HD 篩選、每方 15 手、全量聚合 | 已確認 |
| rank_1d、正確貼目、二十維輸出及 P2 二十二維線性 B2 | 本版主流程；純二十維是拿掉 P2 的消融 |
| 玩家切分 | 第 9 節四折 OOF、M=開發玩家分層 10%、U 一次評估、R 保留；不永久排除全部 S/U 只留 854 人 |
| Seed | 玩家／題目／B2=2027，模型初始化／位置抽樣=2026；實際 manifest 待 prepare，不能宣稱已發布 |
| 抽盤 | 各級獨立輪次／游標，不截短；同 update 同級無重複、跨輪衝突延後，保存可續跑狀態 |
| 上游監控 | M 固定 5,000 題、每方 15 手，等權雙分支十維平均＋score_rank；每 5,000 updates／最後一步驗證，三次未提升早停 |
| 候選／完整評估 | 只留 best/final 兩組在 M 做完整位置＋B2 比較；設定鎖定後 U 一次評估 |
| B2 抽題 | 先玩家、再合格顏色、再 k 盤；每 k 20,000 題，C=1、max_iter=5000，玩家層級 bootstrap |
| 最終部署 | release 重新四折 OOF 擬合 S＋U B2，測試用全 P、固定 s1/s2 重訓的單模型，非四折 ensemble |
| H200 batch 起點 | 8 卡／global 128 組，每卡 microbatch 1 組（300 位置）、累積 16 次，SyncBN；需 100 步資源驗證後固定 |
| 輸入快取／teacher | 第 11.1 節：不建全量輸入磁碟快取，有界預取；只補算來源無法證明或新增的必要 teacher |
| 每階段 optimizer／LR／schedule／最大更新數 | 留作短測配置與長作業預算的數值決策，不直接照抄 SE3 的 LR=0.01／100,000 步；尚未核准任何長作業 |
| 100 步短測及正式長作業時間、GPU-hours、分 k 增益 | 未知；本次不提交作業 |

不得把本文寫出的未定事項或未執行驗收改標為完成；後續決策直接更新本文件，實際結果登記於 RESULTS.md。

## 13. 2026-11-09 程式實作與工程驗證

- 依使用者要求分給三個 subagent 完成模型／loss、資料／teacher、特徵／B2；主 agent 整合訓練器、設定、Slurm、文件及跨模組回歸。沒有複製舊 SE3 trainer。
- 第一／二階段入口為 `python -m go_aicup.tasks.strength.humansl_residual_b2.train --stage 1|2`，第三階段為 `.b2`。可追蹤 Slurm 置於本目錄 `slurm/stage1.slurm`、`stage2.slurm`、`stage3.slurm`，另有 prepare、teacher、features、baseline 及兩階段 smoke100。
- 完成獨立 HD／原玩家 fold 稽核、D/M/U/R 與 release 四折、雙方串流編碼／D4、各級獨立抽盤與消費者 durable state、強 teacher 來源／視角／每盤 checksum、梯度累積、凍結 BN、嚴格改善早停、best/final/latest、監控／發布失敗重試、OOF 全位置二十維、P2 線性 B2、受控 H＋C 對照、分 k gate／玩家 bootstrap、M 信心尺度診斷、U 鎖定及單一全 P release 模型／CSV 接線。
- H＋C 對照沿用 likelihood／C 向量數學與 Adam／cosine，為避免持久化輸入 packs，改用全盤位置均勻抽樣的有界更新；不是舊 epoch 遍歷逐步重現。此差異記入對照身分，正式更新預算亦未決定。
- 短測 config 為 `configs/strength/humansl_residual_smoke100.json`：兩階段各 100 updates、AdamW LR=0.001，C 為 100 updates、LR=0.01；僅屬可執行的工程起點。正式 `humansl_residual_train.json` 的 LR／updates 保留 null，需短測後明訂。smoke 禁止鎖定 U／release；沒有核准或提交長作業。
- 整合回歸 **108 項通過，108.24 秒**，涵蓋本版與計畫指定的既有資料／teacher／checkpoint／HD／CSV 回歸；其後新增成本／鎖定檢查由最新 B2 測試另行驗證，詳見 RESULTS。實際兩程序 CPU Gloo 與單程序在固定 BN 下，兩階段 loss／參數一致；中斷續跑的模型、sampler、RNG 一致。實際 spawn workers／預取的續跑亦通過。這不驗證 GPU SyncBN 的數值或吞吐。
- 另唯讀抽取既有 HD0／HD1 各三個位置，新串流 spatial／global／action 與既有 encoder 逐元素完全一致。teacher 真引擎未啟動；使用 fake Engine 驗證完整落子前 prefix、兩種 komi、1 visit、黑白 target 及部分失敗後續跑。
- 本次查得 `data/processed/cv/` 為空，預設原始 seed=2027 完整 fold 表不存在。prepare 必須取回原切分表，不因缺檔重抽；因此未建立官方新題庫／manifest。未生成正式模型、分 k 分數或提交檔；Equal-k 增益、GPU／SyncBN、顯存、吞吐、teacher／全位置推論及完整 GPU-hours 均未知。

輸出隔離於 `outputs/rank/humansl_residual_b2/<run_id>/`；操作、設定欄位、來源前置條件及成本紀錄見 [README.md](README.md)。
