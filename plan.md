# 完整 HumanSL policy：新資料五臂重訓與持久快取計畫

更新日期：2026-11-17（台灣時間）。本版取代原 A0／R10／Z10／Z32 四臂的下一輪規劃。
使用者本次授權更新計畫；新資料選取、快取與五臂訓練尚未全部實作，未因本次文件更新提交作業。
既有實驗與快照保留；本版不改寫舊 run 的資料身分或成績。

## 1. 目的與比較範圍

更換 TRAIN／monitor／dev 後，A0、原 set R10、原 set Z32 都必須從新 TRAIN 的共同起點重訓，
再與新增 CNN-R10、CNN-Z32 比較；不續接785715／788616的權重、optimizer、scaler或sklearn head。
本輪共五臂，不重跑Z10。預期官方 Equal-k 增益未知；3,000步是固定實驗預算，不保證收斂。

主要問題：CNN保留局部空間配置後，是否比原候選集合編碼更有效；同一編碼器下，獨立z支路是否優於改寫q。
卷積與set比較同時改變參數量、聚合及輸入表示，不能單獨歸因於空間資訊，也不證明32維足以無損保留policy。

## 2. 資料來源與選盤

### 2.1 輸入來源

- Canonical棋局metadata：`data/processed/reproduce_paper/seed2026/games.parquet`。
- 原始HD標註：`data/raw.zip` 內 `raw/training/handicap.csv`，可直接讀取，不必整包解壓。
- 棋步來源：`data/processed/boards/strength_v1/` 的games／moves／colors，用於缺少輸入planes的棋局。
- C4x清單：`data/processed/strength_features/humansl/c4x/learn_{A,B}_hd{0,1}.parquet`
  與 `monitor_hd{0,1}.parquet`。
- 已取回的組裝輸入：同目錄 `learn_A.zip`、`learn_B.zip`、`monitor.zip`，合計約0.75 GiB。
- 凍結teacher：既有HumanSL plain checkpoint與 `outputs/rank/humansl/cvec_v2/learn/vectors.pt`。
  不使用C4x微調權重；C4x只供玩家來源與已組裝planes。

### 2.2 最新切分

只用HD≤1。先按玩家分組，TRAIN／monitor／dev玩家嚴格互斥：

| 集合 | 玩家來源 | 每級玩家 | 玩家總數 | 每人棋局 | 棋局總數 | 題數 | 每個k題數 |
|---|---|---:|---:|---:|---:|---:|---:|
| TRAIN | C4x learn A＋B | 100 | 1,000 | 同色20盤 | 20,000 | 8,000 | 2,000 |
| monitor | C4x learn A＋B，排除TRAIN玩家 | 20 | 200 | 同色20盤 | 4,000 | 4,000 | 1,000 |
| dev | 原C4x monitor玩家 | 36 | 360 | 同色20盤 | 7,200 | 8,000 | 2,000 |

k固定為5／10／15／20，各題同玩家、同色、同級別，題內棋局互異。
同一棋局可以跨題重用，但不能跨資料集合。四桶均按級別平衡出題。
不混合黑白來湊20盤；每盤仍使用本人及對手的全部著手位置，不篩ply≥30，也不只取每側50手。

選取規則與順序固定，seed=2027：

1. 合併learn A＋B清單，找出其玩家在canonical中全部HD≤1棋局；原monitor另作獨立來源。
2. 每玩家只保留至少有20盤的顏色；優先已有pack盤數最多者，其次可用總盤數，再同分取黑。
3. rank_id由0至9，各級玩家先按player_id排序再隨機排列；A＋B每級前100人TRAIN、接著20人monitor。
4. 所選顏色的棋局先按game_index排序並隨機排列，再穩定地將已有pack的棋局排前，逐玩家取20盤。
5. 接著處理原C4x monitor來源，以同一RNG狀態各級選36人dev，依相同規則補足20盤。
   RNG消費順序包含TRAIN／monitor的補盤shuffle，必須固定，以重現下表試算。
6. 三組固定題組分別用seed=2037／2038／2039建立；split只在 `src/go_aicup/data/cv.py` 產生。

A＋B中3,517人、原monitor中476人有至少一色可補足20盤；各級足以支持上述規模。
以此流程得到的manifest交集試算如下；實作時保存並核對hash及實際數字，不能把試算當完成快取：

| 集合 | 可重用盤面 | 需補建盤面 | 需補建位置 |
|---|---:|---:|---:|
| TRAIN | 12,451 | 7,549 | 1,683,401 |
| monitor | 2,536 | 1,464 | 324,192 |
| dev | 1,419 | 5,781 | 1,286,332 |
| TRAIN＋monitor | 14,987 | 9,013 | 2,007,593 |
| 全部 | 16,406 | 14,794 | 3,293,925 |

目前已生成的 `policy_cnn/cached_A_seed2027/` 是先前A-only中間方案：TRAIN1,000人、monitor200人、dev542人。
它不是本版資料；不能直接拿來訓練本輪。現有 `scripts/strength/select_cached_policy.py` 仍需更新到本版規則，
產物改存新版本目錄，不覆寫舊清單。

### 2.3 分組與洩漏限制

A／B是玩家切分，pack的0–7只是I/O分片，不能作為CV fold。
原C4x monitor未參與C4x A／B微調，但不自動等於HumanSL／C v2皆未見玩家；
須重新查核並逐組報告C v2向量學習玩家重疊。舊190／200的數字不能套用到新monitor。
題目重複使用同一玩家／棋局，不增加獨立樣本數；最終bootstrap以玩家為單位。

## 3. ZIP重用、補建與持久policy

### 3.1 已組裝的planes

先核對ZIP完整性、manifest與pack的game_index／手序對應、HD／貼目、輸入格式與來源身分。
既有相容pack直接解包，不重算氣、提子、劫與歷史；缺少的棋局才由完整棋步重播，逐手組裝HumanSL輸入。
普通盤面快取不能直接替代HumanSL完整輸入planes。

ZIP內壓縮的npy不能直接作一般memory map；逐片解壓至計算節點RAM暫存，
讀出需要的棋局後供teacher推論，再釋放暫存。保留原ZIP，不將全部A／B解壓到/work。
本輪依使用者2026-11-17執行指示，補建planes只保留於有界RAM預取流水，與GPU推理重疊；
不另存輸入planes（包含bit-packed補建pack）。中斷時僅重播尚無已驗證policy分片的棋局。
若實體pack缺檔、設定不同或驗證失敗，先回報並修正覆蓋清單，不假裝已可重用。

### 3.2 完整逐手C policy快取

先只建立TRAIN＋monitor快取，全部五臂共用一份：

| 欄位 | 格式 | 說明 |
|---|---|---|
| logP | FP16 [位置,10,362] | 十級完整policy；非法位置保留可恢復遮罩的表示 |
| q | FP32 [位置,10] | 實際著手的原始精確logP |
| target | int16 [位置] | 實際著手索引，含pass |
| side | uint8 [位置] | 本人／對手 |
| offsets／game_index | 整數 | 棋局邊界及固定順序 |

P由logP轉FP32重新正規化後取exp，不另存P；著手圖由target建立，不另存十份標記。
q必須保存teacher的原始FP32結果，不由FP16重建。
HD0 komi=6.5，HD1 komi=0.5；遮罩沿用C（空點、排除ko、含pass，不額外排除suicide／superko）。

分片原子發布，包含checksum、位置數、來源hash及完成標記；重試只接受驗證過的分片，
整組完整後才能供訓練。抽取程式、teacher／vectors／題組身分固定，不能跨版本混用。
共享快取放data/processed，與模型／optimizer所在outputs分開。

新的TRAIN＋monitor約需36.3 GiB；五臂不各存一份。2026-11-17補回ZIP後實查剩51.77 GiB，
扣policy後約15.5 GiB，尚須容納補建pack、模型／log與其他作業，寫入前重新檢查容量。
這是儲存量估算，不是已產生的快取。CNN逐手z會隨參數更新，不能預先固定來代替policy訓練。

## 4. 五臂架構

| 臂 | 編碼器 | 每手輸出 | 每盤／題特徵 | 分類頭 |
|---|---|---|---:|---|
| A0 | 無 | q[10] | 20 | Linear(20,10) |
| R10 | 原F set attention | q＋Δ[10] | 20 | Linear(20,10) |
| Z32 | 原F set attention | [q,z][42] | 84 | Linear(84,10) |
| CNN-R10 | 兩層5×5 CNN | q＋Δ[10] | 20 | Linear(20,10) |
| CNN-Z32 | 兩層5×5 CNN | [q,z][42] | 84 | Linear(84,10) |

所有臂先本人／對手各自逐手平均，再題內各盤等權平均；長局不增加整盤權重。
不加入H、HD旗標或非線性題目分類頭。

### 4.1 原set R10／Z32

沿用788616架構：每候選點P10＋logP10＋is_pass＋visible為22維，
共用phi為22→32／ReLU→32。取實際著手chosen[32]，
scorer讀[v_j32,chosen32,q10]，74→32／ReLU→1，合法點softmax後加權平均得summary[32]。
[q10,chosen32,summary32,log合法數1]共75維，75→64／ReLU→10或32。
實際著手token的P／logP用精確exp(q)／q。
set不讀座標，對候選點同步排列不變；不改成CNN或其他聚合。

### 4.2 新CNN-R10／CNN-Z32

```text
十級P棋盤圖[10,19,19]
＋十級logP棋盤圖[10,19,19]
＋實際著手one-hot標記，重複十份[10,19,19]
                        ↓
                   [30,19,19]
                        ↓
       Conv2d(30,32,kernel=5,stride=1,padding=0) → ReLU
                        ↓
                   [32,15,15]
                        ↓
       Conv2d(32,32,kernel=5,stride=1,padding=0) → ReLU
                        ↓
                   [32,11,11]
                        ↓ 全域平均池化，不flatten
                      c[32]
                        ↓ 串接
     [c32, q10, exp(q)10, pass logP10, is_pass1, log合法數1]
                        ↓ 共64維
                   Linear(64,64) → ReLU
                  ┌─────┴─────┐
           Linear(64,10)   Linear(64,32)
                  ↓             ↓
                Δ[10]         z[32]
                  ↓             ↓
               q＋Δ[10]      [q,z][42]
```

新增數值輸入是實際著手P，不是pass P；保留pass logP。pass不放入19×19格點，
實際pass時著手圖全零，以is_pass表示。非法格點logP進卷積前填有限值。
同步D4變換P／logP／著手標記；q、pass與候選數等數值不隨方位改變。
不使用候選scorer／attention／chosen embedding。

兩層卷積每個末層位置的感受野是9×9；全域平均彙整全盤局部特徵，不代表每個位置已經看過19×19。
卷積本體49,664參數；含hidden／輸出／分類頭，CNN-R10為54,684、CNN-Z32為56,786參數，實作後再核對。

## 5. 新起點、loss與固定預算

先用新TRAIN全部8,000題的原始20維C擬合共同sklearn分類頭與q scaler；
scaler與head不讀monitor／dev。五臂q欄載入同一份初始化，z不套q scaler。
R10／CNN-R10的Δ末層weight／bias設零；Z32／CNN-Z32的z正常隨機初始化，分類頭讀z欄設零。
原set兩臂共用相同set主體初始權重；CNN兩臂共用相同CNN／hidden初始權重。
不同架構不宣稱共用同一編碼器。五臂step0的機率、CE及官方分數須在容差內相同。

| 項目 | 固定設定 |
|---|---|
| training seed | 2027 |
| batch | 32題，k=5／10／15／20各8題 |
| 每epoch | 固定8,000題無放回遍歷，250 updates |
| 每臂預算 | 12 epochs／3,000 updates |
| optimizer | AdamW，LR=.0003，固定LR，沿用betas／eps |
| encoder weight decay | .0001 |
| head weight decay | 0；loss另加head weight L2 |
| loss | 題目平均CE＋分類頭weight平方和／(2×8,000)，bias不加L2 |
| 運算 | BF16 encoder，FP32聚合／loss，凍結HumanSL與vectors |
| 監控 | step0及每125步完整TRAIN／monitor；25點含初始 |
| 選模 | monitor官方Equal-k嚴格改善；同分保留較早步，step0可勝出 |
| early stopping | 關閉；不自動增加步數或seed |

所有臂相同題序與D4排程，各有獨立模型、optimizer及RNG；執行先後不能改變另一臂的抽樣。
重用有界記憶體分塊反傳，棋局去重後梯度仍需正確累加；不加梯度clipping或額外loss。
保存initial／monitor best／latest，latest含完整optimizer／RNG／題序進度及當時best。
半輪中斷從輪界確定性重播，回復輪界best，不能把重播中的半輪best錯誤保留。

## 6. 執行順序與資源分離

以下是後續執行順序，不表示本次已啟動任何階段：

1. **檢查來源**：核對ZIP、pack結構與資料身分，確認原始棋步、HD、HumanSL及C v2權重可讀；
   查C-vector玩家重疊與可用磁碟。保留所有原始ZIP與舊run。
2. **凍結新切分／題組**：按§2建立新games、reuse／supplement清單、三組題組及hash；
   驗證玩家互斥、同色20盤、HD≤1、各k／rank數量及試算覆蓋。dev只建清單，不抽特徵、不看成績。
3. **多卡抽取TRAIN＋monitor**：規劃5×H200、60 CPU、500 GiB RAM；
   從learn A／B ZIP重用相容planes，補建9,013盤的planes，產生並驗證持久policy分片。
   完成即釋放多卡配置，不把五卡一直占到訓練結束。
4. **新TRAIN起點與短測**：建立共同TRAIN-only scaler／sklearn head；
   在獨立短測輸出量測單卡batch32的set／CNN更新及完整評估時間、顯存、I/O、續跑，
   驗證step0一致與分塊梯度。短測模型不用來挑架構或當正式起點；正式從共同初始化開始。
   先回報短測成本，正式訓練資源上限據此確定。
5. **單卡五臂正式重訓**：同一張H200依序 **A0 → R10 → Z32 → CNN-R10 → CNN-Z32**，
   每臂完整3,000步，對相同TRAIN／monitor評估、各自選best。
   五臂共用同一policy快取，無需同時保留五份模型計算圖；個別完成後可由下一臂接續，
   不重算teacher、不提前使用dev。此處將先前「單卡兩臂」擴充為本輪五臂。
6. **最後才處理dev**：五臂均完成預算並鎖定monitor best之後，
   使用monitor.zip重用1,419盤，補建5,781盤；凍結teacher抽dev policy，
   僅對五臂selected checkpoint各做一次dev評估。不推論所有歷史checkpoint。
   dev policy優先放RAM，或逐分片供五個凍結模型共用；不與36.3 GiB持久TRAIN／monitor policy
   另存整份dev到/work，避免超出配額。保存dev逐題預測與selected逐盤特徵即可。
7. **配對比較與報告**：全部比較用同一dev題組，做玩家bootstrap、分k與成本報表；
   更新RESULTS，不自動改正式基底、不自動啟動多seed或上傳。

CPU新邏輯／來源驗證通過才進GPU。長作業前先告知估時；提交後不輪詢，完成後依使用者要求分析。
中斷可用已驗證policy和各臂latest續跑，禁止因缺RAM快取而無條件重抽teacher。

## 7. 比較、判定與紀錄

每次TRAIN／monitor評估報CE、head L2、完整objective、四k官方分數、Equal-k與相對step0差值。
逐步記錄分組gradient／update norm與ratio、缺梯度及非有限梯度；NaN／Inf立即失敗。
記錄q／新增支路對logits的RMS、實際玩家／棋局覆蓋、選中步數、參數量、顯存、牆鐘與配置GPU-h。
即時batch loss不拿來直接比較不同batch的收斂。

| 比較 | 問題 |
|---|---|
| R10−A0、Z32−A0 | 原set分支在新資料上的增益 |
| Z32−R10 | set內獨立支路與殘差 |
| CNN-R10−R10 | 殘差接線下CNN與set |
| CNN-Z32−Z32 | 獨立支路下CNN與set |
| CNN-Z32−CNN-R10 | CNN內獨立支路與殘差 |
| CNN-R10−A0、CNN-Z32−A0 | 新架構整體實用增益 |

所有比較都報告，不只挑最高分；主要CNN比較事前固定，不依dev改選比較方式。
dev按玩家配對bootstrap2,000次，每次重算四k平均；報Δ各k、ΔEqual-k、95%區間、Pr_boot(Δ>0)。
候選gate沿用：ΔEqual-k≥+.005、各k≥−.005、Pr_boot(Δ>0)≥.90。
聲稱CNN優於set須對應CNN-R10−R10或CNN-Z32−Z32通過；列整體候選另須相對A0通過。
聲稱獨立支路優於殘差須同架構的Z32−R10通過。

dev不選LR／步數／checkpoint，不依結果自動延長訓練；要延長應另訂monitor-only預算。
單seed過gate僅列候選；2028／2029確認另行決定。不能把重複題目或多seed視為獨立玩家。
新切分不能與785715／788616舊成績直接配對；原比較僅作歷史背景。

## 8. 成本、目錄與狀態

- TRAIN＋monitor約534萬位置；policy持久快取約36.3 GiB。
- 補建planes總量：TRAIN＋monitor約201萬位置，dev約129萬位置。
  CPU組裝與GPU teacher推論可流水重疊，不能把兩者時間直接相加。
- 多卡抽取粗估20–40分鐘、5卡約1.7–3.3配置GPU-h；重用ZIP後真實吞吐待量測。
- 既有set分支約1.8秒／步（含定期評估），CNN速度仍未知；四個神經臂改為單卡依序跑，
  正式訓練粗估6–9小時、約6–9配置GPU-h，A0另加少量時間；不是原五卡全程2–3小時的配置。
- dev抽取／五臂評估／bootstrap的時間另計，尚未量測。短測後重新報總成本與Slurm分段上限。
- 不另存FP32完整policy、不為五臂複製policy、不把全部ZIP解壓到持久磁碟。

建議新版本路徑（待實作建立）：

- 共用清單／題組／TRAIN-monitor policy（補建planes僅在RAM）：
  `data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/`。
- 模型／optimizer／log／報表：`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/<run_id>/`。
- 保存config、計畫與程式快照、teacher／vectors／ZIP／清單／題組hash、cache schema、
  分片checksum／完成標記、環境及提交紀錄。只寫本輪獨立目錄，舊run維持原樣。
- 模型與診斷先預留200 MiB；新增planes與policy另外估算，不把200 MiB誤當全部快取預算。

### 實作／驗收狀態

- 已完成：舊四臂實作與歷史CPU驗收；A-only選盤工具及清單；A＋B／原monitor方案的唯讀補盤試算。
- 本次完成：更新本計畫，固定新資料與五臂範圍、架構、預算和執行順序。
- 尚待：A＋B選盤工具與正式清單、ZIP相容性檢查、持久policy抽取／續跑、CNN新架構、
  單卡五臂入口與獨立RNG、dev最後評估入口、CPU／GPU短測及正式提交。
- 驗收須含：五臂step0一致、5×5／padding0尺寸與pass處理、D4同步、精確q、
  ZIP與重播planes的抽樣一致性、補盤覆蓋、玩家／棋局互斥、分塊梯度、cache損壞拒絕／中斷續跑、
  monitor獨立選模、dev延後、bootstrap及gate邊界。沒有完成標記不能宣稱成功。

### 2026-11-17 本輪抽取入口

使用者授權先抽TRAIN＋monitor、5卡送出後結束對話，不持續監看；本輪不訓練、不抽dev。
選盤已生成：TRAIN 20,000盤／4,459,699位置，monitor 4,000盤／884,079位置，
重用14,987盤、補建9,013盤／2,007,593位置；三組重用與補盤數已逐項重現§2試算。
完整輸出5,343,778位置，約36.25 GiB；只持久保存logp FP16、q FP32、target、side、offsets及game_index。

入口：`python -m go_aicup.tasks.strength.policy_residual.persistent_cache`；
設定：`configs/strength/policy_cache.json`；
Slurm：`src/go_aicup/tasks/strength/policy_residual/persistent_cache.slurm`。
5×H200／60 CPU／500 GiB／2小時上限（最多10配置GPU-h）；粗估20–40分鐘，實測尚未知。
清單及快取在`data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/`，
policy分片在其`policy/`；程序來源、設定、檢查與worker紀錄在
`outputs/rank/humansl/policy_cnn/cached_AB_mondev_seed2027/extract/<job_id>/`。

每GPU以一個有界預取批次重疊CPU補建／ZIP解包與GPU推理；實體ZIP逐pack在RAM解壓，
每32盤原子發布，checksum及來源身分一致才接受續跑。CPU先核對A/B兩種HD的ZIP／重播抽樣，
GPU啟動前檢查兩份ZIP全部CRC；每pack另抽樣比對。整份完成標記僅在全分片校驗通過後發布。
完成狀態須看`policy/complete.json`；作業提交本身不代表快取完成。


### 2026-11-17 取消後的CPU/GPU流水線優化

使用者先取消792644，再要求盡量保持GPU推理與CPU重建持續運作；本輪只修改與CPU驗證，不重新提交。
已取消作業留下25個完整分片（約1.3 GiB），保留原`policy/`及舊來源紀錄。
優化版使用schema 2與獨立`policy_stream_v2/`，log／來源快照改為`extract_stream_v2/<job_id>/`，
不跨抽取程式版本混用舊分片，選盤與teacher不變。

1. 每GPU兩條獨立供料線從一開始並行：ZIP解包最多緩衝4片；10個CPU重播worker持續補建缺盤，
   ProcessPool同時保持兩片待處理。補盤的RAM佇列上限由該GPU的缺盤片數決定，足以容納其全部補盤，
   避免CPU因GPU正在推理ZIP資料而停工；全部補盤壓縮輸入約2 GiB，全部24,000盤約5.4 GiB。
2. 重播worker在跨程序傳輸前packbits，ZIP直接保留既有bits；H2D仍為bits，在GPU使用既有unpack。
   plane部分的IPC與H2D資料量約降為原本1/8；不落盤保存任何額外輸入plane。
3. 一片的輸入一次傳入GPU，推理批次間不做CPU scalar／逐批D2H同步；logP與原始q先保留GPU buffer，
   再以獨立CUDA copy stream非同步複製到pinned RAM，以event保證CPU讀取前複製已完成。
4. 每GPU背景writer負責精度gate、原子寫片、checksum與發布後重新讀取校驗；最多2片輸出在途，
   I/O落後會施加有界backpressure，錯誤會傳回主流程。GPU可同時推理下一片。
5. 完整結束時核對writer驗證收據與marker hash，避免把全部新policy再次讀過一遍；
   續跑舊片仍完整校驗。保留來源檢查、ZIP CRC、精度與資料完整性gate。
6. 記錄input wait、writer backpressure及CUDA event推理／D2H耗時，用於後續短測找出供料不足；
   這些不是硬體GPU utilization百分比。啟動檢查／編譯和最後drain仍有固定成本，不保證零空檔。

配置仍為5×H200／60 CPU／500 GiB／2h上限；20–40分鐘仍是舊粗估，優化後吞吐未知。
CPU合成驗收覆蓋精確q／mask／pass／尾批、打包解包一致、兩條producer獨立供料／容量界限／
中途停止／失敗傳遞、async writer等待copy event後才讀取及原子checksum損壞拒絕。
正式GPU加速、CUDA跨stream與實際等待比例尚未驗收，不自動重送作業。


### 2026-11-17 使用原檔名重新提交（最新執行指示）

使用者明確要求提交、刪除已取消作業792644的產出並以原檔名繼續。
因此清除該次`policy/`的25個舊分片與舊identity、`extract/792644/`及其Slurm log；
同步移除已失效的`extract/preflight/`、未使用的`policy_stream_v2/`與`extract_stream_v2/`預檢產物。
保留共用選盤／題組、原始ZIP、棋步快取與teacher權重；RESULTS保留歷史實驗紀錄。
優化後schema 2抽取器改回原`policy/`輸出，run／來源／log紀錄回到`extract/<job_id>/`。
清除後重新建立來源身分，從零抽取TRAIN＋monitor，不混用已取消作業的舊分片。
配置維持5×H200／60 CPU／500 GiB／2h上限；20–40分鐘仍為粗估，優化後實際效能未知。
重新跑CPU測試與真實來源預檢後提交，依使用者要求提交成功即結束、不監看、不接續訓練或dev。
