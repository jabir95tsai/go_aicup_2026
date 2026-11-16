# 完整 HumanSL policy：A0／R10／Z10／Z32 四臂實驗

日期：2026-11-16。狀態：使用者確認更新設計；本四臂尚未實作、驗收或提交。

本文件定義四臂的架構、loss 傳遞、固定訓練預算與判定方式。
既有 policy_residual 的五臂與六臂 LR 實驗只作歷史依據，不能當成本輪結果。
既有接線見 [policy_residual/README.md](src/go_aicup/tasks/strength/policy_residual/README.md)。
本次只更新計畫，不啟動訓練、產生提交檔或更換正式基底。

## 1. 問題、假設與歷史依據

現有 C 只保留每手實際著手在十個棋力假設下的 logP，再分別平均本人／對手，
得到每盤20維。既有 F 將完整 policy 編碼成十維修正，先做 e=q+Δ，再交給線性頭。
本輪檢查：把新增資訊保留為獨立 z 支路，是否比改寫 q 更容易學習及泛化；
獨立支路由10維增加至32維是否再有增益。預期官方 Equal-k 增益：未知。

歷史六臂 job 787069：batch 32、4 epochs／1,000 updates，S／F 兩個 LR 都選回
step 0。初始 TRAIN CE=.984291、monitor CE=.981758、monitor Equal-k=.726117。
S／LR .0003 在 epoch 3／4 的 TRAIN CE 為 .982510／.981896，低於初始；
monitor CE 為 .983815／.982307，雖後段下降，仍未優於初始。
F／LR .0003 的 epoch 1 monitor CE=.980651 優於初始，Equal-k=.723706 卻較低。
這些數字不能證明十維不足、完整 policy 無資訊，或四輪已足夠收斂。

十維中間表示在理論上仍可能足以形成十級 logits。本輪比較的是接線、維度及其
最佳化效果；32維不代表無損壓縮，也不預設它會勝出。

## 2. 固定資料、輸入與聚合

唯讀重用來源：outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/。
沿用其 games、TRAIN／monitor／dev 固定題組、raw C 與初始 sklearn head；
不重抽玩家或題目，不使用 dev 選預算、LR、架構或 checkpoint。

| 集合 | 玩家 | 棋局 | 全部著手位置 |
|---|---:|---:|---:|
| TRAIN | 1,000 | 20,000 | 4,460,508 |
| monitor | 200 | 4,000 | 898,462 |
| dev | 762 | 12,974 | 2,896,761 |

沿用 A/B 玩家來源、HD≤1、各玩家較多盤的顏色（同數取黑）、每人20盤。
三組玩家互斥；既有 TRAIN 題組實際涵蓋998人／19,784盤，須區分來源池與更新覆蓋。
全部著手位置參與每盤平均，不做每側50手或 ply≥30 篩選。
TRAIN 固定8,000題，k=5／10／15／20 各2,000題；四桶可出題玩家覆蓋分開報告。

每手共同輸入：

- P[10,362]、logP[10,362]：十個棋力假設，361個棋盤點加 pass。
- a：實際著手索引；q[10]：實際著手的原始 FP32 logP。
- 合法遮罩：沿用 C 的空點、排除 ko 禁點、包含 pass；不另排除 suicide／superko。

HumanSL 與 C v2 vectors 凍結且 eval，policy forward 不保留梯度。
HD0 komi=6.5，HD1 komi=.5；不加入 H、HD 旗標或其他模型特徵。
保留原始 FP32 q；快取 FP16 logP 轉 FP32 後依既有方法重新正規化，
不以低精度 policy 重建 q。非法位置在 softmax 中遮蔽，進入 MLP 前填有限值。

各臂均先將本人／對手著手各自等權平均，再將題內 k 盤等權平均。
固定順序為本人特徵、對手特徵；串接臂各側內部為 [q,z]。
長局不因手數較多而增加整盤權重。題內棋局互異；批次引用同盤可去重運算，
但反向梯度須累加。每臂獨立一個跨 k 共用的線性分類頭。

## 3. 四臂與共同 F 編碼器

| 臂 | 每手表示 | 每盤／題特徵 | 線性分類頭 |
|---|---|---:|---|
| A0 | q[10] | 20 | Linear(20,10) |
| R10 | q[10]+Δ[10] | 20 | Linear(20,10) |
| Z10 | [q[10],z[10]] | 40 | Linear(40,10) |
| Z32 | [q[10],z[32]] | 84 | Linear(84,10) |

A0 只訓練原始 C 線性頭。其餘三臂完整沿用現有 F 的候選點處理與 attention，
不改回不同的候選聚合架構：

1. 每候選點22維為十級 P、十級 logP、is_pass、visible；合法候選 visible=1。
   實際著手 token 的 P／logP 欄使用精確 q.exp()／q，沿用既有 F 的處理。
2. phi：Linear(22,32) → ReLU → Linear(32,32)，所有候選點共享權重。
3. 另取實際著手嵌入 chosen[32]。
4. scorer 讀取 [候選嵌入32, chosen32, q10] 共74維，經
   Linear(74,32) → ReLU → Linear(32,1)，合法點 softmax 後加權平均為 summary[32]。
5. 串接 [q10, chosen32, summary32, log合法候選數1] 得75維，經
   Linear(75,64) → ReLU 得共同的 hidden[64]。
6. R10 以 Linear(64,10) 輸出 Δ；Z10／Z32 分別以 Linear(64,10／32) 輸出 z。

Z32 的完整路徑：

```text
完整 policy → F 編碼器 → z[32]
固定 q[10] ───────────→ [q,z][42]
                            ↓
                  本人／對手各自逐手平均
                            ↓
                         每盤[84]
                            ↓
                     題內各盤等權平均
                            ↓
                       B3：84→10
                            ↓
                    logits → 題目 CE
```

F 不讀座標，對候選點同步排列不變；D4 同步處理 policy、實際著手及遮罩，pass 不變。
不加入卷積、flatten、其他候選遮蔽臂或非線性分類頭。

## 4. 初始化與公平性

- 全部載入相同 TRAIN-only sklearn head；原始 q 的 mean／scale 固定不重擬合。
  R10 對 q+Δ 使用此原始 scaler；串接臂 q 欄同樣標準化，z 直接讀入，不擬合 z scaler。
- R10／Z10／Z32 的 phi、scorer、hidden 使用完全相同初始權重與對應參數名稱。
  scorer 最後一層設零，從均勻候選加權開始；R10／Z10 的輸出層形狀相同，
  但依以下公平起點要求採不同初始化。
- R10 的 Δ 輸出層權重及 bias 設零，使初始 e=q。
- Z10／Z32 的 z 輸出層正常隨機初始化；分類頭 q 欄載入共同 sklearn 權重，
  bias 相同，z 欄權重設零。不能將 z 輸出層與讀取權重同時設零。
- 四臂 step 0 的機率、CE、官方分數須在預定浮點容差內一致。
- 每臂獨立更新分類頭與新增編碼器；A0 也依相同規則更新分類頭，不能只更新某些臂。
- 題目順序、batch、D4、預算與選模規則一致；資料統計只可使用 TRAIN。
- 所有新增參數量逐臂報告；初始化與參數化差異本身也是解讀限制。

## 5. Loss 與反向傳遞

唯一監督標籤是題目的十級 rank；不為每手設定額外分類目標。
沿用既有目標（N=8,000個 TRAIN 題目，W 為各臂分類頭全部輸入權重）：

```text
L = batch_mean CrossEntropy(logits, rank) + ||W||² / (2N)
```

分類頭 bias 不加 L2；optimizer 對分類頭 weight_decay=0，避免重複懲罰。
新增編碼器另外使用 AdamW weight_decay=.0001；此為 optimizer 更新規則，
不併入報表的 CE＋head L2。推論用 softmax＋metrics.score_rank；
官方分數只用於評估與選模，不反向傳遞。

以串接臂表示分類頭 logits=W_q Q+W_z Z+b，省略固定 q 標準化的記號：

```text
dL_CE/dlogits = p - one_hot(rank)
第 g 盤、第 s 側、第 t 手：
dL_CE/dz[g,s,t] = W_z,s^T (p-one_hot(rank)) / (k*n[g,s])
```

k 是題內盤數，n[g,s] 是該盤該側有效著手數；batch mean 再除 batch 題數。
相同盤側的各手有相同上游梯度，但編碼器對不同 policy 的導數不同；
梯度經 z 回到 phi、scorer、hidden、輸出層，再按所有題目累加。
R10 的 q 與 Δ 共用分類頭讀取權重；串接臂的 q 與 z 有獨立讀取權重。
head L2 直接作用於 W，不經特徵回傳；梯度不進入 HumanSL、C vectors 或 q。

Z10／Z32 第一步 W_z=0，因此編碼器的 CE 梯度為零，但 W_z 通常有非零梯度；
更新後才開始傳回編碼器。R10 第一步 Δ 輸出層可收梯度，更前層因零輸出權重
暫無 CE 梯度。不得把這些預期的啟動行為誤報為斷梯度。

重用既有有界記憶體分塊反傳：先算每盤特徵與題目 loss，再依每盤特徵梯度
重算逐手 chunk，累加精確鏈式法則梯度。分塊與直接 autograd 須合成驗證一致。

## 6. 固定訓練與選模

| 項目 | 設定 |
|---|---|
| 第一輪 training seed | 2027 |
| batch | 32題，每個 k 各8題 |
| 每 epoch | 8,000題無重複完整遍歷，250 updates |
| 預算 | 12 epochs／3,000 updates，共96,000題目出現次數 |
| optimizer | AdamW，沿用現有預設 betas／eps，LR=.0003 |
| LR schedule | 固定，不新增 scheduler |
| encoder weight decay | .0001 |
| head weight decay | 0；使用明示 head L2 |
| early stopping | 關閉，完整跑完固定預算 |
| 完整 TRAIN／monitor 評估 | step 0、每125 updates，共24個訓練後評估點 |

每 epoch 對每個 k 的2,000題獨立 shuffle、無放回取樣，核對每題恰好一次。
不做梯度累積或 clipping，沿用 CUDA BF16 編碼器運算、FP32 聚合與 CE。
每臂使用相同題序與 D4 排程；12輪是預定預算，不保證已收斂。

只以 monitor 官方 Equal-k 嚴格改善選模型，step 0 可勝出；同分保留較早步。
每 epoch 保存 latest optimizer／模型／RNG／抽樣進度以續跑；半輪評估可保存 best，
但若從 epoch 邊界重跑，必須維持 best 與消費進度的可重現性。
保留 initial、monitor best、latest；曲線保留全部評估點，不保留所有 checkpoint。
全部訓練完成、模型選定後，才對每臂 selected checkpoint 做一次 dev 評估。
不因 dev 結果自動延長訓練、調參、改選其他 checkpoint 或重跑題組。

## 7. 紀錄、比較與 gate

每次完整評估均報 TRAIN／monitor：CE、head L2、CE＋head L2、
k=5／10／15／20 官方分數與 Equal-k；記錄各指標相對 step 0 的差值。
即時 batch loss 僅作執行紀錄，不與不同 batch 比較來判斷收斂。

每次更新保留分組 gradient norm、update norm／ratio、缺失及非有限梯度計數；
NaN／Inf 立即失敗。評估時分開報 q 與新增支路對 logits 的 RMS，
使用固定 TRAIN／monitor 題組，確認支路實際使用程度，不用其數值新增選模規則。
R10 額外項為分類頭作用於聚合 Δ／原始 scale，bias 歸原始 q 分支；
串接臂額外項為 W_z Z，bias 同樣歸 q 分支。
最終報每臂參數量、選中步數、玩家／棋局覆蓋、GPU-hours、顯存、磁碟 bytes。

| 配對比較 | 主要問題 | 解讀限制 |
|---|---|---|
| Z10−R10 | 獨立保留資訊是否比改寫 q 有效 | 參數化、初始化及分類頭約束亦不同 |
| Z32−Z10 | 增加至32維是否有額外價值 | 表示維度與參數量同時增加 |
| Z32−A0 | 整套方案是否提供實用增益 | 不單獨證明其他候選點的新增資訊 |

dev 使用相同題目作配對玩家 bootstrap 2,000次，每次重算四 k 平均；
各比較報分 k 差值、Equal-k 差值、95%區間及 Pr_boot(Δ>0)，不稱傳統 p-value。
第一輪候選 gate：ΔEqual-k≥+.005、各 k≥−.005、Pr_boot(Δ>0)≥.90。
Z32 成為候選須相對 A0 與 R10 均通過；支持32維值得保留另須 Z32−Z10 通過。
所有比較均報告，不只保留勝出的數字。

單 seed 通過只列候選；候選與其必要對照再以 training seeds 2028／2029 確認，
重用相同資料、題組與 sklearn 起點。後續執行另行確認，不自動啟動。
多 seed 確認沿用平均 ΔEqual-k≥+.005、平均各 k≥−.005，並報各 seed 方向與
逐 seed 玩家 bootstrap，不能把三個 seed 或重複題目當成獨立玩家擴大樣本數。

判讀：TRAIN 改善而 monitor 未改善，只支持擬合改善；三個新增臂均選回 step 0，
表示本設定尚未證明有效。若末尾仍下降，只記錄收斂未知，不自動追加預算。
即使勝出，也不能單獨證明十維在理論上不足，或增益必然來自其他候選機率；
後者需未來同架構遮蔽對照，第一輪不增加此臂。

## 8. 算力、產物與已知限制

規劃5張 H200、60 CPU、500 GiB主機 RAM、2小時作業上限：
五卡共同抽取 policy；訓練階段三卡各跑 R10／Z10／Z32，另一卡跑 A0。
配置 GPU-hours 按5卡全作業時間計，不只計忙碌卡。

參考 job 787069 的48.6分鐘／4.05配置 GPU-h，12輪與更密評估粗估90–120分鐘、
7.5–10配置 GPU-h；新架構吞吐、評估成本與真實四桶增益未知。
超時或中斷保留續跑狀態，先回報，不自動追加作業或預算。
完整 FP16 logP 約55.67 GiB，另加 q／索引／模型，供四臂共用 RAM-backed 暫存；
不寫四份 policy、不刪除既有資料，失去快取則需重新抽取。
持久產物先預留200 MiB，實作時依 checkpoint／診斷格式估算，超出時須回報。

新增 run 依 [SHARED_DATA_LAYOUT.md](SHARED_DATA_LAYOUT.md) 放置：
共用固定資料沿用 data/processed/ 的既有來源或其相對連結，run 專屬的權重、
狀態、log、曲線與報表放獨立 outputs/ 目錄；不覆寫785715／787069。
具體新 config、入口與輸出子路徑於實作時固定並記錄，不能借用舊完成標記。

限制：monitor／dev 已歷史重用；C-vector 學習涵蓋190／200位 monitor 玩家；
dev 僅部分玩家可出 k=20，且玩家與題目不是獨立重複樣本。
逐手與多盤仍取平均，順序及變異未直接保留；線性頭不能在聚合後新增 q-z 交互。
本輪 C-only 結果不直接證明正式 H＋C 的互補性或 LB 增益，不自動換正式基底。

## 9. 實作驗收

- 核對來源身分、題組 hash、TRAIN／monitor／dev 玩家互斥與棋局 membership。
- policy 遮罩、pass、精確 q、FP16誤差、機率質量及同步 D4 通過。
- 四臂 step 0 重現同一 C 預測；共用編碼器初始參數一致。
- 核對20／40／84維與本人／對手、q／z 欄位順序，q scaler 不套到 z。
- 平均與梯度權重對變長棋局、同盤多題、batch 去重累加均正確。
- 分塊反傳與直接 autograd 一致；零初始化的首步梯度及後續啟動通過。
- head L2／AdamW decay、FP32 loss、非有限值拒絕及支路 logits 紀錄正確。
- 每輪題組完整無重複、四臂排程一致、半輪評估與 epoch 續跑／best 發布一致。
- monitor 獨立選模、step 0可勝出、同分取早、dev 不參與訓練或選模。
- gate 邊界、配對 bootstrap、分 k 指標與完成標記通過合成驗證。

尚無本四臂的測試結果、GPU量測、模型權重或提交紀錄。
