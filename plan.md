# HumanSL policy：200k TRAIN、八卡、十網路與二十個 B3

2026-11-19 最新使用者核准方案。取代先前 20k／400k 與四卡規劃。
使用者授權實作並直接提交完整訓練，跳過 GPU 短測；CPU 驗證、來源與容量檢查仍需完成。
不產生提交 CSV、不上傳、不自動改正式基底。

## 資料與來源

從完整官方 1M canonical metadata 重新選 HD≤1 棋局；棋步只讀現有
`data/processed/boards/strength_v1/`。切分唯一來源為 `data/cv.py`，split seed=20261119。
C-v2 learn_hd0／learn_hd1 包含 learn 與 monitor，合計涉及 4,174 位玩家；
兩種角色全數從新 monitor/dev 排除。凍結 HumanSL plain checkpoint 與 C v2 vectors。

| 集合 | 玩家 | 棋局 | 位置 | 題數 | 每 k 題數 |
|---|---:|---:|---:|---:|---:|
| TRAIN | 4,298 | 200,000，每級20,000 | 44,481,777 | 80,000 | 20,000 |
| monitor | 200，每級20 | 4,000 | 893,253 | 4,000 | 1,000 |
| dev | 400，每級40 | 8,000 | 1,776,590 | 8,000 | 2,000 |

以上已由實際 metadata 唯讀重現，正式執行時再次核對來源／表格身分。
三組玩家及棋局互斥。monitor/dev 每人選同色20盤；TRAIN 依玩家／顏色輪替，
從可供 k=20 的顏色組先取20盤，再公平補盤，避免重度玩家主導。
各 k/rank 等量；題內同玩家、同色、棋局互異；覆蓋優先建題，200,000盤全進TRAIN題組。
TRAIN含3,435位C上游玩家；monitor/dev與C上游零重疊。
新切分不宣稱歷史上未使用，不能與舊dev直接配對。

## 八卡執行與不落盤快取

單節點8×H200／96CPU／200G RAM，8gpus partition。
GPU0–3保留H九profile policy分片，GPU4–7保留C十級policy分片。
每個family依位置數平衡切四份，整盤不跨分片；共享同一process的GPU tensor，
透過跨卡copy取得batch，不為每個模型複製完整policy。

1. 從board cache有界重播，沿用HumanSL encode_game完整planes／歷史。
   HD0 komi6.5、HD1 komi0.5；本人／對手依實際顏色處理。
2. H、C可並行，各完整推理一次：全部47,151,620位置，分別9／10條件。
   FP16 logP[N,P,362]、原始FP32著手q、target／side直接放所屬GPU。
   只將小型每盤本人／對手q均值、counts放RAM。
   保留非法位置-inf；mask沿用空點／排除ko／含pass；P在使用時FP32正規化重建。
   不額外做teacher parity重推；不足整批的尾端eager計算，不補重複位置。
3. 各family先完成A0；八個實驗模型分別在八卡同時訓練，各讀自己的family四片cache。
4. 全十網路monitor選模固定後，從同份policy聚合各網路每盤特徵到RAM，釋放全部policy。
5. 固定每盤特徵訓練五組×四k＝20個B3。全部選模固定後才讀取並報告dev分數。

H快取309,078,869,100 bytes（約287.85GiB）；C343,405,248,460 bytes（約319.82GiB）。
合計約607.67GiB；每卡約72／80GiB，另保留teacher、activation與batch空間。
CPU僅有界plane暫存、metadata、每盤／B3特徵；不保留全部輸入planes，H/C各自重播同一board cache。
不把policy／q／planes／每盤／B3特徵寫入磁碟或大型RAM tmpfs；僅模型、optimizer、scaler、
清單／題組、來源／程式快照、曲線、dev預測與報表可落盤。

## 四臂各加一層，A0不變

H／C均從新TRAIN的共同LR/scaler起點訓練，不載入舊五臂權重。

- R10／Z32：phi改為 input→32→ReLU→32→ReLU→32，多一個Linear(32,32)。
- CNN-R10／CNN-Z32：第三個Conv(32,32,5,padding0)＋ReLU，19→15→11→7，
  感受野13×13，再接原全域平均與原MLP。
- R輸出q+delta（H實為9維、C10維）；Z輸出[q,z32]，各側分開平均，再題內盤等權。
- 殘差末層為0；z分類頭新增欄初值0；paired set／CNN共享各自主體初始值。
  五臂step0機率須與各自共同A0一致。

| 臂 | H總參數，含題目頭 | C總參數，含題目頭 |
|---|---:|---:|
| A0 | 190 | 210 |
| R10 | 10,760 | 11,005 |
| Z32 | 12,895 | 13,075 |
| CNN-R10 | 77,639 | 80,316 |
| CNN-Z32 | 79,774 | 82,386 |

## 更新與早停

seed2027、AdamW LR=.0003；encoder WD=.0001，head WD=0。
上游batch32（每k8），80,000題完整無放回遍歷：2,500 updates/epoch，最多12epochs／30,000updates。
分類頭loss＝CE＋W²/(2×8,000)，bias不加L2；不因題數放大而減弱正則。
TRAIN-only sklearn warm start C=8,000/80,000=.1，scaler固定，encoder BF16、聚合／loss FP32。
每批同一D4，set排列不變；各臂共同題序與D4排程。

step0及每500updates monitor，最多61次。完整monitor；TRAIN曲線取固定rank/k平衡8,000題診斷，
不把它稱為完整TRAIN loss。完整TRAIN特徵於選模固定後聚合。
早停看monitor CE＋head L2：第一epoch只更新歷史最低值，step2500耐心為0；
之後每次嚴格改善才歸零，同分也算未改善；連續5次未改善即停止，最早step5000。
NaN/Inf直接失敗。最終權重仍按monitor官方Equal-k最高選模，同分保留較早步。
保存最低loss、耐心、當時best、optimizer與RNG／D4狀態；記錄實際更新數與停止原因。

## B3

凍結十網路，同名H/C配對，各四個k獨立Linear(d,10)，共20頭：
A0＝原始qH+qC，38維；R10／CNN-R10＝rH,qH,rC,qC，76維；
Z32／CNN-Z32＝zH,qH,zC,qC，166維，z每側32，不讀hidden64。
每盤與每題盤等權；TRAIN-only逐k scaler，五臂同k共同原始38維LR起點，新增欄權重0。
每頭20,000題、batch8、2,500updates/epoch、最多12epochs／30,000updates。
LR=.0003，head WD0，CE＋W²/(2×2,000)，warm start C=.1。
與上游相同每500步monitor、loss耐心5與一epoch暖期；每頭各自monitor官方分數選模。

## dev與gate

全部30個checkpoint固定後，各自只評一次dev；報十網路、五組B3的四k／Equal-k與CE。
同一dev題組做玩家配對bootstrap2,000，報delta各k、Equal-k、95%CI與正向比例。
候選gate：delta Equal-k≥.005、各k≥−.005、Pr(delta>0)≥.90。
相對各自A0及set/CNN對照全部報，不只挑最高者；單seed只列候選，不自動改提交基底。
假設：更多玩家／棋局與較深逐手壓縮器能提供增量訊號；預期收益未知。
限制：單seed、歷史題組使用、B3使用上游in-sample TRAIN特徵、兩階段共用monitor。

## 成本、完成與中斷

八卡預留約8–16小時、64–128配置GPU-h；為舊吞吐外推，跨卡cache與新增層未GPU量測。
Slurm上限36h＝288配置GPU-h；依使用者指示不做GPU短測，直接正式作業。
完整流程在同一次allocation內，不拆成會丟失VRAM的相依作業、不自動requeue。
若節點／程序中斷，模型checkpoint仍保存，但無法同時滿足「不落盤快取」與「不重推teacher」；
不對不完整run自動重推，需另行決定新run。完成重開只驗身分／讀報表，不重跑。

入口：`python -m go_aicup.tasks.strength.policy_large.run`；CPU預檢加`--prepare-only`。
設定：`configs/strength/policy_large_200k.json`；Slurm：`src/go_aicup/tasks/strength/policy_large/train.slurm`。
輸出：`outputs/rank/humansl/policy_large/200k_deep_seed2027/<job_id>/`。
只以根目錄complete.json＋summary.json確認全流程完成；提交成功不代表已有成績。
