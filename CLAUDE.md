# AI CUP — 圍棋棋力與風格預測競賽（競賽一：玩家辨識 / 競賽二：棋力預測）

> 兩個競賽**共用同一份訓練資料**，因此本專案是 **一個 repo、共用資料管線、兩條任務訓練與提交線；是否共用預訓 encoder 由對照實驗決定**。
> **兩個競賽同等重要（2026-10-20 使用者決定）：** 規劃、算力、時程與上傳額度兩賽等量對待，不以任一賽為主、另一賽為輔。
> 本檔案是專案唯一事實來源。標記「待確認」的項目在開工第一天必須查清並回填。
> 第 0 節已於 2026-09-27 依官方競賽頁回填（見 `RESULTS.md` A-006）。

---

## 0. 官方設定（2026-09-27 回填；✅ = 已確認，⏳ = 測試集 11/04 發布後才能確認）

### 評估指標 ✅

**競賽一（玩家辨識）— Top-5 指數衰減：** 正解在 top-5 第 r 名得 `e^{-(r-1)}`，不在榜得 0；400 題平均。

| 名次 | 1 | 2 | 3 | 4 | 5 | 榜外 |
|---|---|---|---|---|---|---|
| 得分 | 1.000 | 0.368 | 0.135 | 0.050 | 0.018 | 0 |

→ **本質上是 top-1 accuracy**，第 2–5 名合計最多只值 0.57，且衰減比 MAP@5（1/r）陡得多。
→ 期望分數對名次是線性的且權重遞減 → **top-5 依後驗機率由高到低排就是最佳**，不需要特別的排序技巧；戰場在把正解推上第 1 名。

**競賽二（棋力）— 精確 / 差一級：** 完全正確 1 分、差 ±1 級得 `1/e ≈ 0.368`、其他 0；題平均。

→ **不是 Accuracy、不是 MAE、不是 QWK。** 差兩級以上 = 0 分，所以 ordinal 期望值解碼不划算。
→ **Bayes 最佳解碼：** 猜 r 的期望分數 = `p[r] + (p[r-1] + p[r+1]) / e` → 取這個的 argmax（`metrics.score_rank`），不是 `p` 的 argmax。
  邊界級（D、6D）只有一個鄰居可分分數，會被自然降權。
→ head 選擇：**10-way softmax（校準好的機率）+ `score_rank` 解碼**。ordinal 結構用 label smoothing 到鄰級／CORAL 當正則化來實驗，不是必需。
→ 實測（A-006，線性探針）：`score_rank` 比 argmax +0.009、比期望值解碼 +0.022。

### Baseline 分數 ✅（= 報告／獎狀門檻，Private LB 須超過）

| | 官方 baseline | 我們目前的本地數字（2026-10-22 更新） |
|---|---|---|
| 競賽二（棋力） | **0.4631** | **Public LB 最高 0.7508820（2026-11-09，`ahc-p3-noA-t20-allp-prior_736589_rank.csv`：P1＋P2＋P3 拿掉 A-024、以 Valid-S＋train20＋Valid-U 擬合、B2 合成題先驗對齊）；演進 AH＋C 0.7103 → P1＋P2 0.7165 → P3 0.7197 → noA 0.7271 → noA-allp 0.7386 → ＋train20 0.7405 → ＋先驗對齊 0.7509（＋794 排除玩家 0.7468，未採用），見 `RESULTS.md` 11/08–11/09。** **11/11 使用者決定不理 gate、以 `751959-cvec2.csv`（C 換為 v2 重學向量；本地 E1 −0.0008／E2 +0.0015，持平）為新基底；LB 11/11 0.7484（＋L10 0.7500），兩者各比舊版 −0.0025，雜訊內。** 本地：**A-024 每盤特徵＋KataGo HumanSL likelihood＋B2 題目層級 LR（2026-10-25 更新，A-044）：Valid-U 1,000 題 Equal-k 0.6833**（k=5/10/15/20：0.617/0.679/0.722/0.715）；同題 V2 ResNet（A-024）＋C2 為 0.5532（全部 10,000 題 0.5553）→ 已過 baseline（本地合成題目，不是 LB）。舊的手工特徵線性探針為 0.4117（A-006） |
| 競賽一（玩家辨識） | **0.2845** | **Public LB 基底 `820880-likeb.csv` 0.8542891（2026-11-21，依事前保留規則；`RESULTS.md` 11/21「經驗貝氏先驗＋逐 k 學習融合權重」）：似然項改用全部 50 維逐手特徵（G1–G4 加回），候選人 θ 的先驗由 N(θ_pop, 3²I) 改為經驗貝氏 N(θ_pop, Σ)，Σ 為驗證台 pool 0 玩家四組 10 盤 θ 的交叉共變異（玩家間共變異，c＝1），λ＝4；封存 pool 1 對 likrich +0.0109（z 5.9，k=1/2/3/5 +0.012/+0.016/+0.010/+0.005），Public +0.0007。**最終上傳用 `820880-likeb.csv`**。前一基底 `818024-likrich.csv` 0.8535888（`RESULTS.md` 11/21「似然項加入更豐富的逐手特徵」）：在下述 `817757-lik2lk.csv` 上，似然項的逐手特徵再加局部棋形 G3（碰／長立／尖／一間跳／小飛，各 ×手數≥100，10 維，以盤面重播計算，共 37 維）；驗證台封存 pool 1 對 lik2lk +0.0109（z 7.8，k=1/2/3/5 +0.013/+0.010/+0.010/+0.011），pool 0 +0.011；Public +0.0074。前一基底 `817757-lik2lk.csv` 0.8461584（`RESULTS.md` 11/20 兩節「玩家條件走子似然第 0 段」「似然項擴大 λ／σ 網格＋逐 k 風格倍率疊加」）：新增集在 ogsa12w 融合上加玩家條件走子似然比 3·LLR。LLR 的基底 policy 是凍結 HumanSL＋C v2 四個級別條件（C／A／2D／5D），每位候選人以 10 盤參考盤擬合 27 維對數線性殘差（先驗往族群收縮，σ＝3）。風格項另用逐 k 倍率 0.75／1.25／2／3（原為 1.5）。驗證台封存 pool 1 對 ogsa12w：只加 2·LLR +0.030（z 17.6），再加逐 k 與 λ＝3 共約 +0.035。HumanSL／C v2 現在也用於競賽一，外部資料申報見 `EXTERNAL_RESOURCES.md`。再前一基底 `817714-lik.csv` 0.8464716（只加 2·LLR、不改風格倍率；與 lik2lk Public 差 −0.0003，雜訊內）。更早的基底 `ogsa12w.csv` 0.8388310（2026-11-20 使用者決定，`RESULTS.md` D-008：新增集風格項＝1.5×(0.25 seqB＋0.75·mean(A1, A2))，A1／A2 為幽玄＋OGS 從頭訓練的 E2E 編碼器；換檔門檻改為封存 pool 1 Δ>0 且 z≥2.5、各 k≥−0.005、pool 0 Δ>0；OGS 授權已於 11/20 確認）。前一基底 `hung-seqB-e2emix-ml.csv` 0.8245254（2026-11-18：在 `hung-seqB.csv` 0.8012 上，新增集風格項改為 B 頭與從頭端到端 E2E9600 編碼器 cosine 各半，再加開局多層比對 L1 角／L2 應手；驗證台封存題組 pool 1 對 seqB +0.045，`RESULTS.md` 11/17–11/18）。** 舊數字：開局 pp+slot、D4 正規化（A-029）：合成驗證 0.9494（0.838/0.971/0.989/1.000），加 L6 Hungarian 0.9829 → **遠高於 baseline，但合成驗證已飽和，且與 baseline 的落差未解（見下方警訊）**。零訓練的黑前 4 手 naive Bayes 為 0.929（A-006） |

⚠️ **競賽一的落差是警訊，不是好消息：** 零訓練計數模型在合成驗證拿 0.93，官方 baseline 卻只有 0.28。
可能原因（**未知**，測試集發布後第一件事查）：(a) 官方 baseline 很弱；(b) 真實候選庫／查詢盤的切法（例如依時間切）讓開局習慣不穩定；
(c) 真實候選人不像合成庫（≥120 盤重度玩家、第一手集中度 0.95）那麼有習慣性；(d) 測試盤做過方位打亂。
→ 在查清前，**合成驗證只支持目前情境內的比較；絕對分數與相對排序都不保證轉移到正式測試**。
→ **最糟情況規劃（2026-10-23 使用者確認，`RESULTS.md` D-005；原 main 分支編號 D-002，10/28 合併改號）：** 合成驗證已飽和（A-027 ≥0.99），競賽一改以最糟情況下的保底分數規劃：壓力測試台 S1–S4、不靠開局的風格模型、gate G1–G4、**Hungarian 預設關閉直到確認一對一**、11/04 先交保底檔。現況：S1／S3／S4、風格模型（GE2E 保險）、G1／G2 已完成；**G3 FAIL**（10/28 在 `t1-pipeline-t2-models` 分支完成，11/14 補登 `RESULTS_ARCHIVE.md` A-063：查詢盤開局全換掉時合併 0.126 ≪ 只用風格 0.936，換 60% 時已低 0.19、換 30% 時仍高 0.017）→ 原測試集若有大範圍開局漂移，開局權重過大；穩健開局聚合 gate 未過（A-064：無漂移時 −0.014）；漂移偵測程式未搬進 main（tag `archive/t1-pipeline-1030`）；指派已改為 Hungarian 與 Sinkhorn 都預設關閉（D-004）。**11/14 LB 探針改判原測試集部分：`772748-hung.csv`（只把原測試集換 Hungarian、改 Public 4 題）Public +0.0045，約等於 4 題全改對 → 原測試集為一對一，競賽一基底改為 `772748-hung.csv`；新增集官方明文非一對一，不指派（`RESULTS.md` 11/13 Hungarian 探針節）。** **11/17 新增集風格項改用 B 逐手 MLP 聚合頭（封存題組三 seed 全過 gate，新增集 +0.023～+0.028），競賽一基底改為 `submissions/pid/hung-seqB.csv`（Public 0.8012）；舊提交檔已移至 `submissions/pid/archive/`。** 逐項對照見 `EXECUTION_PLAN.md` §9.1。

### 時程 ✅（台灣時間）

| 日期 | 事件 |
|---|---|
| 2026/09/18 | 訓練集開放 |
| 2026/11/03 23:59 | 報名截止 |
| **2026/11/04 11:00** | **測試集發布 + 開放上傳**（此前無 LB 可用） |
| 2026/11/06 11:00 | **競賽一**暫停上傳並清空排行榜（官方 11/06 公告，為「提升鑑別度」新增測試集） |
| **2026/11/11 11:00** | **競賽一**重新開放：下載更新後資料，提交須同時含原有題與新增題，格式當天公布；11/04 版提交檔作廢 |
| **2026/11/24 23:59:59** | **上傳截止**（逾期不收） |
| 2026/11/25 16:00 | 公布 Private LB |
| 2026/11/26 11:00 – **12/06 23:59:59** | 優勝隊伍繳交報告、程式碼、外部資料（逾期不收、不可補交修改） |
| 2027/01/04 11:00 | 公布最終名次 |

→ LB 窗口原為 **21 天 × 5 次 = 每賽 105 次上傳**。11/04 前的所有決策只能靠本地驗證。
→ **競賽一（2026-11-07 依官方 11/06 公告更新）：** 有效窗口縮為 11/11 11:00–11/24，**14 天 × 5 = 最多 70 次**；排行榜已清空，最終仍以最後一次上傳計分。新增題目的內容未知；若候選庫仍為 400 人而題數增加，則必定**不是一對一**（指派維持關閉，`_hungarian`／`_sinkhorn` 探針可能失效）。截至 11/07，本機（Nano4）沒有任何競賽一真實候選庫或測試檔，Day-0 稽核尚未執行。
→ **競賽一 11/11 官方公告（2026-11-12 回填，《更新版測試集資料及提交說明》）：** 排行榜全面重置、舊提交不計分。題目包＝原測試集（`pub_`／`priv_`，不變）＋新增測試集（`pub2_`／`priv2_`）。新增集：候選庫 `player_identification_candidates2_{public,private}.csv` **每人只有 10 盤、黑白都有**（`color`＝該玩家實際執色）；測試題 `player_identification_test2_{public,private}.csv` 各 400 題，**k ∈ {1,2,3,5} 各 100 題、黑白混合**。官方明文「同一玩家可能對應多題、候選庫可能有不對應任何題目的玩家」→ **新增集確定不是一對一**。提交格式見第 0 節「仍待確認」表。**11/12 資料到手並完成 Day-0 稽核（`RESULTS.md` 11/12）：** 新增集候選庫為 **600 人 × 10 盤**（每人黑棋 0–10 盤、平均約 5）；八個競賽一檔案 HD 只有 0／1、**沒有 HD≥2**；原測試集題目未被旋轉、候選人第一手集中度與合成庫相同，top-1 幾乎各指向不同的人（396／395 個不同候選人）→ 原測試集很可能一對一。第一份合法保底檔 `submissions/pid/756494-floor.csv`（新增集只用黑棋盤，約 25% 題目無黑棋而 0 分）；新增集的雙色模型尚未實作。
→ **競賽二：** 公告頁只有 11/02 的 HD 訓練資料更新，沒有換測試集公告，11/04 測試檔視為仍有效；11/06 暫停未波及競賽二：當天 13:03、16:24 仍成功上傳並計分（`RESULTS.md` 11/08 補登）。

### 計分 / 排名 ✅
- Public LB 在上傳期間可見；**Private LB 不公布，最終分數取最後一次上傳**
- Public 同分 → **先上傳者名次在前**
- 獎項（兩賽各自獨立）：學生組前 16 名有獎金（4 萬 / 2.5 萬 / 1.5 萬 / 5 千×3 / 3 千×10）+ 教育部獎狀；
  前 30 名且前 25% 且超過 baseline → 繳報告審查後發電子獎狀。報告評分 = 完整性 + 正確性 + **程式原創性**
- 得獎隊伍須提交**可重現模型的原始程式碼**

### 算力 ✅（2026-10-10 更新支援範圍）

- 必須支援的目標環境只有 **本機 RTX 3060 Ti 8GB（CUDA 12.4）與國網中心「晶創 26」**；取消 AMD 單卡／ROCm 相容性要求
- → 兩個目標都以 NVIDIA／CUDA 為基準，可使用 CUDA 專屬 kernel，無須為 ROCm 提供替代實作；訓練入口預留 DDP／torchrun（單節點多卡；無須支援跨節點訓練）；資料路徑、batch、precision、裝置都走 config
- → `EXECUTION_PLAN.md` 的算力預算是依 3060 Ti 估的，**需要依 H200 實測重估**（尤其 P3 盤面 CNN、P4 KataGo 特徵的可行規模）；帳號剩餘 GPU 小時、儲存配額與實際軟體版本仍未知

#### 晶創 26（Nano4）H200 環境（2026-10-10 查核）

**使用範圍：** 本專案只使用 H200 的 `dev` 與 `8gpus` 佇列，每個作業限單節點、最多 8 GPU。

- **硬體：** H200 節點共 220 台；每台 8 張 NVIDIA H200、2 顆 Intel Xeon Platinum 8480+（共 112 CPU 核心）、2TB 主記憶體、8 個 InfiniBand 400Gb/s 網路埠。H200 節點與登入節點為 x86_64，作業系統為 Red Hat Enterprise Linux。[官方系統規格](https://man.twcc.ai/@nano4-manual/SJuKzVlwbx)
- **顯存與互連：** Nano4 的 GPU 型號為 H200 SXM5 141GB；H200 每卡為 141GB HBM3e、4.8TB/s 顯存頻寬。節點內 8 卡透過 NVLink 互連，跨節點走 InfiniBand。[TOP500 系統登錄](https://www.top500.org/system/180421/)、[NVIDIA H200 規格](https://www.nvidia.com/en-us/data-center/h200/)、[官方系統規格](https://man.twcc.ai/@nano4-manual/SJuKzVlwbx)
- **節點與 GPU 數：** 8 GPU 完整配置 = 1 個 H200 節點，本專案只規劃單節點作業。顯存仍分屬各 GPU，不能當成單卡共享顯存。`8gpus` 是佇列名稱，實際配置以作業申請與 Slurm 分配為準。

**排程器：Slurm Workload Manager。** 以下只列本專案使用的 H200 佇列；查核時官方政策如下。[官方 Slurm 佇列表](https://man.twcc.ai/@nano4-manual/SJM_FuxDWe)

| 佇列 | 單一 account 可用 GPU 總數 | 每 job 最少 GPU | 每 job 最長時間 | 每 user 同時 running jobs | 每 user 同時提交作業（含 running／pending） |
|---|---:|---:|---:|---:|---:|
| `dev` | 32 | 1 | **1 小時**（2026-11-20 `scontrol` 實測 MaxTime；原官方表為 4 小時） | **2** | **7** |
| `8gpus` | 32 | 1 | 48 小時 | 8 | **13** |

**2026-11-12 現場查核：** `scontrol show partition` 與 `sacctmgr show qos` 顯示 `dev` 套用 `p_dev`（`MaxJobsPU=2`、`MaxSubmitJobsPU=7`），`8gpus` 套用 `p_8gpus`（8／13）。上表作業數已依現況修正；`QOSMaxJobsPerUserLimit` 表示該佇列同時執行的作業數已滿，與是否還有空 GPU 是不同限制。`8gpus` 可只申請 1 張 GPU。

H200 佇列限制**每張 GPU 最多 12 CPU 核心、200GB 主記憶體**；200GB 是主機 RAM 配額，與 141GB GPU 顯存不同。佇列名稱不等於每 job 的最少 GPU 數；提交前以 `sinfo`／`scontrol show partition` 查當下政策與實際 job 上限。

**本專案使用方式：** 本機 3060 Ti 8GB 做開發與小規模驗證；H200 `dev` 做環境、單卡與單節點 DDP 短測；通過後以 `8gpus` 做單節點最多 8 卡訓練或分片推論。這是工作分配規劃，尚未實測吞吐量或多卡加速比。長作業須定期 checkpoint 並可續跑；NRQ 工作可能被中斷、重新排程。預算以 GPU 小時記帳（GPU 數 × 執行小時），不能由卡數直接假定線性加速。[官方 NRQ 說明](https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=54)

**連線與環境：** 登入 `nano4.nchc.org.tw:22`；SFTP 資料傳輸使用同主機的 `2222` 埠，掛載 `/home`、`/work`。計算工作交給 Slurm，不在登入節點執行。本專案已使用 Conda 建置名為 **`go`** 的環境，執行前以 `conda activate go` 啟用。Job 如需系統模組，先 `module purge`，再按 `module avail`／`module spider` 查核並載入所需模組，接著啟用 `go`；Conda 初始化方式以現場為準；CUDA、driver、PyTorch／NCCL 的實際版本須在計算節點記錄，不能直接沿用本機 CUDA 12.4 的假設。[官方登入與傳輸說明](https://man.twcc.ai/@nano4-manual/BydP-_lvZg)、[官方 Modules 說明](https://man.twcc.ai/@nano4-manual/BJyI6dgw-g)

**算力額度查核（2026-10-25）：** `wallet` 顯示專案 GOV115094（名稱「量子計算」，可能與其他成員共用）SU 餘額 **108,387.0087**；`sreport` 顯示本帳號 2026-10-01～10-25 共用 **254 GPU-h**；`sacctmgr` 的 QoS 為 normal、無 GrpTRESMins 上限。H200 每 GPU 小時的 SU 費率在官方公開頁面查不到，無法換算剩餘 GPU 小時；換算方式：下次跑已知 GPU 時數的作業前後各記一次 `wallet`（有其他成員同時使用時會有雜訊），或向國網中心客服詢問。

**儲存配額（2026-10-26 查得）：** `/work/jason1122` 與 `/home/jason1122` 各有 **100 GB** 目錄配額（WekaFS，`df -h /work/jason1122` 可見）。寫滿時所有寫入都會失敗，Slurm log 也寫不進去，作業會在沒有任何錯誤訊息的情況下以 exit 1 結束（A-057：5 個作業在 40 秒內同時失敗）。大檔要先估容量；刪除後配額要過約 1 分鐘才更新。

**仍未知：** 專案 account／QoS、剩餘 GPU 小時、RQ／NRQ 身分、實際 driver／CUDA／PyTorch／NCCL 版本與排隊時間。晶創 26 另有 GB200／Arm 資源；本專案目前的 HPC 支援範圍限定上述 H200 佇列。

**依賴安裝基準（2026-10-11）：** 本機 Windows 與 Nano4 Linux x86_64 共用 `requirements.txt` 的直接依賴 pins，使用標準 CPython 3.11–3.13、`torch==2.6.0+cu124` 與官方 CUDA 12.4 額外索引。Linux CUDA wheel 的 NCCL 依賴由 pip 安裝；系統 module 不必恰好是 CUDA 12.4，但 driver 必須相容。Nano4 的 Conda **`go`** 環境已建置（2026-10-11 使用者確認）；上述依賴 pins 是否已完整安裝、實際套件版本與多卡測試結果仍待驗證。安裝、driver／glibc 檢查與 `scripts/environment/check_torch_ddp.py` 的 1／2／8 卡驗證方式見 [README 的 Nano4 段落](README.md#nano4-linux單節點多卡)；環境紀錄放 `outputs/rank_env_nano4/`。2026-10-12 已依確認的 V1 計畫撰寫 DDP 模型訓練入口；使用者要求先寫程式不驗證，因此單／多卡 loss 與梯度、SyncBN、尾批及續跑皆尚未驗收，環境通過不代表訓練程式通過。

**棋力 V1 已確認例外與程式狀態（2026-10-12）：** 第一版依 [STRENGTH_V1_PLAN.md](STRENGTH_V1_PLAN.md) 與 [STRENGTH_V1_IMPLEMENTATION.md](STRENGTH_V1_IMPLEMENTATION.md) 使用固定黑／白 18 層歷史輸入、256-channel／20-block scalar ResNet、全域級別平均排序 loss、每級按棋局 80,000／20,000 切分與 training-only 切點校準；本檔以下我方／對方、softmax 與 player-disjoint 設定保留為其他路線規範，不取代本版已確認設定。V1 模型、DataLoader、索引、loss、DDP／checkpoint 及評估程式已撰寫；本次未執行驗證、全量資料處理或訓練，未產生權重、索引或分數。操作與路徑見 [README.md](README.md#棋力-v1資料準備訓練與評估)，後續驗收仍依兩份 V1 計畫執行。

**棋力 V1 資料要求更新（2026-10-14）：** 依使用者決定，取消保留 pass 時間位置與禁止按奇偶判色的要求；訓練資料仍依 `board_cache.py` 保存 `shard_XXXXX_colors.npy`（`int8[T]`，黑=0、白=1），維持現有快取格式。訓練集已確認黑先、黑白交替且 pass=0（`RESULTS.md` A-002），可依零起算 `ply % 2` 推得當手顏色、選取指定玩家樣本及組合顏色層；固定黑／白 18 層輸入仍依 V1 計畫。已完成相關程式精簡與部分測試（`RESULTS.md` A-011），其後依使用者要求停止測試；未啟動全量轉檔或訓練。

**棋力 V1 校準時機更新（2026-10-17）：** 依使用者決定，只在全部設定 epochs 完成後，以最終模型建立一次 training-only 的 9 個切點並評估 validation，不再每 epoch 校準／評估。訓練期間仍定時與每 epoch 保存最新 checkpoint；中途達到更新／時間上限不建立切點，最終評估中斷則續跑重試。

**棋力 V1 精簡流程更新（2026-10-18）：** 依使用者確認，各 seed 自行切分、訓練 1 epoch；每 100 steps 與最後一步完整推論 training／validation，以全資料各 rank 平均 score 計 loss。processed 統一放每 seed 的 strength_v1 目錄，取消獨立 cv／calibration 子集；切點改由全部 training 逐手分數擬合。成功後 run 僅留 model.pt、losses.npy、train_scores.npy、valid_scores.npy；評估直接讀保存的分數。此更新取代上方 10/17 校準流程，詳見 README；未啟動正式訓練。

**棋力 V1 切點更新（2026-10-18）：** 依使用者要求，切點改用全部 training 每盤的指定玩家逐手 score 平均值擬合，每盤等權，最佳化官方得分；不使用 validation 標籤。此更新取代先前逐手等權切點設定，排序 loss 與 loss 抽樣不變。

**棋力 V2 計畫（2026-10-19 更新）：** 依使用者要求，沿用 V1 ResNet，最後線性 head 由 `361→256→1` 改為 `361→256→10`，以 softmax 輸出十級機率、逐樣本 cross entropy 訓練（訓練 loss 直接接收 logits）。此方案取代原錨點 MSE＋間隔懲罰；單盤分類取指定玩家所有手的 softmax confidence 總和最大類別；多盤題目已確認先算每盤十類平均機率，再按每盤等權平均；多盤題目的正式解碼已確認使用相鄰級加權的 `metrics.score_rank`，不擬合 scalar 切點。每 100 steps 記錄最近 training batch loss；validation 改為從既有 split 每級固定抽取 5,000 盤、共 50,000 盤，推論指定玩家全部落子（手數不固定）計算逐手平均 loss 與 validation 官方分數；每 100 steps 與最後一步的 validation 官方分數已確認採單盤：每盤指定玩家全部落子的 confidence 總和 argmax 分類，再按盤等權平均官方公式得分，作為單盤診斷成績，中途不另建立多盤題目；最終全量評估維持原 200,000 盤 validation，全部訓練結束保存完整 validation 每盤分類結果，輸出目錄固定為 `outputs/v2_strength/`，正式檔案為 `model.pt`、`losses.npy`、`validation_category.npy`；`model.pt` 不保存題目雜湊、聚合規則或驗證報告，不額外計算或保存最終全量逐手 CE；分類報表於評估時在 CPU 重算，loss 使用 `losses.npy`；保存每盤 confidence 總和與手數，不要求保存全量逐手機率或最終 training 推論。最終評估規劃由 CPU 直接讀取已存每盤 confidence、重用固定 validation 題目，依使用者要求不另做棋局、手數、confidence 或資料雜湊的完整性檢查，報告單盤與 k=5／10／15／20 成績；題目 argmax／score_rank 分開列報，`score_rank` 為已確認的正式解碼，argmax 僅作診斷，不依 validation 自動挑選解碼。V2 依使用者要求不計算或呈現 MAE。視覺化規劃包含 loss／score 曲線、單盤與各 k 混淆矩陣、各級成績、平均 confidence 熱圖，圖表由保存產物重建，評估後僅以 `.png` 格式直接保存至 `output-dir/plot/`（預設 `outputs/v2_strength/plot/`），不使用 `plt.show()`，不另設顯示或匯出目錄選項。詳見 [STRENGTH_V2_PLAN.md](STRENGTH_V2_PLAN.md)；依使用者要求不進行 V1／V2 比較，不安排 V1 對照訓練、比較圖或相對增益 gate；已實作 V2 第二步 config／資料準備入口與整盤監控索引，以及第三步十類 logits 模型與 FP32 全域樣本平均 CE loss（含尾批 mask）；CPU／兩程序 Gloo loss 與梯度驗證通過，CUDA／SyncBN 尚未驗收，操作見 `src/go_aicup/tasks/strength/README.md`；推論聚合、訓練與評估待實作，未啟動官方全量資料準備或訓練。（以上為 10/19 當時狀態；推論、訓練、評估與提交路徑其後皆已完成（A-018～A-024、A-030、A-054），現行設定以下方 10/22 之後的更新與第 11 節第 6 項為準。）

**棋力 V2 首輪切分與 gate（2026-10-19 使用者確認）：** 改 player-disjoint：fold 0＝Valid-U、fold 1–4 每玩家顏色組留出 10%＝Valid-S、fold −1 不使用；監控每級 U／S 各 2,500 盤，S8b 實測單次監控占 100 steps 訓練 32.4%，依規則改每 200 steps（A-023）。模型選擇與 baseline 一律看 Valid-U 四桶等權 `score_rank`（≥0.4631 過 baseline、≥0.47 列提交候選）；G＝Equal-k(S)−Equal-k(U)：≤0.03 維持、≤0.08 注意、>0.08 須加反記憶對策。每盤另存 log-prob 總和，機率平均與 log-prob 聚合的 CPU 比較不論分數都要做。S4 推論聚合與 S4b 切分已完成（`RESULTS.md` A-018、A-019）；正式 1 epoch run 提交前須先回報實測估時並經使用者確認。

**棋力 V2 正式解碼與切分 seed 更新（2026-10-22 使用者確認，`RESULTS.md` D-002）：** 題目聚合改用 A-025 的 C2：每盤指定玩家落子的 log-prob 平均，題內各盤等權平均後，乘上每個 k 各自的溫度 T_k（4.42／6.24／7.67／8.13，只在 Valid-S 的 S-fit 擬合，存於 `outputs/v2_agg_calibration/result.json`），softmax 後以 `score_rank` 解碼；取代上方 10/19 的「每盤十類平均機率」規則，舊規則只保留作對照。A-024 模型在 Valid-U 的 Equal-k 由 0.4910 升為 0.5553，四個 k 桶都提升。之後的 V2 run 固定切分 seed 2027（與 A-024 相同），只改初始化 seed；A-028（seed2026）的切分不同，不能當 ensemble 元件。

**棋力主線改為 HumanSL＋B2（2026-10-24～25，使用者同意提前 D3，`RESULTS.md` A-039～A-047、D-003）：** KataGo v1.18.2 analysis engine 以 HumanSL 網路 `b18c384nbt-humanv0` 為主模型，每個位置 1 visit、15 個 `rank_*` profile、日本規則貼目 6.5，取實際著手的 log-likelihood；每盤特徵＝本人／對手各 15 個平均，再加 A-024 每盤平均 log-prob（10 維）。題目聚合 B2：每個 k 一個 logistic regression，在擬合玩家的合成題目（同一玩家抽 k 盤、特徵取平均）上擬合，`score_rank` 解碼。正式模型 `outputs/humansl_model/model.pkl` 以全部 Valid-S 擬合（65,223 盤、5,000 人），Valid-U 從不擬合。A-024＋C2 保留為對照提交與 HumanSL 失敗時的備援。外部資源見 `EXTERNAL_RESOURCES.md`。D-003：本地沒過 gate 的方向（只用 HumanSL＋train 玩家、強引擎特徵、貼目 0、平衡解碼、主幹表示等）保留，T5 以 LB 探針判斷。

**棋力首選改為 AH＋C（2026-10-26，使用者決定，`RESULTS.md` A-055、A-056）：** C2：凍結 HumanSL，為每個幽玄級別學兩個 384 維條件向量（本人／對手，取代 metadata 編碼器輸出），每盤 C＝10 個級別假設下本人／對手的平均 log p（20 維）。B2 以全部 Valid-S 擬合 AH＋C（`outputs/humansl_rankvec_model/model.pkl`）：S-select 0.7209 vs AH 0.7101，Valid-U 0.6934 vs 0.6833（+0.0102，玩家 bootstrap 95% [−0.0003, +0.0211]）。AH 保留為對照提交。提交流程輸出 `rank_<part>.csv`（AH＋C）、`rank_<part>_ah.csv`（AH）、`rank_<part>_allplayers.csv`（AH＋C 以 Valid-S＋Valid-U 6,442 人擬合，LB 探針候選，A-058；本地交叉擬合增益 +0.0017，量不出來）、`rank_<part>_v2_fallback.csv`。C4／C4b（全網路微調 HumanSL）試跑都未過 gate（A-057），原因推測是可施力的是玩家數而非算力。

**SE 幽玄十級移植第一、二步（2026-11-01）：** 使用者指定本版預設 seed=2026，包含 game-level 80/20 split，取代 reproduce_paper 舊計畫的 2027；不改主線 V2 的 seed。已新增獨立 config、沿用唯一 CV 模組的資料準備與有效位置索引、落子前本人／對手八步歷史及同步 D4 policy 樣本組裝；操作與驗收見 `src/go_aicup/tasks/strength/reproduce_paper/README.md`。未處理官方全量資料、未生成 teacher 標籤、未啟動訓練。

**SE 幽玄十級訓練預算更新（2026-11-02，使用者決定）：** `reproduce_paper` 改為 global 128 組／update、35,000 optimizer updates；初始 LR=0.04，完成第 25,000 次更新後降為 0.02，完成第 32,500 次後降為 0.002，最後再跑 2,500 步。以作者 32 組／130,000 updates 為參考，前 32,500 步組數放大四倍、updates 除四、LR 乘四，再加低 LR 尾段；取代先前 512 組／10,000 updates。十級七手共 313,600,000 次位置使用（較原 32,500 步增加 7.69%），8×H200 各 16 組／1,120 個位置。Loss 維持平均，SGD momentum=0.9、weight_decay=0.0001、seed=2026 不變，未加入 warmup。這是近似方案，收斂與加速比未知；正式長作業仍依 §9 回報估時。設定及驗收見 `configs/strength/se_seed2026.json`、`src/go_aicup/tasks/strength/reproduce_paper/plan.md`；未啟動訓練。

**reproduce_paper 共用切分規則（2026-11-01，使用者限定範圍）：** 僅本目錄所有訓練 run 共用同一份每級 80,000／20,000 棋局切分，split seed 固定 2026。初始化／抽樣 seed 預設 2026、可獨立修改，但不重新切分；各 run 記錄相同切分雜湊、重用固定 validation 題組。共用表為 `data/processed/reproduce_paper/seed2026/games.parquet`。本次僅更新規則，`prepare.py` 的 split seed 解耦與跨 run 共用驗收待實作；不改 V1／V2、HumanSL 或玩家辨識流程。細節見 `src/go_aicup/tasks/strength/reproduce_paper/plan.md`。

**reproduce_paper 固定題組實作（2026-11-01）：** 最新資料準備流程第五步已實作 `questions.py`，從既有 seed=2026 切分表生成共用 validation 題組；固定 question seed=2026、預設 10,000 題，各 k／rank cell 等量，題內同玩家同色且棋局互異、與 train 零重疊。保存題組／eligibility／獨立 manifest，跨訓練 seed 重用與完整性驗證通過；合計 17 項 CPU 合成測試通過。此項不代表八階段計畫第五步的模型／loss 已實作；`prepare.py` 的 split seed 解耦仍待完成。操作見本目錄 README，未對官方全量生成題組。

**reproduce_paper 資料準備第二步已完成（2026-11-01）：** `split_seed` 固定整數 2026，與訓練 `seed` 分開；首次 prepare 及後續跨 seed 共用皆已驗證，已存在時驗證並重用同一切分／位置索引，不重抽或覆寫。Dataset manifest identity version 2 使用 split_seed，舊版本拒絕靜默遷移；`PaperDataset` 明確接受獨立訓練 seed，`questions.py` 綁定相同固定切分。合計 24 項 CPU 合成測試通過（7.17s），含不同訓練 seed 初建一致、重用不呼叫切分器、產物雜湊與修改時間不變、損壞拒絕及 cache→prepare→questions 整合驗收。此更新完成上方所列 split seed 解耦待辦；官方全量資料尚未準備，未啟動 GPU 作業。

### 仍待確認 ⏳

| 項目 | 為什麼重要 | 何時／去哪查 |
|------|-----------|---------|
| 提交檔格式（欄位、public/private 題是否同一檔、top-5 分隔方式、rank 用字串還是編號） | 格式錯 = 浪費一次上傳 | 競賽二已確認：Public＋Private 合併一份 800 列、`question_id,pred_rank`、級別字串（`data/raw/tests/sample_submission_rank.csv`）；**競賽一已確認（11/11 公告）：** 單一 `submission_player.csv`、1,600 列（`pub_`／`priv_`／`pub2_`／`priv2_` 各 400）、欄位 `question_id,top_1,top_2,top_3,top_4,top_5`，ID 原樣照抄且 player_id 前綴須與 question_id 一致、同題 5 人不重複、不可缺題／重複題／未知題，否則整份不計分；前綴正確但不在候選庫的 ID 只是該名次不算 |
| Public LB 用的是 public 題、Private LB 用的是 private 題？（推測是） | 決定 public 分數對 private 的代表性 | 11/04 / 討論區 |
| 測試集 rank 是否平衡（每級 40 題？） | L4 先驗校正能不能用 | 無法直接看，用預測分布反推 |
| 競賽一 baseline 為何只有 0.28（見上方警訊） | 決定合成驗證能不能信 | 11/04 測試集 + 真實候選庫 EDA |

---

## 1. 資料

來源：日本棋院幽玄網站 19 路圍棋棋譜（已授權），2021–2023。官方發布：[AILAB-NDHU/aicup2026-go-dataset](https://github.com/AILAB-NDHU/aicup2026-go-dataset)。**2026-11-06 已更新為 [v1.1](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/tag/v1.1)**，`aicup2026_go_training_v1.1.tar.xz`，SHA-256 `e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e`；解壓出的十份 CSV 位於 `data/raw/training/`，另有 `handicap.csv`／`handicap_manifest.json`。已全量確認移除新增 `HD[n]` 後與原始資料逐列相同；既有 processed／teacher 不重建，尚未將 HD 接入訓練特徵。舊 v1.0 壓縮檔 SHA-256 `959ebc602b9638d665cb063a6269e0d5aef73956b77913e725763eb335ced94b` 留作來源紀錄。資料僅限本競賽使用。已匿名化——**移除姓名、段位、勝負、註解，只留棋盤設定與落子**；玩家代號不可回推身分。

**棋力 10 級（由弱至強，ordinal）：** `D < C < B < A < 1D < 2D < 3D < 4D < 5D < 6D` → 內部一律編碼為 `0..9`

### 檔案

| 檔案 | 內容 | 規模 | 欄位 |
|------|------|------|------|
| `data/raw/training/train_{D,C,B,A,1D..6D}.csv` | 訓練集（兩賽共用，v1.1 含 HD） | 共 1,000,000 盤；每級 100,000 盤；黑白各半；**8,007 位玩家** | `player_id, game_id, rank, color, sgf_content` |
| `player_identification_candidates_public.csv` | 競賽一 Public 候選庫 | 400 人 × 100 盤 = 40,000 盤，**全黑** | `game_id, player_id, sgf_content, color(=B)` |
| `player_identification_candidates_private.csv` | 競賽一 Private 候選庫 | 同上，**另外 400 人** | 同上 |
| `player_identification_test_public.csv` | 競賽一 Public 測試題 | 400 題 | `question_id, num_games, sgf_1..sgf_20, color_1..color_20(恆 B)` |
| `player_identification_test_private.csv` | 競賽一 Private 測試題 | 400 題 | 同上 |
| `player_identification_candidates2_{public,private}.csv` | 競賽一新增集候選庫（11/11） | **每人 10 盤、黑白都有**；人數未公布（可有不對應任何題的玩家） | `game_id, player_id, sgf_content, color(B/W)` |
| `player_identification_test2_{public,private}.csv` | 競賽一新增集測試題（11/11） | 各 400 題，**k ∈ {1,2,3,5}** 各 100 題、黑白混合 | `question_id, num_games, sgf_1..sgf_20, color_1..color_20(B/W)` |
| `rank_prediction_test_public.csv` | 競賽二 Public 測試題 | 400 題 | `question_id, num_games, sgf_1..sgf_20, color_1..color_20` |
| `rank_prediction_test_private.csv` | 競賽二 Private 測試題 | 400 題 | 同上 |

**測試題結構（兩賽相同）：** 每題 `num_games ∈ {5,10,15,20}`，各 100 題。不足 20 的欄位補空。
- 競賽一：**同題全黑**，正解必在對應候選庫的 400 人之中
- 競賽二：**同題同色**（全黑或全白），顏色隨題給

### 稽核已確認的事實（2026-09-24，見 `RESULTS.md` A-001）

- **v1.0 訓練集 SGF 僅有 `SZ[19]` 與落子（歷史結果）。** v1.1 新增 `HD[n]`，其餘內容相同；仍無 `HA`、`KM`、`AB/AW`。2026-11-06 稽核：訓練 HD=0／1／≥2 分別 792,610／170,969／36,421 盤；棋力 Public／Private 各 5,000 盤皆有 HD，且全部為 0／1。讓子表以 `(rank, game_id)` join，不能只用 game_id。分析、重跑方式與待確認實驗見 [HANDICAP_V11.md](HANDICAP_V11.md)。
- **HD 語意（官方 Q&A Q13，2026-10-22 版；該題標「待確認」，2026-11-07 以訓練資料驗證，`scripts/data/audit_handicap_semantics.py`）：** 全部為日本規則。HD0＝互先、貼 6.5 目；HD1＝讓先、不貼目、和棋白勝（≈ KataGo 貼目 0.5），SGF 黑白標記與實際一致；**HD2–9＝讓子棋，讓子石不在 SGF、須依 Q13 星位自行補黑子，且 SGF 著手 B／W 標記與實際相反**（第一手標 B 的其實是白棋）。驗證：36,421 盤 HD≥2 的前 10 手**零盤**落在對應星位，第一手為白棋掛角（HD2 最常見 `pp`／`qp`／`pq`）。**`color` 欄位是實際顏色，不需對調**（Q13 此點未定案；資料推論：HD≥2 中 6D 的 color＝B 僅 27.5%，D 為 63.7%，符合強者執白讓子）。
- **⚠️ 既有管線在 HD≥2 棋局取錯玩家（2026-11-07 發現）：** 本人落子以「SGF 標記顏色＝`color` 欄位」選取（`board_cache.py`），HD≥2 的標記相反、欄位卻是實際顏色 → **36,421 盤（3.6%）的本人／對手落子整組互換，且盤面缺讓子石**。V2、HumanSL、C、B2、SE 系列的訓練／擬合與 Valid-U 本地驗證都含這些盤。**棋力 Public／Private 測試題沒有 HD≥2，推論本身不受影響**；競賽一以 SGF 偶數手當黑方開局，若候選庫／查詢題含 HD≥2 也會取到白方（11/11 資料到手後先統計）。修正與相關實驗須依 §9 先確認。
- **唯一鍵是 `(rank, game_id)`**，不是 `game_id`——1M 列只有 100,000 個 distinct `game_id`。只用 `game_id` join/去重會 10:1 靜默塌縮
- **100 萬盤來自 8,007 位玩家，每人只屬於一個 rank**（跨 rank 玩家數為 0）→ 棋力標籤是玩家級屬性，評估跨玩家泛化時的獨立單位是玩家，不能把 100 萬盤當成獨立樣本；須留意模型記住玩家習慣而非學會棋力的風險
- **各 rank 玩家數差 3.6 倍**（1D 僅 433 位、A 有 1,582 位）→ 少數 rank 極易被「記住玩家」而非「學會棋力」
- **只有 1,281 位玩家有 ≥120 盤黑棋** → 競賽一的忠實合成驗證人數上限
- 棋局長度 **100–445** 手（中位數 228、p99 323），全為 19 路，無損毀，**全量零重複棋譜**
- → 序列位置上限 **448**；因 p99 僅 323，用 length bucketing 而非固定 padding
- 黑方第一手集中：`pd` 54.9% + `qd` 19.9%

### 棋力訊號 EDA（2026-09-25，見 `RESULTS.md` A-005 E3）

**對手走法也有棋力訊號，強度約與本人相當。** 距離／應手類特徵的玩家成分較低，例如 `d_opp` 的玩家成分為對手 **0.063**、本人 **0.151**（約一半）→ 棋力模型值得納入雙方落子，利用對手訊號並降低對本人習慣的依賴。提子類特徵須另行解讀：「對手提子」等同「本人被提」，仍反映本人表現。這是描述性分析，實際模型增益仍須驗證。

### 規模現實
1M 盤 × 約 200–250 手 ≈ 2 億手。**絕對不要每個 epoch 重新 parse CSV/SGF。**
第一週就要做一次性轉檔：SGF → 緊湊二進位（每手一個 `int16` = `row*19+col`，範圍 0～360），存成 `npy`/`memmap` + 一份 metadata parquet。之後所有訓練吃這份。

---

## 2. 兩個任務

### 競賽一：玩家辨識（re-identification）
輸入一題的 5/10/15/20 盤黑方棋譜 → 從**該題對應候選庫的 400 人**中輸出信心度前 5 名。

**關鍵結構：Public 與 Private 是兩組完全不同的 400 人**，而且**兩個候選庫都附 `player_id` 標註**（400 人 × 100 盤全有標）。所以這是 **transductive** 問題：
- 可以直接在 public 候選庫上訓一個 400-way 分類器（40,000 筆有標資料）
- 同樣流程要**在 private 候選庫上獨立再跑一次**（分類頭不可共用權重）
- 也可以在 1M 訓練集上用 metric learning（ArcFace / triplet，以 `player_id` 為類別）預訓風格 embedding，再把候選庫 400 人各做 centroid，查詢盤去比對

三條路線（A 純 pool 分類 / B 純 embedding 檢索 / C 預訓 embedding + pool 微調）要**實測比較**。不得預先認定 C 最強；先以文獻式 GE2E 檢索建立對照，再測分類與微調的增量效果。

**指標 = Top-5 指數衰減（第 1 名 1.0、第 2 名 0.368…）→ 幾乎等於 top-1 accuracy。** 優化目標是 top-1，top-5 依後驗排序即可。

### 競賽二：棋力預測（ordinal 10-class）
輸入一題的 5/10/15/20 盤同色棋譜 → 預測該玩家 10 級之一。

**這是 ordinal 任務**，D→6D 有序。**指標 = 完全正確 1 分、差一級 1/e、其他 0**（第 0 節）。
→ softmax 輸出校準過的 10 級機率，再用 `metrics.score_rank` 做 Bayes 最佳解碼（`p[r] + (p[r-1]+p[r+1])/e` 的 argmax）。不要用期望值解碼（差兩級 = 0 分，實測較差）。

### 共用
- 共用解析、快取與驗證；先比較盤面與座標序列表示，預訓是否共用另測，兩賽各自微調
- 競賽一：embedding → 玩家比對頭
- 競賽二：embedding → 棋力頭
- **跨任務加成**：候選庫沒給 rank，但可用競賽二的模型推估「查詢題的棋力」與「每位候選人的棋力」（候選人有 100 盤），棋力對不上的候選人降權。這是兩賽合辦的最大結構紅利，列為 P2 實驗。

---

## 3. 預測單位是「題」不是「盤」——訓練必須模擬這件事

測試最小單位是一題 = 同一玩家的 k 盤棋（k ∈ {5,10,15,20}）。所以：

- 可先做單盤／手級預訓；題級模型訓練、聚合選擇與最終驗證要**合成題目**：從同一 `player_id` 抽 k 盤組成一題，k 在 {5,10,15,20} 均勻取樣
- 聚合先用平均，attention 作後續消融。同一組 logits 經 softmax 且各盤等權時，logit 平均與 log-prob 相加的候選排序相同；信心尺度與後續融合仍可能不同
- **驗證分數一定要分 k 桶報告**（k=5/10/15/20 各一個數字）。整體平均會掩蓋「k=5 崩潰但 k=20 很好」這種問題，而測試集四個桶各佔 25%
- 若某方法只在 k 大時有效，它在真實分數裡只值 50%

---

## 4. 洩漏與陷阱清單（每個新特徵都要過一遍）

### 本賽專屬
- [ ] **SGF header 稽核**：parse 出所有 SGF property 並列表。特別檢查 `HA`(讓子)、`KM`(貼目)、`TM`/`OT`(時間)、`DT`(日期)、`AP`(軟體)、`SZ`、`RE`(結果，宣稱已移除)。
      → **v1.0 歷史稽核（2026-09-24）只有 `SZ`；2026-11-06 已確認 v1.1 train 與棋力 Public／Private 都有 `HD`。** `HA`／`KM`／`AB/AW` 仍缺（Q&A Q14：RU、KM、HA、AB、AW、PL 皆已移除）。HD 語意已依 Q13 確認並驗證（見 §1）。L3 可重新評估；HD 是測試輸入本身就有的欄位，不算洩漏，但尚未加入 feature policy 白名單或任何模型，須依 §9 先確認實驗。玩家辨識候選庫／測試檔本次未提供，HD 覆蓋未知。
- [ ] **檔名 / `game_id` / `player_id` 不得成為特徵。** test 題目只有 `question_id`，這些欄位根本不存在 → 任何用到它們的 CV 分數都是假的。（已確認 `game_id` 在 10 個 rank 檔之間重複使用，唯一鍵是 `(rank, game_id)`。）
- [ ] **玩家記憶 = 本賽的頭號過擬合風險（競賽二）。** rank 是玩家級屬性且 1D 只有 433 位玩家，模型很容易學成「這是玩家 X → X 是 1D」。測試玩家未見過就崩盤。對策：player-adversarial 頭（gradient reversal）或每批次每位玩家只取一盤，並嚴格用 player-disjoint 驗證。
- [ ] **重複盤稽核**（合法且必做）：檢查測試題的 SGF 是否逐字出現在候選庫／訓練集；檢查 `player_id` 空間在 train 與兩個候選庫之間是否重疊。這不是作弊，是必要的資料理解。結果無論如何都要寫進 `RESULTS.md`。
- [ ] **候選庫的查詢盤與候選盤必須互斥**：自建驗證集時，同一玩家的查詢盤絕不能出現在他自己的 100 盤候選庫裡，否則等於直接比對，分數完全失真。
- [ ] **D4 對稱增強是兩面刃**：棋盤有 8 重對稱。
      - 競賽二（棋力）：棋力與絕對方位無關 → D4 增強應該有效，用
      - 競賽一（玩家辨識）：**「這個人開局習慣走右上小目」本身就是指紋**。D4 增強會洗掉這個訊號
      - **競賽一預設 = D4 正規化（2026-10-19 使用者決定，見 `RESULTS.md` E0-001）：** 查詢盤一旦被旋轉，raw 會從 0.929 崩到 0.09–0.16；D4 正規化不受影響，代價 −0.022。11/04 稽核真實查詢盤與候選庫的方位分布後，再決定是否改回 raw
      - → 兩賽的增強策略必須分開實驗，不要共用一組設定
- [ ] **顏色正規化**：盤面統一成「我方／對方」兩個 plane，而不是「黑／白」。競賽一全黑、競賽二黑白都有，不正規化的話競賽二的白方樣本會變成另一個分布。

### 上屆（AI CUP 2026 桌球賽）的通用鐵律
上屆 V1–V4 累積 6 個洩漏 bug，OOF 從 0.3678 修正後掉到 0.3256——0.04 全是假的。每個新特徵都問：
- [ ] 有沒有用到**同 fold 驗證集**的統計量？（target encoding / 全域統計 / 頻率表都要 fold-safe）
- [ ] 有沒有用到**預測時刻之後**才知道的資訊？
- [ ] 有沒有用到**目標欄位的代理**？（上屆 test.csv 竟含真實標籤欄位 → **test 裡任何疑似標籤的欄位一律覆寫掉，永不當 truth 或特徵**）
- [ ] 特徵選擇／超參搜尋是不是在**全資料**上做的？必須在 fold 內或獨立 holdout 上做
- [ ] 有沒有依賴 **test 對不上的 ID**？（上屆的玩家層級特徵整條線 OOF→LB 落差擴大到 −0.026，永久排除）

---

## 5. CV 設計

**群組鍵 = `player_id`。全專案只有 `src/go_aicup/data/cv.py` 一處產生 split，並附 assert 檢查 fold 間 player 零重疊。**

### 競賽二（棋力）
`GroupKFold(5) by player_id`。同一玩家的棋譜跨 fold 會讓模型背玩家而不是學棋力 → CV 虛高。

### 競賽一（玩家辨識）——驗證集要**複刻測試結構**
不能用一般 KFold。正確做法：
1. 從訓練集抽 400 位 held-out 玩家（模擬候選庫）
2. 每人取 100 盤黑棋當「合成候選庫」
3. 每人**另外**取盤組成查詢題，k ∈ {5,10,15,20} 各 100 題
4. 查詢盤與候選盤嚴格互斥
5. 做多組（不同 400 人抽樣）降低單一抽樣的雜訊

### 通則（上屆實測）
- 除了 KFold OOF，另留一個**固定 holdout** 做第二意見
- **CV 只對「乾淨特徵」可信**，上屆的落差規律：
  - 乾淨結構特徵：OOF→LB 落差 ≈ −0.005 ~ −0.016（正常）
  - 玩家／使用者層級特徵：−0.022 ~ −0.026（不轉移，整條線排除）
  - blend 元件數 ≥ 6：落差近乎翻倍 → **blend 上限 5**
  - threshold / temperature 落在搜尋網格邊界：過擬合 OOF，不轉移
- 隊友的 AutoGluon 內部驗證 0.4149 → LB 0.3152，原因是 **holdout 沒依群組分**。任何外部 AutoML 結果，先問它的 split 怎麼切的

---

## 6. 上傳策略（本賽規則與上屆不同，最重要的一節）

> 官方原文（2026-09-27 核對）：測試集下載與上傳是**單一連續視窗 2026/11/04 11:00 – 2026/11/24 23:59:59**（不是每天獨立重置），
> 每日上傳至多 5 次，「以**最後一次**上傳之預測結果獲得之分數進行排名」。

**這句「最後一次」指的是整個 21 天視窗結束那一刻的最後一次上傳，不是每天各自的最後一次。**
→ 11/4–11/23 期間的任何上傳，只要之後還會再傳，完全不影響最終成績；真正決定成績的只有 **11/24 23:59:59 前的那最後一筆**。
（此前一版寫的「每天收工前必須是最佳檔、沒驗證完寧可 0 次上傳」是誤把上屆桌球賽的每日紀律套用過來，過度保守，已修正。）

### 硬性規則
1. **唯一鐵律：11/24 23:59:59 前的最後一次上傳 = 最終成績。** 中途 20 天可以放心用滿 5 次額度做探針，不用每天收尾留最佳檔
2. 越接近視窗末端（尤其 11/24 當天）風險越高，此時才需要「每次上傳前先確認是不是想留的那份」
3. **真正要停手前（不論是 11/24 當天或你自己決定提前收工），確保手上有一份驗證過的最佳檔案，並用它完成那次上傳，再到平台確認顯示的就是它**
4. 兩個競賽各有獨立的 5 次/日額度（21 天 × 5 = 每賽最多 105 次），**分開記帳**；額度不是無限，避免在還沒到關鍵階段就浪費在無意義的重複探針上
5. **隊員都可上傳（平台允許同隊不同成員帳號上傳）。** 每日 5 次額度是全隊共用的，上傳前先在 `RESULTS.md` 登記，避免兩人互相蓋掉「最後一次」

### 實驗紀律
- 上傳前 `RESULTS.md` 先有一行：tag / 驗證分數（分 k 桶）/ 預期 LB / 假設。上傳後補實際 LB 與 Δ
- **一次探針只改一個變數**，否則結果無法歸因
- 新元件的高驗證增益 → 先做**同尺寸受控探針**再 promote，不要直接塞進 best blend
- 最終選擇（2026-11-09 使用者修訂；最終排名看 Private LB，泛化優先）：**以本地 player-disjoint 多 seed 證據（分 k）＋能推廣到未見玩家的機制為主；Public LB（400 題，標準誤約 0.02）只作方向確認**。Public 差距小於約 0.02、本地又沒有支持的改動，不升為基底或最終檔；每次上傳前先寫好 LB 升／降／持平時各保留哪個檔。**2026-11-20 使用者修訂：** 有理論依據（機制能推廣到未見玩家）且 Public LB 上升的改動可以保留為基底，不要求 LB 差距超過標準誤；但本地 player-disjoint 結果明確為負（例如低於 −0.01，或 bootstrap 區間整段在 0 以下）時仍不保留，因為此時 LB 上升較可能是 400 題的雜訊。上屆避開 AutoGluon 災難的教訓仍適用：本地驗證必須依玩家分組，外部 holdout 先問切法

---

## 7. 模型路線圖（2026-10-22 修訂，詳見 EXECUTION_PLAN.md）

10/05 的 E0–E6 文獻對照路線（D-001）已由 EXECUTION_PLAN.md 的 T1–T6 時程取代（D-002）。E0、E1 已完成；E2 以下依下表改排或延後。

| 階段 | 日期 | 內容 |
|------|------|------|
| T1 管線 | 10/22–10/27 | 兩賽端到端提交管線（官方 CSV → 提交檔），用假測試檔彩排；Day-0 稽核腳本 |
| T2 保險模型 | 10/23–11/01 | 競賽一：V2 盤面 ResNet 當逐手 encoder → 盤內平均 → GE2E，作為不靠開局的保險（A-035 gate 通過；10/25 接進提交管線並彩排通過，A-053）；競賽二：多初始化 seed ensemble 未過（A-033），改由 HumanSL＋B2 成為主線（A-039～A-047） |
| T3 凍結彩排 | 11/02–11/03 | 凍結程式重跑；報告骨架 |
| T4 Day-0 | 11/04–11/06 | 真實資料稽核、格式上傳、真實候選庫自驗、L10／L4／方位／一對一判斷 |
| T5 校準 | 11/07–11/20 | 受控 LB 探針、ensemble（≤5 元件）；D-003 保留的候選在此用 LB 判斷（Public 400 題，標準誤約 0.02） |
| T6 收尾 | 11/21–11/24 | 凍結；使用者手動做最後一次上傳 |

已砍：競賽二 GBDT（E1-001）、座標 Transformer 預訓與 scaling curve、家族 B 生成式。延後到 T5：SE 排序、ArcFace、attention 聚合、gallery 微調、L9。具體提案仍須依第 9 節先討論數值 gate、預期增益（未知可明寫）及成本。

---

## 8. 官方規則要點（會影響做法的）

- 上傳：每日 5 次 / **以最後一次計分** / 全隊共用額度
- 外部資料、開源程式、生成式 AI 工具**可用**，但須遵守著作權法，且**必須在報告中申報來源與貢獻**
- **Private LB 前 25% 且超過 baseline（棋力 0.4631／玩家辨識 0.2845）者（不超過 30 隊）必須繳交完整報告**（11/26–12/06）：成員、模型演算法、資料前處理原始碼說明、訓練原始碼與說明、各項參數（含訓練權重）、執行環境、外部資源。**報告完整性與正確性影響獲獎資格與名次核定。**
  → 這不是賽後才做的事。從第一天起：固定 seed、每個 run 有 config 檔、`requirements.txt` 鎖版本、記錄硬體與執行時間
- **禁止人為修正預測結果**
- **隊伍間不可私下共享程式與特徵值**，報告須獨立撰寫（官方討論區的公開討論不在此限）
- 主辦方保有比賽途中調整資料集的權利 → 轉檔腳本要能一鍵重跑，不要手動改過資料

---

## 9. 工作方式（給 Claude 的指示）

1. **先計畫再動手。** 新版本／新特徵／新模型，先給我：假設是什麼、預期在哪個指標上有多少增益、用什麼 gate 判成敗、要花多少算力。我確認後才寫 code
2. **每個結論都要有數字**，而且**分 k 桶**（5/10/15/20）。「應該會比較好」不算
3. **解釋機制**，我要理解為什麼，不只知道做什麼
4. **繁體中文溝通**，code 與 commit message 用英文
5. **不要幫我上傳。** 你的責任是產出檔案 + 在 `RESULTS.md` 登記候選與預期分數；上傳由我手動執行（真正停止上傳前確認最後一次為已驗證最佳檔）
6. 長時間訓練前先估時間並告訴我；1M 盤的 job 動輒數小時，不要默默開下去
7. 不知道就寫「未知」，不要事後編機制（上屆的 v11plus 為何不可替代至今無解，就記著無解）

---

## 10. 專案結構

**2026-11-17：使用者已下載備份並移除 Nano4 的 `data/processed/strength_train_hc.zip`；本機確認 ZIP 與原目錄均不存在。下載備份的位置與完整性未驗證；重做依賴舊 H/C 特徵的流程前，須先將 ZIP 傳回並解壓還原。**

**共用資料放置（2026-11-16）：** 跨實驗重用的盤面／落子快取、固定切分與題組、HumanSL／引擎特徵、embedding 放 `data/processed/`；本機從 `outputs/` 搬移 3,357 個實體檔案（32.882659 GiB）至 `strength_features/`、`identity/artifacts/`、`release/`，保留舊路徑相對連結。權重、訓練狀態、log 與結果報表仍在 `outputs/`；執行中 job 787069 的 epoch run 專屬資料未搬。既有 `strength_train_hc/` 等資料保留原位；新共用資料直接寫入對應 `data/processed/` 版本目錄，不覆蓋其他來源。細節及完整路徑對照見 [SHARED_DATA_LAYOUT.md](SHARED_DATA_LAYOUT.md)。 **2026-11-17：使用者已下載備份並移除 Nano4 的 `data/processed/strength_features/humansl/c4x/` 整個目錄（刪除前約 24.01 GiB，含 pack、切分表與監控資料）；本機確認目錄不存在，備份位置與校驗結果未驗證。重做依賴該資料的 prepare／實驗前須還原所需檔案與路徑，詳見共用資料文件本日紀錄。**

```
project/
├── CLAUDE.md                  # 本檔案（唯一事實來源；AGENTS.md 只指向這裡）
├── README.md                  # 新成員上手：安裝、資料放置、重建步驟、協作規則
├── RESULTS.md                 # 結果總覽、架構比較與現行提交候選
├── RESULTS_ARCHIVE.md         # 完整實驗帳本（只增不刪，失敗也記）
├── STRATEGY.md / EXECUTION_PLAN.md / EXTERNAL_RESOURCES.md / reference.md
├── pyproject.toml             # 讓 go_aicup 可 import：pip install -e .
├── requirements.txt           # 鎖版本，報告要用
├── data/                      # 不進 git
│   ├── raw/training/          # 官方 v1.1 train_*.csv 與擷取的 handicap.csv
│   ├── raw/tests/             # 官方候選庫／測試題（依實際下載檔案）
│   └── processed/             # SGF／盤面快取、cv/；strength_features/、identity/artifacts/、release/ 共用資料
├── src/go_aicup/              # 可重用程式（import 用），兩人分工以子目錄為界
│   ├── paths.py               # repo 路徑只在這裡算一次
│   ├── metrics.py             # 兩賽官方指標，一律分 k 桶回報（兩賽共用，不拆）
│   ├── data/
│   │   ├── sgf.py             # SGF parser + property 稽核（唯一 parser）
│   │   ├── preprocess.py      # raw CSV → processed，一次性轉檔
│   │   ├── store.py           # GameStore：memmap 讀取 + 我方／對方視角
│   │   └── cv.py              # 唯一 split 來源：GroupKFold by player_id + 合成候選庫
│   ├── features/
│   │   ├── feature_policy.py  # 禁用欄位清單（game_id / player_id / 檔名衍生…）
│   │   ├── game_stats.py      # 盤面重播統計（numba）
│   │   └── features_vN.py     # 手工特徵，一版一檔
│   └── tasks/
│       ├── strength/          # 競賽二：shared/ 共用基礎、v1/ 專用模型與訓練評估
│       └── identity/          # 競賽一：train_pid_vN、推論、提交檔
├── scripts/                   # data／environment／identity／strength／release；索引見 scripts/README.md
├── tests/
├── configs/{strength,identity}/  # 每個 run 一個 yaml（報告要用）
├── outputs/<run-id>/          # 不進 git；新 run 一律放這：config、log、OOF、權重、分數（run-id 以 rank_/pid_ 開頭）
├── logs/                      # 不進 git；W1 舊 log，新 run 不再寫這裡
├── submissions/{rank,pid}/    # 不進 git
└── notebooks/                 # EDA 只讀，不放訓練邏輯
```

執行方式：可重用模組用 `python -m go_aicup.<子套件>.<模組>`（例如 `python -m go_aicup.data.cv`），一次性報告用 `python -m scripts.<分類>.<name>`；分類見 [scripts/README.md](scripts/README.md)。

---

## 11. 第一週里程碑

1. **資料稽核（Day 1）**
   - SGF property 全表：train / candidates / test 三邊的存在率比對 → 決定 `HA`/`KM` 能不能用
   - 訓練集 distinct `player_id` 數、每人盤數分布、一人是否只對應一個 rank
   - `player_id` 空間在 train / public pool / private pool 之間是否重疊
   - 測試題 SGF 是否逐字出現在候選庫或訓練集
   - 棋局長度分布、是否有異常短局／損毀 SGF
   - 回填本檔第 0 節的所有待確認項（✅ 指標／baseline／時程已於 9/27 回填；提交格式待 11/04）
2. **`preprocess.py`（Day 1–2）**：一次性轉檔，1M 盤 → memmap，之後所有訓練吃這份
3. **`cv.py`（Day 2）**：GroupKFold by player_id + 競賽一的合成候選庫產生器，附零重疊 assert
4. **`feature_policy.py`（Day 2）**：把第 4 節的檢查清單寫成 code（禁用欄位名單 + fold-safe encoder）
5. **後續工作**：依第 7 節與 EXECUTION_PLAN.md 的 T1–T6 執行；11/04 前僅本地驗證，不安排官方上傳。
6. **主力現況（2026-10-26；競賽二 11/08 更新）**：競賽二首選為 **P1＋P2＋P3**（AH＋C 的 B2 只以 Valid-S HD≤1 擬合、加兩個讓先旗標、讓先局 H／C 以貼目 0.5 重算；`scripts/strength/humansl_hd_filter.py`、`scripts/strength/humansl_p3_komi.py`；Public LB 0.7197）。原首選為 A-024＋HumanSL＋C2 級別向量＋B2（AH＋C，A-055／A-056，LB 0.7103），對照 AH（A-024＋HumanSL＋B2），備援 V2 ResNet＋C2；競賽一為 D4 開局模型＋不靠開局的 GE2E 保險模型（λ＝8，A-053），**預設不做 L6 指派**（D-004：在確認一對一之前假設不是一對一；Hungarian 版只作 LB 探針，A-059）。一鍵提交流程 `scripts/release/run_release.sh`（A-054；10/26 加入 AH＋C 與 AH 對照檔，彩排通過，A-056）。


### 2026-11-01：reproduce_paper 第三步逐有效盤面 v script

- 使用者明確指定逐有效落子前盤面各自分析，取代末盤共用 teacher target；有效位置仍為指定玩家且 ply >= 30。
- 新增 `src/go_aicup/tasks/strength/reproduce_paper/label_value.py`、`configs/strength/se_value_analysis.cfg`，並接入 `PaperDataset(label_dir=...)`。分片逐位置紀錄勝率、v、目數、不確定性（如有）、visits 與狀態，支援重試／續跑、來源與 teacher 身分檢查及完整匯出。每個 split 的 value.npy 與原有效位置索引一對一對齊。
- 依使用者「不要執行計算」要求，僅撰寫程式／文件／測試，未執行任何測試、資料準備、標籤生成或訓練；不宣稱驗收通過。visits／品質政策與全量成本仍待短測；k=5/10/15/20 模型分數與增益未知。操作見 reproduce_paper README。


### 2026-11-01：逐盤面 value 完成後只留 value.npy

依使用者要求，reproduce_paper 的 `label_value --export` 在驗證全部位置並原子發布後，自動清除該 split 的分片、續跑狀態、log、暫存檔與 manifest，只留下 value.npy。未成功匯出保留續跑資料；清理中斷可重跑 export 接續。Dataset 改為單檔讀取，保留 shape／dtype／樣本值檢查；完成後不再保留來源或 teacher 雜湊驗證能力。僅修改程式與文件、補寫未執行的測試；沒有計算、測試或刪除既有產物。


### 2026-11-01：value 標籤僅處理 training

使用者指定本次只需要 training。`slurm/label_value.slurm` 預設改為 train，處理 800,000 局、76,763,475 個有效位置；完成後僅保留 training 的 value.npy。本次不生成 validation 標籤，未提交作業或執行計算／測試。


### 2026-11-01：training value teacher 改用 1 visit

依使用者決定，`configs/strength/se_seed2026.json` 的 teacher.visits 設為 1，`slurm/label_value.slurm` 預設亦為 1，可直接不帶參數提交。仍只處理 training；未執行計算、測試或提交作業，品質與實際耗時尚未驗收。


### 2026-11-02：reproduce_paper 第四步完成

使用者回報全部 training target 已取得；prepared 索引、固定 validation 題組及 `value_v1/train/value.npy` 已存在。第四步已實作 `GroupBatchSampler`／`collate_groups`，按盤等權抽每級同盤七手、完整組 DDP 分配及消費端 step 可重建抽樣／D4，沿用逐位置 value。資料／題組／teacher protocol 合成回歸 41 項通過（21.50s），含 spawn workers、prefetch、8 rank 分組及續跑一致性；未啟動 GPU、teacher 或訓練，未逐值重新稽核官方標籤。`plan.md`／README 與 RESULTS 已更新，下一步為第五步三頭模型／loss；正式品質、分 k 成績及成本未知。


### 2026-11-02：reproduce_paper 第五步完成

依固定作者版本查核，新增專用三頭 ResNet 與 FP32 分組 BT／policy CE／soft value MSE，保留卷積 bias、residual 相加後 ReLU、預設 BN／初始化與 value tanh；不使用 V1／V2 的不同骨幹，省略作者未啟用的 learned weight head。支援第四步完整組輸入及逐位置推論，loss 等權且只接受完整 teacher 標籤。完整 reproduce_paper CPU 合成回歸 67 項通過（27.04s），兩程序 Gloo 固定 BN 的全參數梯度向量 relative L2=1.37e-6、max abs=3.69e-7；近零分量 max abs=2.68e-8。詳細方式、對齊／偏離及測試見本目錄 README／plan 與 RESULTS。下一步第六步訓練器／checkpoint／H200 短測；未啟動 GPU、optimizer 訓練或 teacher，分 k 分數及成本未知。


### 2026-11-02：reproduce_paper 只做 8 卡／100 步短測

依使用者最新限定，不實作／安排本版單卡 GPU 短測。新增 `reproduce_paper/train.py`、`configs/strength/se_smoke100.json`、`slurm/se_smoke100.slurm`：8×H200 dev、100 updates、global 128 組／每卡 16 組、BF16＋SyncBN；預設每卡整批 1,120 位置。前 5 步只作量測暖機，LR 不變。量測 loss、梯度／更新、分項耗時、顯存／RAM／I/O、checkpoint 與 GPU-hours，保存可續跑 checkpoint，使用獨立 `outputs/rank/reproduce_paper/se_smoke100_8gpu/`。CPU 小模型／Gloo 僅作工程驗證；未提交 GPU 工作，實際效能未知。正式 35,000 步與四小步／一大步近似診斷仍待完成。操作與量測限制見本目錄 README。


### 2026-11-02：SE 大顯存前兩方向

使用者指定實作固定訓練 batch 的運算對照與加大推論 batch。新增 cuDNN benchmark／channels-last 2×2 的 8 卡 100 步設定（保持 global 128 組及模型／LR），另建同版本基準；新增 `reproduce_paper/infer.py` 與 `se_infer_batch_benchmark.json`、`slurm/se_infer_benchmark.slurm`，使用既有 checkpoint 在固定位置集合比較每卡 1,120／2,240／4,480／8,960 的推論 batch。推論 eval／無 D4，逐候選報吞吐、顯存、score 差異，OOM 停止更大候選。只實作及 CPU 工程驗證，未提交新 GPU 作業，正式每盤聚合／centroid／評估仍待完成；操作與數據解讀限制見 reproduce_paper README。


### 2026-11-02：選定 channels-last，關閉 cuDNN 搜尋

依使用者決定，reproduce_paper 訓練採 `channels_last=true`、`cudnn_benchmark=false`，同步 `configs/strength/se_seed2026.json` 與 `se_smoke100.json`。預設短測改用獨立 `outputs/rank/reproduce_paper/se_smoke100_default_8gpu/`，不改寫已完成的 baseline／對照 config 或輸出。模型、128 組、LR schedule、35,000 步預算不變；正式完整訓練入口尚未完成。

依據：job 673150（channels-last）5.133 steps/s、allocated 峰值 16.85 GiB；job 673193（兩者合用）5.152 steps/s、18.70 GiB，額外吞吐僅約 0.37%。此為效能設定決策，不以 100 步 loss 宣稱棋力改善；尚無同版本新 baseline 或重複量測。未提交新 GPU 作業。


### 2026-11-02：reproduce_paper 正式訓練入口

依使用者確認，先完成正式訓練、續跑與最終 model.pt；centroid／評估留待下一步。train.py 預設改讀 se_seed2026.json，完成 CUDA／BF16／SyncBN／channels-last 的單節點 8 卡、35,000 updates 入口及 slurm/se_train.slurm。每 1,000 步保存 checkpoint，以自前次保存後的平均 training total loss 比較；只有新 loss 嚴格較低才刪緊接前一份，不清更早歷史，相等／較高保留兩份。保留最新 checkpoint 作續跑，正式訓練取消資源用量與更新 norm 監控。提前停止／Slurm 停止檔在完整更新後同步保存；全部更新完成才發布經重載驗證的 model.pt，支援最終發布重試。已完成逐位置推論與 batch 效能測試，本輪依使用者要求不跑 GPU 短測、不提交正式訓練或推論。CPU 合成工程驗證見 RESULTS.md；GPU 中斷續跑、正式收斂及分 k 成績未知。最新契約與操作見 reproduce_paper PLAN.md 第 5 節及 README。V1／V2 未傳 checkpoint loss 參數時保留原 latest-only 行為。


### 2026-11-02：reproduce_paper 完整訓練後清除續跑殘留

依使用者要求，正式訓練完成、最終模型驗證及完成摘要發布成功後，自動刪除 checkpoints/ 全目錄、last.json、停止請求與已知暫存，保留 model.pt／marker、設定、資料身分、loss、環境與 session 摘要。清理前再次驗證最終 checksum／config／完成步數；失敗不刪檔。提前停止及最終發布失敗仍保留續跑資料。完成清理中斷可使用相同 config 重跑補清，不依賴 last.json、不增加更新；probe 路徑不清除 checkpoint。本次僅程式／CPU 合成回歸，不啟動官方訓練或刪除既有實驗產物。


### 2026-11-02：reproduce_paper 完訓後自動全量評估與圖表

依使用者追加要求，預設正式入口在全部訓練、最終模型及續跑清理成功後，自動呼叫 evaluate.py。全部 800,000 training 棋局有效位置按位置等權建立十級 centroid、統計及全量 histogram；全量 validation 推論後以固定題組所有位置平均＋nearest-centroid（並列取較低 index）評估 k=5／10／15／20 的官方公式分數與 Equal-k，另存各級／單盤診斷、confusion matrices 與 training loss learning curve。預設 evaluation.per_gpu_batch_size=2240；不再做 GPU 短測或 batch benchmark。產物在 run/evaluation/，包含可重用的全部逐位置分片、每盤／每題統計、CSV／JSON／NPY／NPZ 及 10 張 PNG；可用 evaluate --report-only 在 CPU 重建。原訓練 checkpoint 清理政策不變；評估失敗不重新訓練。本輪不啟動正式訓練或全量推論，不宣稱幽玄真實分 k 成績。最新規格／驗收見本目錄 PLAN.md 第 6 節與 RESULTS.md。


### 2026-11-02：reproduce_paper 保留最低 loss 的兩個 checkpoint

依使用者最新要求，取代先前相鄰比較及完訓刪除全部 checkpoint：正式訓練每次保存後保留全歷程 training total loss 保存窗口平均最低的兩份，另保留最新一份供續跑；loss 相同優先較晚 optimizer step。完訓先原子留存並驗證 best_1.pt／best_2.pt（完整 checkpoint）及 complete markers、best_checkpoints.json，成功後才刪來源 checkpoints/、last.json、best.json 與暫存；失敗保留來源並可重試，不增加更新。不足兩份時只留實際份數，舊規則已刪除的檔案無法還原。model.pt 仍是最後一步，完訓後自動 centroid／評估不切換至最佳 training loss 權重。V1／V2／probe 不傳 loss 時仍維持原 latest-only 行為；本輪不啟動 GPU 工作或對既有實驗檔操作。

### 2026-11-03：reproduce_paper_2 AdamW＋全訓練集無重複棋局實驗

使用者已核准實作，新增 src 全部置於 `src/go_aicup/tasks/strength/reproduce_paper_2/`，重用原 SE 模型、loss、資料及評估報表。來源固定為 `se_seed2026/best_1.pt`（35,000 步及 checksum）；載入模型／BN／每 rank RNG，重建 AdamW 狀態，抽樣從新一輪開始。新增 70,000 updates：前 50,000 LR=0.002、後 20,000 LR=0.001；betas=(0.9,0.999)、eps=1e-8、weight_decay=0.0001，無 warmup，其他模型／global 128 組／8 卡設定沿用。

每級各 80,000 盤分別 shuffle，8 卡按全域排列分配，每輪全部 800,000 棋局用完才重抽；每盤仍抽七個有效位置，不要求遍歷每盤所有位置。625 steps／輪，共 112 輪，第一段 80 輪、第二段 32 輪；依完成 step 重建輪次／游標，不以預取進度續跑。新增步數與累計步數分開記錄，最終分別為 70,000／105,000。沿用 game-disjoint；實際 validation 的 7,855 位玩家全在 training 的 8,007 位內，未驗證新玩家泛化。

獨立設定 `configs/strength/se_seed2026_adamw70k.json`、入口 `python -m go_aicup.tasks.strength.reproduce_paper_2.train`、Slurm `slurm/se_adamw70k.slurm`（8 卡 6 小時），輸出 `outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/`。新增 50,000 步發布 milestone 並做全量 centroid／validation，完成後續訓；70,000 步完訓後再全量評估。各階段共用 process group，避免中途重建通訊停滯。依使用者最新要求，SE2 訓練期間只留最新一份完整 checkpoint 供續跑，不保存最低 loss 的 best_1／best_2；最終 model.pt 驗證成功後才清除續跑狀態。原 run 唯讀，來源 best_1.pt 保留。原 SGD 速度外推純訓練約 3.57h／28.6 GPU-h，AdamW 實測成本與四桶增益未知。僅 CPU／合成資料驗證，不提交 GPU 作業；操作及驗證範圍見新套件 README。


### 2026-11-03：reproduce_paper_3 十級 CE 實作

依使用者核准的 SE3 plan，分五項完成獨立設定／18-block 三頭分類模型與 FP32 loss、每盤三十有效位置且全輪無重複棋局抽樣、固定分層監控與 log-prob 分類評估、AdamW 訓練／early stopping／可重試 best／latest 發布及 Slurm／文件。預設 seed=2026、channels=256、18 blocks、361→256→10；保留 policy／soft value 輔助 loss。正式八卡 global 128 組、每卡 4,800 位置，100,000 updates，前 70,000 LR=0.01、後 30,000 LR=0.005；每 5,000 步固定最多 5,000 題、每盤三十位置監控。盤內平均 log_softmax、題內各盤等權平均、argmax；同分更新較晚 best 並重設 patience，三次嚴格較低早停。早停／達預算皆發布最佳監控模型後，完整 validation 全部有效位置＋固定 10,000 題評估；不推論 training、不使用 centroid／score_rank。

舊 prepared manifest 的七位置身分經完整驗證後唯讀重用，三十位置另記本 run 設定。官方 train 800,000／valid 200,000 盤最少有效位置均為 35，不足三十均為 0。SE3 CPU／Gloo 合成測試 20 項及既有相關回歸 87 項通過；涵蓋預取續跑、全域平均／梯度、四處驗證失敗注入、半行 log、早停／最佳模型發布／最終發布失敗及全量分類報表重試。未提交 GPU 短測、正式訓練、官方全量推論或上傳，真實四桶分數、CUDA／SyncBN、顯存／吞吐、長程 LR 穩定性未知。

入口 `python -m go_aicup.tasks.strength.reproduce_paper_3.train`，正式／smoke 設定 `configs/strength/se3_seed2026.json`／`se3_smoke100.json`；獨立輸出 `outputs/rank/reproduce_paper_3/se3_seed2026/`／`se3_smoke100_8gpu/`。Slurm 為 `slurm/se3_train.slurm`／`se3_smoke100.slurm`，仍依現有規則被 Git 忽略。詳見新套件 README／plan 與 RESULTS；八卡短測實測估時後才確認長作業。原 SE／SE2 檔案未修改。


### 2026-11-04：SE3 每盤抽樣改為二十位置

依使用者要求，SE3 訓練與固定監控由每盤三十改為二十個無重複有效位置；config、sampler／collate、model／loss shape、trainer reshape 與監控身分同步修改。global 128 組／八卡各 16 組維持，每組 200、每步 25,600、每卡 3,200 位置，十萬步最多使用 2,560,000,000 次位置樣本。min_ply=30、teacher／split、LR／更新預算及最終全量 validation 使用全部有效位置均沿用。

二十位置使用獨立 `outputs/rank/reproduce_paper_3/se3_seed2026_20pos/`、`se3_smoke100_20pos_8gpu/`，保留三十位置 smoke，不混用舊 checkpoint／監控。job 691911 已完成三十位置八卡 100 步與 5,000 題驗證，1.47 steps/s、allocated 峰值 63.56 GiB、監控 22.32s，k=5／10／15／20=0.1595／0.1516／0.1662／0.1639、Equal-k=0.1603；僅為歷史短測，不能視為二十位置效能。二十位置 GPU 顯存／吞吐／四桶分數與長程收斂未知，本次未提交 GPU 作業。現行操作見 SE3 README／plan §12。

二十位置版本 CPU／Gloo 合成回歸 20 項通過（68.11s），包含二十位置抽樣／監控、十九位置拒絕、spawn 預取精確續跑及兩程序 loss／梯度／訓練／評估；兩份 config 與 Slurm 語法檢查通過。未重跑 GPU。


### 2026-11-04：SE3 每盤抽樣改為十五位置

依使用者最新要求，SE3 訓練與固定監控改為每盤十五個不同有效位置；config、sampler／collate、model／loss shape、train reshape 與監控身分同步更新，不足十五位置報錯。每組十級共 150 位置，global 128 組共 19,200 位置／update，八卡每卡 16 組／2,400 位置；十萬步最多使用 1,920,000,000 次位置樣本。min_ply=30、LR／組數／更新預算、split／teacher 及最終全量 validation 使用全部有效位置均沿用。

十五位置改用獨立 `outputs/rank/reproduce_paper_3/se3_seed2026_15pos/`／`se3_smoke100_15pos_8gpu/`，保留舊三十／二十位置產物，不混用其 checkpoint／監控集合。二十位置 smoke job 692064 已完成八卡 100 步＋5,000 題監控，2.15 steps/s、allocated 峰值 42.55 GiB、監控 16.51s；四桶 0.1718／0.1685／0.1678／0.1577，Equal-k=0.1665，僅為歷史短測。十五位置 GPU 效能與分數未知；本次不提交 GPU 工作。現行規格及操作見 SE3 README／plan §13。

十五位置版本 CPU／Gloo 回歸 20 項通過（47.81s），十五位置邊界、固定監控、worker 預取續跑與兩程序梯度／訓練／評估皆通過；兩份 config、Slurm 語法與三十／二十位置 smoke 保留檢查通過，未重跑 GPU。


### 2026-11-06：SE3＋H＋C＋B2 兩層分類頭實作

使用者已核准實作，目錄依指定為 `src/go_aicup/tasks/strength/SE_H_C_B2/`（加號改為底線）。凍結 SE3／HumanSL／C，四個 k 各使用 `Linear(60,32) → ReLU → Linear(32,10)`，訓練 CE 接原始 logits、推論 Softmax＋score_rank；僅訓練分類頭與擬合 scaler。獨立設定、同色固定題組／來源稽核、凍結特徵重用與分片補算、CPU AdamW／cosine／逐 k 早停／續跑／固定 epoch 最終重訓、配對 bootstrap／報表、整題 AH＋C fallback、官方 Public／Private 合併 CSV 及套件內可追蹤 Slurm 已建立。計畫見 [STRENGTH_SE3_HC_B2_PLAN.md](STRENGTH_SE3_HC_B2_PLAN.md)，操作／驗證與成本限制見 [新目錄 README](src/go_aicup/tasks/strength/SE_H_C_B2/README.md)。CPU 合成與既有 release／SE3 回歸共 41 項通過（71.21 秒），來源／快取格式唯讀查核通過；未發布正式題組、啟動官方 GPU 推論、B2 正式訓練、提交作業或上傳。SE3 玩家重疊與 B2 in-sample 信心分布偏移限制保留，分 k 增益、LB、CUDA／KataGo 接線及完整成本未知；完整 GPU 作業前仍先回報短測估時。

### 2026-11-06：SE3＋H＋C＋B2 單次提交訓練

依使用者要求，新增可追蹤的 `src/go_aicup/tasks/strength/SE_H_C_B2/slurm/train.slurm`，一次提交依序執行 prepare、八卡開發 SE3 特徵、兩臂 B2 開發訓練／S-select、開發 gate，再視結果補算最終 SE3／H／C、最終兩臂訓練與 U_newgames 評估；gate 不通過時保留開發報表並結束。預設 8 卡／12 小時上限，即最多 96 GPU-h（含 CPU 階段）；實際總耗時與 SU 未知。修正特徵補算在多程序啟動時間不同時可能漏分片的問題，以固定 game_index 分配；含既有完整／限量分片及單一作業成功、gate 停止、worker 失敗的 21 項 CPU 測試通過。未提交 Slurm、執行 GPU 推論或正式訓練；歷史吞吐的局部估算與重跑限制見該目錄 README。


### 2026-11-09：HumanSL residual B2 三階段實作

依使用者明確授權，三個 subagent 與主 agent 已完成 `humansl_residual_b2/PLAN.md` 的資料／雙方串流輸入、兩階段模型／loss、獨立級別 sampler、teacher、DDP／梯度累積／嚴格改善早停／精確續跑、OOF／線性 B2、受控 H＋C 對照、M 選模／U 鎖定、release 固定步數重訓與 CSV 接線，及套件內可追蹤的三階段／短測 Slurm。原 CV helper 僅在唯一 `data/cv.py` 增補，既有 export 只擴充相容檔名參數；原實驗權重／索引未改寫。

CPU／Gloo 整合 108 項（108.24 秒）與最新 B2 15 項（1.83 秒）通過，合計 109 個不同測試；包括實際兩程序更新／累積梯度、stage2 凍結、RNG／sampler／spawn 預取續跑、失敗發布／監控重試及測試 store 接線。既有 HD0／HD1 各三個位置的編碼 parity 通過。未執行 GPU、真 teacher、正式全量 prepare／訓練／推論或提交作業；真實各 k 分數、CUDA／SyncBN 與成本未知。

短測 `humansl_residual_smoke100.json` 兩階段各 100 步／AdamW LR=0.001（C 對照 100 步／LR=0.01），只屬工程起點；正式 `humansl_residual_train.json` 的 LR／updates 留 null，須短測後依 §9 明訂預算。原 seed=2027 完整 fold 表在本機預設路徑缺失，prepare 不會重抽；須取回原切分身分。輸出為 `outputs/rank/humansl_residual_b2/<run_id>/`，完整操作／差異／未驗收事項見該目錄 README、PLAN §13 及 RESULTS 本日紀錄。


### 2026-11-10：HumanSL residual B2 修訂

依使用者「全部依建議修改」：B2 主臂改 log 機率平均（保留 prob 對照）、新增 H＋C＋新特徵的 stack 臂、B2 擬合擴大到全部 D 玩家（`ext_v1/fit_all`，每人每色最多 20 盤）、兩階段改 3,000／6,000 步加 cosine 與較密監控、不訓練全 D 模型（M／U／測試用四折平均）、取消 release 重訓（release B2 加入 M＋U）、新增低成本探針 `probe.py` 作 go/no-go。teacher 身分與 base prepared 未變。62 項 CPU 測試通過；未提交任何作業，分 k 增益與成本未知。詳見套件 PLAN §14 與 RESULTS。


### 2026-11-17：raw 與 strength_train_hc 僅保留 ZIP

依使用者要求，已將 `data/raw/` 與 `data/processed/strength_train_hc/` 分別封存為同層的 `data/raw.zip`（413,620,651 bytes）與 `data/processed/strength_train_hc.zip`（2,268,546,012 bytes），並刪除兩個原目錄。兩份封存均以逐檔 SHA-256 比對來源、核對完整項目清單及符號連結，確認來源未變後才刪除；合計一般檔案邏輯容量減少 1,437,097,326 bytes（約 1.34 GiB）。ZIP 內保留頂層目錄，還原分別執行 `unzip data/raw.zip -d data/` 與 `unzip data/processed/strength_train_hc.zip -d data/processed/`。讀取原始 CSV、重做前處理或讀取舊 H/C 特徵前須先還原；既有指向 strength_train_hc 的相容連結未移除，還原前無法讀取。未改動其他資料或模型。


### 2026-11-17：e2e_adam 僅保留 ZIP

依使用者要求，`data/processed/identity/e2e_adam/` 已完整打包為同層 `e2e_adam.zip`（4,224,570,410 bytes），再次逐檔 SHA-256 核對封存內容與來源一致、確認來源未變後刪除原目錄（原一般檔案共 5,376,719,380 bytes，約 5.01 GiB）。ZIP 包含 lr1e5／lr3e5 兩臂的完整快取；重做依賴它們的分析前，執行 `unzip data/processed/identity/e2e_adam.zip -d data/processed/identity/` 還原。其他實驗目錄與模型未刪除。
