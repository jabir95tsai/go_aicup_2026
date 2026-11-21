# 外部資源申報清單

> 官方規則：外部資料、開源程式、生成式 AI 工具可以用，但**報告中必須申報來源與貢獻**。
> 每次用到新的外部資源，**當下**加一列；「實際貢獻」欄在用進提交檔後補上數字。
> 狀態：`候選`（已調查、未使用）／`使用中`／`已棄用`。

## 模型與權重

| 資源 | 版本 | 來源 | 授權 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|---|---|
| KataGo Human SL 網路 | `b18c384nbt-humanv0.bin.gz`（2024-07；sha256 `637746e4…4ab5`） | [KataGo v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0) | KataGo Neural Network License（MIT 式，須保留版權聲明）—[已驗證](https://katagotraining.org/network_license/) | 競賽二：15 個 `rank_*` profile 下實際著手的 log-likelihood，本人／對手各取每盤平均（A-039～A-047） | **使用中**（2026-10-24 起；正式候選模型的特徵） | Valid-U 1,000 題：A-024＋C2 0.5532 → A-024＋HumanSL＋B2 0.6833（+0.130，A-044）；單用 HumanSL 0.657（A-045） |
| KataGo Human SL 網路（PyTorch checkpoint） | `b18c384nbt-humanv0.ckpt`（sha256 `84b23476…c8b27`；使用一般權重，非 SWA） | [KataGo v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0) | 同上 | PyTorch 重現引擎 policy（A-050）、主幹表示（A-051、A-052）。**競賽一（2026-11-20 起）**：凍結網路，metadata 換成我方自行學得的 C v2 幽玄級別向量（C／A／2D／5D 四個條件），取每個本人落子位置的前 32 名 policy 當玩家條件走子似然的基底（`scripts/identity/pid2_likelihood.py`）；只用於特徵抽取，不微調 | **使用中**：競賽一最終提交 `submissions/pid/821010-comb-probe.csv` 的新增集似然項 | 競賽一驗證台封存 pool 1：ogsa12w 0.7114 → 加似然項 0.7418（+0.030，z 17.6）→ 再加逐 k 倍率 0.7459；Public 0.8388310 → `817714-lik.csv` 0.8464716 → `817757-lik2lk.csv` 0.8461584 → `818024-likrich.csv` 0.8535888（加局部棋形特徵）→ `821010-comb-probe.csv` 0.8623864（經驗貝氏先驗＋逐 k 學習融合權重；`RESULTS.md` 11/20–11/21）。競賽二主幹表示加入主線後 S-select −0.035（A-052），未用於競賽二提交 |
| KataGo 強網路 | `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`（sha256 `9d7a6afe…a51f1d`） | [katagotraining.org/networks](https://katagotraining.org/networks/) | KataGo Neural Network License（同上） | 競賽二：每手目數／勝率損失等著手品質特徵（A-048） | 使用中（實驗；未進正式模型，D-003 保留待 LB） | 加入主線後 S-select −0.0024（A-048） |

## 開源程式

| 資源 | 版本 | 來源 | 授權 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|---|---|
| KataGo 引擎（analysis engine，預編譯執行檔） | v1.18.2 `katago-v1.18.2-cuda12.8-cudnn9.8.0-linux-x64.zip`（2026-08-30；sha256 `2cb197fc…6c02`） | [github.com/lightvector/KataGo/releases](https://github.com/lightvector/KataGo/releases/tag/v1.18.2) | MIT | 批次推論 HumanSL 與強網路（`features/humansl.py`、`features/katago_engine.py`）；1 visit、日本規則、貼目 6.5 | **使用中**（正式候選模型的特徵抽取） | 同 HumanSL 網路列 |
| KataGo 原始碼（Python 部分） | v1.18.2 `v1.18.2.tar.gz`（sha256 `fefd554b…f8bf9`） | [github.com/lightvector/KataGo](https://github.com/lightvector/KataGo/archive/refs/tags/v1.18.2.tar.gz) | MIT | 執行時匯入 `python/katago`（Board、Features、SGFMetadata、model_pytorch、load_model）；profile metadata 依 `cpp/neuralnet/sgfmetadata.cpp` 重寫成 Python（`features/humansl_torch.py`）；未修改原始碼 | 使用中（實驗） | 見 PyTorch checkpoint 列 |
| NVIDIA CUDA／cuDNN 執行期函式庫 | `nvidia-cublas-cu12==12.8.4.1`、`nvidia-cuda-runtime-cu12==12.8.90`、`nvidia-cuda-nvrtc-cu12==12.8.93`、`nvidia-cudnn-cu12==9.8.0.87`（PyPI wheel，只解出 `.so`） | [PyPI](https://pypi.org/project/nvidia-cudnn-cu12/) | NVIDIA 軟體授權（隨 wheel 附帶） | 讓 KataGo CUDA 12.8 版在 Nano4 執行（`data/external/cuda-libs-12.8/`）；另用 `go` 環境的 libstdc++ | 使用中（執行環境） | 環境依賴，非模型增益 |
| scikit-learn | 1.7.2（`requirements.txt`） | [scikit-learn.org](https://scikit-learn.org/) | BSD-3 | B2 題目層級 logistic regression、PCA | 使用中 | 正式候選模型的分類器 |
| Ponytail | 5.3.3 | [dietrichgebert/ponytail](https://github.com/dietrichgebert/ponytail) | MIT | Codex plugin，提供開發與程式審查 skills | 已安裝（2026-11-02；lifecycle hooks 待使用者於 Codex 信任） | 開發工具，無模型分數貢獻數字 |
| CodeGraph | 1.6.2 | [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph) | MIT | Codex MCP 本機程式碼索引與呼叫關係查詢；索引位於 `.codegraph/`，遙測已關閉 | 使用中（2026-11-02；109 個 Python 檔案索引與查詢通過） | 開發工具，無模型分數貢獻數字 |

## 外部資料

| 資源 | 來源 | 授權 | 用途 | 狀態 |
|---|---|---|---|---|
| OGS（Online-Go.com）棋譜合集 2013–2025-05，by-username SGF（`sgfs-by-username.tar.gz`，16.2 GB） | [za3k.com/ogs](https://za3k.com/ogs/ogs_games_2013_to_2025-05/sgfs-by-username.tar.gz)（za3k 整理） | 發布頁載明經 OGS 同意重散布，README 記載 OGS 同意研究使用；**用於本競賽與得獎交付已於 2026-11-20 經使用者確認** | 競賽一：只用於訓練風格編碼器。過濾條件為 19 路、ranked、HA≤1、無擺子、已完成、≥60 手、對手是真人。A1 用 40,000 人×20 盤，A2 用全部 82,481 人×10 盤，幽玄抽樣比例 0.5。不參與推論、候選庫或測試題 | 使用中：最終提交 `submissions/pid/821010-comb-probe.csv`，A1／A2 仍為新增集風格項的一部分（2026-11-20 前的最終檔為 `ogsa12w.csv`：新增集風格項 1.5×(0.25 seqB＋0.75·mean(A1, A2))）；Public 0.8388310，對不用 OGS 的 `hung-seqB-e2emix-ml.csv`（0.8245254）+0.0143（`RESULTS.md` 11/18–11/20、D-008） |

## 方法參考（引用，不含程式碼）

| 文獻 | 用到的想法 | 用在哪 |
|---|---|---|
| McIlroy-Young et al., *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess*, NeurIPS 2021 | 多盤 centroid 檢索、GE2E loss、開局必須入模 | 競賽一 |
| Kuboki et al., *Policies of Multiple Skill Levels for Better Strength Estimation in Games*, arXiv 2505.00279 (2025) | 多段位 HumanSL policy 特徵 + LightGBM；同一玩家驗證設定 | 競賽二 |
| Chen, Shih, Wu, *Strength Estimation and Human-Like Strength Adjustment in Games*, ICLR 2025 | 每手 strength score + Bradley-Terry、多盤平均 | 競賽二 |
| Wan et al., *Generalized End-to-End Loss for Speaker Verification*, ICASSP 2018 | GE2E loss | 競賽一 |

## 生成式 AI 工具

| 工具 | 用途 | 範圍 |
|---|---|---|
| Claude Code（Anthropic，Claude Opus 5.5） | 程式撰寫與審查、資料稽核與 EDA 腳本、文獻調查、文件撰寫 | 全專案；所有程式由團隊審閱、執行並驗證 |
| OpenAI Codex（2026-10-12） | 依已確認的 V1 設計撰寫 ResNet、packed DataLoader、資料索引、全域排序 loss、DDP 訓練與校準評估程式及操作文件 | `models/strength_resnet.py`、`tasks/strength/`、V1 `data/cv.py` API 與 config；依使用者要求未執行驗證，尚無訓練／分數貢獻數字 |

## 算力環境與官方操作文件

| 資源 | 來源／查核日期 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|
| 國網中心晶創 26（Nano4）H200 官方文件 | [系統規格](https://man.twcc.ai/@nano4-manual/SJuKzVlwbx)、[Slurm 佇列](https://man.twcc.ai/@nano4-manual/SJM_FuxDWe)、[登入與傳輸](https://man.twcc.ai/@nano4-manual/BydP-_lvZg)、[Modules](https://man.twcc.ai/@nano4-manual/BJyI6dgw-g)、[NRQ 說明](https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=54)；2026-10-10 | 執行環境與作業限制回填至 CLAUDE.md §0 | 使用中（文件引用；尚未執行 HPC 作業） | 工程環境參考，非模型／資料；無分數增益數字 |
| H200 硬體規格與 Nano4 系統登錄 | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)、[TOP500 Nano4](https://www.top500.org/system/180421/)；2026-10-10 | 查核 GPU 型號與顯存規格 | 使用中（文件引用） | 硬體規格參考，非效能實測 |
| PyTorch 官方安裝與分散式文件 | [歷史版本安裝](https://docs.pytorch.org/get-started/previous-versions/)、[Distributed](https://docs.pytorch.org/docs/2.6/distributed.html)、[torchrun](https://docs.pytorch.org/docs/2.6/elastic/run.html)；2026-10-11 | Windows／Linux CUDA 12.4 依賴來源與單節點 NCCL／DDP 環境檢查 | 使用中（文件引用與 API 使用） | 環境相容性設定，非模型增益；Nano4 尚未實測 |
| PyTorch 2.6 DataLoader／可微 collective／SyncBN 官方文件與原始碼 | [DataLoader](https://docs.pytorch.org/docs/2.6/data.html)、[可微 all-reduce](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/distributed/nn/functional.py)、[DDP](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/parallel/distributed.py)、[SyncBN](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/_functions.py)；2026-10-11；2026-10-12 補閱 DataLoader／可微 SUM API | STRENGTH_V1_IMPLEMENTATION.md 與 V1 程式：批量讀取、全域級別平均梯度尺度與尾批 BN；閱讀 API，未複製外部程式碼 | 使用中（程式已撰寫，未驗證） | 無模型分數或 Nano4 效能增益數字；梯度、SyncBN 與續跑待實測驗收 |
| NVIDIA CUDA 12.4 Release Notes | [driver 相容表](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)；2026-10-11 | 查核 Linux CUDA 12.4 driver 基準與 minor compatibility 限制 | 使用中（文件引用） | 環境參考，非效能實測 |

## 2026-10-30：Strength Estimator 重現前置查核

- 論文：本機 `reference/2502.17109v2.pdf`（ICLR 2025，含附錄 A–C）；既有方法參考列的延伸查核。
- 作者程式：[rlglab/strength-estimator](https://github.com/rlglab/strength-estimator/tree/d9498964e2ba84b5798c40e3f5b8a91a8791d66c)，`iclr2025`／commit `d9498964e2ba84b5798c40e3f5b8a91a8791d66c`，GPL-3.0；僅閱讀 README、Go configs、網路、訓練、抽樣與評估邏輯，未複製或納入程式。
- 用途：盤點現有 V1／shared 管線可重用處，辨識資料、分組 loss、auxiliary heads 與 centroid 解碼差異。產物：`src/go_aicup/tasks/strength/reproduce_paper/README.md`。
- 狀態：候選／計畫待確認；未新增外部訓練資料、未訓練、無分數貢獻。OpenAI Codex 協助閱讀與撰寫本次盤點文件。

## 2026-11-01：SE 第一、二步實作查核

沿用上列作者固定版本，補閱 `strength/misc/game_wrapper.cpp`、`strength/trainer/st_data_loader.cpp` 及內嵌 MiniZero 的 `minizero/minizero/environment/{base/base_env.h,go/go.cpp}`，確認落子前輸入、當手 policy target、本人／對手歷史順序與零 padding。僅閱讀參考，未複製或納入作者程式碼；本專案沿用既有 packed cache，自行撰寫設定、索引準備、樣本轉換與合成驗證。沒有外部訓練資料或新模型分數。OpenAI Codex 協助本次實作與驗證。來源連結與契約見 `src/go_aicup/tasks/strength/reproduce_paper/README.md`。


## 2026-11-02：SE 第五步三頭模型／loss 實作查核

沿用 GPL-3.0 作者固定版本 [d9498964e2ba84b5798c40e3f5b8a91a8791d66c](https://github.com/rlglab/strength-estimator/tree/d9498964e2ba84b5798c40e3f5b8a91a8791d66c)，閱讀 `strength/trainer/bt_network.py`／`train.py`、同 tree 內 MiniZero `network/py/alphazero_network.py`／`network_unit.py` 與 `cfg/se_go.cfg`。依查核的架構及數學公式重新撰寫 `reproduce_paper/model.py`／`losses.py`（非直接複製來源檔），包含 bias、post-add ReLU、預設 BN、policy 2 channels、scalar heads、value tanh 與 listwise 正規化；省略未啟用的 weight head，改以本專案分組／DDP 介面整合。未納入作者權重、外部訓練資料或作者執行程式依賴；不宣稱作者 checkpoint／二進位逐值相容。

OpenAI Codex 協助來源查核、程式與合成測試及文件。CPU／Gloo 合成回歸 67 項通過，僅代表工程驗證；分 k 成績與對競賽分數的貢獻未知，未啟動正式訓練。詳細來源連結、架構對齊／偏離及 API 見 `src/go_aicup/tasks/strength/reproduce_paper/README.md` 第五步。


## 2026-11-02：SE 8 卡／100 步量測入口

OpenAI Codex 協助撰寫 `reproduce_paper/train.py`、短測 config／Slurm 腳本、CPU 合成測試及操作文件。沿用本專案 shared runtime／checkpoint、既有三頭模型及分組 loss；本次沒有新增外部資料、模型權重或第三方依賴。僅完成工程驗證，尚未提交 GPU 短測；分 k 成績、增益與實測成本未知。


## 2026-11-02：SE 卷積／布局對照與推論 batch 量測

參考 [PyTorch 2.6 cuDNN benchmark 文件](https://docs.pytorch.org/docs/2.6/backends.html#torch.backends.cudnn.benchmark) 與 [官方 channels-last 教學](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)，查核模型／4D 輸入布局轉換及卷積演算法搜尋介面。僅使用既有 PyTorch 公開 API，未新增依賴、外部資料或權重；未複製外部實作。OpenAI Codex 協助本專案 train 布局接線、獨立對照 config、infer.py 與多程序／數值合成測試。GPU 加速、BF16 差異與競賽分數貢獻未知，未提交新作業。
