# 外部資源申報清單

> 官方規則：外部資料、開源程式、生成式 AI 工具可以用，但**報告中必須申報來源與貢獻**。
> 每次用到新的外部資源，**當下**加一列；「實際貢獻」欄在用進提交檔後補上數字。
> 狀態：`候選`（已調查、未使用）／`使用中`／`已棄用`。

## 模型與權重

| 資源 | 版本 | 來源 | 授權 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|---|---|
| KataGo Human SL 網路 | `b18c384nbt-humanv0`（2024-07） | [katagotraining.org/extra_networks](https://katagotraining.org/extra_networks/)；[KataGo v1.15.0](https://github.com/lightvector/KataGo/releases/tag/v1.15.0) | KataGo Neural Network License（MIT 式，須保留版權聲明）—[已驗證](https://katagotraining.org/network_license/) | 競賽二：p(手 \| 盤面, 段位) 特徵（STRATEGY L5 / §1b） | 候選 | — |
| KataGo 引擎（analysis engine） | v1.15 以上 | [github.com/lightvector/KataGo](https://github.com/lightvector/KataGo) | MIT | 批次推論 HumanSL 網路 | 候選 | — |

## 開源程式

| 資源 | 版本 | 來源 | 授權 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|---|---|
| （尚無） | | | | | | |

## 外部資料

| 資源 | 來源 | 授權 | 用途 | 狀態 |
|---|---|---|---|---|
| （尚無；目前只用官方資料） | | | | |

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
| OpenAI Codex | 依賴相容性查核、Nano4 安裝文件與 CUDA／NCCL／DDP 環境檢查腳本 | requirements.txt、pyproject.toml、scripts/check_torch_ddp.py 與環境文件；Nano4 GPU 驗證待執行 |

## 算力環境與官方操作文件

| 資源 | 來源／查核日期 | 用途 | 狀態 | 實際貢獻 |
|---|---|---|---|---|
| 國網中心晶創 26（Nano4）H200 官方文件 | [系統規格](https://man.twcc.ai/@nano4-manual/SJuKzVlwbx)、[Slurm 佇列](https://man.twcc.ai/@nano4-manual/SJM_FuxDWe)、[登入與傳輸](https://man.twcc.ai/@nano4-manual/BydP-_lvZg)、[Modules](https://man.twcc.ai/@nano4-manual/BJyI6dgw-g)、[NRQ 說明](https://iservice.nchc.org.tw/nchc_service/nchc_service_qa.php?target=54)；2026-10-10 | 執行環境與作業限制回填至 CLAUDE.md §0 | 使用中（文件引用；尚未執行 HPC 作業） | 工程環境參考，非模型／資料；無分數增益數字 |
| H200 硬體規格與 Nano4 系統登錄 | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/)、[TOP500 Nano4](https://www.top500.org/system/180421/)；2026-10-10 | 查核 GPU 型號與顯存規格 | 使用中（文件引用） | 硬體規格參考，非效能實測 |
| PyTorch 官方安裝與分散式文件 | [歷史版本安裝](https://docs.pytorch.org/get-started/previous-versions/)、[Distributed](https://docs.pytorch.org/docs/2.6/distributed.html)、[torchrun](https://docs.pytorch.org/docs/2.6/elastic/run.html)；2026-10-11 | Windows／Linux CUDA 12.4 依賴來源與單節點 NCCL／DDP 環境檢查 | 使用中（文件引用與 API 使用） | 環境相容性設定，非模型增益；Nano4 尚未實測 |
| NVIDIA CUDA 12.4 Release Notes | [driver 相容表](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)；2026-10-11 | 查核 Linux CUDA 12.4 driver 基準與 minor compatibility 限制 | 使用中（文件引用） | 環境參考，非效能實測 |
