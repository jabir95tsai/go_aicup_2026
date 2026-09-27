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
