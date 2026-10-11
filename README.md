# AI CUP 2026 圍棋競賽

兩個競賽共用一份訓練資料、一個 repo：**競賽一 玩家辨識**（`tasks/identity/`）、**競賽二 棋力預測**（`tasks/strength/`）。

| 文件 | 內容 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **唯一事實來源**：規則、評分、時程、資料稽核結果、CV 設計、洩漏清單、上傳紀律、專案結構 |
| [RESULTS.md](RESULTS.md) | 實驗帳本，只增不刪，失敗也記 |
| [STRATEGY.md](STRATEGY.md) / [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | 決勝槓桿與 8 週計畫 |
| [reference.md](reference.md) | 文獻與工具整理 |
| [EXTERNAL_RESOURCES.md](EXTERNAL_RESOURCES.md) | 外部資源申報清單（報告必備） |
| [AGENTS.md](AGENTS.md) | 只指向 CLAUDE.md（給 Codex 等 agent） |

## 目前進度

W1 完成：SGF 稽核、1M 盤轉檔、CV 切分與合成題組、兩賽官方指標、訊號 EDA（細節見 `RESULTS.md` A-001～A-006）。下一步是兩賽的 P0 baseline。

## 上手

使用標準版 CPython **3.11–3.13**（目前套件組合不支援 Python 3.14）；本機參考版本為 3.13.12。
`requirements.txt` 同時使用 PyPI 與官方 CUDA 12.4 索引，Windows／Linux x86_64 共用同一組直接依賴版本。

```bash
python -m pip install -r requirements.txt  # 版本已鎖，報告要用
python -m pip install -e .                 # 讓 go_aicup 可 import
python -m pytest -q                 # 應全數通過
```

**GPU 環境：** 本機 RTX 3060 Ti 8GB 與晶創 26 H200；HPC 佇列、節點規格、連線方式與待查額度見 [CLAUDE.md 第 0 節](CLAUDE.md#算力-2026-10-10-更新支援範圍)。

### Nano4 Linux／單節點多卡

`torch==2.6.0+cu124` 有 Linux x86_64 wheel，作為本機與 H200 待驗證的共同環境基準；GPU 通訊使用 **NCCL + DDP／torchrun**。Linux CUDA wheel 的 NVIDIA／NCCL 依賴交給 pip 解析，無須另加 `nccl`、`torchvision` 或 `torchaudio`。參考 [PyTorch 2.6 安裝方式](https://docs.pytorch.org/get-started/previous-versions/)、[DDP backend 說明](https://docs.pytorch.org/docs/2.6/distributed.html)。

Nano4 已使用 Conda 建置本專案環境，名稱為 **`go`**；使用前先執行 `conda activate go`，不需另建 `.venv`。若作業需要系統模組，先依 Nano4 官方文件執行 `module purge`，用 `module avail`／`module spider` 查核並載入需要的模組，再啟用 `go`；模組名稱與 Conda 初始化方式以現場為準。環境中的 Python 版本須符合 3.11–3.13。以下驗證放在 Slurm 分配的 **H200 計算節點**，先用 `dev` 做 1／2 卡短測，再驗證同節點 8 卡。實際 driver、glibc、可見 GPU 與 NCCL 通訊尚未在 Nano4 實測。

```bash
conda activate go
python --version
python -m pip install --upgrade pip
python -m pip install --dry-run --ignore-installed --only-binary=:all: -r requirements.txt
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip install -e .
python -m pip check
python -m pytest -q
```

`--only-binary=:all:` 讓 Python／glibc 沒有相容 wheel 時直接報錯，避免在 HPC 意外編譯大型套件。CUDA runtime 來自 `cu124` wheel；系統 CUDA module 與 `nvidia-smi` 顯示的 CUDA 上限不必恰好是 12.4，但 NVIDIA driver 必須相容。以 Linux driver **550.54.14 或更新**作為 CUDA 12.4 GA 的驗證基準；更舊 driver 的 minor compatibility 有限制，需額外查核。[NVIDIA CUDA 12.4 driver 表](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)

以下指令在配置 **2 張 GPU、1 個 Slurm task、1 個節點**的 job 內執行。由單一 `torchrun` 啟動每張 GPU 一個 process；不要再用 2 個 Slurm tasks 各啟動一次 `torchrun`。`--nproc-per-node` 依實際配置改成 1、2 或 8，保留 Slurm 設定的 `CUDA_VISIBLE_DEVICES`。

```bash
set -euo pipefail
conda activate go
mkdir -p outputs/rank_env_nano4
nvidia-smi > outputs/rank_env_nano4/nvidia-smi.txt
ldd --version > outputs/rank_env_nano4/libc.txt
python -m pip freeze > outputs/rank_env_nano4/pip-freeze.txt
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/check_torch_ddp.py 2>&1 | tee outputs/rank_env_nano4/ddp-2gpu.log
```

通過條件：所有 process 正常退出，log 的 JSON 顯示 `status: passed`、`world_size: 2`、`all_reduce_sum: 3.0`、`ddp_gradient: 1.5`。改跑 8 卡時，使用 `ddp-8gpu.log`，預期 sum 為 36.0、gradient 為 4.5；`pip-freeze.txt` 記錄包含間接依賴的實際版本。這是環境檢查；目前模型訓練入口尚未實作 DDP，後續訓練仍須包裝 `DistributedDataParallel` 並按 rank 分配資料。

**資料**（不進 git）：從 [官方 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) 下載 `aicup2026_go_training.tar.xz`，核對 SHA-256（見 CLAUDE.md 第 1 節），解壓出的 `training/` 放到 `data/training/`。11/04 發布的候選庫與測試題放 `data/raw/`。

**重建衍生資料**（全 CPU，依序執行，結果是確定性的，兩人重建出的檔案會逐位元相同）：

```bash
python -m go_aicup.data.preprocess train         # SGF → int16 memmap，約 2.5 分鐘
python -m go_aicup.data.cv                       # CV 切分 + 合成題組，數秒
python -m go_aicup.features.game_stats           # EDA 用的盤面統計，約 10 秒
python scripts/eda_report.py                     # 重產 notebooks/02_eda.md，約 2 分鐘
```

## 結構

詳見 CLAUDE.md 第 10 節。重點：

- `src/go_aicup/` 放可重用程式（用 `python -m go_aicup....` 執行）；`scripts/` 放一次性報告與命令列入口
- 兩賽共用 `data/`、`features/`、`models/`、`metrics.py`；各自的訓練、推論、提交檔放 `tasks/strength/`、`tasks/identity/`
- 切分只能由 `go_aicup/data/cv.py` 產生；特徵要過 `go_aicup/features/feature_policy.py`
- 新的實驗 run 一律寫到 `outputs/<run-id>/`（run-id 以 `rank_`／`pid_` 開頭），存 config、seed、git commit、log、OOF、權重、分k桶分數
- 提交檔只由推論程式寫入 `submissions/`，不手動改答案

## 協作規則

1. **預設分支 `main`。** 改動開 branch → PR → 另一人看過再 merge；不要 force push `main`
2. push 前先 `git pull --rebase`
3. **`RESULTS.md` 只在檔尾新增**，不改別人的舊紀錄；編號接續（A-xxx 稽核、L-xxx 文獻、實驗用 tag）
4. 資料、快取、權重、提交檔一律不進 git（`.gitignore` 已擋）；外部資源一用就登記到 `EXTERNAL_RESOURCES.md`
5. **AI CUP 提交只用同一個帳號**（帳號：**待定**）。兩人都可以操作上傳，但**絕不能用各自的個人帳號**（規則：隊員各自上傳 = 多重帳號 → 喪失資格）。
   每日 5 次額度兩人共用、計分看最後一次上傳 → 上傳前先在 `RESULTS.md` 登記 tag／預期分數，11/24 當天先講好由誰做最後一次
6. 隊伍之間不分享程式與特徵；repo 保持 private
