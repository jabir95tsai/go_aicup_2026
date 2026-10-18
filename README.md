# AI CUP 2026 圍棋競賽

兩個競賽共用一份訓練資料、一個 repo：**競賽一 玩家辨識**（`tasks/identity/`）、**競賽二 棋力預測**（`tasks/strength/`）。

| 文件 | 內容 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | 競賽規則、評分、時程、既有資料稽核與專案紀律；第一版棋力模型以以下計畫為準 |
| [STRENGTH_V1_PLAN.md](STRENGTH_V1_PLAN.md) | **第一版棋力辨識實作計畫**：兩層盤面動態組合 18 層、ResNet、級別平均排序 loss、Nano4 DDP；目前設定檔為 10 epochs、全域 batch 65,536 |
| [RESULTS.md](RESULTS.md) | 實驗帳本，只增不刪，失敗也記 |
| [STRATEGY.md](STRATEGY.md) / [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | 決勝槓桿與 8 週計畫 |
| [reference.md](reference.md) | 文獻與工具整理 |
| [EXTERNAL_RESOURCES.md](EXTERNAL_RESOURCES.md) | 外部資源申報清單（報告必備） |
| [AGENTS.md](AGENTS.md) | 只指向 CLAUDE.md（給 Codex 等 agent） |

## 目前進度

W1 完成：SGF 稽核、1M 盤轉檔、CV 切分與合成題組、兩賽官方指標、訊號 EDA（細節見 `RESULTS.md` A-001～A-006）。2026-10-12 已依 [第一版計畫](STRENGTH_V1_PLAN.md) 撰寫 Nano4 盤面 ResNet、資料索引／DataLoader、全域級別平均排序 loss、DDP 訓練／續跑與 training-only 切點評估。**依使用者要求，本次只寫程式，未執行任何新程式或測試；正確性、顯存、吞吐與分數尚未驗證。**

`features/sgf_planes.py` 輸出 `T×2×19×19` 的 float32 黑／白盤面；T 為雙方總落子數。唯一 parser 接收 `(;SZ[19];B[點位];W[點位];...)` 的線性棋譜，落點為 0～360，不支援 pass、setup 或其他 SGF 格式。訓練讀取時才組合目前與前 7 手盤面，再加這一手的黑／白顏色層，形成 `18×19×19`，不預先存全量 18 層 tensor。只在樣本索引過濾指定玩家的手，基礎盤面仍保留雙方歷史。

V1 的位元打包轉檔、續跑與按需解包程式已完成，使用方式見下方「棋力 V1：盤面快取與 DataLoader」。目前尚未執行官方百萬盤盤面轉檔；本次也沒有產生正式 train／validation 索引或權重。新程式的執行方式見「棋力 V1：資料準備、訓練與評估」。

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

通過條件：所有 process 正常退出，log 的 JSON 顯示 `status: passed`、`world_size: 2`、`all_reduce_sum: 3.0`、`ddp_gradient: 1.5`。改跑 8 卡時，使用 `ddp-8gpu.log`，預期 sum 為 36.0、gradient 為 4.5；`pip-freeze.txt` 記錄包含間接依賴的實際版本。這是環境檢查；2026-10-12 新寫的模型訓練入口已有 DDP 與全域 batch 分配程式，**尚未執行單／多卡梯度、SyncBN、尾批與續跑驗收**，環境通過仍不代表訓練入口通過。

**資料**（不進 git）：從 [官方 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) 下載 `aicup2026_go_training.tar.xz`，核對 SHA-256（見 CLAUDE.md 第 1 節）。目前原始 training CSV 位於 `data/raw/training/`，V1 盤面快取直接使用此路徑。以下既有 `preprocess train` 指令仍讀取 `data/training/`；使用 V1 時直接執行下一節的 `board_cache convert`，無須搬動資料或先執行舊轉檔。11/04 發布的候選庫與測試題放 `data/raw/`。

**重建衍生資料**（全 CPU，依序執行，結果是確定性的，兩人重建出的檔案會逐位元相同）：

```bash
python -m go_aicup.data.preprocess train         # SGF → int16 memmap，約 2.5 分鐘
python -m go_aicup.data.cv                       # CV 切分 + 合成題組，數秒
python -m go_aicup.features.game_stats           # EDA 用的盤面統計，約 10 秒
python scripts/eda_report.py                     # 重產 notebooks/02_eda.md，約 2 分鐘
```

## 棋力 V1：盤面快取與 DataLoader

程式位置：[board_cache.py](src/go_aicup/data/board_cache.py) 負責轉檔、打包與解包；[dataset.py](src/go_aicup/tasks/strength/dataset.py) 提供 PyTorch DataLoader 使用的介面。下列命令在專案根目錄執行，環境先依「上手」安裝依賴與 `pip install -e .`。Windows 根目錄為 `D:\college\go_aicup_2026`；Nano4 使用自己的 checkout 路徑與 Conda `go` 環境。

### 1. 轉檔與壓縮

確認 `data/raw/training/` 包含 `train_D.csv`、`train_C.csv`、`train_B.csv`、`train_A.csv` 與 `train_1D.csv`～`train_6D.csv`。預設檢查每級 100,000 盤，共 1,000,000 盤；異常棋譜或盤數不符會報錯，不靜默跳過。

```bash
python -m go_aicup.data.board_cache convert
```

這個命令會實際開始全量轉檔。轉檔使用 CPU，逐批讀取 CSV，由各 worker 解析 SGF、逐盤重播並寫入自己的分片。預設每片最多 4,096 盤、不跨 rank；worker 數為可用 CPU 數與 8 的較小值，也會考慮 CPU affinity 與 `SLURM_CPUS_PER_TASK`。Nano4 請在 Slurm 分配的計算節點執行，使用單一轉檔主程序；耗時尚未全量實測。

可以明確指定來源、目的與平行數；以下單行命令可用於 PowerShell 或 Bash，`--workers 8` 需要配置至少 8 個可用 CPU：

```bash
python -m go_aicup.data.board_cache convert --input-dir data/raw/training --output-dir data/processed/boards/strength_v1 --workers 8 --games-per-shard 4096
```

### 2. 資料格式與輸出位置

預設輸出至 **`data/processed/boards/strength_v1/`**；本機完整路徑為 **`D:\college\go_aicup_2026\data\processed\boards\strength_v1\`**。

```text
data/processed/boards/strength_v1/
  build.json                       # 來源／轉換程式 SHA-256、格式與分片設定
  manifest.json                    # 全部成功後才發布的完成清單
  games.parquet                    # game_index → shard_id、local_offset、length 等
  shard_00000.npy                  # uint8[該片總手數, 2, 46] 打包盤面
  shard_00000_moves.npy            # int16[該片總手數] 落點 0～360
  shard_00000_colors.npy           # int8[該片總手數] 實際落子顏色，0 黑、1 白
  shard_00000_games.parquet        # 該片棋局 metadata
  shard_00000.done.json            # 該片檔案大小、shape 與 SHA-256
  ...
```

T 為每盤雙方總落子數，各盤 T 不同；邏輯上是一百萬個 `[T,2,19,19]`，實際以 offset／length 存入分片，沒有把所有棋局補成同一個 T。channel 0 固定黑棋、channel 1 固定白棋，每個盤面是該手落子與提子後的狀態。

每層按 row-major 把 361 個 0／1 值打包成 46 bytes，固定 `bitorder="little"`，最後 7 個 padding bits 為零；因此每手兩層共 92 bytes。以 2 億手估算，打包盤面 payload 約 **18.4 GB**，另外還有 moves、colors、metadata 與暫存；實際大小依總手數而定。解包結果是 `uint8[T,2,19,19]`，值為 0／1。

`games.parquet` 保留 `(rank, game_id)`、player_id、指定玩家顏色、source_row 與指定玩家手數供索引／稽核使用。`game_index` 是零起算的 metadata 列號；不可把跨級別重複的 `game_id` 當陣列索引。

### 3. 中斷續跑與完整性檢查

中斷後以相同來源、rank、預期盤數、分片大小與目的目錄重跑原命令即可續跑；worker 數可以調整。程式核對來源與轉換程式 SHA-256，再檢查已完成分片的檔案大小、shape 與 hash；完整分片跳過，缺失或損壞的分片重做。來源、轉換程式或分片設定改變時，請用新的 `--output-dir`，避免混用不同版本。

整份成功後才會產生 `manifest.json`，讀取端要求 manifest 與各片完成標記一致。完成後可執行全檔 hash、shape、padding 與顏色檢查，並固定抽每個 shard 一盤，核對合法性稽核重播與快取；指定自訂快取路徑時，`verify` 也要傳入相同的 `--output-dir`：

```bash
python -m go_aicup.data.board_cache verify --output-dir data/processed/boards/strength_v1
```

成功時回報 `num_games`、`total_ply`、`payload_bytes` 與 `sampled_games`。一般轉檔不再逐盤解包比對，也不檢查占點、自殺與立即劫；可用 `sgf_to_planes(content, validate_moves=True)` 做明確的棋步稽核。`verify` 會讀取全部快取檔案，需要 I/O 時間；不產生未壓縮副本。

### 4. Python 按需解包

快取完成後即可使用下面的介面。程式只 memmap 分片並解包所需資料，不重新解析 SGF。

```python
from go_aicup.data.board_cache import BoardCache

cache = BoardCache(max_open_shards=8)
try:
    boards = cache.game(0)                 # uint8[T, 2, 19, 19]，第 0 盤
    first = cache.game(0, [0])             # uint8[1, 2, 19, 19]，只讀第 0 手
    colors = cache.colors(0)              # int8[T]，實際落子顏色
    history = cache.history(0, 0)         # uint8[8, 2, 19, 19]
    inputs = cache.training_batch([0], [0])  # uint8[1, 18, 19, 19]
finally:
    cache.close()
```

`history()` 依目前手、前一手到前七手排序，包含雙方每一手；不足的歷史補零，禁止跨棋局。`training_batch()` 將這 8 個兩層盤面展成 channel 0～15，再加當手實際黑／白顏色的兩個全盤 channel 16／17。這個底層介面可讀任意合法 ply；下面的 Dataset 另檢查樣本是否為指定玩家落子。

已有記憶體陣列時，可以直接使用 codec：

```python
from go_aicup.data.board_cache import pack_planes, unpack_planes

packed = pack_planes(boards)       # uint8[T, 2, 46]
restored = unpack_planes(packed)   # uint8[T, 2, 19, 19]
```

### 5. 訓練時的 DataLoader

`PackedBoardDataset` 接收兩個等長的一維整數樣本索引：`game_indices` 與 `plies`。每一組 `(game_index, ply)` 必須是指定玩家落子後的盤面。依訓練集黑先、黑白交替且 pass=0 的已知條件，允許用零起算 ply 的奇偶選取。訓練資料仍依 `board_cache.py` 保存 `shard_XXXXX_colors.npy`；現行程式沿用 colors 選取指定玩家並組合顏色層，Dataset 仍檢查對手 ply。18 層組裝由快取與 Dataset 共用 `assemble_inputs()`。

以下示範在快取完成後取第 0 盤的指定玩家盤面，用於確認解包與 loader 介面。**這是單盤讀取範例，沒有建立正式 train／validation 切分，也沒有啟動模型訓練。** 將程式存成獨立 `.py` 後執行；Windows 多程序需要保留 `__main__` guard。

```python
import numpy as np
import torch
from torch.utils.data import DataLoader

from go_aicup.data.board_cache import BoardCache
from go_aicup.tasks.strength.dataset import PackedBoardDataset


def main():
    cache = BoardCache()
    try:
        game_index = 0
        player_plies = np.flatnonzero(
            cache.colors(game_index) == cache.player_colors[game_index]
        ).astype(np.uint16)
    finally:
        cache.close()

    game_indices = np.full(len(player_plies), game_index, dtype=np.uint32)
    dataset = PackedBoardDataset(game_indices, player_plies)
    loader = DataLoader(
        dataset, batch_size=32, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in loader:
        # CPU 保持 uint8；傳到裝置後再轉成模型需要的 dtype。
        inputs = batch["planes"].to(device, non_blocking=True).float()
        print(inputs.shape, batch["rank_id"].shape)
        # inputs: [B, 18, 19, 19]；rank_id: [B]，D=0 ... 6D=9。
    dataset.close()


if __name__ == "__main__":
    main()
```

Dataset 回傳 `planes`、`rank_id`、`game_index`、`ply`、`sample_index` 與 `valid`；只有 `planes` 作為模型輸入。批次取樣按 shard 分組讀取，再恢復原樣本順序；各 worker 自行開啟 memmap，預設最多保留 8 個 shard handle 組。`valid` 用於分散式尾批補齊，補齊項在 forward 前移除；`sample_index` 用於按原樣本順序保存推論分數。

正式的大型樣本索引建議傳入 `.npy` 路徑，讓 worker 按需 memmap，避免 Windows spawn 複製完整索引。例如，**在另行產生以下兩個檔案之後**：

```python
dataset = PackedBoardDataset(
    "data/processed/strength_v1/YOUR_DATASET_ID/train_game.npy",
    "data/processed/strength_v1/YOUR_DATASET_ID/train_ply.npy",
    directory="data/processed/boards/strength_v1",
)
```

`YOUR_DATASET_ID` 換成實際資料版本，正式索引為 `uint32[N]` 的 `train_game.npy` 與 `uint16[N]` 的 `train_ply.npy`。這些索引檔由新增的 `prepare_rank_v1` 產生，**不由 `board_cache convert` 產生**。上述 `batch_size=32` 是讀取範例，不代表 V1 的正式全域 batch。正式 loader 使用 `GlobalBatchSampler`，同一 epoch 的 shuffle 排列保存在 run 目錄；D4 由 `(seed, epoch, sample_index)` 決定，支援 persistent workers 與續跑。若自行設 `num_workers=0`，同時移除 `persistent_workers=True`。

### 6. 已完成的驗證

```bash
python -m pytest tests/test_board_cache.py tests/test_sgf_planes.py -q
```

測試使用無 pass 的合成棋譜，涵蓋打包／解包、little bit order、提子、顏色、歷史不跨局、分片續跑與損壞檔案檢查。2026-10-14 清理期間已完成分組測試與官方各級前 20 盤（共 200 盤）的一般／稽核模式對照，結果見 `RESULTS.md` A-011。之後依使用者要求停止測試，最終整套測試尚未完成；目前 Python 缺少 torch，DataLoader 多程序與 DDP 尚未驗收。未執行官方全量轉檔或 Nano4 效能驗證。

## 棋力 V1：資料準備、訓練與評估

2026-10-18 更新：三個 seed（2027／2028／2029）各自切分、各訓練 1 epoch。預設設定 `rank_cnn_v1.json` 與 seed2027 相同；原 seed2026 資料與輸出不遷移、不刪除。以下是操作說明，修改程式不代表已啟動正式訓練。

啟用 `go` 環境；先完成上方共用 board cache，再在計算 allocation 依序執行：

```bash
# 每個 seed 的 split、validation 題目與逐手索引全部放在自己的 dataset_dir。
for seed in 2027 2028 2029; do
  python -m go_aicup.tasks.strength.prepare_rank_v1 --config configs/strength/rank_cnn_v1_seed${seed}.json
done

# 下例為單節點 8 卡，三個 seed 依序訓練。正式啟動前先量測耗時與顯存。
for seed in 2027 2028 2029; do
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.train_rank_v1 --config configs/strength/rank_cnn_v1_seed${seed}.json
done

# 中斷續跑需要相同 config、程式、dataset、world_size。
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.train_rank_v1 --config configs/strength/rank_cnn_v1_seed2027.json --resume outputs/rank_cnn_v1_seed2027/last.json

# 完成後 CPU 直接讀保存的 scores，不重新推論或擬合切點，不額外寫檔。
python -m go_aicup.tasks.strength.evaluate_rank_v1 --run-dir outputs/rank_cnn_v1_seed2027
```

使用同一份 Slurm 腳本分別提交三個獨立的單節點作業（每份獨占 1 節點／8 GPU；不會由腳本替你提交其他作業）：

```bash
sbatch --job-name=rank2027 --export=ALL,SEED=2027 scripts/train_rank_v1.slurm
sbatch --job-name=rank2028 --export=ALL,SEED=2028 scripts/train_rank_v1.slurm
sbatch --job-name=rank2029 --export=ALL,SEED=2029 scripts/train_rank_v1.slurm
```

腳本預設 `SEED=2027`，也可直接修改該行。每份作業先準備該 seed 資料，再啟動訓練；存在 `last.json` 時自動續跑。預設 24 小時、64 CPU、1400 GiB RAM；是否同時啟動由排程器決定。需要 account 時，在 `sbatch` 加 `-A <account>`。若改節點路徑，須同步調整腳本 `--chdir` 與 `PROJECT_DIR`；`PYTHON_BIN` 預設使用 `$HOME/.conda/envs/go/bin/python`。Slurm 日誌為 repo 下的 `slurm-<job-name>-<job-id>.out`。


每級 training／validation 各 80,000／20,000 盤，允許玩家重疊。seed 同時控制切分、模型初始化、shuffle 與 D4。預設 global batch 為 65,536，8 卡時每卡 8,192；3060 Ti 開發需另設小模型／batch 的 CPU 或 GPU 設定。

`training.evaluate_every_steps=100`：在 step 100、200……與 epoch 最後一步，凍結模型／BN、關閉 D4，完整推論 training 和 validation。loss 由整份資料各 rank 的平均 scalar score 計算，並非 batch loss 平均。中途只保留總和／計數；最後一次同時保存完整逐手 score。`losses.npy` 為 structured array，欄位為 `step:int64, train_loss:float64, valid_loss:float64`。不額外保存訓練 batch loss。

| 位置 | 內容 |
|---|---|
| `data/processed/boards/strength_v1/` | 共用 packed board cache 與 metadata |
| `data/processed/strength_v1/seed{seed}/` | `build.json`、`manifest.json`、`strength_v1_games.parquet`、`strength_v1_validation_questions.parquet`、train／valid 的 `*_game.npy` 與 `*_ply.npy` |
| `outputs/rank_cnn_v1_seed{seed}/` | 成功後只有 `model.pt`、`losses.npy`、`train_scores.npy`、`valid_scores.npy` |

不建立獨立 cv 目錄、calibration 子集或 training 校準題。分數是 `float32[N]`，`scores[i]` 精確對應 processed 中的 `game[i]`／`ply[i]`，涵蓋指定玩家的所有落子位置。processed 索引需保留，各 seed 的陣列不能直接逐列對齊。

切點使用所有 training 每盤平均 score 與 rank 標籤，以每盤等權官方 score 的有序區間 DP 擬合 9 個切點；平手 score 不拆分，允許重複切點。回溯表使用暫存 memmap，排序和向量運算仍需要額外主機 RAM 與磁碟。validation 不參與切點擬合。每盤平均指定玩家全部位置，再將題內各盤等權平均後套用同一組切點。評估報告包含 training／validation 逐手及每盤成績，validation 題目 k=5／10／15／20 與 equal-k 成績；不產生 training 題目成績。

`model.pt` 保存權重、BN buffers、seed、設定、dataset ID、索引／score 雜湊、程式／環境資訊及切點，不含 optimizer 等續跑狀態。評估會核對 processed 與 score 雜湊。搬移資料後可加 `--dataset-dir`／`--cache-dir` 指定新位置。

訓練期間只保留最新 checkpoint、排列、暫存設定與進度；SIGINT／SIGTERM、時間上限會在共同推論／訓練 batch 邊界保存，完整評估中斷會續跑重試。loss 記錄以 checkpoint 內資料為準，續跑丟棄未提交分支。成功寫出及驗證四份最終檔案後才清理暫存；共享 processed 和舊 run 保留。`--max-steps` 是總更新上限；`--skip-evaluation` 只供明確短測，保留 checkpoint，不執行最終清理。

以舊 seed2026 規模估計，每 seed 約 1,355 updates、14 次完整評估；三個 seed 共約 46.6 億位置推論，另加訓練。每 seed score 約 444 MB。正式耗時、顯存及 CUDA／SyncBN／8 卡效能尚需在計算節點量測，不在登入節點訓練。

### 棋力 V1 視覺化

```bash
python scripts/plot_rank_v1.py --run-dir outputs/rank_cnn_v1_seed2027
# 尚未完成時可只讀已完成的完整資料 loss。
python scripts/plot_rank_v1.py --run-dir outputs/rank_cnn_v1_seed2027 --loss-only
# 只畫 loss 與十級每盤平均分數分布（兩張）
python scripts/plot_rank_v1.py --run-dir outputs/rank_cnn_v1_seed2027 --skip-confusion
```

新格式直接讀取 `.npy` 與保存的切點，預設在 `<run-dir>/plots/` 產生 `loss_vs_step.png`、`confusion_matrix.png`、`score_distributions_cutpoints.png`；圖表只在執行此指令時產生。目前 losses.npy 的 train_loss 是當步訓練 batch loss（train mode），valid_loss 是固定抽樣 100,000 手（每級 10,000 手、不放回、由 seed 決定）的 rank-mean loss（eval mode）；舊版 legacy_full_losses.json 定義不同，不混接。混淆矩陣分 k 桶，每盤分布由指定玩家的逐手 score 先取盤內平均，每盤等權。切點由 training 每盤平均分數擬合。可用 `--output-dir`、`--bins`，或搬移資料時用 `--dataset-dir`／`--cache-dir`。新格式需 CPU PyTorch 讀取 `model.pt`，不使用 GPU。

舊 seed2026 run 的 JSONL／evaluation 格式仍可繪圖：保留 `--evaluation-dir`、`--smooth-window` 與原有 hash 檢查；validation batch loss 缺少或為 null 時警告並略過，仍畫其他資料。舊格式不依賴 PyTorch。

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

既有 run 改以 training 每盤平均分數重新擬合切點（只更新 model.pt 的切點，保留模型權重與保存的 score）：

```bash
python -m go_aicup.tasks.strength.evaluate_rank_v1 --run-dir outputs/rank_cnn_v1_seed2027 --refit-cutpoints
```
