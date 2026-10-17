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

**2026-10-12 狀態：以下入口已撰寫，未執行、未驗證。所有命令是後續操作說明，本次沒有啟動它們。** 先完成上述 board cache 轉檔，再依序產生索引與訓練。設定檔為 [configs/strength/rank_cnn_v1.json](configs/strength/rank_cnn_v1.json)；相對資料路徑以 repo 根目錄解析，Nano4 可改為絕對路徑。

```bash
# CPU 資料準備；在計算 allocation 執行，不用 torchrun。
python -m go_aicup.tasks.strength.prepare_rank_v1 --config configs/strength/rank_cnn_v1.json

# Nano4 單節點 8 卡；目前設定檔預設 10 epochs、global batch 65536，每卡 8192。
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.train_rank_v1 --config configs/strength/rank_cnn_v1.json

# 明確短跑；max-steps 是總 optimizer updates 上限，不是本次新增 steps。
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.train_rank_v1 --config configs/strength/rank_cnn_v1.json --max-steps 100 --skip-evaluation

# 相同 config／資料／程式／world_size，從最後完成的 batch cursor 續跑。
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.train_rank_v1 --config configs/strength/rank_cnn_v1.json --resume outputs/rank_cnn_v1_seed2026/last.json

# 單獨評估已有完整 checkpoint；output-dir 必須是新的或空的目錄。
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 -m go_aicup.tasks.strength.evaluate_rank_v1 --config configs/strength/rank_cnn_v1.json --checkpoint outputs/rank_cnn_v1_seed2026/last.json --output-dir outputs/rank_cnn_v1_seed2026/evaluation/manual_001
```

新 run 更換 config 的 `paths.output_dir`；不得覆蓋既有非空 run。目前設定檔預設訓練 **10 epochs**，training 與 evaluation 的 **global batch 均為 65,536**，單節點 8 卡時每卡 8,192；更改 epochs 會改變 scheduler，應作為新 run，不能冒充相同設定的精確續跑。`--skip-evaluation` 僅供明確短跑；完整 run 預設在每 epoch 後評估。`training.max_runtime_seconds` 可設定這次作業允許的執行秒數，程式預留 `shutdown_margin_seconds=180` 並在共同 batch 邊界保存；SIGINT／SIGTERM 同樣請求保存。精確續跑要求相同 world_size、設定、程式指紋、索引及排列。 每次新 checkpoint 與 `last.json` 完整發布後，只保留最新一份 `.pt` 與完成標記，刪除舊 checkpoint；不另存最佳權重。`best_score.json` 只記錄歷史最佳分數與評估路徑，歷次評估結果仍保留。BF16 不使用 GradScaler；FP16 模式保存 scaler。

完整 256-channel／20-block 網路的預設 batch 面向 Nano4 H200。若要在本機撰寫小型開發設定，可明確修改 channels、blocks、batch、precision、workers，並使用不同 run／設定檔；這不代表正式 V1 的 H200 batch 已能在 8GB 顯存執行。正式顯存、步長秒數與完整 epoch 時間均未知，先依計畫短測再啟動長作業。

資料與輸出預設位置如下，**本次只有新增／修改程式與文件，以下衍生檔尚未產生**：

| 位置 | 執行後的內容 |
|---|---|
| `data/processed/boards/strength_v1/` | 沿用 `board_cache` 的 packed shards、colors、moves、games metadata；不複製兩份快取 |
| `data/processed/cv/strength_v1_seed2026/` | `strength_v1_games.parquet`、固定 training calibration／validation 題組、eligibility |
| `data/processed/strength_v1/seed2026/` | train／valid／calibration 的 `*_game.npy`、`*_ply.npy`；build／manifest、hash、盤面數與玩家重疊率 |
| `outputs/rank_cnn_v1_seed2026/` | resolved config、環境／程式指紋、資料 manifest、`train.jsonl`、session 時間與顯存紀錄、`last.json`／`best_score.json`／`finished.json` |
| `outputs/rank_cnn_v1_seed2026/permutations/` | 每 epoch 的 uint32 排列與 hash；續跑不使用 loader 預取游標 |
| `outputs/rank_cnn_v1_seed2026/checkpoints/` | 原子發布的 `.pt` 和完成標記；模型、optimizer、scheduler、AMP、各卡 RNG／buffers、epoch／cursor／資料指紋 |
| `outputs/rank_cnn_v1_seed2026/evaluation/epoch_0000/` | 手分數 npy（可設不保留）、盤／題預測、training-only `cutpoints.json`、`metrics.json`、完成標記 |

本機輸出根目錄的完整路徑為 **`D:\college\go_aicup_2026\outputs\rank_cnn_v1_seed2026\`**。

`cv.py` 的 V1 API 是**按棋局**切分，保留既有 player-disjoint API。樣本索引依保存的 colors 選取指定玩家的 ply；整個歷史仍保留雙方。每 epoch 全域打亂所有 training 盤面索引，各卡讀取同一 batch 的互斥部分。尾批缺級會明確記錄並跳過更新，optimizer／scheduler／BN 都不前進；補齊項不進入 SyncBN。

訓練 loss 只約束十級的全域平均分數。評估固定模型與 BN、不使用 D4；指定玩家各手平均成盤，再由同玩家同色的 k 盤等權平均成題。校準只推論固定 training 題組所需的唯一棋局，不把這個子集的分布宣稱為全量 training 分布；validation 推論整份指定玩家盤面。九個切點以四個 k 桶等權官方 score 的有序區間動態規劃擬合，平手分數不能拆分，切點綁定 checkpoint SHA-256。`metrics.json` 分別保存 batch loss、整體 rank mean loss、分布、k=5／10／15／20 的分數／accuracy／MAE／混淆矩陣和四桶等權總分。

本次沒有單／多卡 loss 與梯度、完整 SyncBN 尾批、checkpoint 續跑、cutpoint DP 或小樣本學習的測試結果；正式資料數、玩家重疊率、四桶分數與 Nano4 效能均須後續執行才知道。


### 棋力 V1 視覺化

完整評估後，以 CPU 讀取既有日誌、validation 題目預測與每盤分數，不重新推論：

```bash
python scripts/plot_rank_v1.py --run-dir outputs/rank_cnn_v1_seed2026
```

預設選擇 run 內最新、含 `complete.json` 的 `evaluation/epoch_XXXX/`，在該目錄的 `plots/` 產出三張 PNG：

- `loss_vs_step.png`：全域 optimizer step 對 training ranking loss，附最近 50 筆已記錄更新的移動平均；加上所有已完成評估的 validation batch loss（按 batch 樣本數加權平均），依 checkpoint 檔名的精確 optimizer step 對齊。這不是整份驗證集十級均值計算的 loss，且未使用 D4；續跑重播的 training step 以新分支取代。
- `confusion_matrix.png`：k=5／10／15／20 四個 10×10 子圖；橫軸預測、縱軸真實級別，每格顯示列百分比與題數。
- `score_distributions_cutpoints.png`：單一座標圖呈現十個真實級別的每盤 scalar 分數密度直方圖。每盤只計一筆，使用完整 validation 推論後指定玩家所有手的平均值（`valid_game_scores.parquet`）；不按 k 分組、不重複計入同盤。保留 training-only 題目校準的 9 個切點作參考：切點適用題目平均分數，不能視為已針對單盤校準。重合切點以 t1–t9 分別標示，不重新擬合切點。

可用 `--evaluation-dir <path>` 指定已完成評估、`--output-dir <path>` 指定圖檔目錄、`--smooth-window 1` 關閉平滑、`--bins 50` 設定分布圖區間數。腳本核對評估完成標記、metrics hash、checkpoint hash 與切點解碼結果；不需 torch 或 GPU。圖表反映該評估的 split，現行 V1 的 game-disjoint 分數不能直接解讀為未見玩家表現。

只有短跑日誌時可先產出一張 loss 圖（若已有完成評估則也會加入 validation loss），預設存至 `<run-dir>/plots/`：

```bash
python scripts/plot_rank_v1.py --run-dir outputs/rank_v1_short_100_20261017_135242/run_socket --loss-only
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
