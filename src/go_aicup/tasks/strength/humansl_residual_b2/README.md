# HumanSL residual B2

依 [PLAN.md](PLAN.md) 實作三階段：固定 HumanSL 的雙分支分類、凍結第一階段的十層殘差修正，以及四個 k 各自的線性 B2。這裡的命令是操作說明；本次沒有提交 Slurm 或產生正式權重。

## 設定與前置資料

- `configs/strength/humansl_residual_smoke100.json`：兩階段各 100 optimizer updates，探索用 AdamW LR=0.001；C 對照 100 updates、LR=0.01。這些數字只供工程短測，不是已選定的正式超參數。
- `configs/strength/humansl_residual_train.json`：正式設定（2026-11-10 修訂）。第一階段 3,000 updates、LR 0.001；第二階段 6,000 updates、LR 0.0003；兩者都是 cosine 衰減到 5%（`lr_decay`、`min_lr_ratio`），各自每 300／600 步監控（`monitor_every`）、patience 3。B2 預設 `aggregation=logp`、`fit_pool=fit_all`（全部 D 玩家、每人每色最多 20 盤）。C 對照 20,000 updates、LR=0.01。全部是未經正式訓練驗證的起點，長作業前仍須依短測估算成本。`lr_schedule=[[完成步數,新LR],...]` 可與 cosine 疊加。
- 兩者預設 8 GPU、global 128 組、每卡 microbatch 一組／300 位置、累積 16 次。每方 15 手、seed=2026、玩家／題目 seed=2027；實際 world size、microbatch、precision 都綁定續跑身分。3060 Ti 使用獨立縮小 config／輸出目錄，不能混用八卡 checkpoint。
- `data.folds_path` 必須指向**原完整資料 seed=2027** 的玩家表，至少有 `player_id,fold`，fold 為 −1、0、1、2、3、4。本次唯讀查核時 `data/processed/cv/` 為空，預設的 `strength_folds_seed2027.parquet` 尚不存在；須從原切分保存處取回並設定路徑。程式不自動重抽原 fold。
- `data.legacy_split_paths` 重用既有 `humansl_q1000`／`humansl_qsel`／`humansl_vs_rest` 的棋局表，合併 S-fit／S-select／S-extra 為 Valid-S、U 為 Valid-U，其餘棋局僅作上游來源。也可用 `legacy_split_path` 指向完整、明列 `split` 的原棋局表；兩種方式擇一。
- 既有 board-cache 的 moves／colors 唯讀使用，以 `(rank,game_id)` 對齊 store；HD 從官方 handicap.csv 合併，HD≥2 整盤排除。監控／U 每 k/rank 固定 125 題；B2 每 k 固定 20,000 題。資格不足、缺 HD、任一方不足 15 手都停止。
- 需要已申報的 HumanSL ordinary checkpoint、KataGo 原始碼、強 teacher 網路及執行檔；不安裝新依賴。

所有命令從 repo 根目錄執行：

```bash
conda activate go
export PYTHONPATH="$PWD/src"
export CONFIG=configs/strength/humansl_residual_smoke100.json
export JOBS=src/go_aicup/tasks/strength/humansl_residual_b2/slurm
```

先準備獨立索引及題目，再產生必要 teacher。teacher 的 `--limit-games` 可用於量測；限量標籤不保證足供訓練，缺 target 會報錯。

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.prepare --config "$CONFIG"
# 或 sbatch "$JOBS/prepare.slurm"
python -m go_aicup.tasks.strength.humansl_residual_b2.teacher --config "$CONFIG" --shard 0 --limit-games 10
# 全部八個 teacher 分片：sbatch --array=0-7 "$JOBS/teacher.slurm"
```

teacher 只分析 D（release 為 P）所需的 ply≥30 位置，雙方各自取落子前 target；HD0 貼 6.5、HD1 貼 0.5、1 visit、黑方勝率轉當手視角。每盤原子完成狀態、來源、索引及數值 checksum 支援重試；舊 SE 的裸 `value.npy` 不直接重用。teacher 索引預留 HD0/1 棋局的緊湊數值空間，未要求的玩家維持未完成，不會啟動其分析。dev/release 可延用完全相同身分的既有 target。

**寫入方式（2026-11-10 起）：** 不再用 `np.memmap` 原地寫 `values/done/checksums.npy`（作業 742066 在 Nano4 WekaFS 上 `flush()` 回報成功卻幾乎全未落地，見 RESULTS 同日紀錄）。每 `teacher.chunk_games`（預設 500）盤寫一個 `chunks/chunk_*.npz`（一般寫入＋`fsync`＋原子 rename＋目錄 fsync）；結束（成功、停止或失敗）時以同法原子發布三個 `.npy`，丟棄本節點快取後重讀驗證，與已提交數一致才刪 chunk。續跑先逐盤重驗既有 `.npy`，checksum 不符或非有限值的列丟棄重算（`discarded_games`），再重放 chunk。只有 D（release 為 P）全部經磁碟重讀驗證才標 `complete`；否則 `incomplete`／`stopped`／`failed` 且程序非 0 結束。Slurm 另起新程序執行 `--verify`，並把 USR1 轉送給 label 程序。`scripts/strength/hrb2_teacher_progress.py` 從登入節點計入已提交 chunk，作為外部進度檢查。

## 三個訓練入口與 Slurm

| 階段 | Python 入口 | Slurm |
|---|---|---|
| 第一階段 | `python -m go_aicup.tasks.strength.humansl_residual_b2.train --config "$CONFIG" --stage 1 --phase dev --fold 1` | `stage1.slurm` |
| 第二階段 | 同入口 `--stage 2 --source <同折第一階段候選.pt>` | `stage2.slurm` |
| 第三階段 | `python -m go_aicup.tasks.strength.humansl_residual_b2.b2 --config "$CONFIG" --stage 1 --candidate best --arm new --aggregation logp` | `stage3.slurm` |

`smoke100.slurm` 依序量測兩階段；第二階段只在第一階段完成後載入它的 `final.pt`。短測也在第 100 步做固定 M 監控。`summary.json` 保存時間、steps/s、位置/s、資料等待／計算／監控／存檔時間、顯存、RAM、I/O、重播手數及 GPU-hours。前五個 updates 只從純訓練吞吐量統計剔除，仍正常更新；不改 LR。計算與資料等待是同步牆鐘量測，CPU 重播手數是所選最大 ply 導出的重播工作量，不是 CPU 指令的獨立 profiler。

```bash
sbatch "$JOBS/smoke100.slurm"
```

Slurm 腳本均放在本套件下，會被 Git 追蹤。訓練／特徵預設八卡；teacher／C 對照單卡。prepare／B2 使用 dev 的最少一張 GPU 配額執行 CPU 工作，仍須計入配置 GPU-hours。預設四小時（teacher 為六小時）是排程上限，沒有實測完成時間含義；長作業須先回報成本，依實測調整 `sbatch --time=...`，dev 最多四小時、8gpus 最多 48 小時。不會因完成程式就自動提交。

正式開發每個階段只跑四個 fold（2026-11-10 起不再訓練全 D 模型），命令見下方「2026-11-10 修訂後的開發流程」第 2 步。

每個 run 的目錄是 `<run_dir>/dev/fold_<fold>/stage1/` 或 `stage2/`，保存 `best.pt`、`final.pt`、`latest.pt` 及 complete markers。`best` 僅在 M 融合 Equal-k 嚴格增加時更新；相等保留較早者，連續三次未增加早停。每 `monitor_every`（正式設定 300／600）updates 及預算最後一步監控，各分支另報分 k 分數／CE；另報 `fused_logp_t1`（log 機率平均、T=1）僅供診斷，早停仍看機率平均融合。訓練 loss 不選 best。

重跑相同命令自動從 `latest.pt` 續跑；更換 config、資料、程式、world size 或階段來源會拒絕。抽樣 durable state 只按已消費完整更新保存，與 worker 預取隔離；不同級別獨立洗牌，跨輪衝突延後，不刪盤、不截短。停止檔為該 stage 目錄的 `.stop-requested`，Slurm 預告／TERM／INT 會在完整 update 後保存。監控或最終發布失敗，可重跑補做，不增加已完成的 updates。續跑 log 會裁掉超出 durable step 的紀錄。

## 2026-11-10 修訂後的開發流程

修訂內容與理由見 [PLAN.md](PLAN.md) 第 14 節。重點：不再訓練全 D 模型，也不做 release 重訓；M、U、測試一律用四個 fold 模型的每盤特徵平均；B2 主臂改用 log 機率平均，並擴大到全部 D 玩家；先跑低成本探針再決定是否投入第一階段。

### 0. 擴充題庫（CPU，一次）

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.prepare --config "$CONFIG" --extension
# 或 EXTENSION=1 sbatch "$JOBS/prepare.slurm"
```

在 `prepared_dir/ext_v1/` 產生 `pool_fit_all`（全部 D 玩家、每人每色最多 20 盤，seed 2027）與 `questions_fit_all`（每 k 20,000 題）。既有 base manifest、dataset_id 與 teacher 身分都不動。

### 1. 探針（go/no-go，先做）

```bash
sbatch "$JOBS/probe_cache.slurm"   # 8 卡：fit_all、fit 與 M 每盤固定 30 手，一次 HumanSL 前向，存 768 維 fp16 讀出
sbatch "$JOBS/probe_run.slurm"     # 1 卡：四折 OOF 線性／MLP 頭 → 每盤 prob／logp 平均 → B2 → M 分 k 報告
```

結果在 `<run_dir>/probe/result.json`，每個臂（linear／mlp × prob／logp × fit／fit_all）包含 k=5/10/15/20、Equal-k；若對照 H＋C 的 M 預測已存在，另附配對 gate 與玩家 bootstrap。架構比較看 `*_fit` 臂（與 H＋C 同擬合池）。探針只用每盤 30 手，分數是同一讀出的保守下界。快取約 600 萬位置、約 9 GB。

### 2. 第一／二階段只訓練四折

```bash
export CONFIG=configs/strength/humansl_residual_train.json
for FOLD in 1 2 3 4; do export FOLD; sbatch "$JOBS/stage1.slurm"; done
```

`--fold all` 仍可用（smoke100 用它），但正式流程不需要。第二階段同樣只跑 fold 1..4，來源固定為同 fold 第一階段的 M 選定候選。

### 3. 全位置特徵

每個候選（best、final）各抽一次：fit 池每折只抽自己的 OOF 玩家；M 由四個 fold 模型各抽一次，讀取時平均。

```bash
export CANDIDATE=best STAGE_DIR=stage1
for FOLD in 1 2 3 4; do
    for POOL in fit fit_all monitor; do
        CHECKPOINT=<run_dir>/dev/fold_$FOLD/$STAGE_DIR/$CANDIDATE.pt FOLD=$FOLD POOL=$POOL sbatch "$JOBS/features.slurm"
    done
done
export STAGE=1
sbatch "$JOBS/stage3.slurm"   # best/final × {fit, fit_all} × {prob, logp} 的 new 臂 B2，再以 fit_all＋logp 選 best/final
```

每盤特徵同時保存 prob 平均（`own_r`）和 log 機率平均（`own_lp_r`），格式為 `humansl_residual_features_v2`；舊 v1 特徵不能用於 logp 臂。

**本實驗只比較架構：新模型 B2 只用新 20 維＋P2（22 維），不做 72 維 stack。** config 的 `b2.arms=["new"]`，要求 stack 臂時程式直接拒絕；`probe.py` 也只在 `arms` 列出 stack 時才跑。ResNet 對 H＋C 的配對比較用 `--fit-pool fit`：兩者在同一個擬合池（Valid-S∩D 的固定題）、同一份 M 題目上比較，差異只來自上游特徵。`fit_all` 是另一個「多玩家擬合 B2」的變因，分開報告：

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" --stage 1 --candidate <selected> --arm new --aggregation logp --fit-pool fit
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" \
    --reference <run_dir>/baseline/dev/evaluation/monitor.parquet \
    --candidate-predictions <run_dir>/evaluation/stage1/<selected>/new_logp_fit/monitor.parquet \
    --output <run_dir>/gate_stage1_vs_hc.json
```

`evaluate --diagnostics` 比較各單折與四折平均在同一 M 上的熵、NLL 與 B2 分數。

### 4. 對照、gate 與 U 鎖定

H＋C 對照沿用 `baseline.py`（四折 C 加全 D C，命令同下節舊說明；`fit` 使用 `stack_fit_pool`）。gate 仍為 Equal-k ≥ +0.005、各 k ≥ −0.005，另報玩家 bootstrap。鎖定時每個臂都要寫明 stage、candidate、arm、aggregation；鎖定紀錄同時綁定四個 fold 的 checkpoint：

```json
{
  "s1_new_logp_fit": {"stage": 1, "candidate": "best", "arm": "new", "aggregation": "logp", "fit_pool": "fit"},
  "s1_new_logp_fit_all": {"stage": 1, "candidate": "best", "arm": "new", "aggregation": "logp", "fit_pool": "fit_all"},
  "baseline": {"model": "<run_dir>/baseline/dev/model.pkl"}
}
```

鎖定後才可抽 `POOL=U`（四折各一次），再以 `evaluate --pool U --arm ... --aggregation ...` 各評一次。

### 5. Release（不重訓）

```bash
PHASE=release sbatch "$JOBS/prepare.slurm"   # 鎖定後：mu_all（M＋U 每人每色最多 20 盤）與 release 題庫
for FOLD in 1 2 3 4; do CHECKPOINT=... FOLD=$FOLD POOL=mu_all sbatch "$JOBS/features.slurm"; done
PHASE=release STAGE=<s> ARM=new AGGREGATION=logp CANDIDATE=<selected> sbatch "$JOBS/stage3.slurm"
```

release B2 以開發 OOF 特徵（fit 池）加上四折平均的 M＋U 特徵擬合，等於把 allp 的做法帶進本實驗；上游模型就是鎖定的四個 fold 模型，沒有獨立 U 分數。測試推論：

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.predict --prepare --store-dir <test-store> --name rank_public --work <release-root>/rank_public
python -m go_aicup.tasks.strength.humansl_residual_b2.predict --config "$CONFIG" --work <release-root>/rank_public --arm-name s1_new_logp
python -m go_aicup.tasks.strength.humansl_residual_b2.predict --work-root <release-root> --sample data/raw/tests/sample_submission_rank.csv --output submissions/rank/<job>-hrb2.csv
```

每個 fold 的測試特徵分別快取，讀取時平均。release 時 `FIT_POOL` 依鎖定的臂而定（`stage3.slurm` 的 release 分支用 config 預設，必要時直接呼叫 `b2 --phase release --fit-pool ...`）。

## 舊流程說明（2026-11-09，已被上一節取代的部分僅供追溯）

下列 H＋C 對照命令仍有效；其中「全 D 模型抽 M」「release 四折重訓」「固定步數 s1/s2」的描述已由上一節取代。

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command train-c --fold 1
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command extract --pool fit --fold 1
# 其餘三折相同；全 D 的 C 使用 --fold all，之後抽 --pool monitor（以及鎖定後的 U、mu_all）。
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command fit
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command evaluate --pool monitor
```

## 工程驗證與限制

新增測試位於 `tests/test_humansl_residual_{model,data,train,ddp,b2,baseline,improvements}.py`，涵蓋模型凍結、梯度、雙方抽樣、預取／消費者續跑、teacher 來源／視角、OOF 洩漏拒絕、B2/scaler、分 k gate、發布重試及 CSV 接點。執行方式：

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/test_humansl_residual_*.py
```

尚未執行真實 GPU／SyncBN、100 步資源短測、官方 teacher、完整訓練／推論或上傳；真實分 k 成績、吞吐、顯存與 GPU-hours 未知。`summary` 的磁碟空間取檔案系統數值，不取代 Nano4 每目錄配額；正式作業前須另查 100GB 配額剩餘量。成本需包含 dev/release 各五套兩階段模型、五份 C 對照、teacher、兩組候選全位置特徵、M 監控及 CPU B2。


## SE3 匯入與八分片續跑（2026-11-10）

兩份 config 的 `teacher.se3_reuse` 已指向既有 SE3 value、prepared 索引、訓練身分紀錄及 `outputs/value_resume_672603_recovery.json`。啟動先驗證 value／索引 checksum、原八分片 teacher 身分、KataGo 引擎／網路／設定、規則、貼目、visits 與完整歷史條件；缺證據或不一致時停止，不默默混用。HD0 且在舊 training 的棋局，逐盤核對 rank、game_id、player_color、length，逐位置驗證原 ply 為指定玩家落子前位置。HD1、舊 training 未涵蓋棋局和對手位置仍由 KataGo 補算。

直接續跑八個分片：

```bash
sbatch --array=0-7 src/go_aicup/tasks/strength/humansl_residual_b2/slurm/teacher.slurm
```

已完成棋局驗證 checksum 後略過；未完成棋局重新匯入相容的 SE3 位置並補算缺值，整盤成功才提交 checksum／done。每片 `se3_import.json` 保存匯入來源 checksum，`metrics.json` 分開記錄 `imported_positions` 與新分析的 `new_positions`。匯入發生在續跑時，並非一次性將全部舊值預先寫入。

新 `.teacher.lock` 使用作業系統檔案鎖：共用身分初始化等待鎖；同分片重複 writer 立即拒絕。程序結束會自動釋放鎖，檔案保留不代表仍被鎖定。舊 `.writer.lock` 仍會明確拒絕，必須先確認舊程序停止才清理。2026-11-10 已確認原八個作業取消、extern 完成且沒有 teacher 排程後，清除八片殘留的舊鎖。

原 labeler 的特定 SHA-256 被列為相容版本，保留既有 identity 及全部分片 manifest，不重標或覆寫已完成棋局；其他資料／模型／設定／encoder 身分差異仍拒絕。SIGUSR1／TERM／INT 會請求在當前棋局完成後停止，狀態記為 `stopped`；Slurm 腳本以 exec 傳遞訊號。若取消作業同時終止 KataGo 子程序，或發生硬終止，當前未完成棋局重做。四小時作業上限維持原值，續跑後的節省時間仍待 GPU 實測。
