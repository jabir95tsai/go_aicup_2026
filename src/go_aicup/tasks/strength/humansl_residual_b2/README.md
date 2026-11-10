# HumanSL residual B2

依 [PLAN.md](PLAN.md) 實作三階段：固定 HumanSL 的雙分支分類、凍結第一階段的十層殘差修正，以及四個 k 各自的線性 B2。這裡的命令是操作說明；本次沒有提交 Slurm 或產生正式權重。

## 設定與前置資料

- `configs/strength/humansl_residual_smoke100.json`：兩階段各 100 optimizer updates，探索用 AdamW LR=0.001；C 對照 100 updates、LR=0.01。這些數字只供工程短測，不是已選定的正式超參數。
- `configs/strength/humansl_residual_train.json`：正式設定，兩階段各 20,000 updates，AdamW LR 分別為 0.001／0.0003；C 對照 20,000 updates、LR=0.01。這些是未經正式訓練驗證的起點，長作業前須依短測估算成本。可用 `lr_schedule=[[完成步數,新LR],...]` 設定各階段的學習率變化。
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

## 三個訓練入口與 Slurm

| 階段 | Python 入口 | Slurm |
|---|---|---|
| 第一階段 | `python -m go_aicup.tasks.strength.humansl_residual_b2.train --config "$CONFIG" --stage 1 --phase dev --fold all` | `stage1.slurm` |
| 第二階段 | 同入口 `--stage 2 --source <同折第一階段候選.pt>` | `stage2.slurm` |
| 第三階段 | `python -m go_aicup.tasks.strength.humansl_residual_b2.b2 --config "$CONFIG" --stage 1 --candidate best` | `stage3.slurm` |

`smoke100.slurm` 依序量測兩階段；第二階段只在第一階段完成後載入它的 `final.pt`。短測也在第 100 步做固定 M 監控。`summary.json` 保存時間、steps/s、位置/s、資料等待／計算／監控／存檔時間、顯存、RAM、I/O、重播手數及 GPU-hours。前五個 updates 只從純訓練吞吐量統計剔除，仍正常更新；不改 LR。計算與資料等待是同步牆鐘量測，CPU 重播手數是所選最大 ply 導出的重播工作量，不是 CPU 指令的獨立 profiler。

```bash
sbatch "$JOBS/smoke100.slurm"
```

Slurm 腳本均放在本套件下，會被 Git 追蹤。訓練／特徵預設八卡；teacher／C 對照單卡。prepare／B2 使用 dev 的最少一張 GPU 配額執行 CPU 工作，仍須計入配置 GPU-hours。預設四小時是排程上限，沒有實測完成時間含義；長作業須先回報成本，依實測調整 `sbatch --time=...`，dev 最多四小時、8gpus 最多 48 小時。不會因完成程式就自動提交。

正式開發依序執行五個第一階段 runs，`FOLD=1/2/3/4/all`；第二階段同樣五個，不能用全 D 來源替代 fold 來源：

```bash
export CONFIG=configs/strength/humansl_residual_train.json
for FOLD in 1 2 3 4 all; do
    export FOLD
    sbatch "$JOBS/stage1.slurm"
done
```

每個 run 的目錄是 `<run_dir>/dev/fold_<fold>/stage1/` 或 `stage2/`，保存 `best.pt`、`final.pt`、`latest.pt` 及 complete markers。`best` 僅在 M 融合 Equal-k 嚴格增加時更新；相等保留較早者，連續三次未增加早停。每 5,000 updates 及預算最後一步監控，各分支另報分 k 分數／CE。訓練 loss 不選 best。

重跑相同命令自動從 `latest.pt` 續跑；更換 config、資料、程式、world size 或階段來源會拒絕。抽樣 durable state 只按已消費完整更新保存，與 worker 預取隔離；不同級別獨立洗牌，跨輪衝突延後，不刪盤、不截短。停止檔為該 stage 目錄的 `.stop-requested`，Slurm 預告／TERM／INT 會在完整 update 後保存。監控或最終發布失敗，可重跑補做，不增加已完成的 updates。續跑 log 會裁掉超出 durable step 的紀錄。

## 全位置特徵、B2 與開發選擇

對 `best`／`final` 各做一條完整管線。各折模型在 `POOL=fit` 只抽自己未見過的 B_dev 玩家；全 D 模型在 `POOL=monitor` 抽 M。模型／階段／fold／候選／HD／貼目／資料身分一併保存。特徵逐盤、有界批次抽取，只有每盤二十維機率、計數及 marker 落盤，不存全量輸入或浮點主幹。

```bash
export FOLD=1 CANDIDATE=best POOL=fit
export CHECKPOINT=<run_dir>/dev/fold_1/stage1/best.pt
sbatch "$JOBS/features.slurm"
# 同樣完成其餘三折 fit；fold_all 對 POOL=monitor 做 M 特徵。
# 再對 final 候選重做以上步驟。
export STAGE=1
sbatch "$JOBS/stage3.slurm"
```

`stage3.slurm` 不重做特徵：它在全部特徵完成後擬合 best／final 各自四個 B2，再以 M 的完整管線 Equal-k 選擇，同分取 best，保存 `selection/stage1.json`。B2 每盤／每題算術等權、二十維各方機率和為 1，加 P2 兩個旗標後為 22 維；固定題目直接交既有 StandardScaler／GameLR，C=1、max_iter=5000，不呼叫舊的另抽題 B2 fit。

正式第二階段的 `stage2.slurm` 自動讀取這個 selection，以同 fold 第一階段候選為來源。安排第二階段前先檢視第一階段對照 gate 與短測成本；程式不替使用者決定是否花費後續預算。第二階段的特徵／B2／selection 流程相同，設定 `STAGE=2`、checkpoint 改為 `stage2/`。

為量測 OOF／全 D 信心尺度差異，另外對每個 fold 的模型抽 `POOL=monitor`，再執行：

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" --stage 1 --candidate best --diagnostics
```

輸出各 fold 與全 D 在同一 M 的本人／對手熵、NLL，以及完整 B2 分 k 成績。

特徵與 B2 首次成功的成本另存 `.metrics.json`；重用快取寫 `.reuse.metrics.json`，不覆蓋原量測，也不把重用盤數計為新增吞吐。每個特徵 rank 各記配置 GPU-hours；CPU B2 亦計入 Slurm 持有的 GPU 配額。teacher 的 `metrics.json` 另列本次新分析位置，失敗重試不把舊完成盤灌入吞吐。

## H＋C 對照、gate 與 U 鎖定

`baseline.py` 建立 D 內四折及全 D 共五份 C 向量；H/C 使用原 policy likelihood 數學、全盤位置及正確貼目，沒有 A-024。C 使用有界輸入的 Adam／cosine、全盤位置均勻抽樣；不是複製會建立大輸入 packs 的舊訓練器。實際更新預算仍須以獨立短測決定。

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command train-c --fold 1
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command extract --pool fit --fold 1
# 其餘三折相同；全 D 的 C 使用 --fold all，之後抽 --pool monitor。
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command fit
python -m go_aicup.tasks.strength.humansl_residual_b2.baseline --config "$CONFIG" --command evaluate --pool monitor
```

也可設定 `COMMAND=train-c|extract|fit|evaluate`、`FOLD`、`POOL` 使用 `baseline.slurm`。對照輸出在 `<run_dir>/baseline/dev/`；不可載入舊全玩家 C 快取充當本版 OOF 對照。

配對比較用相同 M 題目，gate 為 Equal-k 至少 +0.005、每 k 不低於 −0.005，另報玩家層級 bootstrap：

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" \
    --reference <run_dir>/baseline/dev/evaluation/monitor.parquet \
    --candidate-predictions <run_dir>/evaluation/stage1/best/monitor.parquet \
    --output <run_dir>/gate_stage1.json
```

範例的 `best` 必須換成 `selection/stage1.json` 實際選中的候選；第二階段增量與相對 H＋C 的差值分開比較。

全部選擇完成後，建立 arms JSON，將 candidate 換成實際 M 所選者；baseline model 路徑見其 fit 輸出：

```json
{
  "stage1": {"stage": 1, "candidate": "best"},
  "stage2": {"stage": 2, "candidate": "best"},
  "baseline": {"model": "<run_dir>/baseline/dev/model.pkl"}
}
```

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" --lock-arms arms.json
# 鎖定後才允許抽 POOL=U；對每個已登記 arm 一次完整評估。
python -m go_aicup.tasks.strength.humansl_residual_b2.evaluate --config "$CONFIG" --stage 1 --candidate best --pool U
```

`selection_lock.json` 綁定選擇、B2／checkpoint checksum、資料及全 D 選定步數 s1/s2。U 結果可重讀，身分改變時拒絕覆寫；不能把選完 M 的數字稱為獨立泛化。對照 U 亦須先登記 baseline model，再抽 baseline U 特徵、執行其 evaluate。

## Release 重訓與提交檔

鎖定後使用同一 config：`prepare --release` 建立 P 的新四折玩家映射，排除 R；teacher `--phase release` 只補缺少的 P 標籤。設定 `PHASE=release`，依序使用 stage1／stage2 Slurm 對 1..4/all 重訓；步數由 lock 的 s1/s2 讀取，不監控、不早停。release stage2 固定載入同 fold 的 stage1/final.pt。smoke config 禁止 release。

release 特徵使用 `PHASE=release,POOL=fit,CANDIDATE=final,FOLD=1..4`；B2 使用 `PHASE=release,STAGE=2` 的 `stage3.slurm`。只用 S＋U 棋局的 OOF 特徵擬合，測試使用全 P 單模型，不平均四折。release 沒有獨立 U 分數。

測試資料先沿用既有 preprocessing 建立有明確 HD 的 GameStore，接著使用本版 prepare／推論／CSV 匯出：

```bash
python -m go_aicup.tasks.strength.humansl_residual_b2.predict --prepare \
    --store-dir <test-store> --name rank_public --work <release-root>/rank_public
python -m go_aicup.tasks.strength.humansl_residual_b2.predict --config "$CONFIG" \
    --work <release-root>/rank_public --checkpoint <run_dir>/release/fold_all/stage2/final.pt
# private 同樣處理；兩者完成後合併並驗證 800 列與官方 sample 順序。
python -m go_aicup.tasks.strength.humansl_residual_b2.predict \
    --work-root <release-root> --sample data/raw/tests/sample_submission_rank.csv \
    --output submissions/rank/humansl_residual_b2_rank.csv
```

完整推論覆蓋 ply≥30 的雙方全部位置，缺分支／不支援 HD／缺棋局直接報錯。舊無 HD 的測試 store 須從原官方輸入重建 metadata；不默認 HD0。CSV 沿用既有 export 的格式、機率、rank、順序及 checksum 檢查。輸出檔登記 RESULTS 後由使用者手動上傳。

## 工程驗證與限制

新增測試位於 `tests/test_humansl_residual_{model,data,train,ddp,b2,baseline}.py`，涵蓋模型凍結、梯度、雙方抽樣、預取／消費者續跑、teacher 來源／視角、OOF 洩漏拒絕、B2/scaler、分 k gate、發布重試及 CSV 接點。執行方式：

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
