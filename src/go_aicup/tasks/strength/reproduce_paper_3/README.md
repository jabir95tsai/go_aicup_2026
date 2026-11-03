# SE3：十級 CE 分類實驗

依 [plan.md](plan.md) 實作，與 SE／SE2 使用獨立模型、設定、checkpoint 與輸出。先完成下列五個小任務，再進行 GPU 驗收。正式四桶分數與增益仍未知。

| 小任務 | 實作 |
|---|---|
| 1. 設定／模型／loss | `config.py`、`model.py`、`losses.py`；18-block、21,754,485 parameters、361→256→10，保留 policy／value；三項 FP32 loss 等權 |
| 2. 資料／固定監控 | `dataset.py`；每級棋局無重複全輪排列、每盤三十位置、同步 D4；`evaluate.prepare_monitor` 固定最多 5,000 題與每盤三十位置 |
| 3. 分類推論／報表 | `evaluate.py`；位置 log_softmax→盤內平均→題內各盤等權平均→argmax；保存逐盤均值與位置數，不保存逐位置十維向量 |
| 4. 訓練／選模／續跑 | `train.py`；AdamW、70,000／70,001 LR 邊界、每 5,000 步監控、同分換較晚 best、三次嚴格較低早停、原子 best／latest 與可重試驗證 |
| 5. 驗收／入口／紀錄 | `tests/test_reproduce_paper_3.py`、兩份 config、兩份 Slurm；工程驗收與成本限制見下方 |

正式模型從 seed=2026 初始化，不載入 SE／SE2 權重。CUDA 入口要求單節點八卡、每卡完整 16 組／4,800 位置；不做梯度累積或自動縮 batch。CPU 小模型設定只用於合成工程驗證。

## 共用資料的使用方式

只重用既有 `data/processed/reproduce_paper/seed2026/`，不生成／重切資料。`validate_prepared` 先要求既有 prepared／question manifests，再以原七位置準備契約驗證全部來源身分及 checksum；另將本 run 的三十位置設定記錄於 `data_identity.json`。不改寫既有 manifest。缺資料、來源不相容或任一 train／valid 棋局不足三十位置均失敗。

2026-11-03 唯讀索引檢查：training 800,000 盤、validation 200,000 盤，兩者有效位置最少均為 35，不足三十位置均為 0。這不代表重新驗證了 teacher 的來源；程式會在啟動時檢查既有 float32 `value.npy` 的 shape、有限值、範圍並記錄 checksum。完成後的 teacher 單檔本來就不含原引擎來源 manifest。

監控取既有題庫每個 k／rank cell 最多 125 題；若 cell 較少則所有 cell 使用共同可用下限，缺任一 cell 報錯。題內同玩家、同色、同級、棋局互異且均來自 validation。題目與每盤固定抽樣位置保存在本 run 的 `monitor/`，source／selection checksum 改變時拒絕續跑。不同題目共用棋局時只推論一次。

## 八卡短測

設定：[se3_smoke100.json](../../../../../configs/strength/se3_smoke100.json)。Slurm 檔案：`slurm/se3_smoke100.slurm`，dev／8 GPU／1 小時上限。命令：

```bash
sbatch slurm/se3_smoke100.slurm
```

只完成獨立的 100 updates 與一次最多 5,000 題監控；不自動接續正式 run，也不在 smoke 後做全量 validation。`smoke=true` 明確區分短測與正式設定。輸出：`outputs/rank/reproduce_paper_3/se3_smoke100_8gpu/`。

`train.jsonl` 記錄三項 loss、LR、資料等待及步耗時、梯度 norm；smoke 額外保留每 rank 更新 norm、峰值 allocated／reserved 顯存、主程序 RSS。`checkpoint_metrics.json` 記錄最新保存耗時及位元組；`validation/step_00000100.json` 記錄一次監控耗時、四桶題數／exact／adjacent／官方得分與 Equal-k。DataLoader worker 的 RAM 不計入主程序 RSS；`data_seconds` 是 DataLoader 等待，並非磁碟實際讀取位元組。

短測完成後用實測重估：純訓練 `平均每步秒數 × 100,000`，再加至多二十次監控與最終全量推論；GPU-hours＝八卡 wall hours×8。正式模型 raw state 約 83.1 MiB，含 AdamW 的完整 checkpoint 約三倍；latest／best／model 及原子發布的暫存還會增加占用，先預留至少 2 GiB，並以短測檔案大小確認。全量逐盤均值與報表另計。舊的二十小時／160 GPU-h 估計未經本版 GPU 量測確認。

## 正式訓練與續跑

設定：[se3_seed2026.json](../../../../../configs/strength/se3_seed2026.json)。Slurm：`slurm/se3_train.slurm`，8gpus／八卡／36 小時上限；此上限不是成本實測。正式長作業仍須先回報短測估時並經使用者確認，再提交：

```bash
sbatch slurm/se3_train.slurm
```

預設輸出：`outputs/rank/reproduce_paper_3/se3_seed2026/`。每 1,000 步保存 latest，驗證邊界先保存 `validation_pending`。驗證成功後發布完整 best、驗證結果，再提交含已更新 counters 的 latest；中斷重試從 pending 狀態重建同一個決策，不重複累加 patience。每桶精確計數及未四捨五入分數保留，固定 k 順序計算 Equal-k，避免 JSON 排序造成浮點加總差異。

```bash
sbatch slurm/se3_train.slurm --resume outputs/rank/reproduce_paper_3/se3_seed2026/last.json
```

續跑只接受本 run 的 latest、完全相同設定／程式／資料／監控集合／world size，恢復 AdamW、scheduler 與每 rank RNG／BN buffer。完成 step 決定排列輪次、游標、落子抽樣及 D4，不採用 worker 預取進度。驗證 eval 模式後還原原訓練模式與 RNG／BN buffer；八卡以 SyncBN 訓練，CPU Gloo 測試使用一般 BN，不能代替 SyncBN 驗收。

Slurm 的 USR1／TERM／INT 或 `.stop-requested` 在完整更新後保存 latest；中斷屬暫停，不觸發最終評估。可用 `--stop-after N` 主動保存至指定 update。若剛好達預算或 patience 已觸發則仍屬完成。SIGKILL 無法執行清理；確定舊 job 已結束後才人工移除殘留 `.writer.lock`。

只有早停或滿預算才重載並發布最佳權重為 `model.pt`，保留 best_step、stopped_step、stop_reason 與監控分數。latest、完整 best 與來源恢復檔保留。程式不使用 training loss 的 best_1／best_2 政策，不以全量評估分數回頭換模型。

## 全量評估及報表重建

正式訓練完成自動執行；失敗可獨立重試：

```bash
srun --ntasks=1 /home/jason1122/.conda/envs/go/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 \
  -m go_aicup.tasks.strength.reproduce_paper_3.evaluate \
  --config configs/strength/se3_seed2026.json
```

此命令須在已取得八卡的 Slurm allocation 內執行。只推論完整 validation 的全部有效位置，以原固定 10,000 題評估；不推論 training，不建立 centroid。合法早停模型也可載入。每 rank 保存逐盤均值分片與 checksum，綁定 model／data／code／world／推論 batch 身分，完整分片直接重用；中途失敗的 rank 重做自己的分片，不混入不同權重結果。

全部分片完成後，可在 CPU 重建報表，沿用原 CUDA 設定檔即可：

```bash
python -m go_aicup.tasks.strength.reproduce_paper_3.evaluate \
  --config configs/strength/se3_seed2026.json --report-only
```

`evaluation/` 包含：

- `validation_games.parquet`：逐盤十維 log-prob 平均、有效位置數及原 metadata。
- `validation_predictions.parquet`：逐題聚合向量、argmax 預測與級別。
- `result.json`、`official_scores.csv`、`per_rank.csv`：四桶、Equal-k、各級與單盤診斷。
- `confusion_matrices.npz`、`plot/*.png`：單盤、全題與各 k 混淆矩陣、training loss／監控分數曲線。

argmax 並列取較低 rank index。官方 exact／相鄰級得分與解碼方法分開，本版不使用 `score_rank`、溫度、機率算術平均或 centroid。監控只用每盤三十位置／最多 5,000 題；最終報表用全部位置／10,000 題，兩者不能直接混比。

## 工程驗收

使用 `go` Conda 環境、CPU、合成棋譜與兩程序 Gloo。測試涵蓋 shape／原始 logits CE／policy CE／soft value MSE、全參數梯度、八 rank 棋局全輪覆蓋與三十位置互異、spawn workers 預取續跑、prepared 唯讀與不相容／不足位置拒絕、固定分層題組、穩定 log_softmax、盤間等權及 argmax 並列、LR 邊界、同分與三次較低、失敗注入重試、RNG／BN 還原、最佳權重發布、全量評估重試。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_reproduce_paper_3.py -q
```

實際最新測試數字見 `plan.md` 的進度表與根目錄 `RESULTS.md`。未提交 GPU 短測、正式訓練或官方上傳；顯存、吞吐、LR 長期穩定性及真實 k=5／10／15／20 分數均未知。SE 系列沿用 game-disjoint，不能據此宣稱新玩家泛化。

`/slurm/` 仍被 Git 忽略；本次已在工作區建立兩個 launcher，未變更追蹤政策。搬到另一台機器時須另外帶入這兩個檔案。
