# SE 第二次實驗：AdamW＋每輪不重複棋局

從 `outputs/rank/reproduce_paper/se_seed2026/best_1.pt`（35,000 步）載入模型與 BN buffers，額外訓練 70,000 步，累計 105,000 步。原 run 唯讀，獨立輸出至 `outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/`。

設定檔：`configs/strength/se_seed2026_adamw70k.json`。來源 checkpoint 固定 SHA-256；啟動前驗證來源 checksum、設定、資料身分、world size 與程式雜湊。保留每 rank RNG／buffers，SGD momentum 不轉移；AdamW 的一、二階動量及 optimizer step 從零開始。抽樣從本實驗第一輪開始，不能回溯保證舊 SGD 歷史無重複。

| 新增步數（含端點） | 累計步數 | LR | 完整輪數 |
|---|---|---|---|
| 1–50,000 | 35,001–85,000 | 0.002 | 80 |
| 50,001–70,000 | 85,001–105,000 | 0.001 | 32 |

AdamW：betas=(0.9, 0.999)、eps=1e-8、weight_decay=0.0001，無 warmup；對所有可訓練參數套用相同衰減。保持 global 128 組、每組十級各一盤七個有效位置、8×H200／BF16／SyncBN／channels-last、三頭模型與三項 loss 等權。不新增依賴，不更動原 SE 訓練器。

## 抽樣與步數

每級 training 80,000 盤分別 shuffle，8 rank 按全域組號分配；每步每級取 128 盤、總共 1,280 盤，625 steps 正好完整遍歷 800,000 盤。全輪無棋局重複，全部用完才重新 shuffle。每次出現仍重新抽七個不同位置與同步 D4，不要求把同一盤全部位置送入模型。拒絕各級盤數不等、不能整除 batch 或重複 game index 的索引，避免靜默漏盤／補抽。

`step`、checkpoint 的 `optimizer_steps` 及 model.pt 的 `optimizer_steps` 一律是本實驗新增 AdamW 步數；log 的 `cumulative_step`、checkpoint 的 `cumulative_steps`、summary 的 `cumulative_updates` 另記來源＋新增步數。seed／輪次決定棋局排列，已完成 step 決定下一批及位置／D4；DataLoader 預取不推進 durable cursor。中斷後未保存的更新會回退並重做，與一般 checkpoint 語意相同。

## 執行

首次提交（此文件不代表已提交）：

```bash
sbatch slurm/se_adamw70k.slurm
```

中斷續跑，必須使用新 run 的 latest；不要把舊 `best_1.pt` 傳給 `--resume`：

```bash
sbatch slurm/se_adamw70k.slurm \
  --resume outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/last.json
```

直接入口（須由計算節點 8 卡啟動）：

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m go_aicup.tasks.strength.reproduce_paper_2.train \
  --config configs/strength/se_seed2026_adamw70k.json
```

可用 `CONFIG=/absolute/path/config.json sbatch ...` 選擇設定，`PYTHON_BIN` 預設為 `$HOME/.conda/envs/go/bin/python`。`--stop-after N` 指本實驗已完成的 AdamW 步數；此選項供中斷驗證，達停止點後不自動做最終評估。Slurm signal 寫 `.stop-requested`，訓練於完整 update 後保存退出。若在評估期間收到停止要求，現有評估不支援逐 batch 優雅退出；完成後不啟動下一段，強制中止時可重用已完成推論分片重試。

## 保存與評估

每 1,000 步、停止點及段落邊界保存完整 AdamW／scheduler／每 rank RNG；保留保存窗口平均 total training loss 最低兩份＋最新 checkpoint。完訓使用既有原子發布／驗證／留存流程，留下 `model.pt`、`best_1.pt`、`best_2.pt` 及 markers，再清理 checkpoints／last.json。最終模型是最後一步，不依 validation 選權重。

新增 50,000 步先發布獨立 `milestones/update_00050000/model.pt`（新增步數 50,000、累計 85,000），重用原 SE evaluate 完整重建 training centroids 與固定 validation 題組分數／圖表，再由剛保存的 AdamW checkpoint 接續第二段。這段恢復也還原 RNG，評估建立模型不影響下一批訓練。新增 70,000 步完成後，在 run 根目錄執行同樣評估。兩次報表分別在 milestone 的 `evaluation/` 與 run 的 `evaluation/`。

訓練及兩次評估在同一個 process group 內進行；只在整個入口退出時關閉通訊。

中途評估或 snapshot 發布失敗可使用同一 config 與新 run `last.json` 重試，不多做已保存的更新；最終評估失敗可直接重跑首次命令，不重訓。原 evaluator 支援 CPU report-only，使用 milestone 或 run 內保存的 config.json：

```bash
python -m go_aicup.tasks.strength.reproduce_paper.evaluate \
  --config outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/config.json --report-only
```

## 實驗判讀與成本

原模型 k=5/10/15/20 為 0.5453／0.5922／0.6108／0.6432，Equal-k=0.5979。建議 gate：Equal-k≥0.6029，且各桶退步不超過 0.01；這是比較門檻，不作自動早停，增益未知。同時改 optimizer、抽樣與訓練長度，不能單獨歸因。沿用 game-disjoint：實際 validation 7,855 位玩家全在 training 8,007 位玩家內，不代表新玩家泛化。

依舊 SGD 實測外推，純訓練約 3.57 小時／28.6 H200 GPU-h；AdamW 實際速度未知。Slurm 申請 8 卡 6 小時，包含兩次全量推論、存檔與餘裕；不包含排隊，SU 費用未知。

驗證：`tests/test_reproduce_paper_2.py` 使用合成資料與 CPU／Gloo，涵蓋八 rank 全輪覆蓋、跨輪 shuffle、worker 預取恢復、LR 邊界、AdamW reset、精確中斷續跑、失敗重試與全量評估接線。不把 CPU 通過視為 8×H200 實測通過。

目前 `.gitignore` 忽略整個 `/slurm/`；新腳本已建立於工作目錄，預設不納入 Git。
