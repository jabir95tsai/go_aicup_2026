# reproduce_paper smoke test 結果

整理日期：2026-11-02（台灣時間）。本次僅整理既有產物，未重跑訓練。

## 結論

**8×H200、BF16＋SyncBN、global 128 組的 100 步訓練已完成**。逐步紀錄包含連續的 step 1–100，每步均有 8 個 rank；loss、梯度 norm、參數 norm 與更新 norm 均為有限值，且有非零參數更新。第 50／100 步皆有 checkpoint 寫入紀錄，目前保留第 100 步權重。

排除前 5 步量測暖機後，平均 **0.2643 秒／update、3.783 updates/s、33,897 positions/s**。相同條件下，35,000 步的訓練迴圈外推約 **2.57 小時／20.56 GPU-hours**。這只支持本設定可完成短測；正式收斂、驗證成績與 GPU 中斷續跑一致性尚未驗收。

## 測試設定與環境

| 項目 | 實際設定／紀錄 |
|---|---|
| Run | `se_smoke100_8gpu`；Slurm job `673013`，`dev` partition |
| 硬體 | 單節點 8×NVIDIA H200；每卡顯存 143,771 MiB；96 CPUs |
| 軟體 | Python 3.13.12、PyTorch 2.6.0+cu124、CUDA runtime 12.4、cuDNN 9.1.0、NCCL 2.21.5、driver 580.65.06 |
| 模型 | 256 channels／20 residual blocks；scalar、policy（362 類）、value 三頭 |
| 資料 | 固定 split seed=2026，每級 train／validation 為 80,000／20,000 盤；訓練 seed=2026 |
| 位置與增強 | 指定玩家、ply ≥30 的落子前盤面；本人／對手歷史輸入、D4 增強 |
| Teacher | 逐有效落子前位置的 value；1 visit、日本規則、貼目 6.5 |
| 批次 | global 128 組；每組十級各七手，共 8,960 positions／update；每卡 16 組／1,120 positions |
| 精度與 BN | BF16＋SyncBN；每卡一次處理 16 組，未跨 microbatch 累積 |
| 最佳化 | SGD，LR=0.04、momentum=0.9、weight decay=0.0001；三項 loss 權重均為 1 |
| 其他 | 每 rank 4 個 loader workers；TF32 關閉、cuDNN benchmark 關閉、deterministic=false |
| 量測窗口 | 100 updates；前 5 步僅排除於穩態統計，並非 LR warmup；100 步 LR 均為 0.04 |

保存環境的 git commit 為 `07e576bdd0d983bc4ba20b4a4d2dd4e826ceba82`，但執行當時有未提交變更，不能只靠 commit 重建此次程式狀態；需併看 `environment.json` 的工作樹紀錄與 checkpoint 的程式身分。

## 效能與資源

下表主要取自 `probe_summary_00000_00100.json`；穩態統計為 step 6–100，共 95 步，包含診斷、logging／跨 rank 收集及窗口內的 checkpoint 成本。

| 指標 | 結果 |
|---|---:|
| 本 session wall time | 40.564 秒 |
| Setup | 8.011 秒 |
| 全 session GPU-hours（wall ×8） | 0.09014 |
| 穩態平均／p50／p95 每步耗時 | 0.2643／0.2493／0.2546 秒 |
| 吞吐量 | 3.783 updates/s；33,897 positions/s |
| 單卡 PyTorch allocated 峰值 | 16.98 GiB |
| 單卡 PyTorch reserved 峰值 | 17.70 GiB |
| 節點取樣 RSS 峰值 | 43.72 GiB |
| 第 50／100 步 checkpoint 耗時（各步最慢 rank） | 0.6065／0.7681 秒 |
| Checkpoint 總耗時 | 1.3748 秒 |
| 最後 checkpoint 大小 | 194,378,646 bytes（185.37 MiB） |
| 35,000 步訓練迴圈線性估時 | 2.5698 小時；20.5588 GPU-hours |

由 `train.jsonl` 重算，step 6–100 各步取最慢 rank 後，資料等待、forward/backward、optimizer、診斷平均分別為 **3.223 ms、238.297 ms、1.298 ms、4.021 ms**。各項最大值可能來自不同 rank，不能直接相加當成端到端耗時；目前紀錄中 forward/backward 為主要耗時。

100 步所有 rank 的 `/proc` I/O 增量合計：`read_bytes=0`、`write_bytes=388,907,008`、`rchar=11,387,420,874`、`wchar=399,576,018`。這是程序與 loader workers 的 OS counter；包含 cache 行為，不能將 `read_bytes=0` 解讀成未讀資料，也不能用它推估冷快取或平行檔案系統的實體頻寬。

顯存數字僅涵蓋 PyTorch allocator，並非整卡用量；RSS 是主程序與存活 workers 的取樣總和，共享頁面可能重複計算。Wall time 是訓練函式的 session 計時，非 Slurm 完整配置時間，GPU-hours 也非排程器帳務值。正式估時未包含 teacher 標籤生成、centroid／validation 推論與排隊，亦未驗證長跑 I/O 或效能變化。

## Loss 與參數更新

Loss 為跨卡平均的 training batch 指標；前／後 10 步平均分別使用 step 1–10／91–100，各步資料不同。

| 指標 | Step 1 | Step 100 | 前 10 步平均 | 後 10 步平均 |
|---|---:|---:|---:|---:|
| Total | 8.44590 | 7.56413 | 8.37292 | 7.77734 |
| BT | 1.68258 | 1.66832 | 1.67731 | 1.66217 |
| Policy CE | 5.96591 | 5.11710 | 5.90360 | 5.33947 |
| Value MSE | 0.79740 | 0.77871 | 0.79201 | 0.77570 |

Total 的前後十步平均下降 **0.59558**，其中 policy CE 下降 **0.56413**，為主要來源；BT／value 分別下降 0.01514／0.01631。這是早期訓練趨勢，尚不能推論棋力排序能力或泛化增益。

以 rank 0 的逐步紀錄計，gradient L2 norm 範圍為 **0.337995–6.887647**；update L2 norm 為 **0.044739–0.279323**，update／parameter norm 為 **0.000374–0.002341**。第 100 步 gradient norm=1.348397、update norm=0.164316、update／parameter=0.001378。這些數字與 finite checks 支持短測中有正常的有限值更新，不能視為長期穩定性保證。

## 驗收範圍與待確認項目

- 已確認：100 步／8 ranks 的紀錄完整、BF16＋SyncBN 訓練走完、loss／梯度／更新數值有限、checkpoint 寫入及最後檔案存在。
- 未確認：GPU 中斷續跑與連續執行的參數一致性；本次只有 start_step=0 至 100 的單次 session，未重新載入 checkpoint 驗收。
- 未確認：正式 LR 在 25,000／32,500 步的切換、35,000 步收斂，以及四個 32 組／LR=0.01 小步與一個 128 組／LR=0.04 大步的近似誤差。
- 未確認：單卡加速比、其他 microbatch 或 cuDNN／channels-last 設定；本目錄目前只有這一組已完成的 smoke test 結果。

本次未執行 centroid／validation 評估，不能與官方 baseline 0.4631 比較：

| Validation 官方分數 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 本次 smoke test | 未知 | 未知 | 未知 | 未知 | 未知 |

## 證據路徑

以下路徑均相對 repo 根目錄：

- `outputs/rank/reproduce_paper/se_smoke100_8gpu/`：`config.json`、`environment.json`、`data_identity.json`、`train.jsonl`、`probe_summary_00000_00100.json`、`last.json` 與 `checkpoints/`。
- `outputs/slurm-se-smoke100-8gpu-673013.out`：GPU／driver 與逐步輸出；最後一筆為 step 100。
- 最後權重：`outputs/rank/reproduce_paper/se_smoke100_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793595607442517821.pt`。

本次整理以保存的有效 config 與結果為準；沒有覆寫原始產物、啟動作業或修改訓練程式。


## 2026-11-02：已記錄短測產物清理

依使用者要求，已刪除本檔 Git 版本所記錄的四組已完成 training smoke test（jobs 673013／673105／673150／673193）之輸出目錄、全部 checkpoint、執行時 config／環境／逐步 log／摘要及對應 Slurm `.out`。上方證據路徑為歷史紀錄，檔案已不存在。

刪除範圍：

- `outputs/rank/reproduce_paper/` 下的 `se_smoke100_8gpu/`、`se_smoke100_cudnn_8gpu/`、`se_smoke100_channels_last_8gpu/`、`se_smoke100_cudnn_channels_last_8gpu/`。
- `outputs/slurm-se-smoke100-8gpu-{673013,673105,673150,673193}.out`。
- `configs/strength/se_smoke100_cudnn.json`、`se_smoke100_channels_last.json`、`se_smoke100_cudnn_channels_last.json`。

保留共用資料、teacher value 標籤、程式與 Slurm 入口，以及尚未執行的新預設 `se_smoke100.json`、`se_smoke100_perf_baseline.json`、`se_infer_batch_benchmark.json`。後者仍引用已刪除的 job 673013 checkpoint；再次使用推論 benchmark 前須指定新的 checkpoint。README 的舊對照／續跑指令亦屬歷史用法，專用 config 與 checkpoint 已刪除。本次未執行訓練或測試。


## 2026-11-02：8 卡推論 batch benchmark（job 673590）

**四個 batch 均成功，無 OOM；每卡 batch 從 1,120 增至 8,960，穩態吞吐僅增加 4.56%。** 2,240 的含初始化吞吐最高；本次不自動變更正式推論設定。

設定：8×H200、BF16、channels_last=false、eval mode、無 D4。使用 training 固定連續分片，每卡 71,680 個盤面、全節點每個候選共 573,440 個盤面；每個候選先將第一批重複暖機兩次。報告 wall time 為 48.591 秒，非 Slurm 完整配置時間。

| 每卡 batch | 全節點盤面／秒（排除 setup） | 相對 1,120 | 含 setup 盤面／秒 | 每卡 allocated 峰值（GiB） | 每卡 reserved 峰值（GiB） | score 最大絕對差 | 各 rank 最大相對 L2 差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,120 | 108,710 | +0.00% | 62,818 | 1.36 | 1.72 | 0.0000000 | 0.000e+00 |
| 2,240 | 110,816 | +1.94% | 63,890 | 2.54 | 2.85 | 0.0078125 | 6.195e-05 |
| 4,480 | 107,975 | -0.68% | 45,653 | 4.91 | 5.20 | 0.0078125 | 6.195e-05 |
| 8,960 | 113,671 | +4.56% | 47,417 | 9.64 | 10.11 | 0.0156250 | 1.614e-03 |

score 差異以同一 checkpoint、相同位置的 batch=1,120 結果為基準。8,960 的各 rank 相對 L2 差約 0.152%–0.161%，高於 2,240／4,480 的最大約 0.0062%；是否影響最終棋力級別仍未知。

解讀與限制：

- 穩態吞吐約 10.8–11.4 萬盤面／秒，未隨 batch 明顯增加。各 rank 的 compute_seconds 約 4.46–4.67 秒，表示本次加大 batch 未大幅縮短計算時間；未做 kernel profiling，不能據此斷定硬體瓶頸。
- 2,240 是本次較均衡的候選：穩態比 1,120 快 1.94%，含 setup 吞吐最高，allocated 峰值 2.54 GiB。8,960 穩態最快，但 allocated 增至 9.64 GiB，數值差異也較大；4,480 未見速度優勢。
- 每個候選的正式量測僅約 5 秒，只有單次、固定由小至大的比較；後續候選可能受益於 OS cache，不能將幾個百分點的差異視為穩定增益。setup 包含 worker 啟動、第一批讀取與暖機。
- 本次 channels-last 關閉，尚未驗證目前選定 channels-last 布局的推論效能；顯存統計僅涵蓋 PyTorch allocator。
- 僅推論固定盤面子集，未執行全量每盤聚合、centroid 或 validation；k=5／10／15／20 與 Equal-k 官方分數均未知，無法判定數值差異對品質的影響。

證據路徑（相對 repo 根目錄）：

- `outputs/rank/reproduce_paper/se_infer_batch_benchmark_8gpu/benchmark.json`：各候選／rank 的耗時、吞吐、顯存、位置範圍與數值差異。
- 同目錄的 `config.json`、`environment.json`、`reference_rank_00.npy` 至 `reference_rank_07.npy`。
- `outputs/slurm-se-infer-batch-8gpu-673590.out`。
- 本次實際 checkpoint：`outputs/rank/reproduce_paper/se_smoke100_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793602368217443709.pt`；SHA-256：`531839c8ebbedd1f7ecb24d548b1f435e0435bba662742fbd1f8bff70b3db306`。此檔名與上方已清理的歷史 job 673013 權重不同。

本次僅整理既有 benchmark 產物，未重跑、提交作業或修改推論設定。
