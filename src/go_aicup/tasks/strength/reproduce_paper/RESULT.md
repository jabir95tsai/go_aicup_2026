# reproduce_paper 全部短測結果

整理日期：2026-11-02（台灣時間）。本次僅整理既有產物，未重跑訓練。

## 全部已完成短測總覽（2026-11-02 更新）

已查核四份 training smoke test 的 `environment.json`、有效 `config.json`、摘要、逐步 log、`last.json` 與 Slurm `sacct`。四個作業均為 **COMPLETED、exit 0:0**；每份都有連續 step 1–100、各步完整 8 ranks，loss／梯度／參數／更新 norm 均有限。第 50／100 步均有 checkpoint 寫入紀錄，保留的第 100 步檔案已重算 SHA-256，與 `last.json` 相符。這不是 GPU 恢復訓練測試。

**選定：`channels_last=true`、`cudnn_benchmark=false`（使用者 2026-11-02 決定）。** 合用 cuDNN 搜尋的吞吐僅再提高約 0.37%，allocated 峰值卻多約 1.85 GiB；目前保留單獨 channels-last。短測 training loss 不作棋力品質選擇依據。

| 設定 | Job ID | channels-last | cuDNN 搜尋 | 完成步數 | Slurm 配置時間 |
|---|---|---|---|---|---|
| 舊 baseline | 673013 | false | false | 100 | 57 秒 |
| cuDNN 搜尋 | 673105 | false | true | 100 | 59 秒 |
| channels-last（選定） | 673150 | true | false | 100 | 49 秒 |
| cuDNN＋channels-last | 673193 | true | true | 100 | 46 秒 |

四組共用：8×H200、256 channels／20 blocks、BF16＋SyncBN、global 128 組／每卡 16 組（1,120 個位置）、seed=2026、SGD momentum=0.9／weight decay=0.0001、LR=0.04、三項 loss 等權、每 rank 4 workers、TF32 關閉、不使用跨 microbatch 累積。前 5 步僅作量測暖機，不是 LR warmup；穩態窗口為 step 6–100。軟體版本與下方原始 baseline 環境表相同。

### 吞吐與資源比較

| 設定 | 平均 ms/update | p50／p95 ms | updates/s | positions/s | allocated／reserved GiB（單卡峰值） | 節點取樣 RSS GiB |
|---|---|---|---|---|---|---|
| 舊 baseline | 264.327 | 249.268／254.590 | 3.783 | 33,897 | 16.98／17.70 | 43.72 |
| cuDNN 搜尋 | 259.757 | 244.958／252.061 | 3.850 | 34,494 | 18.27／18.83 | 45.72 |
| channels-last（選定） | 194.814 | 181.578／190.189 | 5.133 | 45,993 | 16.85／17.15 | 44.02 |
| cuDNN＋channels-last | 194.102 | 179.772／185.757 | 5.152 | 46,161 | 18.70／18.81 | 45.41 |

相對舊 baseline，cuDNN 搜尋／channels-last／兩者合用的吞吐分別增加 **1.76%／35.68%／36.18%**。合用相對單獨 channels-last 僅增加 **0.37%**。這是單次作業間比較，不是重複實驗的加速保證；新版同程式 baseline（`se_smoke100_perf_baseline_8gpu`）尚無產物。舊 baseline 後曾調整參數診斷的非連續布局支援與多餘拷貝，不能把全部差異嚴格歸因於兩個開關。

### 執行成本與正式預算外推

| 設定 | Session wall／setup 秒 | 程式 session GPU-h | Slurm 配置 GPU-h¹ | 35,000 步外推小時／GPU-h |
|---|---|---|---|---|
| 舊 baseline | 40.564／8.011 | 0.09014 | 0.12667 | 2.5698／20.5588 |
| cuDNN 搜尋 | 40.321／7.981 | 0.08960 | 0.13111 | 2.5254／20.2033 |
| channels-last（選定） | 33.356／7.723 | 0.07412 | 0.10889 | 1.8940／15.1522 |
| cuDNN＋channels-last | 30.794／6.889 | 0.06843 | 0.10222 | 1.8871／15.0968 |

¹ Slurm 配置 GPU-h＝頂層 job Elapsed ×8／3,600；不加總 batch／extern／step，以免重複記帳，也不等於 SU 扣款。四份作業合計配置 211 秒，約 **0.46889 GPU-hours**。程式 session 計時不含全部 Python／Slurm 啟動與結束成本。35,000 步僅按暖機後平均耗時線性外推，包含當前診斷／logging／每 50 步保存的攤提成本，不包含 teacher、centroid、validation、排隊，亦非長跑保證。

### 分項耗時

| 設定 | 資料等待 ms | Forward/backward ms | Optimizer ms | 診斷 ms |
|---|---|---|---|---|
| 舊 baseline | 3.223 | 238.297 | 1.298 | 4.021 |
| cuDNN 搜尋 | 3.129 | 234.148 | 1.260 | 3.689 |
| channels-last（選定） | 3.454 | 169.887 | 1.281 | 4.378 |
| cuDNN＋channels-last | 3.350 | 168.214 | 1.306 | 4.389 |

每個分項均先取各步最慢 rank，再平均 step 6–100；最大值不一定來自同一 rank，不能直接相加當作端到端耗時。channels-last 的 forward/backward 由約 238 ms 降至 170 ms，資料等待仍約 3 ms；目前主要改善在模型計算。

### Loss：前／後各 10 步平均

| 設定 | Total 前→後 | BT 前→後 | Policy CE 前→後 | Value MSE 前→後 |
|---|---|---|---|---|
| 舊 baseline | 8.372920 → 7.777341 | 1.677309 → 1.662166 | 5.903604 → 5.339473 | 0.792007 → 0.775701 |
| cuDNN 搜尋 | 8.372692 → 8.099800 | 1.677229 → 1.666120 | 5.903505 → 5.657793 | 0.791959 → 0.775887 |
| channels-last（選定） | 8.372536 → 7.604230 | 1.677393 → 1.639304 | 5.903108 → 5.191901 | 0.792035 → 0.773025 |
| cuDNN＋channels-last | 8.372647 → 7.753849 | 1.677411 → 1.660216 | 5.903170 → 5.318796 | 0.792066 → 0.774837 |

### Loss：第 1／100 步

| 設定 | Total 1→100 | BT 1→100 | Policy CE 1→100 | Value MSE 1→100 |
|---|---|---|---|---|
| 舊 baseline | 8.445898 → 7.564128 | 1.682583 → 1.668319 | 5.965910 → 5.117100 | 0.797405 → 0.778710 |
| cuDNN 搜尋 | 8.445898 → 7.943127 | 1.682583 → 1.669808 | 5.965909 → 5.494535 | 0.797405 → 0.778783 |
| channels-last（選定） | 8.445858 → 7.437150 | 1.682582 → 1.651179 | 5.965744 → 5.009324 | 0.797531 → 0.776647 |
| cuDNN＋channels-last | 8.445858 → 7.594135 | 1.682582 → 1.660477 | 5.965744 → 5.154770 | 0.797531 → 0.778887 |

各步為不同隨機 training batch，並非固定 validation 題目。四組早期下降主要來自 policy CE。演算法／布局與 BF16 可改變浮點歸約順序，差異經 BN、ReLU、SGD momentum 累積後可能走出不同更新軌跡；這是可能機制，本次未做同權重／同 batch 的 GPU 單步對照，不能證明差異全由捨入造成。較低的短測 loss 不代表更高的棋力成績。

### 梯度與參數更新

| 設定 | Gradient L2 min–max | Update L2 min–max | Update／parameter min–max（%） |
|---|---|---|---|
| 舊 baseline | 0.337995–6.887647 | 0.044739–0.279323 | 0.03745–0.23410 |
| cuDNN 搜尋 | 0.343865–6.339716 | 0.046390–0.260310 | 0.03883–0.21828 |
| channels-last（選定） | 0.417691–7.850758 | 0.050278–0.324584 | 0.04208–0.27212 |
| cuDNN＋channels-last | 0.385343–13.085187 | 0.045819–0.470991 | 0.03835–0.39489 |

以上為 rank 0 的逐步 norm。合用設定最大 gradient norm=13.085，高於單獨 channels-last 的 7.851，但所有紀錄均有限；單一最大值不能直接判定不穩定，100 步有限性也不能保證長期收斂。

### Checkpoint 與 I/O

| 設定 | Step 50／100 秒（最慢 rank） | Checkpoint 總秒數 | 最後檔案 bytes | SHA-256 |
|---|---|---|---|---|
| 舊 baseline | 0.606499／0.768103 | 1.374776 | 194,378,646 | 已驗證 |
| cuDNN 搜尋 | 0.636801／0.720469 | 1.357458 | 194,378,646 | 已驗證 |
| channels-last（選定） | 0.538555／0.676800 | 1.215530 | 194,378,838 | 已驗證 |
| cuDNN＋channels-last | 0.587679／0.733655 | 1.321512 | 194,378,838 | 已驗證 |

| 設定 | read_bytes | write_bytes | rchar | wchar |
|---|---|---|---|---|
| 舊 baseline | 0 | 388,907,008 | 11,387,420,874 | 399,576,018 |
| cuDNN 搜尋 | 0 | 388,907,008 | 11,378,181,208 | 399,573,623 |
| channels-last（選定） | 0 | 388,907,008 | 11,375,050,298 | 398,898,213 |
| cuDNN＋channels-last | 0 | 388,907,008 | 11,424,559,364 | 399,061,271 |

I/O 為 100 步所有 rank／loader workers 的 `/proc` counter 增量合計；cache 命中與檔案系統行為會影響數字。四組 `read_bytes=0` 不代表未讀取資料，不能據此推算冷快取頻寬。RSS 為取樣且共享頁可能重複計算，顯存僅為 PyTorch allocator，非整卡用量。

### 證據檔案

- **舊 baseline／job 673013**：`outputs/rank/reproduce_paper/se_smoke100_8gpu/`；Slurm log：`outputs/slurm-se-smoke100-8gpu-673013.out`。
  - 最後 checkpoint：`outputs/rank/reproduce_paper/se_smoke100_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793595607442517821.pt`；SHA-256：`1095a967a67c0cddef0fa8d6bf15da85294cb134011ae866e57635cb19f22673`。
- **cuDNN 搜尋／job 673105**：`outputs/rank/reproduce_paper/se_smoke100_cudnn_8gpu/`；Slurm log：`outputs/slurm-se-smoke100-8gpu-673105.out`。
  - 最後 checkpoint：`outputs/rank/reproduce_paper/se_smoke100_cudnn_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793596922929113735.pt`；SHA-256：`50ab7b6fbded3134d0ac9010d83a832406d8115b19eea23e10abd04e2a953964`。
- **channels-last（選定）／job 673150**：`outputs/rank/reproduce_paper/se_smoke100_channels_last_8gpu/`；Slurm log：`outputs/slurm-se-smoke100-8gpu-673150.out`。
  - 最後 checkpoint：`outputs/rank/reproduce_paper/se_smoke100_channels_last_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793597289873461422.pt`；SHA-256：`c00fb723c04ecc1889fdf11af6a833f5ae407c72c522b06c1868717a1fc4ddf6`。
- **cuDNN＋channels-last／job 673193**：`outputs/rank/reproduce_paper/se_smoke100_cudnn_channels_last_8gpu/`；Slurm log：`outputs/slurm-se-smoke100-8gpu-673193.out`。
  - 最後 checkpoint：`outputs/rank/reproduce_paper/se_smoke100_cudnn_channels_last_8gpu/checkpoints/update_00000100_epoch_0000_batch_000100_1793597608503070254.pt`；SHA-256：`c444c7252d4d346272041447ab74a4506c3a88142f5ee3c100cb088d113ca43b`。

每個 run 目錄均保存 `config.json`、`environment.json`、`data_identity.json`、`train.jsonl`、`probe_summary_00000_00100.json`、`last.json`。本報告以執行時保存的有效 config 為準，不以後來改過的預設 config 回推歷史設定。

### 尚無結果的短測與工程驗收

- 新版同程式 baseline：設定已存在，目前未找到完成摘要。
- 推論 batch benchmark：入口與設定已存在，目前未找到 `benchmark.json`；每卡 batch 1,120／2,240／4,480／8,960 的吞吐、顯存與 score 誤差仍未知。
- 選定設定的新預設短測目錄 `se_smoke100_default_8gpu`：目前未找到完成摘要；採用依據為 job 673150，不另宣稱新作業已執行。
- CPU 合成工程驗收依根目錄 `RESULTS.md`：完整 reproduce_paper 回歸 **79 passed（46.50s）**；預設設定選定後訓練測試 **7 passed（19.77s）**。這些是既有測試紀錄，本次未重跑；不替代 CUDA／NCCL 恢復訓練驗收。
- GPU 中斷續跑、跨布局同權重單步對照、25,000／32,500 步 LR 邊界與35,000 步收斂、四個小步／一個大步近似診斷、centroid／validation 評估均仍待完成。

| Validation 官方分數 | k=5 | k=10 | k=15 | k=20 | Equal-k |
|---|---|---|---|---|---|
| 舊 baseline | 未知 | 未知 | 未知 | 未知 | 未知 |
| cuDNN 搜尋 | 未知 | 未知 | 未知 | 未知 | 未知 |
| channels-last（選定） | 未知 | 未知 | 未知 | 未知 | 未知 |
| cuDNN＋channels-last | 未知 | 未知 | 未知 | 未知 | 未知 |

本次僅讀取產物、查詢 Slurm 與重算統計，沒有重跑訓練或修改原始輸出。

## 原始 baseline 結論（job 673013）

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
- 未確認：單卡加速比與其他 microbatch；cuDNN／channels-last 的三組對照已完成，見上方全部短測總覽。

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
