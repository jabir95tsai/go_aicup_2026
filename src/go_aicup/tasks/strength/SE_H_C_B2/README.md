# 凍結 SE3＋H＋C，重新訓練 B2

架構與資料依根目錄 [實驗計畫](../../../../../STRENGTH_SE3_HC_B2_PLAN.md)。本目錄為獨立實驗，不修改 SE3／SE2、V2、舊 sklearn B2 或既有提交模型。

每盤 SE3 十維平均 log-prob＋H 三十維＋C 二十維，先在題內各盤等權平均。k=5／10／15／20 各有獨立 scaler 與 `Linear(60,32) → ReLU → Linear(32,10)`。訓練直接對 logits 做 CE；推論 Softmax＋`score_rank`。每頭 2,282 個參數，只有隱藏層有 ReLU。

## 設定與來源

設定為 `configs/strength/se3_hc_b2.json`，所有路徑相對 repo root 解讀。預設輸出 `outputs/rank/se3_hc_b2/se3_30k_mlp32_seed2027/`。SE3／A／C 權重有固定 SHA-256；HumanSL bin／checkpoint、engine、analysis config、上游與本目錄程式、版本、dataset／board manifest、重用特徵與題組均記錄身分。檔案損壞或來源改變會失敗，不自動重抽、搬移或覆寫舊 run。

變更來源／程式後，使用獨立 output 重新 prepare；不要刪 manifest 來繞過來源檢查。多個程序可以讀同一 prepared，寫入同一 head／同一 feature worker 會被 lock 拒絕。作業收到 SIGTERM 或 SIGKILL 時可能留下 `.writer.lock`，只能在確認舊程序結束後人工移除。checkpoint 與舊 pickle／torch 權重只應讀本專案可信產物。

上游全程 eval／no-grad，凍結檢查包含 BN buffers；C 向量也檢查不變。訓練 B2 只讀快取，不載入 teacher 或上游 optimizer。SE3 使用指定玩家、ply≥30 的全部有效位置及落子前本人／對手八步歷史；H／C 維持原位置範圍。

## 一次提交訓練任務

在 repo 根目錄執行一次：

```bash
sbatch slurm/SE3_H_C_B2/train.slurm
```

任務依序完成 `prepare`、八卡補算開發 SE3 特徵、兩臂開發 B2 訓練、S-select 評估；若開發 gate 通過，再補算最終 SE3／新棋局 H／C 特徵、重新訓練兩臂並評估 `U_newgames`。全部 k 退步時依計畫停止，保留開發報表；Slurm 顯示完成，log 會明確寫出 gate 停止。任何子程序失敗則任務失敗並指出 log 路徑。已完成分片與訓練 checkpoint 會在重新提交同一命令時重用；因強制終止留下的 lock，須確認舊作業結束後依上節處理。

預設配置是單節點 8×H200、96 CPU、128 GB、`8gpus` 佇列與 12 小時上限，即配置上限 **96 GPU-h**；CPU 準備及 B2 訓練期間也占用這八張卡。可用 `sbatch --time=HH:MM:SS .../train.slurm` 依量測調整上限。這是一次提交的配置上限，實際耗時及 SU 消耗未知；目前沒有此流程的 GPU 短測數據。

供排程參考：以現有 seed2027 題組與快取唯讀估算，新棋局 H／C 各缺 **6,642 盤、1,481,462 個落子前位置**。若假設八卡負載均衡且能達舊實驗單卡吞吐，H 的 15 profiles 約需 **9.5–13.2 分鐘**，C 約需 **3.1–3.2 分鐘**。這只是歷史吞吐的算術外推，不含 SE3 補算、CPU 準備／訓練、I/O、各批 engine 啟動、編譯及排隊；本流程與舊實驗的設定不同，無法據此可靠預測總耗時或保證在 12 小時內完成。任務只產生模型與評估報表，不產生正式測試 CSV 或上傳。

主 log 在 `outputs/slurm-se3-hc-b2-train-<job-id>.out`；八卡補算的各程序 log 在 run 目錄的 `slurm_logs/<job-id>/`。完整結果位於 `reports/final_newgames/result.json`，若開發 gate 停止則查看 `reports/dev_select/result.json`。若要自訂 config，可在提交時設定 `CONFIG` 環境變數；執行者不需先啟動其他作業。

## 執行順序

以下從 repo root、既有 `go` 環境執行。Nano4 的 CPU 準備／訓練也請在已取得的計算資源內執行，不在登入節點跑完整工作。下面命令不會自動上傳。

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON_BIN="${HOME}/.conda/envs/go/bin/python"
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --help
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 prepare
```

`prepare` 固定兩份各 80,000 道 synthetic fitting questions（每 k 20,000）、原 S-select／舊 U 及新 1,000 題 U_newgames，寫入 manifest／audit 後即可唯讀重用。題內同玩家、同色、不同棋局；U_newgames 排除 SE3 training、SE3 monitor 全棋局及 C 學習／監控棋局，與 B2 fit 玩家互斥。樣本不足、某 k 缺十類或玩家重疊會拒絕，沒有縮 k 或換 training 棋局的替代路徑。

若希望在正式任務前另外量測，可先執行：

```bash
sbatch slurm/SE3_H_C_B2/benchmark.slurm
```

八卡共用全域分層 limit：SE3 從開發缺失池取最多 1,000 盤，H／C 從新題組缺失池各取最多 100 盤。已完成 feature shards 可重用。`features/train/<component>/timing_XX.json` 記錄各 worker 的遊戲數、位置數、end-to-end 秒數、吞吐與 PyTorch allocated 顯存。時間包含載入／編譯／I/O，不宣稱穩態速度；H engine 的顯存不在 PyTorch 計數中。估總 wall time 時用最慢 worker 與實際完成量，GPU-h 以實際配置的八卡 wall hours×8 記帳。

短測 Slurm 上限 30 分鐘／4 GPU-h；補算與正式推論 launcher 的 4 小時是排程上限，並非成本估計或執行授權。全量前先依量測回報估時。

### 開發階段

確認補算成本後，只補開發題目及 S-select 所需的 SE3 特徵，舊 A／H／C 可重用：

```bash
COMPONENT=se3 SCOPE=dev sbatch slurm/SE3_H_C_B2/extract.slurm
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 train --stage dev --arm a
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 train --stage dev --arm se3
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 evaluate --stage dev --role select
```

模型從 S-fit＋S-extra 的 38,589 盤合成題目訓練；S-select 1,000 題選模。每 k 的 AdamW lr=0.001、weight_decay=0.0001、batch=512、最多 100 epochs／4,000 updates，cosine 至 0.00001，無 warmup。每 epoch 以該 k 官方得分選 best，並列保留較早 epoch，十次未提高早停。尾批保留；latest 保存 AdamW／scheduler／generator／shuffle order／cursor／best／patience，可精確接續已提交的進度。重跑相同命令即可續跑／重試最後模型發布。

查看 `reports/dev_select/result.json`。若所有 k 退步且 Equal-k Δ≤0，停止擴大新 H／C 推論；保留失敗結果。上述分步命令須自行啟動下一步；一次提交的訓練腳本會依 gate 自動繼續或停止。

### 最終重訓與主要評估

開發通過繼續條件且確認成本後：

```bash
COMPONENT=se3 SCOPE=fit sbatch slurm/SE3_H_C_B2/extract.slurm
COMPONENT=h SCOPE=newgames sbatch slurm/SE3_H_C_B2/extract.slurm
COMPONENT=c SCOPE=newgames sbatch slurm/SE3_H_C_B2/extract.slurm
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 train --stage final --arm a
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 train --stage final --arm se3
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 evaluate --stage final --role newgames
```

最終以全部 Valid-S 65,223 盤重新合成固定擬合題、重擬合 scaler、重新初始化分類頭，逐臂／逐 k 執行開發最佳 epoch E_k。cosine 總長仍為 4,000 步，只走前 40×E_k 步，不重新壓縮 schedule，不用已納入 fit 的 S-select 早停。新題組只評估一次，不用它調參。

若選中的新棋局不在舊 A 全量分類快取，會明確失敗；須以原 V2 checkpoint／原輸入規則補齊對照，再依相同題組評估，不能縮成有特徵的子題。本工作區已唯讀驗證 A 有 248,411 盤快取；是否覆蓋最終選中題組在 prepare／assemble 時檢查。

舊 U 僅作受污染診斷，若要報告需另補其 SE3 缺失特徵：

```bash
COMPONENT=se3 SCOPE=old_u sbatch slurm/SE3_H_C_B2/extract.slurm
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 evaluate --stage final --role old_u
```

報表包括每 k／Equal-k、exact／adjacent、各級成績、混淆矩陣、逐題 posterior 及 2,000 次按 rank 分層的配對玩家 bootstrap。候選 gate：Equal-k Δ≥0.005、每 k Δ≥−0.005；bootstrap 下界>0 才稱本評估設計下有正向證據。

SE3 看過評估玩家的其他棋局，且 B2 fitting 大部分是 SE3 已訓練棋局，仍有玩家記憶與信心分布偏移。新題組只能支持已見玩家的新棋局結果，不能證明 unseen-player 泛化，也不自動 promote 現行主線。

## 正式測試與 CSV

最省成本的路徑是重用已完成的 `slurm/rank_ahc_predict.slurm` 工作目錄，其中 `rank_public/`、`rank_private/` 有 cache／store_meta、H／C 分片與完整 `rankvec_predictions.parquet`。新程式只在這些目錄新增 `se3_hc_b2_features/`、`se3_hc_b2_predictions.parquet/json`，不改舊 posterior。先通過本地 gate，再確認正式推論成本後提交：

```bash
BASE_WORK=outputs/rank_test_ahc_<job-id>/rank \
  sbatch slurm/SE3_H_C_B2/predict.slurm
```

新 SE3 推論沿用同一 BoardCache、本人視角、全部 ply≥30 落子前盤面，沒有 V2 十維。每個測試題必須有完整 k 盤六十維特徵；缺任一盤時整題使用相同 ID、相同 k 的既有完整 AH＋C posterior。fallback 本身若有 V2／其他 fallback 或縮題會拒絕；沒有合格 fallback 也直接失敗。fallback 題數／原因保存在逐題結果與 marker，禁止手動改預測。

也可獨立執行各步；`--work` 指向單一分區的既有工作目錄：

```bash
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 extract --component se3 --work "$BASE_WORK/rank_public"
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 aggregate --work "$BASE_WORK/rank_public" \
  --fallback "$BASE_WORK/rank_public/rankvec_predictions.parquet"
# Private 做同樣流程後：
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 export --work-root "$BASE_WORK" \
  --sample data/raw/tests/sample_submission_rank.csv \
  --output submissions/rank/se3hc_rank.csv
```

官方要求 Public＋Private 合併成一份 **800 列**，欄位 **question_id,pred_rank**。export 驗證兩邊使用相同新模型、posterior／rank 一致、ID 與 sample 完全相等，依 sample 順序原子輸出 CRLF CSV。只有使用者手動上傳；正式產檔後另在 RESULTS 登記本地四桶、fallback 與預期 LB（未知）。

如沒有既有 AH＋C work，可以先沿用原 release conversion 與 V2 prepare，再用本入口 `prepare-release --store ... --name rank_public --work ...` 建立相容 cache；缺 H／C 由 extract 補算。此路徑也必須有完整 AH＋C posterior 才能處理 fallback，預設 launcher 使用上方已有完整 AH＋C 的路徑。

## 驗收與產物

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" -m pytest tests/test_se_h_c_b2.py tests/test_se_h_c_b2_slurm.py tests/test_release_pipeline.py tests/test_reproduce_paper_3.py -q
```

涵蓋真實小型 SE3／黑白 pre-action encoder、逐位置 log_softmax／不同盤長均值、抽題與玩家／棋局邊界、不可變資料、凍結 BN buffers、NaN／缺類／錯欄／缺盤拒絕、AdamW／cosine／尾批／精確續跑、早停並列、最終固定 epoch、權重重載、配對 bootstrap、整題 fallback 與 sample 排序合併 CSV。測試只使用合成資料與 CPU；不載入官方權重做新推論，不模擬為 CUDA／KataGo 工程驗收。

三個 Slurm 檔放在本目錄下的 `slurm/`，可隨套件被 Git 追蹤，避免 repo 根目錄 `/slurm/` 忽略規則遺漏入口。模型和分片仍在 outputs 不進 Git。checkpoint 完成後保留 latest 與發布的選定 model，不刪既有來源產物。

2026-11-06 最終工程驗收：上述三個測試檔 **41 項通過（71.21 秒）**；其中本實驗新測試 17 項、既有 release／SE3 回歸 24 項。固定權重與四路舊特徵格式已唯讀核對；真實 metadata 可在記憶體抽出每 k 250 題，共 7,292 盤不同棋局。沒有發布正式題組、執行官方模型新推論、正式頭訓練或提交 GPU 作業；正式四桶／LB 與效能仍未知。
