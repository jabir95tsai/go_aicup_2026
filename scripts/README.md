# Scripts 目錄索引

一次性報告、實驗及命令列入口依用途分類；共用程式仍放在 `src/go_aicup/`。

| 目錄 | 用途 | 檔案數 |
|---|---|---:|
| `data/` | 訓練資料稽核、HD 擷取與語意檢查、EDA | 4 |
| `environment/` | CUDA／NCCL／DDP 環境檢查 | 1 |
| `identity/` | 競賽一：開局、GE2E、HD、壓力測試及保險模型評估 | 6 |
| `strength/` | 競賽二：V1／V2／SE 評估、HumanSL／C／B2、校準與提交檔匯出 | 32 |
| `release/` | 兩賽共同的假資料彩排、Day-0 稽核、發布流程與摘要 | 7 |

## 執行方式

先切到專案根目錄並啟用 `go` 環境。Python 腳本使用完整模組名稱執行，讓跨目錄匯入一致；尚未以 `pip install -e .` 安裝專案時，加上 `PYTHONPATH=src`。

```bash
python -m scripts.data.audit_train --limit 2000
python -m scripts.identity.identity_ge2e_eval --run-dir outputs/pid_ge2e_v1_short
python -m scripts.strength.rank_more_players local-ex
python -m scripts.release.day0_audit --train-self-check
bash scripts/release/run_release.sh <raw-dir> <out-dir>
```

`environment/check_torch_ddp.py` 仍由 `torchrun` 或 `python -m torch.distributed.run` 啟動，詳見專案 README。Slurm 作業入口集中於根目錄 `slurm/` 及各任務套件內的 `slurm/`。

舊的 `scripts/<name>.py` 指令改用 `python -m scripts.<分類>.<name>`；歷史實驗帳本 `RESULTS_ARCHIVE.md` 保留當時路徑。
