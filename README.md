# AI CUP 2026 圍棋競賽

兩個競賽共用一份訓練資料、一個 repo：**競賽一 玩家辨識**（`tasks/identity/`）、**競賽二 棋力預測**（`tasks/strength/`）。

| 文件 | 內容 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **唯一事實來源**：規則、評分、時程、資料稽核結果、CV 設計、洩漏清單、上傳紀律、專案結構 |
| [RESULTS.md](RESULTS.md) | 實驗帳本，只增不刪，失敗也記 |
| [STRATEGY.md](STRATEGY.md) / [EXECUTION_PLAN.md](EXECUTION_PLAN.md) | 決勝槓桿與 8 週計畫 |
| [reference.md](reference.md) | 文獻與工具整理 |
| [EXTERNAL_RESOURCES.md](EXTERNAL_RESOURCES.md) | 外部資源申報清單（報告必備） |
| [AGENTS.md](AGENTS.md) | 只指向 CLAUDE.md（給 Codex 等 agent） |

## 目前進度

W1 完成：SGF 稽核、1M 盤轉檔、CV 切分與合成題組、兩賽官方指標、訊號 EDA（細節見 `RESULTS.md` A-001～A-006）。下一步是兩賽的 P0 baseline。

## 上手

```bash
pip install -r requirements.txt     # 版本已鎖，報告要用
pip install -e .                    # 讓 go_aicup 可 import
python -m pytest -q                 # 應全數通過
```

**資料**（不進 git）：從 [官方 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) 下載 `aicup2026_go_training.tar.xz`，核對 SHA-256（見 CLAUDE.md 第 1 節），解壓出的 `training/` 放到 `data/training/`。11/04 發布的候選庫與測試題放 `data/raw/`。

**重建衍生資料**（全 CPU，依序執行，結果是確定性的，兩人重建出的檔案會逐位元相同）：

```bash
python -m go_aicup.data.preprocess train         # SGF → int16 memmap，約 2.5 分鐘
python -m go_aicup.data.cv                       # CV 切分 + 合成題組，數秒
python -m go_aicup.features.game_stats           # EDA 用的盤面統計，約 10 秒
python scripts/eda_report.py                     # 重產 notebooks/02_eda.md，約 2 分鐘
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
