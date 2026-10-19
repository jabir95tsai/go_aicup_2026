# 第二版棋力辨識計畫：沿用 V1 ResNet 的十級分類

更新日期：2026-10-19。狀態：依使用者要求更新 V2 設計；尚未實作或執行訓練。此版本取代原本的逐樣本錨點 MSE＋間隔懲罰方案。

競賽規則、資料與工作方式以 [CLAUDE.md](CLAUDE.md) 為準。V2 沿用 V1 模型架構與資料流程，將最後線性 head 從 `361→256→1` 改為 `361→256→10`，以 softmax 輸出十級機率，使用 cross entropy 訓練。

## 1. 假設與目的

V1 以各級平均 scalar score 的排序 loss 訓練，再擬合九個切點。V2 改為逐盤面十級分類，直接學習指定玩家的棋力類別機率，並於題目級聚合後依官方得分解碼，不再擬合 scalar 切點。

假設是逐樣本分類監督能改善題目級官方得分；實際增益未知。下列 gate 是待驗證提案，不是預測結果，不能只以 cross entropy 下降判定成功。

## 2. 模型與標籤

沿用 V1 的固定黑／白 18 層輸入、256 channels、20 個 residual blocks、壓縮層與隱藏層：

| 階段 | 操作 | 輸出 |
|---|---|---|
| 輸入 | 8 個歷史黑／白盤面＋當手顏色兩層 | `B×18×19×19` |
| Stem | Conv3×3(18→256)、BN、ReLU | `B×256×19×19` |
| 主幹 | 20 個 V1 residual blocks | `B×256×19×19` |
| 壓縮 | Conv1×1(256→1)、BN、ReLU | `B×1×19×19` |
| 展平 | Flatten | `B×361` |
| 隱藏層 | Linear(361→256)、ReLU | `B×256` |
| 分類輸出 | Linear(256→10) | logits `B×10` |
| 機率 | Softmax(dim=-1) | probabilities `B×10` |

Residual block 沿用 V1，相加後不額外加 ReLU。隱藏層仍使用 ReLU；softmax 只作用於最後十個類別的 logits。預設模型參數由 23,748,611 增加至 23,750,924，增加 2,313 個。

類別順序固定為 `D、C、B、A、1D、2D、3D、4D、5D、6D`，標籤編碼為 `0..9`。沿用架構不代表載入已訓練 V1 權重；首輪從頭訓練，預訓練／微調另列實驗。

## 3. Softmax 與 cross entropy

令樣本 i 的 logits 為 z_i，標籤為 y_i：

$$
p_{ir}=\frac{\exp(z_{ir})}{\sum_{c=0}^{9}\exp(z_{ic})},
\qquad
L_i=-\log p_{i,y_i}.
$$

- 訓練以 `CrossEntropyLoss` 接收原始 logits `B×10` 與整數標籤 `B`；它內含數值穩定的 log-softmax，不先把 softmax 機率傳入 loss。
- 推論以 `softmax(logits, dim=-1)` 取得十級機率，符合本版輸出 activation 的設定。
- 每個指定玩家落子後的盤面各自計算 cross entropy；不先平均各 rank 輸出，也不先聚合成盤或題才計算訓練 loss。
- 使用標準、無類別權重的 cross entropy；label smoothing=0。不混入 V1 排序 loss、MSE、錨點或間隔懲罰。
- loss 為全域有效樣本的平均，各級依實際樣本數計入，不改為各級等權平均。

## 4. 資料、訓練與多卡計算

沿用目前 V1 的盤面快取、指定玩家逐手索引、歷史組合、D4、每級 80,000／20,000 盤切分與每 seed 獨立切分流程。模型不接收 ID。按盤切分仍可能有玩家重疊，須揭露重疊率，不能據此宣稱未見玩家泛化。

首輪固定 seed 2027、global batch size=65,536、1 epoch，optimizer、scheduler、BN／SyncBN、精度、資料排列及增強設定與目前 V1 對照一致。完整設定以對應 run 的 config 保存。

多卡沿用尾批有效樣本 mask，補齊樣本不計入 loss。以 FP32 計算 cross entropy 與累積統計；全域 loss 為各卡有效 loss 總和除以全域有效樣本數。各卡有效數量不一致時，須按數量加權並核對 DDP 梯度平均尺度，不直接平均局部平均 loss。此 loss 不要求每批包含全部十級。

### 訓練中記錄（使用者已確認）

沿用 V1 實際程式的 loss 來源，每 100 optimizer steps 與最後一步記錄：

- training loss：最近一次訓練 batch 的逐樣本平均 cross entropy，不另做完整 training 推論。
- validation loss：每個 seed 固定、每級無放回抽取 10,000 個 validation 盤面，合計 100,000 個，以 eval 模式凍結 BN、關閉 D4 計算平均 cross entropy；不在每次記錄時重新抽樣。
- validation 官方分數：使用同一時點模型的分類結果計算，正確為 1、差一級為 1/e、其他為 0，與兩個 loss 一起按 step 保存。分數的單位（單盤／多盤題目）與中途評分集合範圍待釐清，不自行以逐手分數代替。單盤評分須涵蓋被評分棋局的全部指定玩家落子；多盤評分須涵蓋被評分題目的全部棋局，不能直接用上述零散 100,000 個盤面當作完整棋局或題目。

訓練 loss 與 validation loss 的資料範圍不同，記錄時保留樣本數與來源；取消原文每 100 steps 完整推論 training／validation 的要求。中途官方分數所需的額外推論量，須在短測時計入時間預算。

### 全部訓練結束（使用者已確認）

完成所有設定 steps 後，以最終模型完整推論 validation，將所有 validation 棋局的分類結果保存為 `outputs/v2_strength/validation_catagory.npy`。每盤記錄 game_index、真實 rank_id、指定玩家手數、十類 confidence 總和與預測 rank_id，對齊 validation 棋局清單；單盤類別依第 5 節的 confidence 總和 argmax 決定。不能只保存最後一個 batch 或中途抽樣集合的結果。

V2 不需用 training 推論擬合切點，因此不要求最終完整推論 training，也不要求另存 train／valid 逐手機率。推論時可按棋局累加 confidence 與手數，保留每盤十類資訊供後續多盤等權聚合。

## 5. 單盤分類與題目級聚合

### 單盤分類（使用者已確認）

每手 confidence 定義為該手 logits 經 softmax 後的十類機率。令 T_g 為棋局 g 中指定玩家的所有落子樣本，對每個級別 r 累加：

$$
C_{g,r}=\sum_{t\in T_g}p_{t,r},
\qquad
\hat y_g=\operatorname{argmax}_{r\in\{0,\ldots,9\}}C_{g,r}.
$$

- 該盤所有指定玩家落子都參與，不只取部分手；對手落子保留作為歷史輸入，但不另納入 confidence 累加。
- 直接選 confidence 總和最大的級別，不先將每手轉成類別投票，也不累加 logits 或 log-prob。
- 同分時固定選較小 rank_id。單盤分類直接使用上述 argmax，不加入相鄰級官方得分加權。
- 同一盤每個類別都除以相同手數，因此總和與平均的 argmax 完全相同。

### 多盤題目（每盤等權聚合已確認）

每盤保留完整的十類 confidence 總和，除以該盤指定玩家手數，得到每盤機率。將同一玩家、同色的 k 盤機率等權平均，得到題目機率 p，避免長局取得較高題內權重；不先把單盤類別轉成硬投票。

使用者已確認「先算出每盤再算等權」：先算每盤的十類機率向量，再以每盤 1/k 的權重合併：

$$
q_{g,r}=\frac{C_{g,r}}{|T_g|},
\qquad
p_r=\frac{1}{k}\sum_{g=1}^{k}q_{g,r}.
$$

每盤先除以各自手數，確保長短局等權；保留十類機率再聚合，不平均 rank_id。


題目級仍依官方指標，對候選級別 r 計算 `p[r] + (p[r-1] + p[r+1]) / e`，邊界外機率視為零，取最大者（`metrics.score_rank` 解碼；同分時固定選較小 rank_id）。每盤等權聚合已確認；此處相鄰級加權的最終解碼仍為待釐清提案，與上方已確認的單盤分類規則分開。

V2 不使用 scalar 平均、固定錨點或九個切點，也不以 validation 標籤擬合任何解碼參數。單盤報告使用 confidence 總和 argmax；多盤主要比較使用題目級官方得分解碼。

## 6. 受控比較與 gate

| 組別 | 模型與目標 | 解碼 |
|---|---|---|
| A：V1 | `361→256→1`，級別平均排序 loss | 只以 training 每盤 score 等權擬合九個切點 |
| B：V2 | `361→256→10`，逐樣本 cross entropy | 逐手機率→每盤→題目平均，再依官方得分解碼 |

兩組使用相同 seed、切分、索引、validation 題目、訓練長度與訓練設定；可共用部分的初始參數保持一致，最後輸出層因形狀不同各自初始化並記錄。只有設定與資料來源一致時才能重用 V1 結果。此比較評估整套分類方案，不能將增益單獨歸因於 head、loss 或解碼。

必須回報 k=5／10／15／20 的官方分數及四桶等權平均，另報 Accuracy、MAE、混淆矩陣、cross entropy 與預測類別分布。V1／V2 的訓練 loss 定義不同，不直接比較 loss 數值。

起始 gate 提案：V2 相對 V1 的四桶等權官方分數增加至少 0.005，且各桶退步不超過 0.005；通過後以 2028、2029 做同 seed 配對確認，三 seed 平均增益至少 0.005 且至少兩個 seed 改善，才視為替換成功。這些數字是判定門檻，不代表統計顯著性；不得只選最好 seed 報告。

## 7. 算力、驗收與輸出

首輪新增 V2 一次完整訓練，為 1 個 run-epoch；V1 若不能重用則共 2 個。三 seed 的 V2 共 3 個 run-epochs，若 V1 均須重跑則最多共 6 個。主幹相同，分類 head 增加 2,313 個參數。逐手輸出由 1 個值增為 10 個，但正式產物只保存每盤聚合結果，不保存全量逐手機率，因此不再以逐手機率約 4 GB／seed 作為必要輸出預算。實際吞吐量、全量評估耗時與 GPU 小時未知，須先以 Nano4 短測估算，正式長作業前回報時間與成本。

未來實作需驗收：輸出 shape 為 `B×10`；softmax 各列和為 1；全零 logits 的 cross entropy 約為 `log(10)=2.302585`；提高正確類別 logit 時 loss 下降；類別順序正確；含尾批 mask 的單／多卡 loss 與梯度一致；單盤 confidence 總和 argmax、每盤正規化後的題內等權聚合及題目邊界級官方解碼正確；checkpoint 可續跑。

### 已確認的輸出位置與檔案

正式輸出目錄為 `outputs/v2_strength/`，成功完成後保留：

| 檔案 | 內容與格式 |
|---|---|
| `model.pt` | 最終模型權重，內嵌模型／訓練設定、seed、類別順序、資料／題目雜湊、驗證報告與聚合規則；不保存 scalar 切點 |
| `losses.npy` | 數值陣列，每列 `[step, train_loss, validation_loss, validation_official_score]`，每 100 steps 與最後一步更新；欄位定義及評分集合資訊內嵌 `model.pt` |
| `validation_catagory.npy` | 完整 validation 每盤分類結果，以不需 pickle 的 structured array 保存 `game_index`、`rank_id`、`num_moves`、`confidence_sum[10]`、`predicted_rank_id`；按 validation 棋局清單順序排列 |

`validation_catagory.npy` 的拼字依使用者指定保留。每盤平均機率可由 `confidence_sum / num_moves` 還原，支援多盤等權聚合，不必重新產生盤面或保存逐手機率。訓練期間可保存暫時續跑 checkpoint；最終三個檔案完整驗收後才清理暫存。其他 seed 的受控比較須使用各自獨立輸出目錄，不覆寫此目錄既有結果；額外目錄命名於開跑前記錄。

本次僅修改計畫，不修改 V1 模型、訓練程式或 config，不執行轉檔與訓練；尚未產生上述模型與結果檔案。V2 產物不覆寫 V1。

## 8. 最終評估流程（本次規劃，待實作）

### 8.1 評估入口與輸入

新增獨立入口 `src/go_aicup/tasks/strength/evaluate_rank_v2.py`，規劃命令：

```bash
python -m go_aicup.tasks.strength.evaluate_rank_v2 --run-dir outputs/v2_strength
```

這是預定介面，目前尚無此程式。預設 CPU 評估，不啟動訓練、不重新解析 SGF、不重新產生盤面，也不重新執行模型 forward。輸入為三個正式產物，加上對應 seed 已保存的 metadata、validation 棋局清單與 `strength_v1_validation_questions.parquet`；資料位置由 `model.pt` 內嵌 config 取得，允許以命令列覆寫搬移後的資料目錄，但內容雜湊必須一致。

### 8.2 完整性檢查

評分前必須通過以下檢查，缺漏不得靜默跳過：

- 模型版本、seed、十類順序、資料與固定題目雜湊一致，確認產物來自完成全部設定 steps 的最終模型。
- `validation_catagory.npy` 的 game_index 唯一、順序正確，恰好涵蓋全部 validation 棋局，不含 training 棋局；rank_id 與 metadata 一致。
- 每盤 num_moves 為正，且等於既有索引／metadata 的指定玩家實際手數；不能把對手落子或 DDP 補齊樣本算入。
- confidence_sum 為有限非負十維數值，每盤十類總和在浮點容許誤差內等於 num_moves。除以 num_moves 後機率和約為 1；超出容許誤差時報錯，不任意修補。
- 保存的 predicted_rank_id 等於 confidence_sum 的 argmax，同分依較小 rank_id；類別值皆在 0..9。
- 固定 validation 題目每題包含 k 個不同棋局，全部屬於 validation、同玩家、同顏色、同真實 rank，且 k 屬於 5／10／15／20。不同題目可以重用棋局，須記錄此限制，不視為獨立玩家樣本。
- `losses.npy` 為四欄，step 遞增且不重複，包含最後一步；所有已記錄值有限。中途 validation 官方分數的集合與單位必須有對應 metadata，不能與最終全量成績混用。

### 8.3 單盤評估

使用全部 validation 棋局，每盤等權，依已確認的 confidence 總和 argmax 分類。報告：

| 報表 | 內容 |
|---|---|
| 整體 | 棋局數、官方計分平均、Accuracy、MAE、差一級比例、差兩級以上比例 |
| 每級 | D 至 6D 各級棋局數、官方計分平均、Accuracy、MAE |
| 混淆矩陣 | 固定 10×10，列為真實級別、欄為預測級別；提供計數與逐列比例 |
| 分布 | 真實及預測各級棋局數，黑／白分組的樣本數與官方計分平均 |

單盤官方計分是將競賽的計分公式套用到每盤，用作診斷；不能冒稱為正式多盤題目成績。令誤差 d=abs(predicted_rank_id-rank_id)，單筆得分為 d=0 時 1、d=1 時 1/e、其他 0。

### 8.4 多盤題目評估

直接重用 V1 已保存的固定 validation 題目，與 V1 比較時不重新抽題。由 `confidence_sum / num_moves` 重建每盤十類機率，再依第 5 節每盤等權平均成題目機率。

最終解碼尚待確認，評估程式先規劃同時提供兩份明確命名的報告，不依 validation 分數自動選較高者：

- `argmax`：直接取題目機率最大的類別。
- `score_rank`：使用 `metrics.score_rank` 的相鄰級加權解碼。

兩份皆回報 k=5／10／15／20 的題數、官方分數、Accuracy、MAE、within1，以及各桶混淆矩陣；主要彙總為四桶官方分數等權平均。任一桶為空則不能產出完整四桶結論。`rank_report` 的 all 列為所有題目直接平均，桶大小不同時不可拿它取代四桶等權平均。對應第 6 節 gate 的主要解碼，須於正式實驗前固定；不事後挑選。

### 8.5 Loss 與圖表

讀取 `losses.npy`，分別呈現 training batch CE、固定抽樣 validation CE，以及中途 validation 官方分數的 step 曲線，標明各自資料範圍。不能將 training batch loss 說成完整 training loss，也不能將抽樣 validation loss 說成完整 validation loss。

最終完整 validation 的逐手 CE 可於最後一次模型推論時，以原始 logits 串流累加 loss sum／count，將結果內嵌 `model.pt` 的最終報告。之後 CPU 評估讀取這個值；不能從每盤平均機率還原逐手 CE。若另算每盤／題目的負對數真實類別機率，須標為 game／question NLL，與逐手 CE 分開。

先以終端表格輸出單盤報表、兩種題目解碼的分 k 報表與完整性檢查結果。評估函式回傳結構化 report，供訓練結束時內嵌 `model.pt`；獨立重跑預設只讀三個正式檔案，不修改權重或分類結果，不要求新增正式產物。圖表可先顯示；只有明確指定 export 目錄時才額外輸出 loss／score 曲線與混淆矩陣圖。

### 8.6 與 V1 比較及驗收

只有相同 seed、棋局切分及固定題目雜湊的 V1 結果才做配對比較；列出 V1、V2、差值，各 k 桶與四桶等權平均都需呈現。缺少相符 V1 結果時標示未比較，不補估分數。保留 training／validation 玩家重疊率，說明此切分的泛化限制。正式實驗結果依專案規範登記於 `RESULTS.md`，目前不填入虛構成績。

驗收應涵蓋：同一保存產物重跑結果一致；單盤總和與平均 argmax 相同；長短局在題內各占 1/k；不平均 rank_id；正確／差一級／差兩級的計分正確；D 與 6D 解碼邊界正確；四桶等權彙總正確；不完整棋局、重複索引、錯誤題目成員或雜湊不符時明確失敗。CPU 後處理僅處理每盤十類向量，實際耗時待測；不需要再次執行全量 CNN 推論。

## 9. 評估視覺化規劃（待實作）

### 9.1 呈現順序與圖表

視覺化使用 Matplotlib 產生靜態評估圖，直接讀取三個正式產物與對應 metadata／固定題目，不重新推論模型。依下列順序閱讀：先檢查訓練變化，再看單盤錯誤分布，最後看多盤是否改善。

| 圖表 | 版面與座標 | 要回答的問題 |
|---|---|---|
| 訓練曲線 `training_curves` | 上圖：x=optimizer step、y=cross entropy，並列 training batch／固定抽樣 validation loss；下圖：x=step、y=validation 官方分數（0..1） | loss 是否下降、validation 成績是否跟著改善？ |
| 單盤混淆矩陣 `game_confusion` | 左圖為 10×10 棋局數，右圖為真實級別逐列正規化比例；x=預測級別、y=真實級別 | 哪些棋力互相混淆？錯誤是否集中在相鄰級？ |
| 各級單盤成績 `game_by_rank` | 上圖各級 Accuracy／官方分數分組柱狀圖（0..1），下圖各級 MAE；x 固定為 D 至 6D，各級標示棋局數 | 是否只對少數級別有效？兩端級別是否特別差？ |
| 平均 confidence 熱圖 `game_confidence` | 10×10；x=預測類別、y=真實類別；每列為該真實級別所有棋局平均的十類機率 | 除了最終類別，模型是否仍將機率放在正確級別附近？ |
| 多盤成績 `question_by_k` | 上圖為 k=5／10／15／20 與四桶等權平均的官方分數（0..1），下圖為各 k 的 Accuracy／MAE（分開子圖）；argmax／score_rank 分開標示，附題數 | 增加盤數後是否改善？兩種解碼的差異在哪一桶？ |
| 多盤混淆矩陣 `question_confusion` | 每種解碼各一張 2×2 圖，依序對應 k=5／10／15／20；每格為逐列正規化 10×10 矩陣，使用相同 0..1 色階 | 多盤是否減少跨級錯誤，哪些級別仍難區分？ |

若有雜湊相符的 V1 評估結果，再增加 `v1_v2_comparison`：左圖並列各 k 與四桶等權平均的 V1／V2 官方分數，右圖呈現 V2−V1 差值，畫出零線與 +0.005 gate 參考線。只比較相同定義的評估成績，不疊加 V1 排序 loss 與 V2 cross entropy。

### 9.2 視覺與計算規則

- 類別順序全圖固定為 `D、C、B、A、1D、2D、3D、4D、5D、6D`，混淆矩陣的列／欄方向固定，不依成績排序。
- 每張圖標明 seed、單盤／題目、資料範圍、樣本數及解碼方式；中途抽樣分數與最終完整 validation 不串成同一條曲線。中途分數範圍尚待確認，實作時以保存的 metadata 為準。
- training 曲線呈現原始記錄點，不用平滑掩蓋波動；loss 與官方分數使用不同子圖，不採雙 y 軸。標示最終 step，最後一步不是 100 的倍數也保留。
- 官方分數、Accuracy 與比例圖使用 0..1；MAE 與 CE 各用自己的座標，不與官方分數共用 y 軸。混淆矩陣比例以真實級別逐列正規化，零樣本列標為 N/A。
- confidence 熱圖先將每盤 confidence_sum 除以 num_moves，再在真實級別內按盤等權平均；不能直接把全部手加總，避免長局影響較大。此圖呈現模型機率分布，不宣稱機率已校準。
- 多盤兩種解碼使用不同顏色與圖例，圖名不稱任何一種為已確認的最終方法；四桶等權平均與單桶成績清楚區分，不以不同樣本數加權。
- 不從單一 seed 或重用棋局的合成題目繪製假定獨立樣本的顯著性／誤差棒。若完成三 seed，比較圖可另呈現每個 seed 點與三 seed 平均，不把它當作已證實的統計顯著性。

### 9.3 預定使用介面與輸出

評估入口預留 `--show` 與 `--export-dir`，屬待實作介面：

```bash
# 本機顯示圖表
python -m go_aicup.tasks.strength.evaluate_rank_v2 --run-dir outputs/v2_strength --show

# 無圖形介面的 Nano4 或報告匯出
python -m go_aicup.tasks.strength.evaluate_rank_v2 --run-dir outputs/v2_strength --export-dir outputs/v2_strength/figures
```

預設只輸出評估表格；`--show` 顯示視窗，指定 `--export-dir` 才產生額外檔案。匯出每張圖的 PNG（200 dpi）與 PDF，檔名使用上表名稱，兩種題目解碼加上 `_argmax`／`_score_rank` 後綴。匯出功能於無 GUI 環境使用非互動 backend，不需要 GPU。未指定匯出時，正式目錄仍只要求 `model.pt`、`losses.npy`、`validation_catagory.npy`。

所有圖表可由保存的產物重新建立；缺少相符 V1 資料就略過比較圖並明確說明，不填入虛構數據。本次僅新增視覺化規格，尚未實作繪圖程式或產生結果圖。
