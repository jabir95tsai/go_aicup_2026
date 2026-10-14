"""W1 signal EDA: where the rank / player signal lives, and how it scales with k.

Reads the processed train store, the per-game replay statistics written by
`game_stats.py`, and the CV tables from `cv.py`. Writes
`notebooks/02_eda.md` plus figures under `notebooks/figures/`.

Nothing here trains a submission model. The two probes (a linear rank probe
and an opening naive-Bayes player-id probe) exist only to put a number on
"how much signal is there"; they respect the CV rules (player-disjoint rank
folds, gallery/query-disjoint pid splits).

Usage:
    python -m go_aicup.features.game_stats      # once, ~10 s
    python scripts/eda_report.py
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from numba import njit  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from go_aicup.data import cv  # noqa: E402
from go_aicup.data.store import GameStore  # noqa: E402
from go_aicup.metrics import K_BUCKETS, pid_report, rank_report  # noqa: E402
from go_aicup.paths import ROOT  # noqa: E402

STATS = ROOT / "data" / "processed" / "eda" / "game_stats_train.parquet"
REPORT = ROOT / "notebooks" / "02_eda.md"
FIG_DIR = ROOT / "notebooks" / "figures"

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
SIZE = 19
N_POINTS = SIZE * SIZE
PREFIX = 40          # plies kept for opening analyses
PHASES = (0, 50, 150)  # ply boundaries: opening / middle game / late game
PHASE_NAMES = ["開局 (1–50 手)", "中盤 (51–150 手)", "後盤 (151 手–)"]

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
           "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # D .. 6D
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a98", "#b82f2e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10,
    "font.family": ["Microsoft JhengHei", "DejaVu Sans"],  # DejaVu covers ≥ ≤
    "axes.unicode_minus": False,
})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    return df.round(digits).to_markdown(index=index)


def save_fig(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"![{name}](figures/{name})"


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


FEATURE_LABELS = {
    "own_adj": "貼著自己的子下（連接/長）",
    "d_opp": "與對手上一手的距離",
    "d_own": "與自己上一手的距離",
    "local": "局部應手率（≤2 路）",
    "tenuki": "脫先率（≥6 路）",
    "cap_single": "單子提（打劫類）",
    "refill": "下在曾被提過的點",
    "line1": "一線手比例",
    "cap_rate": "提子手比例",
    "libs": "落子後本塊氣數",
    "cap_big": "大提子次數（≥5 子）",
    "line3": "三線手比例",
    "early_low": "開局 60 手內 1–2 線比例",
    "atari": "叫吃率",
    "low_libs": "落子後 ≤2 氣比例",
    "contact": "接觸戰比例",
    "line4": "四線手比例",
    "self_atari": "自緊氣（送叫吃）",
    "cap_stones": "每手提子數",
    "escape": "逃被叫吃的子",
    "line5p": "五線以上比例",
    "line2": "二線手比例",
}


def label(col: str) -> str:
    if col in ("length", "me_last", "stone_share_end", "me_lost_stones"):
        return {"length": "棋局手數", "me_last": "本人下最後一手",
                "stone_share_end": "終局盤上子數佔比",
                "me_lost_stones": "被提子數/手"}[col]
    who, base = col.split("_", 1)
    return f"{'本人' if who == 'me' else '對手'}：{FEATURE_LABELS.get(base, base)}"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class Data:
    def __init__(self) -> None:
        self.store = GameStore("train")
        self.meta = self.store.meta
        self.stats = pd.read_parquet(STATS)
        self.rank = self.meta["rank_id"].to_numpy(np.int64)
        self.color = self.meta["color"].astype(str).to_numpy()
        self.pid = self.meta["player_id"].astype(str).to_numpy()
        self.features = list(self.stats.columns)
        offsets = self.meta["offset"].to_numpy(np.int64)
        self.prefix = np.asarray(self.store.moves)[offsets[:, None] + np.arange(PREFIX)]
        self.prefix_d4 = canonical_d4(self.prefix)


def d4_tables() -> np.ndarray:
    """(8, 361) point maps for the 8 board symmetries."""
    r, c = np.divmod(np.arange(N_POINTS), SIZE)
    s = SIZE - 1
    maps = [(r, c), (c, r), (s - r, c), (r, s - c), (s - r, s - c),
            (c, s - r), (s - c, r), (s - c, s - r)]
    return np.stack([rr * SIZE + cc for rr, cc in maps]).astype(np.int16)


def canonical_d4(prefix: np.ndarray, key_plies: int = 6) -> np.ndarray:
    """Map every game into the symmetry that makes its first plies lexicographically smallest."""
    tables = d4_tables()
    keys = np.zeros((8, len(prefix)), dtype=np.int64)
    for t in range(8):
        mapped = tables[t][prefix[:, :key_plies]].astype(np.int64)
        for j in range(key_plies):
            keys[t] = keys[t] * N_POINTS + mapped[:, j]
    best = keys.argmin(axis=0)
    out = np.empty_like(prefix)
    for t in range(8):
        rows = best == t
        out[rows] = tables[t][prefix[rows]]
    return out


# --------------------------------------------------------------------------
# section 1 -- length, ending, colour
# --------------------------------------------------------------------------

def section_length(d: Data) -> str:
    s = d.stats.assign(rank=d.rank, color=d.color)
    q = s.groupby("rank")["length"].describe(percentiles=[.1, .25, .5, .75, .9])
    q.index = RANKS
    table = q[["mean", "10%", "25%", "50%", "75%", "90%"]]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(10)
    ax.fill_between(x, q["25%"], q["75%"], color=BLUE, alpha=0.18, linewidth=0,
                    label="四分位距 (p25–p75)")
    ax.plot(x, q["50%"], color=BLUE, marker="o", markersize=5, label="中位數")
    ax.set_xticks(x, RANKS)
    ax.set_ylabel("手數")
    ax.set_title("棋局長度 vs 棋力", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig_md = save_fig(fig, "eda_length_by_rank.png")

    last = s.pivot_table(index="rank", columns="color", values="me_last", aggfunc="mean")
    last.index = RANKS
    last["B-W 差"] = last["B"] - last["W"]
    black_last = ((s["length"] - 1) % 2 == 0).mean()

    return f"""## 1. 棋局長度、終局方式、執色

{fig_md}

{md_table(table, 1)}

- 棋力越高、棋局越短（D 中位數 {q.loc['D', '50%']:.0f} 手 → 6D {q.loc['6D', '50%']:.0f} 手），但**分布幾乎完全重疊**：
  單盤長度對棋力只有 η²≈0.008（見第 2 節）。長度是弱訊號，但它影響 batch 設計：各 rank 的長度分布一致，length bucketing 不會造成 rank 失衡
- **終局方式：** pass 在全量中出現 0 次 → 所有棋局都以中盤認輸 / 超時結束（或 pass 已被主辦方移除）。
  「誰下最後一手」在認輸局中通常是勝方，所以 `me_last` 是**勝負的代理**。
  全體黑方下最後一手的比例 = {black_last:.3f}

「本人下最後一手」的比例（≈ 本人勝率的代理）：

{md_table(last, 3)}

- 各 rank 都在 0.48–0.53，**勝負代理幾乎不帶棋力訊號**（η²=0.0003）。若「最後一手 ≈ 勝方」的假設成立，
  這符合伺服器依棋力配對、勝率收斂 50% 的直覺（這個假設本身無法驗證——勝負已被移除）
- 每個 rank 都是執黑時 `me_last` 略高（+0.003～+0.023）
- 同理，終局盤上子數佔比（`stone_share_end`）平均 0.500、η²≈0 → 無用
- 執色：每個 rank 檔前 50,000 列全是黑、後 50,000 列全是白（`game_id` 1–50000 = B）。
  `game_id` 是打亂過的流水號：重度玩家的盤散佈在整個 id 範圍（span 中位數 99%），
  依 id 排序後相鄰兩盤同玩家的機率 = 隨機期望（D：0.0035 vs 0.0032）。
  → **`game_id` 沒有時間順序資訊，「玩家風格是否隨時間漂移」無法從資料驗證（未知）**
"""


# --------------------------------------------------------------------------
# section 2 -- univariate signal + variance decomposition
# --------------------------------------------------------------------------

def variance_table(d: Data, mask: np.ndarray | None = None) -> pd.DataFrame:
    s = d.stats if mask is None else d.stats[mask]
    rank = d.rank if mask is None else d.rank[mask]
    pid = d.pid if mask is None else d.pid[mask]
    rows = []
    groups_rank = pd.Series(rank)
    groups_pid = pd.Series(pid)
    for col in d.features:
        x = s[col].to_numpy(np.float64)
        total = x.var()
        if total == 0:
            continue
        rank_mean = pd.Series(x).groupby(groups_rank.values).transform("mean").to_numpy()
        player_mean = pd.Series(x).groupby(groups_pid.values).transform("mean").to_numpy()
        pm = pd.Series(x).groupby(groups_pid.values).mean()
        pr = groups_rank.groupby(groups_pid.values).first()
        rows.append({
            "feature": col,
            "rank η²": ((rank_mean - x.mean()) ** 2).mean() / total,
            "玩家(同rank內)": ((player_mean - rank_mean) ** 2).mean() / total,
            "盤內雜訊": ((x - player_mean) ** 2).mean() / total,
            "ρ(玩家均值,rank)": spearmanr(pm.to_numpy(), pr.loc[pm.index].to_numpy()).correlation,
        })
    return pd.DataFrame(rows).set_index("feature")


def section_univariate(d: Data) -> tuple[str, pd.DataFrame]:
    var = variance_table(d)
    var = var.sort_values("rank η²", ascending=False)
    per_rank = d.stats.groupby(d.rank).mean().T
    per_rank.columns = RANKS
    show = var.join(per_rank[["D", "A", "3D", "6D"]])
    show.insert(0, "說明", [label(c) for c in show.index])

    top = [c for c in var.index if c.startswith("me_")][:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    x = np.arange(10)
    for ax, col in zip(axes.flat, top):
        base = col[3:]
        for tag, color, name in (("me", BLUE, "本人"), ("op", ORANGE, "對手")):
            y = per_rank.loc[f"{tag}_{base}"].to_numpy()
            ax.plot(x, y, color=color, marker="o", markersize=4, label=name)
        ax.set_title(FEATURE_LABELS.get(base, base), fontsize=10, loc="left")
        ax.set_xticks(x, RANKS, fontsize=8)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("各 rank 的平均值：本人（藍）與對手（橙）幾乎重合", x=0.01, ha="left")
    fig.tight_layout()
    fig_md = save_fig(fig, "eda_feature_trends.png")

    best = var.index[0]
    txt = f"""## 2. 單盤特徵的棋力訊號與變異分解

用 numba 盤面引擎重播全量 1M 盤（2.2 億手，7.5 秒），每盤對**本人**與**對手**各算 26 種統計
（`src/go_aicup/features/game_stats.py`）。每個特徵的總變異拆成三塊：

- **rank η²**：rank 之間的差異（=棋力訊號）
- **玩家(同rank內)**：同一 rank 內、不同玩家之間的差異（=個人風格指紋；競賽一要的、競賽二要避免記住的）
- **盤內雜訊**：同一玩家不同盤之間的差異（靠 k 盤平均可以壓掉）

{fig_md}

{md_table(show, 4)}

**讀法與結論：**

1. **單盤的手工特徵訊號很弱。** 最強的 `{best}` 也只有 η²={var.iloc[0]['rank η²']:.3f}，
   85–95% 的變異是盤內雜訊。單盤判棋力（k=1）用這類特徵會很差，**k 盤聚合是必需品，不是選配**
2. **棋力的方向（前 10 名特徵在 10 個 rank 上大致單調；長度等弱特徵有局部反轉）：** 越強 → 越少「貼著自己的子下」（解讀：少愚形／少無謂的長）、
   與前一手的距離越大、脫先越多、局部應手越少、一線手越少、**單子提（打劫）越多、大塊被提越少**、棋局越短
3. **對手特徵的棋力訊號與本人一樣強**（例：對手 `d_opp` η²={var.loc['op_d_opp', 'rank η²']:.3f} vs 本人
   {var.loc['me_d_opp', 'rank η²']:.3f}；對手 `local` {var.loc['op_local', 'rank η²']:.3f} vs 本人 {var.loc['me_local', 'rank η²']:.3f}），
   而且在距離／應手／氣數類特徵上，**對手特徵的「玩家」成分明顯更低**（例：`d_opp` 0.063 vs 0.151）。→ **L2 直接獲得資料支持**，見第 4、5 節
4. **本人特徵的玩家成分是 rank 成分的 1.5–5 倍**（`own_adj` 1.5 倍、`d_opp` {var.loc['me_d_opp', '玩家(同rank內)'] / var.loc['me_d_opp', 'rank η²']:.1f} 倍、
   `local` {var.loc['me_local', '玩家(同rank內)'] / var.loc['me_local', 'rank η²']:.1f} 倍；接觸戰比例幾乎全是玩家成分）。同 rank 內玩家之間的風格差異比 rank 之間的差異還大
   → 這就是 A5「記住玩家」風險的量化：模型最省力的解法是認人，不是判棋力
5. 勝負代理（`me_last`）、終局子數、自緊氣、接觸戰比例 ≈ 無棋力訊號
"""
    return txt, var


# --------------------------------------------------------------------------
# section 3 -- k scaling
# --------------------------------------------------------------------------

def load_rank_questions() -> pd.DataFrame:
    parts = []
    for name in ["f0", "f1", "f2", "f3", "f4", "holdout"]:
        q = cv.load(f"rank_questions_{name}")
        q["fold"] = name
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def question_means(d: Data, questions: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = d.stats[cols].to_numpy(np.float32)
    out = np.zeros((len(questions), len(cols)), np.float32)
    for i, games in enumerate(questions["games"]):
        out[i] = x[games].mean(axis=0)
    return out


def section_k(d: Data, var: pd.DataFrame, questions: pd.DataFrame) -> str:
    top = [c for c in var.index][:6]
    qx = question_means(d, questions, top)
    y = questions["rank_id"].to_numpy()
    k = questions["k"].to_numpy()
    rows = {}
    # k = 1: a single random game
    rng = np.random.default_rng(0)
    single = np.array([rng.choice(g) for g in questions["games"]])
    for j, col in enumerate(top):
        vals = {"1": abs(spearmanr(d.stats[col].to_numpy()[single], y).correlation)}
        for b in K_BUCKETS:
            m = k == b
            vals[str(b)] = abs(spearmanr(qx[m, j], y[m]).correlation)
        r2, p2, w2 = var.loc[col, ["rank η²", "玩家(同rank內)", "盤內雜訊"]]
        vals["理論上限 k→∞"] = np.sqrt(r2 / (r2 + p2))
        rows[col] = vals
    table = pd.DataFrame(rows).T
    table.insert(0, "說明", [label(c) for c in table.index])

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    xs = [1, 5, 10, 15, 20]
    for j, col in enumerate(top[:4]):
        ys = [table.loc[col, str(v)] for v in xs]
        ax.plot(xs, ys, color=SERIES[j], marker="o", markersize=5, label=label(col))
    ax.set_xticks(xs)
    ax.set_xlabel("每題盤數 k")
    ax.set_ylabel("|Spearman ρ|（題均值 vs rank）")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.set_title("單一特徵的棋力相關隨 k 成長、邊際遞減", loc="left")
    fig_md = save_fig(fig, "eda_k_scaling.png")

    return f"""## 3. k 盤聚合能把訊號放大多少

在 `cv.py` 的 54,000 道合成棋力題（f0–f4 + holdout，k 各 25%）上，把每題 k 盤的特徵取平均，
算它與 rank 的 |Spearman ρ|。k=1 是每題隨機抽一盤。「理論上限」= √(rank η² /(rank η²+玩家))，
即 k→∞、盤內雜訊完全消失時的相關比（correlation ratio，與 Spearman 不完全同尺度，當近似參考）。

{fig_md}

{md_table(table, 3)}

**機制：** 題均值的變異 = rank + 玩家 + 盤內雜訊/k。盤內雜訊佔 80–90%，所以 k=1→5 相關約翻倍（例：`me_own_adj` 0.23→0.41）；
之後邊際遞減（5→10 +0.08、10→15 +0.045、15→20 +0.03），因為**玩家成分不會隨 k 下降**。
k=20 時已達上限的約 85–95%。

**對建模的意義：**
- 天花板不是盤數，而是「同 rank 內玩家風格差異」。突破它要靠更好的單手／單盤表徵（L1 per-move 監督、L5 引擎特徵）——
  讓特徵本身的 rank 成分變大、玩家成分變小
- k=5 是四個等權桶中最弱的，也是離上限最遠的 → 更好的單盤表徵在 k=5 的邊際效益最大
"""


def section_probe(d: Data, questions: pd.DataFrame) -> str:
    me = [c for c in d.features if c.startswith("me_")]
    op = [c for c in d.features if c.startswith("op_")]
    variants = {
        "本人特徵": me + ["length"],
        "對手特徵": op + ["length"],
        "本人 + 對手": me + op + ["length"],
    }
    train = questions["fold"].isin(["f0", "f1", "f2", "f3"]).to_numpy()
    test = ~train
    y = questions["rank_id"].to_numpy()
    k = questions["k"].to_numpy()
    blocks = []
    for name, cols in variants.items():
        x = question_means(d, questions, cols)
        # k is a legal input (num_games is given in the test file)
        x = np.column_stack([x, np.log(k)])
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(scaler.transform(x[train]), y[train])
        pred = model.predict(scaler.transform(x[test]))
        rep = rank_report(pred, y[test], k[test])
        rep.insert(0, "variant", name)
        blocks.append(rep.reset_index())
    table = pd.concat(blocks).set_index(["variant", "k"])[["n", "acc", "mae", "within1", "qwk"]]
    chance = 0.1
    return f"""## 4. 線性探針：手工特徵 + k 盤平均能做到多少（競賽二）

**不是提交模型**，只為量化訊號。題級特徵 = k 盤特徵平均（+ log k），多類別 logistic regression；
在 f0–f3 的 40,000 題上訓練，在 f4 + holdout 的 14,000 題上評估（玩家完全不重疊）。
亂猜 acc = {chance:.2f}、MAE ≈ 3.3。

{md_table(table, 3)}

**結論：**
- 26 個手工盤面統計 + 線性模型，就已明顯高於亂猜，而且 k 越大越好 → P0 GBDT 值得做，當作第一個校準點
- **只用對手的手，就能達到與只用本人的手接近的棋力判斷力**（acc 0.249 vs 0.258、QWK 0.715 vs 0.738）。
  這是伺服器依棋力配對的證據：對手的棋力 ≈ 本人的棋力，對手的手是關於本人 rank 的**第二份證據**
  （不完全獨立：同盤雙方特徵相關 0.2–0.9，見第 5 節）
- 本人 + 對手合併在四個 k 桶都優於任一單邊（all：acc +0.018、MAE −0.15、QWK +0.04）→ L2（對手的手也餵進 rank 頭）應列為主力設計
- 準確率只有 0.2–0.3，但 QWK 0.69–0.83、MAE 1.2–1.7：錯誤大多落在附近幾級，ordinal 結構很明顯。
  指標若是 MAE/QWK，ordinal head / 期望值解碼會有明顯好處；若是 accuracy，這條線的天花板很低
- k=5 → k=20：acc 0.225→0.313、MAE 1.74→1.23，k 的影響遠大於「本人 vs 對手」的差異
"""


# --------------------------------------------------------------------------
# section 5 -- opponent / matchmaking
# --------------------------------------------------------------------------

def section_opponent(d: Data, var: pd.DataFrame) -> str:
    bases = [c[3:] for c in var.index if c.startswith("me_")][:10]
    rows = []
    rank_series = pd.Series(d.rank)
    for base in bases:
        a = d.stats[f"me_{base}"].to_numpy(np.float64)
        b = d.stats[f"op_{base}"].to_numpy(np.float64)
        a_w = a - pd.Series(a).groupby(rank_series).transform("mean").to_numpy()
        b_w = b - pd.Series(b).groupby(rank_series).transform("mean").to_numpy()
        rows.append({
            "特徵": FEATURE_LABELS.get(base, base),
            "corr(本人,對手) 同盤": np.corrcoef(a, b)[0, 1],
            "扣除 rank 均值後": np.corrcoef(a_w, b_w)[0, 1],
            "本人 rank η²": var.loc[f"me_{base}", "rank η²"],
            "對手 rank η²": var.loc[f"op_{base}", "rank η²"],
            "本人 玩家成分": var.loc[f"me_{base}", "玩家(同rank內)"],
            "對手 玩家成分": var.loc[f"op_{base}", "玩家(同rank內)"],
        })
    table = pd.DataFrame(rows).set_index("特徵")
    return f"""## 5. 對手：同盤雙方高度相關；距離類對手特徵幾乎不帶本人指紋

{md_table(table, 3)}

- 同一盤棋雙方的統計量高度相關，大部分是**棋局本身的性質**（亂戰局雙方都提子多、接觸多）——
  扣掉 rank 均值後相關仍高，說明這不只是「同 rank 配對」造成的
- **距離／應手／氣數類**：對手的玩家成分約為本人的一半以下（`d_opp` 0.063 vs 0.151、`local` 0.059 vs 0.161、`libs` 0.069 vs 0.127）。
  同一位本人的 k 盤對到 k 位不同的對手，對手的風格不會累積成本人的指紋
- **提子類是例外**：對手的提子率玩家成分反而比本人高（0.089 vs 0.050）。機制：「對手提了多少子」=「本人被提了多少子」，
  這是**本人**的屬性（例如某人習慣打到底、大龍常被殺），不是對手的
- **對競賽二的含意：** 對手的非提子類手是比本人的手**更乾淨**的棋力訊號（有棋力、較少身分），是對抗「記住玩家」過擬合的天然工具。
  一個具體設計：per-move rank 頭同時預測雙方的手，loss 對雙方都算
- **對競賽一的含意：** 對手的手是雜訊，不含本人身分 → 玩家辨識的 encoder 只應把本人的手當指紋來源（對手的手只作為盤面脈絡）
"""


# --------------------------------------------------------------------------
# section 6 -- spatial
# --------------------------------------------------------------------------

@njit(cache=True)
def _spatial_counts(moves, offsets, lengths, player_color, rank, phase_edges, out):
    for g in range(offsets.shape[0]):
        off = offsets[g]
        r = rank[g]
        for i in range(lengths[g]):
            p = moves[off + i]
            me = 1 if (i & 1) == player_color[g] else 0
            ph = 0
            if i >= phase_edges[2]:
                ph = 2
            elif i >= phase_edges[1]:
                ph = 1
            out[r, ph, me, p] += 1


def section_spatial(d: Data) -> str:
    counts = np.zeros((10, 3, 2, N_POINTS), np.int64)
    _spatial_counts(np.asarray(d.store.moves), d.meta["offset"].to_numpy(np.int64),
                    d.store.lengths.astype(np.int64), d.store.player_color.astype(np.int64),
                    d.rank, np.array(PHASES, np.int64), counts)
    me = counts[:, :, 1, :].astype(np.float64)
    dens = me / me.sum(axis=2, keepdims=True)

    show = [0, 3, 6, 9]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    vmax = dens[show, 0].max()
    for ax, r in zip(axes, show):
        im = ax.imshow(dens[r, 0].reshape(SIZE, SIZE), cmap=SEQ, vmin=0, vmax=vmax)
        ax.set_title(f"{RANKS[r]}：開局 50 手的落點", fontsize=10, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="落點機率")
    fig_open = save_fig(fig, "eda_heatmap_opening.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ph, ax in enumerate(axes):
        hi = dens[8:10, ph].sum(axis=0) / 2
        lo = dens[0:2, ph].sum(axis=0) / 2
        ratio = np.log2(hi / lo).reshape(SIZE, SIZE)
        lim = np.nanmax(np.abs(ratio))
        lim = min(lim, 1.5)
        im = ax.imshow(ratio, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
        ax.set_title(PHASE_NAMES[ph], fontsize=10, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="log2(5D–6D / D–C)")
    fig.suptitle("強者（5D–6D）相對弱者（D–C）更常下的位置：紅 = 強者偏好、藍 = 弱者偏好", x=0.01,
                 ha="left")
    fig_diff = save_fig(fig, "eda_heatmap_diff.png")

    # absolute orientation of the labelled player's first move (black games)
    first = d.prefix[:, 0]
    black = d.color == "B"
    r, c = np.divmod(first[black].astype(int), SIZE)
    quad = np.where((r < 9) & (c > 9), "右上", np.where((r < 9) & (c < 9), "左上",
                    np.where((r > 9) & (c > 9), "右下", np.where((r > 9) & (c < 9), "左下",
                                                                 "中線/天元"))))
    qt = pd.crosstab(pd.Series(d.rank[black]).map(dict(enumerate(RANKS))), quad,
                     normalize="index")
    qt = qt.loc[RANKS, ["右上", "左上", "右下", "左下", "中線/天元"]]

    return f"""## 6. 空間分布

{fig_open}

{fig_diff}

黑方（本人執黑）第一手落在哪個角（SGF 座標，上方 = row a）：

{md_table(qt, 3)}

- **絕對方位有強烈偏好：** 74–88% 的第一手在右上角。棋理上四個角等價，所以這是**介面慣例／個人習慣**（解讀）。
  → 方位本身帶身分資訊，第 8 節量化它對競賽一值多少
- D、C 的右上比例最低（74%、78%），D 有 19% 在右下；但各 rank 之間不單調，方位作為棋力訊號很弱
- 差異熱圖（log2 比值，色階 ±0.25 ≈ ±19%，差異不大但結構清楚）：
  - **開局 50 手**：強者更常把子下向中腹（中央一圈偏紅），弱者更常下一線（外框深藍）；**天元那一點本身是弱者偏好**（與上表「中線/天元」第一手一致）
  - **中盤**：強者在二、三線偏多，弱者在一線偏多
  - **後盤**：強者在二線與角上偏多（官子），弱者在中腹偏多（解讀：未解決的中腹戰鬥拖到後盤）
  - 這些都是 D4 不變的訊號 → 競賽二用 D4 增強不會丟掉它們
"""


# --------------------------------------------------------------------------
# section 7 -- opening
# --------------------------------------------------------------------------

def uniqueness_depth(prefix: np.ndarray) -> np.ndarray:
    """Plies needed before a game's opening is unique in the dataset (capped at PREFIX+1)."""
    order = np.lexsort(prefix.T[::-1])
    p = prefix[order]
    eq = p[1:] == p[:-1]
    lcp = np.where(eq.all(axis=1), PREFIX, eq.argmin(axis=1))
    left = np.concatenate([[0], lcp])
    right = np.concatenate([lcp, [0]])
    depth = np.maximum(left, right) + 1
    out = np.empty(len(prefix), np.int64)
    out[order] = depth
    return out


def section_opening(d: Data) -> str:
    rows = []
    depth_raw = uniqueness_depth(d.prefix)
    depth_d4 = uniqueness_depth(d.prefix_d4)
    black = d.color == "B"
    for r in range(10):
        m = (d.rank == r) & black
        first = d.prefix[m, 0]
        first_d4 = d.prefix_d4[m, 0]
        two = d.prefix_d4[m, 0].astype(np.int64) * N_POINTS + d.prefix_d4[m, 2]
        rows.append({
            "rank": RANKS[r],
            "H(第一手) bits": entropy_bits(np.bincount(first, minlength=N_POINTS)),
            "H(第一手, D4) bits": entropy_bits(np.bincount(first_d4, minlength=N_POINTS)),
            "H(黑前兩手, D4) bits": entropy_bits(np.unique(two, return_counts=True)[1]),
            "星位/小目佔比(D4)": np.isin(first_d4, star_komoku_points()).mean(),
            "開局唯一化深度 中位數": np.median(depth_d4[d.rank == r]),
            "開局唯一化深度 平均": depth_d4[d.rank == r].mean(),
        })
    table = pd.DataFrame(rows).set_index("rank")
    rho = spearmanr(depth_d4, d.rank).correlation

    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.arange(1, 32)
    for j, (lo, hi, name) in enumerate([(0, 1, "D–C"), (4, 5, "1D–2D"), (8, 9, "5D–6D")]):
        m = (d.rank >= lo) & (d.rank <= hi)
        h = np.bincount(np.minimum(depth_d4[m], 31), minlength=32)[1:32] / m.sum()
        ax.plot(bins, h, color=SERIES[j], label=name)
    ax.set_xlabel("開局在 1M 盤中變成獨一無二所需的手數（D4 正規化後）")
    ax.set_ylabel("盤數比例")
    ax.legend(frameon=False)
    ax.set_title("強者走得更久的「定石」", loc="left")
    fig_md = save_fig(fig, "eda_opening_depth.png")

    return f"""## 7. 開局：熵與「定石深度」

「開局唯一化深度」= 把 1M 盤的前 {PREFIX} 手（D4 正規化，去掉方位差異）字典序排序，
一盤棋需要走到第幾手才與所有其他棋局都不同。越深代表越照著流行定石走。

{fig_md}

{md_table(table, 3)}

- 開局唯一化深度與 rank 的 Spearman ρ = {rho:.3f}（單盤層級），平均從 D 的 9.2 手到 4D–6D 的 11.2–11.4 手：**越強的玩家照流行下法走得越久**。
  這是棋力的「知識」訊號，與第 2 節的「戰鬥風格」訊號互補，且 D4 不變 → 適合競賽二
- 未正規化 vs D4 正規化的第一手熵差距 ≈ 方位帶來的資訊量，約 0.8–1.2 bits（D 1.23、6D 0.77）；對競賽一這是指紋，對競賽二是雜訊
- 第一手是星位／小目的比例隨棋力略降（D 0.949 → 6D 0.919）：高段反而更常下「非主流」第一手
- 開局唯一化深度中位數：raw {np.median(depth_raw):.0f} 手、D4 {np.median(depth_d4):.0f} 手（雙方合計）。一半的棋局在第 10 手左右就與全資料集不同 →
  「整串開局 n-gram」的共享範圍很短；更長的前綴只能當逐手落點處理（見第 8 節探針）
"""


def star_komoku_points() -> np.ndarray:
    pts = []
    for r, c in [(3, 3), (2, 3), (3, 2)]:
        for rr, cc in [(r, c), (r, 18 - c), (18 - r, c), (18 - r, 18 - c)]:
            pts.append(rr * SIZE + cc)
    return np.array(sorted(set(pts)))


# --------------------------------------------------------------------------
# section 8 -- player fingerprint (competition 1)
# --------------------------------------------------------------------------

def nb_opening_scores(prefix: np.ndarray, gallery_games: list[np.ndarray],
                      query_games: list[np.ndarray], plies: np.ndarray,
                      pop: np.ndarray, alpha: float = 5.0) -> np.ndarray:
    """Per-slot categorical naive Bayes; (n_questions, n_candidates) summed log-likelihoods."""
    n_cand = len(gallery_games)
    n_slot = len(plies)
    logp = np.zeros((n_cand, n_slot, N_POINTS), np.float32)
    for i, games in enumerate(gallery_games):
        for j, ply in enumerate(plies):
            cnt = np.bincount(prefix[games, ply], minlength=N_POINTS).astype(np.float64)
            logp[i, j] = np.log((cnt + alpha * pop[j]) / (len(games) + alpha))
    scores = np.zeros((len(query_games), n_cand), np.float32)
    for q, games in enumerate(query_games):
        pts = prefix[games][:, plies]  # (k, n_slot)
        for j in range(n_slot):
            scores[q] += logp[:, j, pts[:, j]].sum(axis=1)
    return scores


def feature_centroid_scores(z: np.ndarray, gallery_games, query_games) -> np.ndarray:
    gal = np.stack([z[g].mean(axis=0) for g in gallery_games])
    qry = np.stack([z[g].mean(axis=0) for g in query_games])
    return -((qry[:, None, :] - gal[None, :, :]) ** 2).sum(axis=2)


def section_fingerprint(d: Data) -> str:
    gallery = cv.load("pid_gallery")
    questions = cv.load("pid_questions")
    black = d.color == "B"
    me_cols = [c for c in d.features if c.startswith("me_")]
    z = StandardScaler().fit_transform(d.stats.loc[black, me_cols])
    z_full = np.zeros((len(d.stats), len(me_cols)), np.float32)
    z_full[black] = z

    variants = {
        "第 1 手": ("raw", np.array([0])),
        "黑前 2 手": ("raw", np.array([0, 2])),
        "黑前 4 手": ("raw", np.arange(0, 8, 2)),
        "黑前 8 手": ("raw", np.arange(0, 16, 2)),
        "黑前 16 手": ("raw", np.arange(0, 32, 2)),
        "黑前 8 手（D4 正規化）": ("d4", np.arange(0, 16, 2)),
        "雙方前 16 手": ("raw", np.arange(16)),
    }
    results = []
    for split in (0, 1, 2):  # the three player-disjoint splits
        g = gallery[gallery["split"] == split]
        q = questions[questions["split"] == split].sort_values("question_id")
        cand = np.array(sorted(g["player_id"].unique()))
        gal_games = [g.loc[g["player_id"] == p, "game"].to_numpy() for p in cand]
        qry_games = list(q["games"])
        truth = q["player_id"].to_numpy()
        k = q["k"].to_numpy()
        # population prior from players outside this split (black games only)
        outside = black & ~np.isin(d.pid, cand)
        for name, (kind, plies) in variants.items():
            prefix = d.prefix if kind == "raw" else d.prefix_d4
            pop = np.stack([np.bincount(prefix[outside, p], minlength=N_POINTS) for p in plies])
            pop = (pop + 1) / (pop + 1).sum(axis=1, keepdims=True)
            scores = nb_opening_scores(prefix, gal_games, qry_games, plies, pop)
            ranked = cand[np.argsort(-scores, axis=1)[:, :5]]
            rep = pid_report(ranked, truth, k)
            rep["variant"], rep["split"] = name, split
            results.append(rep.reset_index())
        scores = feature_centroid_scores(z_full, gal_games, qry_games)
        ranked = cand[np.argsort(-scores, axis=1)[:, :5]]
        rep = pid_report(ranked, truth, k)
        rep["variant"], rep["split"] = "盤面統計 26 維 centroid", split
        results.append(rep.reset_index())
    res = pd.concat(results)
    top5 = res.pivot_table(index="variant", columns="k", values="top5", aggfunc="mean")
    top5_sd = res.pivot_table(index="variant", columns="k", values="top5", aggfunc="std")
    map5 = res.pivot_table(index="variant", columns="k", values="map5", aggfunc="mean")
    top1 = res.pivot_table(index="variant", columns="k", values="top1", aggfunc="mean")
    order = list(variants) + ["盤面統計 26 維 centroid"]
    cols = ["5", "10", "15", "20", "all"]
    top5, top5_sd, map5 = top5.loc[order, cols], top5_sd.loc[order, cols], map5.loc[order, cols]
    top1 = top1.loc[order, cols]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.array([5, 10, 15, 20])
    for j, name in enumerate(["黑前 4 手", "黑前 8 手", "黑前 8 手（D4 正規化）",
                              "盤面統計 26 維 centroid"]):
        ys = top5.loc[name, ["5", "10", "15", "20"]].to_numpy()
        ax.plot(xs, ys, color=SERIES[j], marker="o", markersize=5, label=name)
    ax.axhline(5 / 400, color=INK2, linewidth=1, linestyle=":")
    ax.annotate("亂猜 0.0125", (5, 5 / 400), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xlabel("每題盤數 k")
    ax.set_ylabel("Top-5 命中率（400 選 5）")
    ax.legend(frameon=False)
    ax.set_title("開局落點是極強的指紋；D4 正規化的代價集中在 k=5", loc="left")
    fig_md = save_fig(fig, "eda_pid_probe.png")

    # per-player habit strength among heavy players
    heavy = pd.Series(d.pid[black]).value_counts()
    heavy = heavy[heavy >= 120].index
    hb = pd.DataFrame({"pid": d.pid[black], "first": d.prefix[black, 0],
                       "first_d4": d.prefix_d4[black, 0]})
    hb = hb[hb["pid"].isin(heavy)]
    modal = hb.groupby("pid")["first"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    modal_d4 = hb.groupby("pid")["first_d4"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    pop_modal = hb["first"].value_counts(normalize=True).iloc[0]
    pop_modal_d4 = hb["first_d4"].value_counts(normalize=True).iloc[0]

    # style ICC among heavy players' black games
    mask = black & np.isin(d.pid, heavy)
    var = variance_table(d, mask)
    var["ICC(玩家)"] = var["rank η²"] + var["玩家(同rank內)"]
    icc = var.sort_values("ICC(玩家)", ascending=False)[["ICC(玩家)", "rank η²"]].head(10)
    icc.insert(0, "說明", [label(c) for c in icc.index])

    return f"""## 8. 玩家指紋（競賽一）

### 8.1 習慣的集中度

在 {len(heavy):,} 位重度玩家（≥120 盤黑棋）中：
- 每人「最常用的第一手」佔自己黑棋的比例：中位數 **{modal.median():.2f}**（p25={modal.quantile(.25):.2f}、p75={modal.quantile(.75):.2f}）；
  若大家都照全體分布下，這個值只會是 ≈{pop_modal:.2f}
- D4 正規化後：個人 {modal_d4.median():.2f} vs 全體 {pop_modal_d4:.2f}——正規化把「哪個角」合併掉之後，大家的第一手都塌縮到少數幾個點，
  個人與全體的差距從 {modal.median() - pop_modal:.2f} 縮到 {modal_d4.median() - pop_modal_d4:.2f}

### 8.2 開局 naive-Bayes 探針（**不是提交模型**）

對 `cv.py` 的 3 組玩家互斥合成候選庫（各 400 人 × 100 盤黑棋 gallery，查詢盤互斥），
每位候選人對「第 t 手黑棋落點」估一個平滑類別分布，查詢題的 k 盤 log-likelihood 相加後排序。
另附一個「盤面統計 26 維平均向量、歐氏距離最近」的對照組。數字為 3 組的平均。

{fig_md}

Top-5 命中率（亂猜 = 0.0125）：

{md_table(top5, 3)}

3 組間標準差：

{md_table(top5_sd, 3)}

MAP@5：

{md_table(map5, 3)}

Top-1：

{md_table(top1, 3)}

**結論：**
- **開局本身就是極強的指紋。** 只用黑棋前 4 手的落點（零訓練、純計數、每位候選人 100 盤），
  合成候選庫上 top-5 = {top5.loc['黑前 4 手', 'all']:.3f}、MAP@5 = {map5.loc['黑前 4 手', 'all']:.3f}；k=20 桶 top-5 已飽和
- **加更多手反而變差**（4 手 → 8 手 → 16 手：top-5 {top5.loc['黑前 4 手', 'all']:.3f} → {top5.loc['黑前 8 手', 'all']:.3f} → {top5.loc['黑前 16 手', 'all']:.3f}）。
  機制：逐手獨立的 naive Bayes 假設各手互不相關，後段的手落點分散、100 盤估不準，平滑後的雜訊 log-likelihood 被等權相加，蓋掉了前幾手的強訊號。
  這是**探針模型的缺陷**，不代表後段的手沒有身分資訊——需要能學「位置之間關係」的模型（序列模型）
- **D4 正規化的代價：** 黑前 8 手 top-5 {top5.loc['黑前 8 手', 'all']:.3f} → {top5.loc['黑前 8 手（D4 正規化）', 'all']:.3f}、MAP@5 {map5.loc['黑前 8 手', 'all']:.3f} → {map5.loc['黑前 8 手（D4 正規化）', 'all']:.3f}，
  k=5 桶 top-5 {top5.loc['黑前 8 手', '5']:.3f} → {top5.loc['黑前 8 手（D4 正規化）', '5']:.3f}。方向在四個 k 桶一致，幅度中等（3 組間標準差約 0.01–0.03）。
  → 絕對方位是指紋的一部分，**競賽一禁用 D4 增強**有數字支持；但大部分身分資訊在「相對的下法」裡，正規化後仍保留
- 加入白方（對手）的手沒有幫助（雙方前 16 手 {top5.loc['雙方前 16 手', 'all']:.3f} vs 黑前 8 手 {top5.loc['黑前 8 手', 'all']:.3f}）→ 與第 5 節一致：對手的手不是本人的指紋
- 盤面統計 26 維 centroid 也帶身分資訊（top-5 {top5.loc['盤面統計 26 維 centroid', 'all']:.3f}），但遠弱於開局落點；兩者資訊來源不同，合併可能有增益
- **對指標的含意：** 若競賽一的指標是 top-5 accuracy，簡單方法在 k≥10 就接近滿分，名次差距會集中在 k=5 桶；
  若是 MAP@5，排序品質（top-1）才是戰場。**第 0 節的指標確認對競賽一的策略影響很大**
- ⚠️ 合成候選庫是 ≥120 盤黑棋的重度玩家，習慣可能比一般玩家更固定；真實候選庫是否有相同的習慣集中度，要等候選庫發布後用 8.1 的指標比對

### 8.3 哪些盤面統計最像指紋（重度玩家黑棋，ICC = 玩家間變異佔比，含 rank）

{md_table(icc, 3)}

- ICC 最高的特徵（距離、局部應手、脫先、氣數）是穩定的個人風格；它們的 ICC 遠大於 rank η² → 對競賽一有用，對競賽二是記憶風險
"""


# --------------------------------------------------------------------------
# section 9 -- colour
# --------------------------------------------------------------------------

def section_colour(d: Data) -> str:
    var_b = variance_table(d, d.color == "B")
    var_w = variance_table(d, d.color == "W")
    top = var_b.sort_values("rank η²", ascending=False).index[:8]
    table = pd.DataFrame({
        "說明": [label(c) for c in top],
        "rank η² (執黑盤)": var_b.loc[top, "rank η²"],
        "rank η² (執白盤)": var_w.loc[top, "rank η²"],
    })
    # does a player's style carry across colours?
    s = d.stats.assign(pid=d.pid, color=d.color)
    counts = s.groupby(["pid", "color"]).size().unstack(fill_value=0)
    both = counts[(counts["B"] >= 20) & (counts["W"] >= 20)].index
    s = s[s["pid"].isin(both)]
    means = s.groupby(["pid", "color"])[list(top)].mean()
    corr = {c: np.corrcoef(means.xs("B", level="color")[c],
                           means.xs("W", level="color")[c])[0, 1] for c in top}
    table["玩家均值 corr(黑,白)"] = pd.Series(corr)
    return f"""## 9. 執色

{md_table(table, 3)}

- 執黑盤與執白盤的棋力訊號強度相當 → 競賽二的黑題、白題難度應相近
- 同一玩家執黑與執白的風格均值高度相關（{len(both):,} 位兩色都 ≥20 盤的玩家）→ 風格是跨執色的個人屬性，
  訓練時兩色可以共用（配合「我方／對方」正規化），不需要分色模型
"""


# --------------------------------------------------------------------------
# section 10 -- eligibility
# --------------------------------------------------------------------------

def section_eligibility(d: Data) -> str:
    counts = pd.DataFrame({"pid": d.pid, "rank": d.rank, "color": d.color}).groupby(
        ["pid", "rank", "color"]).size().unstack(fill_value=0).reset_index()
    counts["max_color"] = counts[["B", "W"]].max(axis=1)
    rows = {}
    for kk in K_BUCKETS:
        rows[f"≥{kk} 盤同色"] = counts.groupby("rank").apply(
            lambda g, kk=kk: int((g["max_color"] >= kk).sum()), include_groups=False)
    rows["玩家總數"] = counts.groupby("rank").size()
    table = pd.DataFrame(rows)
    table.index = RANKS
    table["≥20 同色 / 總數"] = table["≥20 盤同色"] / table["玩家總數"]
    return f"""## 10. 可出題玩家數（合成驗證的抽樣偏差）

一位玩家要能出一道 k 盤題，需要某個顏色至少有 k 盤：

{md_table(table, 3)}

- 出一道 k=20 題需要 ≥20 盤同色；各 rank 能出 k=20 題的玩家比例從 A 的 44% 到 1D 的 86% 不等。
  **合成的 k=20 題只來自各 rank 的「較活躍」子集**，而低段（D–A）的活躍子集佔比較小 → 合成驗證對低段休閒玩家的代表性較差
- 若主辦方的測試玩家是從同一母體、以相同門檻抽出的（未知），合成驗證的分布就是對的
"""


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    t0 = time.time()
    d = Data()
    print(f"loaded in {time.time() - t0:.1f}s")
    questions = load_rank_questions()
    parts = []
    steps = [
        ("length", lambda: section_length(d)),
    ]
    for name, fn in steps:
        parts.append(fn())
    uni, var = section_univariate(d)
    parts.append(uni)
    print("univariate done", time.time() - t0)
    parts.append(section_k(d, var, questions))
    print("k done", time.time() - t0)
    parts.append(section_probe(d, questions))
    print("probe done", time.time() - t0)
    parts.append(section_opponent(d, var))
    parts.append(section_spatial(d))
    print("spatial done", time.time() - t0)
    parts.append(section_opening(d))
    print("opening done", time.time() - t0)
    parts.append(section_fingerprint(d))
    print("fingerprint done", time.time() - t0)
    parts.append(section_colour(d))
    parts.append(section_eligibility(d))

    header = f"""# W1 訊號 EDA — 棋力與玩家指紋在哪裡

產生時間：{time.strftime('%Y-%m-%d %H:%M')}　資料：全量訓練集 1,000,000 盤
產生方式：`python -m go_aicup.features.game_stats && python scripts/eda_report.py`（全程 CPU，約 {int(time.time() - t0) // 60 + 1} 分鐘）

> 結構面稽核（欄位、重複、長度、玩家數）見 `01_audit.md` 與 `RESULTS.md` A-001～A-003，本報告不重複。
> 本報告回答的是：**訊號在哪裡、有多強、隨 k 怎麼變、兩賽的需求哪裡衝突。**
> 兩個探針（第 4、8 節）只用來量化訊號，不是提交模型，也不登記為實驗。

## 0. 總結：七個發現與對計畫的影響

| # | 發現（數字見各節） | 對計畫的影響 |
|---|------|------|
| E1 | **單盤手工特徵的棋力訊號很弱**：最強特徵 rank η² 只有 0.057，80–95% 是盤內雜訊（§2） | k 盤聚合是必需品；單盤表徵的品質（L1、L5）才是天花板所在 |
| E2 | **同 rank 內的玩家風格差異 > rank 之間的差異**（本人特徵玩家成分是 rank 成分的 1.5–5 倍）（§2、§3） | 「記住玩家」風險被量化了；k→∞ 也突破不了這個天花板。競賽二的 player-adversarial / 每人一盤取樣是必要的 |
| E3 | **對手的手與本人的手棋力訊號一樣強**，且距離類特徵的玩家成分只有一半（§2、§4、§5）。線性探針：只用對手 acc 0.249 vs 只用本人 0.258，合併 0.276 | **L2 升級為主力設計**：rank 頭同時吃雙方的手。對手的手是更乾淨、較不帶身分的棋力訊號 |
| E4 | 手工特徵 + 線性模型：acc 0.23（k=5）→ 0.31（k=20），QWK 0.69 → 0.83，MAE 1.74 → 1.23（§4） | P0 GBDT 值得做；錯誤呈 ordinal 結構 → 指標若是 MAE/QWK，ordinal head 必做 |
| E5 | **開局落點是極強的玩家指紋**：黑前 4 手逐手 naive Bayes（零訓練）在合成候選庫上 top-1 0.91、top-5 0.97、MAP@5 0.94；k=5 桶 top-5 0.92（§8） | 競賽一的 P0 可以直接用這個計數模型，當天就有強 baseline。名次差距將集中在 k=5 桶與排序品質（top-1 / MAP） |
| E6 | **D4 正規化的代價中等且集中在 k=5**（黑前 8 手 k=5 桶 top-5 0.867 → 0.813，MAP@5 0.910 → 0.863）（§8） | 「競賽一禁用 D4」有數字支持；棋力的空間訊號（線位、中腹 vs 邊）都是 D4 不變的 → 競賽二用 D4 安全（§6） |
| E7 | `game_id` 是打亂的流水號，**沒有時間資訊**；勝負代理（最後一手）與終局子數無棋力訊號（§1） | 「風格隨時間漂移」無法驗證（未知）；若主辦方依時間切候選庫與查詢盤，合成驗證會高估競賽一 |

**需要注意的偏差：** 競賽一的合成候選庫是 ≥120 盤黑棋的重度玩家（習慣可能更固定）；
競賽二的 k=20 題只來自各 rank 的活躍子集（§10）。真實候選庫發布後，第一件事是用 §8.1 的「個人最常用第一手佔比」比對分布。

"""
    REPORT.write_text(header + "\n".join(parts), encoding="utf-8")
    print(f"wrote {REPORT} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
