"""Generate AI CUP 2026 official technical reports (.docx) and publication figures for
Competition 1 (TEAM_10894) and Competition 2 (TEAM_10895).

Usage:
    python scripts/release/build_reports.py

All user-specific fields that are not stored in the git repository (Private LB scores/ranks,
full Chinese/English names, affiliation, phone numbers, advisor, optional model-weight URL)
are centralized in `TEAM_CONFIG` below so they can be updated in one place and regenerated.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Centralized Team & Leaderboard Metadata (fill 【待填】 items before final PDF)
# ---------------------------------------------------------------------------
TEAM_CONFIG = {
    "github_url": "https://github.com/jabir95tsai/go_aicup_2026",
    "advisor": "無",
    "members": [
        {
            "role": "隊長",
            "zh_name": "【待填中文姓名（隊長）】",
            "en_name": "【待填英文名, 姓（例：Chih-Po, Tsai）】",
            "zh_org": "國立臺灣大學【待填系所中文全稱】",
            "en_org": "National Taiwan University Department of 【待填系所英文全稱】",
            "phone": "【待填電話】",
            "email": "jabir95tsai@gmail.com",
        },
        {
            "role": "隊員1",
            "zh_name": "蕭博勁",
            "en_name": "【待填英文名, Hsiao】",
            "zh_org": "【待填學校＋系所中文全稱】",
            "en_org": "【待填學校＋系所英文全稱】",
            "phone": "【待填電話】",
            "email": "jason.hsiao05@gmail.com",
        },
    ],
    "c1": {
        "team_id": "TEAM_10894",
        "title": "圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家",
        "filename": "TEAM_10894_圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家.docx",
        "private_score": "0.8857159",
        "private_rank": "20",
        "public_score": "0.8630809",
        "public_rank": "25",
        "final_submission": "851729-comb-pseudo-a3l.csv",
    },
    "c2": {
        "team_id": "TEAM_10895",
        "title": "圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級",
        "filename": "TEAM_10895_圍棋棋力與風格預測競賽—競賽二：從圍棋棋譜預測玩家棋力等級.docx",
        "private_score": "0.7423199",
        "private_rank": "19",
        "public_score": "0.7744441",
        "public_rank": "15",
        "final_submission": "849173-opp7-l10.csv",
    },
}


def _apply_local_team_config(path: Path) -> None:
    if not path.exists():
        return
    overrides = json.loads(path.read_text(encoding="utf-8"))
    for key, val in overrides.items():
        if key in ("c1", "c2") and isinstance(val, dict) and isinstance(TEAM_CONFIG.get(key), dict):
            TEAM_CONFIG[key].update(val)
        else:
            TEAM_CONFIG[key] = val


_apply_local_team_config(REPORTS_DIR / "team_config.local.json")

# Official word-count limits per section (minimum, maximum)
SECTION_LIMITS = {
    "壹、環境": (200, 600),
    "貳、演算方法與模型架構": (400, 1200),
    "參、創新性": (300, 1200),
    "肆、資料處理": (300, 1500),
    "伍、訓練方式": (400, 1000),
    "陸、分析與結論": (400, 2500),
}


# ---------------------------------------------------------------------------
# Word-count helpers (verify both MS-Word-like word count and CJK char count)
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_NON_CJK_WORD_RE = re.compile(r"[A-Za-z0-9_.\-+/:%]+")


def count_msword_words(text: str) -> int:
    """Approximate MS Word word count: each CJK char = 1 word, each ASCII/num token = 1 word."""
    cjk = len(_CJK_RE.findall(text))
    stripped = _CJK_RE.sub(" ", text)
    ascii_tokens = len(_NON_CJK_WORD_RE.findall(stripped))
    return cjk + ascii_tokens


def count_cjk_chars(text: str) -> int:
    return len(_CJK_RE.findall(text))


# ---------------------------------------------------------------------------
# Matplotlib Figure Generation
# ---------------------------------------------------------------------------
def _setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans", "DFKai-SB", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8


def generate_c1_figures() -> list[tuple[Path, str]]:
    _setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Public LB progression & Sealed Pool 1 paired progressions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    stages = [
        "Official\nBaseline",
        "Floor\n(11/12)",
        "hung-\nseqB",
        "e2emix-\nml",
        "ogsa12w\n(OGS)",
        "817714-\nlik",
        "818024-\nlikrich",
        "820880-\nlikeb",
        "821010-\ncomb",
        "824896-\npseudo",
        "851729-\na3l(終版)",
    ]
    lbs = [0.2845, 0.7565, 0.8012, 0.8245, 0.8388, 0.8465, 0.8536, 0.8543, 0.8624, 0.8620, 0.8631]
    x = np.arange(len(stages))
    ax1.plot(x, lbs, marker="o", lw=2.2, color="#1f5a8a", markersize=6, label="主線迭代 Public LB")
    ax1.axhline(0.2845, color="#888888", ls="--", lw=1.0, label="官方 Baseline (0.2845)")
    for xi, yi in zip(x, lbs):
        dy = -13 if xi in (9,) else 7
        dx = 14 if xi == 0 else (-6 if xi == 10 else 0)
        ax1.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(dx, dy),
                     ha="center", fontsize=7.2, color="#1a1a1a")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=6.7)
    ax1.set_ylim(0.22, 0.93)
    ax1.set_ylabel("Public LB 分數", fontsize=9.5)
    ax1.set_title("(a) 競賽一 Public LB 演進歷程", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, ls=":")
    ax1.legend(loc="lower right", fontsize=7.8, frameon=True)

    # Right panel: Sealed Pool 1 paired comparisons (base vs new on each draw set)
    pairs = [
        "seqB→\ne2emix\n(1700-)",
        "e2emix→\nogsa12w\n(1700-)",
        "ogsa12w→\nLLR(2)\n(2700-)",
        "LK→\nRICH(G3)\n(3100-)",
        "RICH→\nEB50\n(3300-)",
    ]
    base_s = [0.6761, 0.6973, 0.7152, 0.7512, 0.7655]
    new_s = [0.6973, 0.7097, 0.7453, 0.7622, 0.7764]
    px = np.arange(len(pairs))
    w = 0.34
    b1 = ax2.bar(px - w / 2, base_s, width=w, color="#9ab3c9", label="對照基準")
    b2 = ax2.bar(px + w / 2, new_s, width=w, color="#1f5a8a", label="改進版本")
    for i, (bs, ns) in enumerate(zip(base_s, new_s)):
        ax2.text(i + w / 2, ns + 0.003, f"+{ns - bs:.4f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color="#0e3a5c")
    ax2.set_xticks(px)
    ax2.set_xticklabels(pairs, fontsize=7.5)
    ax2.set_ylim(0.64, 0.81)
    ax2.set_ylabel("新增集 Sealed Pool 1 分數", fontsize=9.5)
    ax2.set_title("(b) 新增集驗證台封存池配對提升（各階段新抽題組）", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, ls=":")
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout()
    p1 = FIGURES_DIR / "c1_fig1_progression.png"
    fig.savefig(p1)
    plt.close(fig)

    # Figure 2: Ablation Deltas & Per-k/Color difficulty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    labels = [
        "玩家條件 LLR (27維)",
        "E2E9600+L1/L2",
        "OGS A1/A2 集成",
        "局部棋形 G3",
        "經驗貝氏先驗 EB50",
        "偽標籤兩輪擴充",
        "逐 k 風格倍率 (LK)",
        "逐 k 學習融合權重",
        "中心擴充 EM (未採)",
        "編碼器 D4 TTA (未採)",
        "共用對手分項 (未採)",
        "題目合併 E1 (未採)",
        "反向似然 RLLR (未採)",
        "KGS A3 集成 (未採)",
    ]
    deltas = [0.0301, 0.0212, 0.0125, 0.0109, 0.0109, 0.0068, 0.0041, 0.0019,
              0.0018, 0.0007, 0.0003, 0.0002, 0.000004, -0.0018]
    colors = ["#1f5a8a" if i < 8 else "#b0b7bd" for i in range(len(labels))]
    colors[-1] = "#c0392b"
    y_pos = np.arange(len(labels))[::-1]
    ax1.barh(y_pos, deltas, color=colors, height=0.68)
    ax1.axvline(0, color="#333333", lw=0.8)
    for yp, d in zip(y_pos, deltas):
        off = 0.0006 if d >= 0 else -0.0006
        ha = "left" if d >= 0 else "right"
        ax1.text(d + off, yp, f"{d:+.4f}", va="center", ha=ha, fontsize=7.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7.8)
    ax1.set_xlim(-0.007, 0.0365)
    ax1.set_xlabel("Sealed Pool 1 相對當時基準 Δ", fontsize=9.5)
    ax1.set_title("(a) 競賽一各模組驗證台封存池增益對照", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, ls=":")

    ks = [1, 2, 3, 5]
    black_scores = [0.598, 0.795, 0.865, 0.964]
    mixed_scores = [np.nan, 0.790, 0.891, 0.953]
    white_scores = [0.523, 0.702, 0.834, 0.877]
    ax2.plot(ks, black_scores, marker="o", lw=2.2, color="#1b2631", label="全黑題")
    ax2.plot([2, 3, 5], mixed_scores[1:], marker="s", lw=2.2, color="#2e86c1", label="黑白混合題")
    ax2.plot(ks, white_scores, marker="^", lw=2.2, color="#d35400", label="全白題")
    for k, b, wv in zip(ks, black_scores, white_scores):
        ax2.text(k, b + 0.012, f"{b:.3f}", ha="center", fontsize=8, color="#1b2631")
        ax2.text(k, wv - 0.028, f"{wv:.3f}", ha="center", fontsize=8, color="#d35400")
    ax2.set_xticks(ks)
    ax2.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
    ax2.set_ylim(0.45, 1.03)
    ax2.set_ylabel("新增集開發題組得分 (821010 融合)", fontsize=9.5)
    ax2.set_title("(b) 新增集依題目盤數 k 與執色分布之難度差異", fontsize=10.5, fontweight="bold")
    ax2.grid(True, alpha=0.3, ls=":")
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    p2 = FIGURES_DIR / "c1_fig2_ablation_and_k.png"
    fig.savefig(p2)
    plt.close(fig)

    # Figure 3: C1 Dual-Track Architecture Diagram
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4fa", ec="#2b5c8f", fs=8.3, bold=False):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#111111")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3))

    ax.text(50, 43.5, "競賽一雙軌融合辨識系統總覽（851729-comb-pseudo-a3l.csv）",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#0d2b45")

    # Track 1 (Top)
    box(2, 31, 18, 9, "原測試集 (pub/priv)\n400人×100盤全黑\nk=5,10,15,20 一對一", fc="#f6f8fa", ec="#555555")
    box(25, 31, 22, 9, "D4正規化開局 pp+slot\n+ 族群先驗計數", fc="#e8f1fb")
    box(52, 31, 22, 9, "8× GE2E 保險編碼器\n(ply>=20 盤向量餘弦)", fc="#e8f1fb")
    box(79, 31, 19, 9, "Hungarian 最佳指派\nTop-5 輸出 (400+400列)", fc="#d7ead8", ec="#2e7d32", bold=True)
    arrow(20.5, 35.5, 24.5, 35.5)
    arrow(47.5, 35.5, 51.5, 35.5)
    arrow(74.5, 35.5, 78.5, 35.5)

    # Track 2 (Bottom)
    box(2, 9, 16, 15, "新增集 (pub2/priv2)\n600人×10盤黑白混合\nk=1,2,3,5 非一對一", fc="#f6f8fa", ec="#555555")
    box(21, 18.5, 23, 8.5, "開局與多層計數 (4項)\nOpening / Category / L1角 / L2應手", fc="#e8f1fb", fs=7.8)
    box(21, 9.5, 23, 8.0, "風格編碼器集成 (3分項)\nseqB + A1 + mean(A2, A3L-9600)", fc="#e8f1fb", fs=7.7)
    box(21, 1.0, 23, 7.5, "50維玩家條件似然比 LLR\nHumanSL(Cv2)+局部棋形G3+EB先驗Σ", fc="#fff4e5", ec="#c87d12", fs=7.7)

    box(49, 9.5, 21, 14, "逐 k Conditional Logit\n8 分項學習融合權重\n(k=1,2,3,5 獨立權重)", fc="#eaf2ff", ec="#1f5a8a", bold=True)
    box(75, 9.5, 23, 14, "兩輪留一偽標籤擴充\n後驗>=0.5 擴充參考盤\n8 分項重算 → Top-5 輸出", fc="#d7ead8", ec="#2e7d32", bold=True)

    arrow(18.5, 21, 20.5, 22.5)
    arrow(18.5, 16, 20.5, 13.5)
    arrow(18.5, 11, 20.5, 5.0)
    arrow(44.5, 22.5, 48.5, 18.5)
    arrow(44.5, 13.5, 48.5, 16.5)
    arrow(44.5, 5.0, 48.5, 14.0)
    arrow(70.5, 16.5, 74.5, 16.5)

    fig.tight_layout()
    p3 = FIGURES_DIR / "c1_fig3_arch_overview.png"
    fig.savefig(p3)
    plt.close(fig)

    return [
        (p1, "圖一、競賽一（TEAM_10894）Public LB 演進歷程與新增集驗證台封存池各階段配對增益"),
        (p2, "圖二、競賽一各元件增量比較（含採納與淘汰方向）及新增集依盤數 k 與執色之得分差異"),
        (p3, "圖三、競賽一雙軌融合辨識架構圖（原測試集指派管線與新增集八分項學習融合＋偽標籤擴充管線）"),
    ]


def generate_c2_figures() -> list[tuple[Path, str]]:
    _setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: C2 Local & Public LB Progression
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    local_labels = [
        "手工+\nLR\n(14k)",
        "V2+\nC2\n(10k)",
        "HSL+\nB2\n(1k)",
        "AH+\nB2\n(1k)",
        "AH+C\n+B2\n(1k)",
        "完整B2\n(E1)",
        "B2+\nopp7\n(E1)",
    ]
    local_scores = [0.412, 0.5553, 0.6572, 0.6833, 0.6934, 0.7151, 0.7532]
    lx = np.arange(len(local_labels))
    ax1.bar(lx, local_scores, color=["#9aa5b1", "#5b89b8", "#3d74aa", "#2b6298", "#1f5a8a", "#174b75", "#0f3758"], width=0.58)
    ax1.axhline(0.4631, color="#c0392b", ls="--", lw=1.2, label="官方 Baseline (0.4631)")
    for i, s in enumerate(local_scores):
        ax1.text(i, s + 0.009, f"{s:.4f}", ha="center", fontsize=7.6)
    ax1.set_xticks(lx)
    ax1.set_xticklabels(local_labels, fontsize=7.5)
    ax1.set_ylim(0.32, 0.82)
    ax1.set_ylabel("Valid-U Equal-k 官方分數", fontsize=9.5)
    ax1.set_title("(a) 競賽二本地未見玩家驗證演進", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, ls=":")
    ax1.legend(loc="upper left", fontsize=8, frameon=True)

    pub_labels = [
        "Baseline",
        "AH+C",
        "P1-P3",
        "noA+\nprior",
        "cvec2\n+L10",
        "qmerge\n+L10",
        "ygbm\n+L10",
        "opp1",
        "opp2\nsoft",
        "opp4\n15g",
        "opp5\n40g",
        "opp7\n(終版)",
    ]
    pub_scores = [0.4631, 0.7103, 0.7197, 0.7509, 0.7500, 0.7531, 0.7563, 0.7572, 0.7726, 0.7744, 0.7754, 0.7744]
    px = np.arange(len(pub_labels))
    ax2.plot(px, pub_scores, marker="o", lw=2.1, color="#1f5a8a", markersize=5.5, label="主線 Public LB")
    ax2.scatter([10.5], [0.7758], marker="*", s=110, color="#c0392b", zorder=5, label="opp6 (含A3/KGS: 0.7758)")
    ax2.axhline(0.4631, color="#888888", ls="--", lw=1.0)
    for i, s in enumerate(pub_scores):
        dy = -13 if i in (4, 11) else (10 if i == 10 else 7)
        dx = 14 if i == 0 else 0
        ax2.annotate(f"{s:.4f}", (i, s), textcoords="offset points", xytext=(dx, dy),
                     ha="center", fontsize=7.0)
    ax2.set_xticks(px)
    ax2.set_xticklabels(pub_labels, fontsize=6.8)
    ax2.set_ylim(0.42, 0.83)
    ax2.set_ylabel("Public LB 分數", fontsize=9.5)
    ax2.set_title("(b) 競賽二 Public LB 演進歷程", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, ls=":")
    ax2.legend(loc="lower right", fontsize=7.8, frameon=True)

    fig.tight_layout()
    p1 = FIGURES_DIR / "c2_fig1_progression.png"
    fig.savefig(p1)
    plt.close(fig)

    # Figure 2: Per-k curves & Opponent Identity Link Rank Consistency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ks = [5, 10, 15, 20]
    ax1.plot(ks, [0.5030, 0.5501, 0.5716, 0.5965], marker="s", lw=2.0, color="#85929e", label="V2 ResNet+C2 (Valid-U 10k)")
    ax1.plot(ks, [0.609, 0.685, 0.670, 0.666], marker="^", lw=2.0, color="#5499c7", label="HumanSL+B2 (Valid-U 1k)")
    ax1.plot(ks, [0.6244, 0.6977, 0.7303, 0.7213], marker="o", lw=2.0, color="#2471a3", label="AH+C+B2 (Valid-U 1k)")
    ax1.plot(ks, [0.7022, 0.7494, 0.7762, 0.7740], marker="D", lw=2.2, color="#154360", label="cvec2+先驗 B2 (E2)")
    ax1.axhline(0.4631, color="#c0392b", ls="--", lw=1.0, label="官方 Baseline (0.4631)")
    for k, v in zip(ks, [0.7022, 0.7494, 0.7762, 0.7740]):
        ax1.text(k, v + 0.011, f"{v:.4f}", ha="center", fontsize=7.8, color="#154360")
    ax1.set_xticks(ks)
    ax1.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
    ax1.set_ylim(0.43, 0.84)
    ax1.set_ylabel("官方公式得分", fontsize=9.5)
    ax1.set_title("(a) 各棋局數桶 k=5,10,15,20 之模型表現", fontsize=10.5, fontweight="bold")
    ax1.grid(True, alpha=0.3, ls=":")
    ax1.legend(loc="lower right", fontsize=7.6, frameon=True)

    # Right panel: Opponent-link rank consistency proof
    cats = ["隨機打亂\n基準", "無門檻\nf8 top-1", "q99 門檻\nHD0 連結", "q99.9 門檻\nHD0 連結"]
    exact_0 = [0.10, 0.30, 0.50, 0.51]
    within_1 = [0.29, 0.65, 0.90, 0.94]
    cx = np.arange(len(cats))
    w = 0.34
    ax2.bar(cx - w / 2, [v * 100 for v in exact_0], width=w, color="#5d9cec", label="完全同級 (Δ=0)")
    ax2.bar(cx + w / 2, [v * 100 for v in within_1], width=w, color="#1f5a8a", label="同級或差一級 (|Δ|<=1)")
    for i, (e0, w1) in enumerate(zip(exact_0, within_1)):
        ax2.text(i - w / 2, e0 * 100 + 1.5, f"{e0*100:.0f}%", ha="center", fontsize=8)
        ax2.text(i + w / 2, w1 * 100 + 1.5, f"{w1*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax2.annotate("HD1 讓先方向驗證 (q99):\n擁有者執黑 平均Δ=+0.38 級\n擁有者執白 平均Δ=-0.51 級",
                 xy=(2.1, 88), xytext=(0.15, 76),
                 bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#d48806"),
                 fontsize=7.8)
    ax2.set_xticks(cx)
    ax2.set_xticklabels(cats, fontsize=8.3)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("連到對手級別與本人一致比例 (%)", fontsize=9.5)
    ax2.set_title("(b) 對手身分連結之段位一致度與讓先方向驗證", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, ls=":")
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout()
    p2 = FIGURES_DIR / "c2_fig2_k_and_opp_analysis.png"
    fig.savefig(p2)
    plt.close(fig)

    # Figure 3: C2 End-to-End Architecture Diagram
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef4fa", ec="#2b5c8f", fs=8.1, bold=False):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#111111")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.3))

    ax.text(50, 43.5, "競賽二端到端棋力預測系統架構圖（849173-opp7-l10.csv）",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#0d2b45")

    box(2, 16, 15, 14, "官方測試題\n800題 (Pub/Priv)\nk=5,10,15,20\n同題同色 HD<=1", fc="#f6f8fa", ec="#555555", bold=True)

    # Top branch: B2
    box(21, 26, 26, 12, "支線一：54維逐盤棋力特徵\nH 30維 (貼目 6.5 / 0.5)\n+ Cv2 20維 (自學十級向量)\n+ HD1 旗標 2維 + yH 非線性 2維", fc="#e8f1fb")
    box(52, 26, 20, 12, "B2 分 k 多類 LR\n+ 先驗對齊\n+ 同人測試題合併\n(9 組 / 21 題)", fc="#eaf2ff", ec="#1f5a8a", bold=True)

    # Bottom branch: Opponent Link
    box(21, 6, 26, 13, "支線二：對手身分連結 (opp7)\nf8 開局+三編碼器 (8,007訓練人)\n→ g40 27維 LLR 重排前十 (μ=2)", fc="#fff4e5", ec="#c87d12")
    box(52, 6, 20, 13, "軟連結 DELTA 核\nτ=3.5 權重混合\nValid-S 2萬盤核函數\nα=0.55 對數似然證據", fc="#fff4e5", ec="#c87d12", bold=True)

    # Final output
    box(77, 15.5, 21, 14, "後驗融合與解碼\nlog P_B2 + α·Σ log_e\n→ Bayes score_rank\n→ L10 高信心覆寫 (7題)", fc="#d7ead8", ec="#2e7d32", bold=True)

    arrow(17.5, 25, 20.5, 31)
    arrow(17.5, 20, 20.5, 13)
    arrow(47.5, 32, 51.5, 32)
    arrow(47.5, 12.5, 51.5, 12.5)
    arrow(72.5, 31, 76.5, 24.5)
    arrow(72.5, 13.5, 76.5, 20.5)

    fig.tight_layout()
    p3 = FIGURES_DIR / "c2_fig3_arch_overview.png"
    fig.savefig(p3)
    plt.close(fig)

    return [
        (p1, "圖一、競賽二（TEAM_10895）本地 Valid-U 驗證演進與 Public LB 各階段分數變化"),
        (p2, "圖二、競賽二各盤數桶 k∈{5,10,15,20} 表現曲線與對手身分連結之段位一致度驗證"),
        (p3, "圖三、競賽二雙支線融合架構圖（54 維棋力特徵 B2 模型＋40 盤 LLR 重排軟連結 DELTA 核）"),
    ]


# ---------------------------------------------------------------------------
# Report Text Specifications (carefully calibrated within official bounds)
# ---------------------------------------------------------------------------
def build_c1_sections() -> dict[str, list[str]]:
    return {
        "壹、環境": [
            "本系統開發於 Windows 11 本地端工作站（NVIDIA RTX 3060 Ti 8GB、CUDA 12.4），大規模訓練、特徵抽取與重現性稽核執行於國家高速網路與計算中心「晶創 26」（Nano4）高效能叢集（Red Hat Enterprise Linux x86_64、Slurm 排程之 dev 與 8gpus 佇列，單節點配置最多 8 張 NVIDIA H200 SXM5 141GB 顯卡、2 顆 Intel Xeon Platinum 8480+ 共 112 核心與 2TB 主記憶體，節點內透過 NVLink 互連）。",
            "程式語言採用 Python 3.13.12（Conda 環境 go），核心套件依 requirements.txt 鎖定版本：PyTorch 2.6.0+cu124、NumPy 2.3.5、Pandas 2.3.3、PyArrow 20.0.0、SciPy 1.16.3、scikit-learn 1.7.2、Numba 0.65.0 及 Matplotlib 3.10.8。單節點多卡訓練採用 PyTorch 原生 torchrun 與 NCCL 後端，並以 CUDA 12.8／cuDNN 9.8 執行期函式庫支撐 KataGo Python 推論模組。",
            "外部預訓模型與資料集：一、KataGo HumanSL 網路 PyTorch 權重 b18c384nbt-humanv0.ckpt（GitHub lightvector/KataGo v1.15.0 release），搭配 KataGo v1.18.2 Python 模組於執行期重現 policy 分布並接續自學之 C v2 級別條件向量，僅供新增集 LLR 特徵抽取，未微調。二、OGS 公開棋譜合集 2013–2025-05（za3k.com/ogs，16.2 GB），用於訓練風格編碼器 A1／A2／A3L。三、KGS 高段棋譜庫（u-go.net/gamerecords/，55 檔 228,849 盤），與 OGS 併抽訓練深層風格編碼器 A3L-9600；兩外部棋譜庫均未參與任何推論。",
        ],
        "貳、演算方法與模型架構": [
            "針對官方雙重測試集特性，本隊設計「原測試集指派管線」與「新增集多模態八分項融合＋偽標籤擴充管線」之雙軌系統（最終檔 851729-comb-pseudo-a3l.csv，共 1,600 題輸出）。",
            "第一軌（原測試集 pub_/priv_，400 人×100 盤全黑、一對一）：先對前 60 手進行 D4 棋盤對稱正規化，計算前綴回退（prefix back-off）與逐手位置（slot）Naive Bayes 對數似然，並以訓練集全黑棋譜估計族群先驗。為防範開局漂移，引入不看開局（ply≥20）之 GE2E 保險編碼器（20-block、256-channel 盤面 ResNet 主幹，輸出 256 維逐手向量取盤內平均與餘弦相似度，權重為 8×w），最後以 Hungarian 演算法執行全域一對一最佳指派（--hd-weight 2），輸出前五名候選。",
            "第二軌（新增集 pub2_/priv2_，600 人×10 盤黑白混合、非一對一）：每道查詢題對 600 名候選人計算八個互補分項：一、D4 開局 Naive Bayes；二、輪廓類別統計（category）；三、多層開局比對 L1（定式角部鍵）；四、L2（對手應手鍵）；五、seqB 編碼器餘弦分數（v4_d4_hd_9600 主幹加兩層 256→512→256 逐手殘差 MLP 聚合頭與手數嵌入）；六、端到端 20-block 風格編碼器 A1；七、由 A2 與深化 30-block（35.9M 參數、grow_trunk=True）之 A3L-9600 編碼器取等權平均 mean(A2, A3L-9600) 組成該風格分項；八、玩家條件走子似然比（LLR）。",
            "在 LLR 分項中，凍結 KataGo HumanSL 網路並替換為自學之 C v2 四級別條件（C／A／2D／5D），取該手對數平均指數分布為基底 b(m)，記錄前 32 高機率落點及其餘點「other」。每位候選人以 10 盤參考盤擬合 50 維對數線性殘差參數 θ_c（級別偏移 3 維、共識排名 5 維、對手與己方上一手距離 10 維、邊線 5 維、階段交互 10 維、跟隨銳利度 G1 2 維、絕對區域 G2 8 維、Numba 盤面重播局部棋形 G3 10 維如碰長尖跳小飛、戰術氣數 G4 3 維），並套用經驗貝氏異向共變異先驗 N(θ_pop, Σ)。最終透過逐 k（k∈{1,2,3,5}）conditional logit 學習八分項權重得出後驗，再對後驗 ≥0.5 之查詢題進行兩輪「留一（leave-one-out）」偽標籤擴充（pub2 選 293 題擴充 259 人、priv2 選 305 題擴充 278 人）：將其棋局併入預測候選人參考集並重算全部分項（計算該題自身時剔除該題棋局），依後驗由高至低輸出 top-5。",
        ],
        "參、創新性": [
            "本架構創新聚焦於突破新增集「每人僅 10 盤參考棋、黑白混合且查詢題最小僅 1 盤（k=1）」之極端少樣本瓶頸。首先，原創「以棋力條件網路為基底之玩家條件對數線性殘差模型（LLR）」：傳統風格嵌入將整盤池化為單一 256 維向量，會稀釋局部手筋與步調習慣；本隊跨任務將競賽二自學的 C v2 幽玄級別向量接回凍結之 KataGo HumanSL，抽取出兼具棋力與人類共識的 top-32 基底分布 b(m)，再於其上為每位候選人擬合 50 維低維度殘差向量 θ_c。由於參數少且具有明確圍棋語意（如局部棋形碰長尖跳小飛與跟隨銳利度），單靠 LLR 即能補足嵌入向量與開局計數無法分辨之細微手癖，在封存驗證池帶來全系統最大單項躍升（+0.0301，z=17.6）。",
            "其次，提出「經驗貝氏異向共變異先驗（EB50）」解決 10 盤過擬合問題：實驗發現各維度在玩家間變異差異極大（如級別偏移標準差達 0.57–0.83，局部棋形為 0.14–0.39，區域僅 0.05–0.14），等向先驗會在無差異方向殘留雜訊。我們從獨立驗證台玩家四組互異 10 盤估計玩家間共變異矩陣 Σ 並設特徵值下限 0.01，使 MAP 收縮精準保留個人特質，將同一人兩組 10 盤的 θ 餘弦穩定度由 0.509 大幅推升至 0.690。",
            "再者，設計「多模態八分項留一偽標籤擴充機制」與「殘差恆等深化擴容（grow_trunk）」：前者利用同一玩家可對應多題之結構，在嚴格留一保護下把高信心題併入候選參考庫並重算全部分項；後者於預訓 20-block 主幹上以零初始化末層 BN 權重無縫擴深至 30-block（A3L-9600），再結合 OGS 與 KGS 高段多樣化棋譜提升少樣本表徵泛化力，並採原／新增集雙軌分離指派達成最佳效能。",
        ],
        "肆、資料處理": [
            "訓練集經全量稽核共 1,000,000 盤、8,007 位玩家、221,894,008 手（100–445 手，無重複棋譜，解析錯誤 0），唯一鍵為 (rank, game_id) 而非單獨 game_id（避免 10:1 靜默塌縮）。針對官方 v1.1 新增之讓子欄位 HD[n]，經實證稽核確認 HD0 為互先貼目 6.5、HD1 為讓先貼目 0.5（白勝和棋），而 HD≥2 之 36,421 盤 SGF 黑白標記與實際執色相反且缺讓子星位。因官方競賽二測試集與競賽一可用訓練條件皆以讓先內為主，本任務資料處理一律過濾 HD≤1，避免讓子標記反轉汙染風格特徵。",
            "為加速百萬盤處理並杜絕重複解析 CSV，建立確定性 bit-packed 盤面快取（每手雙層 19×19 打包為 2×46 bytes 與落點、顏色元資料）。針對雙色盤面，所有風格編碼器與 LLR 皆執行執色正規化（as_black：白方落子將雙方盤面通道對調，統一轉換為「我方／對方」視角）；D4 對稱處理則依子題特性區隔：開局模型與 L1/L2 採用旋轉正規化或原始方位對齊，風格編碼器訓練時施以「玩家層級固定 D4 增強」（同一玩家所有盤共用同一變換，保留方位偏好同時擴增樣本）。",
            "外部資料均經嚴格清洗：OGS 2013–2025-05 僅保留 19 路、ranked、HD≤1、無擺子、已完成、手數 ≥60 手且對手為真人之對局（A1 抽 40,000 人×20 盤、A2 用全量 82,481 人×10 盤）；KGS 高段庫（228,849 盤）依同規格與「前 60 手無 pass、排除疑似電腦帳號」條件篩出 135,806 盤／11,627 人（其中 ≥20 盤者 1,463 人），全數強制納入並與 OGS 併抽至 40,000 人×20 盤訓練 A3L-9600，訓練時幽玄抽樣比為 0.5。驗證台由非編碼器訓練玩家建構兩個 591 人×40 盤互斥池（pool 0 開發調參、pool 1 封存檢定），杜絕任何資料洩漏。",
        ],
        "伍、訓練方式": [
            "訓練流程分為風格編碼器預訓、端到端主幹訓練、條件似然擬合與融合校準四階段，全程固定隨機種子與硬體紀錄，並於各階段完成後凍結權重執行離線驗證。",
            "一、風格編碼器與 A3L 深化訓練：基礎骨幹由棋力 V2 20-block ResNet 初始化，先於幽玄資料（排除驗證台 split 0–2）以 GE2E softmax 損失（每批 64 人×8 盤×16 手，D4 玩家層級增強，AdamW lr=3e-4、wd=1e-4、5% warmup＋cosine 衰減、BF16 混合精度）訓練 9,600 步得到 v4_d4_hd_9600，再凍結逐手向量訓練兩層逐手殘差 MLP 聚合頭（head B）。接著以相同超參數與激活檢查點（每批 64 人×8 盤×32 手，WSD 排程：5% warmup 與最後 20% 衰減），混合幽玄與 OGS 資料從頭端到端訓練 20-block 的 A1／A2 各 9,600 步（8×H200 各約 10.5 小時）；A3L-9600 則以 grow_trunk=True 將 20-block 主幹擴為 30-block（6 段激活檢查點，seed 2027），於 OGS＋KGS 混合快取接續 stable_step5760.pt 跑滿 9,600 步並於第 7,680 步啟動 WSD 衰減。",
            "二、LLR 殘差參數與先驗擬合：於驗證台外抽取 600 名訓練玩家（各 10 盤）以 σ=10 擬合族群基準 θ_pop；再由 pool 0（591 人×4 組×10 盤）估計 50×50 玩家間共變異矩陣 Σ。推論階段以 GPU 批次阻尼牛頓法（damped Newton，單步最大位移限 1、至多 25 輪）為每位官方候選人求取收斂之 MAP 解（λ=4、c=1），全程僅費時數分鐘且無數值發散或記憶體溢位問題。",
            "三、逐 k 融合權重訓練與重現稽核：於 pool 0 上以 conditional logit 最大概似（L-BFGS）分 k∈{1,2,3,5} 學習八分項權重，最後套用兩輪後驗 ≥0.5 留一偽標籤擴充。專案所有核心階段皆須通過封存 pool 1 門檻（Δ>0、z≥2.5、各 k≥−0.005 且 pool 0 Δ>0），同時於獨立環境重跑完整管線以檢核跨節點數值穩定性；最終版本 851729-comb-pseudo-a3l.csv 與基底 824896 皆已完成乾淨環境逐位元組重現驗證（archive/restore-855796.log 與 Slurm 839312），確認與官方最終提交檔內容完全一致。",
        ],
        "陸、分析與結論": [
            "最終提交檔 851729-comb-pseudo-a3l.csv 於 Private LB 取得 0.8857159（排名第 20），Public LB 為 0.8630809（排名第 25，較官方 baseline 0.2845 大幅提升 +0.5786），且 824896 基底與最終版 851729 皆已通過乾淨還原環境之逐位元組重現驗證（SHA-256 前八碼 2501f2f4）。如圖一所示，系統分數由單一開局模型與 seqB（0.8012），隨引入 OGS 雙編碼器與多層開局鍵（0.8388）、27 維 LLR（0.8465）、G3 局部棋形（0.8536）、EB50 異向先驗（0.8543）、逐 k 融合與兩輪偽標籤擴充（0.8620），最後納入 30-block 深化編碼器 A3L-9600 推進至 0.8631；主要階段在驗證台封存 pool 1 皆獲高度統計顯著增益（z 值介於 5.66 至 17.6）。",
            "成功關鍵與機制歸因如圖二(a)：玩家條件 LLR 與逐手局部棋形（G3）合計帶來最大突破（+0.041），印證「以棋力模型條件化後之逐手殘差對數似然比」相較整盤平均嵌入更能保留細微手癖；經驗貝氏先驗精準壓縮低訊噪維度，使 10 盤極少樣本也能穩定估計。從圖二(b) 可見誤差結構與成功／失敗案例規律：全黑題在 k=5 達 0.964，但 k=1 全白題僅 0.523（落後全黑 k=1 的 0.598 達 0.075），顯示白棋受對手開局主導較深且資訊量較少；實際檢查誤判案例（如單盤 k=1 短局或與他人口味相近之對局），往往源於候選庫僅 10 盤的參考抽樣變異，而透過偽標籤擴充（官方 pub2 選中 293 題、priv2 選中 305 題）與 A3L 多樣性集成能進一步修正困難對局排名。",
            "同樣重要的是失敗與淘汰方向之排除：我們以嚴格封存驗證淘汰了六條表面合理但無效的路線——一、難負例配對訓練（815766）：診斷顯示誤認為連續分散的相似度，固定誤認對僅占 0.81%，不具結構性突破空間；二、編碼器 8 向 D4 TTA（+0.0007，z=0.53）：編碼器已學習絕對方位習慣，強行八向平均反而稀釋訊號；三、中心擴充 EM（+0.0018，z=2.30）未達顯著門檻；四、反向似然 RLLR（+0.000004）：因不同候選人自身棋局長度與偏離尺度不可比，反遭族群基準偏差淹沒；五、題目合併（E1／v2，AP 0.51–0.67，pool 1 −0.00003）：高分配對多為後驗已一致的題目，無法挽救真正誤判題；六、較短步數之 7,200 步 A3／A3L（pool 1 −0.0018／−0.0010）與最後五向風格混搭 851772（Public 0.8626）皆未優於 851729 而遭排除。",
            "未來改進方向包括：設計針對白棋執色之專屬動態開局／佈局特徵以縮小黑白落差、探索批內線上困難負例挖掘（online hard negative mining），以及引入更強多尺度區域對局網路強化單盤 k=1 辨識力。",
        ],
        "柒、程式碼": [
            f"Github連結：{TEAM_CONFIG['github_url']}",
            "程式碼目錄結構與主要模組輸入／輸出說明如下（詳細安裝與重現指令見專案根目錄 README.md）：",
            "一、資料處理與共用指標（src/go_aicup/data/）：sgf.py 負責 SGF 棋譜解析與屬性稽核；preprocess.py 將官方 CSV 轉為壓縮 memmap 與 GameStore 元資料；board_cache.py 負責 2×46-byte 位元打包與盤面歷史組裝；cv.py 統一產生玩家互斥切分與驗證台抽樣；metrics.py 實作兩賽官方評估指標。",
            "二、競賽一核心與推論（src/go_aicup/tasks/identity/ 與 scripts/identity/）：ge2e.py、train_ge2e.py 與 train_e2e.py 實作 ResNet 風格編碼器、seqB 頭與雙色 GE2E 訓練；predict.py 與 insurance.py 執行原測試集 D4 開局＋GE2E＋Hungarian 指派；pid2_multilevel_export.py 計算 L1/L2 多層開局鍵；likelihood_rich.py 與 pid2_likelihood.py 實作 Numba 盤面重播、50 維特徵、GPU 阻尼牛頓 MAP、EB 先驗、逐 k conditional logit 融合、A3L 整合與兩輪留一偽標籤匯出。",
            "三、環境建置與一鍵重現流程：依 README.md 執行 pip install -r requirements.txt 與 pip install -e . 安裝鎖定套件，配置 KataGo HumanSL checkpoint 與預訓編碼器權重後，依序執行 scripts/release/repro_pid_setup.sh、repro_pid_run.sh 與 SLURM_JOB_ID=850000 python -m scripts.identity.pid2_likelihood export --eb --combine --pseudo --a3 --a3l 即可逐位元組重現最終提交檔 851729-comb-pseudo-a3l.csv。外部預訓權重 b18c384nbt-humanv0 可直接由 KataGo 官方 GitHub Release（v1.15.0）下載；本隊自訓權重與驗證參數可依 README.md 與設定檔重訓產生，完整凍結權重亦已留存備查，如需審核可立即提供。",
        ],
        "捌、使用的外部資源與參考文獻": [
            "一、外部資料集與預訓模型／開源工具：1. KataGo HumanSL 網路 b18c384nbt-humanv0.ckpt（KataGo v1.15.0 release，KataGo Neural Network License）：於競賽一凍結作為玩家條件似然比基底（搭配我方 C v2 級別向量），貢獻驗證台 +0.0301（LLR）至總融合推進至 0.8631。2. OGS（Online-Go.com）公開棋譜合集 2013–2025-05（za3k.com/ogs，經 OGS 授權重散布與研究使用）：用於 A1／A2／A3L 風格編碼器預訓，A1／A2 貢獻 Public +0.0143（0.8245→0.8388）。3. KGS 高段棋譜庫（u-go.net/gamerecords/，55 檔經清洗後保留 135,806 盤）：僅用於訓練 30-block 深層編碼器 A3L-9600，貢獻 Public 0.8620→0.8631（+0.0010）。4. 開源套件 PyTorch 2.6.0（BSD）、Numba 0.65.0（BSD）、scikit-learn 1.7.2（BSD-3）、KataGo v1.18.2 Python 模組（MIT）。",
            "二、生成式 AI 工具使用揭露：本專案開發與報告製作過程如實使用生成式 AI 輔助，包括 Anthropic Claude Code（Claude Opus 5.5）與 OpenAI Codex 協助撰寫實驗腳本、資料稽核、單元測試與自動化重現管線，以及 Google Antigravity（Gemini）協助彙整實驗帳本、繪製分析圖表與編排本報告初稿；所有演算法設計、實驗假設、門檻裁決、程式碼審閱與實測驗證皆由參賽隊員親自執行與確認。",
            "三、參考文獻（APA 格式）：",
            "Chen, Y.-C., Shih, C.-H., & Wu, I.-C. (2025). Strength estimation and human-like strength adjustment in games. International Conference on Learning Representations (ICLR 2025).",
            "Kuboki, R., et al. (2025). Policies of multiple skill levels for better strength estimation in games. arXiv preprint arXiv:2505.00279.",
            "McIlroy-Young, R., Wang, R., Sen, S., Kleinberg, J., & Anderson, A. (2021). Detecting individual decision-making style: Exploring behavioral stylometry in chess. Advances in Neural Information Processing Systems (NeurIPS 2021), 34.",
            "Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2018). Generalized end-to-end loss for speaker verification. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2018), 4879–4883.",
            "Wu, D. J. (2019). Accelerating self-play learning in Go. arXiv preprint arXiv:1902.10565. (KataGo / HumanSL framework).",
        ],
    }


def build_c2_sections() -> dict[str, list[str]]:
    return {
        "壹、環境": [
            "本系統於 Windows 11 本地端工作站（NVIDIA RTX 3060 Ti 8GB、CUDA 12.4）進行前處理與邏輯測試，正式大規模特徵抽取、模型擬合與多卡推論執行於國家高速網路與計算中心「晶創 26」（Nano4）高效能叢集（Red Hat Enterprise Linux x86_64、Slurm dev 與 8gpus 佇列，單節點配置最多 8 張 NVIDIA H200 SXM5 141GB 顯卡、2 顆 Intel Xeon Platinum 8480+ 共 112 核心與 2TB 主記憶體，節點內透過 NVLink 互連）。",
            "執行環境採用 Python 3.13.12（Conda 環境 go），依 requirements.txt 鎖定 PyTorch 2.6.0+cu124、scikit-learn 1.7.2、NumPy 2.3.5、Pandas 2.3.3、PyArrow 20.0.0、SciPy 1.16.3、Numba 0.65.0 與 Matplotlib 3.10.8。外部推論引擎使用 KataGo v1.18.2 預編譯執行檔（CUDA 12.8／cuDNN 9.8 版，搭配從 PyPI 解出之 NVIDIA CUDA 12.8 與 cuDNN 9.8.0.87 執行期函式庫於計算節點穩定執行）。",
            "外部預訓權重與資料來源：一、KataGo HumanSL 網路 b18c384nbt-humanv0.bin.gz 與 PyTorch 權重 b18c384nbt-humanv0.ckpt（來自 GitHub lightvector/KataGo v1.15.0 release）；前者抽取 15 個段位 profile 之每手對數機率，後者供我方學習 10 個幽玄級別專屬條件向量 C v2 與計算對手 LLR 重排。二、OGS 2013–2025-05 公開棋譜合集（za3k.com/ogs）透過競賽一預訓編碼器 A1／A2 間接協助本系統偵測測試題之訓練玩家對手身分，僅用於風格比對。",
        ],
        "貳、演算方法與模型架構": [
            "最終提交檔 849173-opp7-l10.csv 採用「自身棋力 54 維 B2 題級模型」結合「對手身分連結與 LLR 重排（opp7）」、「同玩家測試題合併」及「Bayes 最佳期望分數解碼」，並輔以高信心自身辨識覆寫（L10）。",
            "第一支線（自身棋力 54 維 B2 模型）：針對每盤棋（HD0 貼目 6.5、HD1 貼目 0.5，1 visit），抽取 KataGo HumanSL 在 15 個段位 profile（rank_20k–rank_9d）下實際落子之平均 log-likelihood（本人 15 維＋對手 15 維，共 H 30 維）；同時凍結 HumanSL 主幹並為幽玄 10 個級別各學習本人與對手兩組 384 維條件向量取代原 metadata 編碼（C v2，輸出 10 級假設下本人與對手共 20 維特徵）；再加上 2 維讓先旗標（HD1 指示與修正）與 2 維於幽玄擬合池以玩家分組 5 折 out-of-fold 訓練之非線性 HistGradientBoostingRegressor 迴歸輸出（yH，分別輸入本人／對手之 30 維 H 特徵），組成 54 維盤特徵。於題目層級取 k 盤平均，對每個 k∈{5,10,15,20} 各訓練一個多類 logistic regression（B2）並執行訓練至測試集之先驗對齊。",
            "第二支線（對手身分連結 opp7）：利用日本幽玄網站依棋力配對與讓子規則（HD0 同級、HD1 差一級），先以競賽一混合偵測器 f8（D4 開局＋3 組風格編碼器 B／A1／A2）對 8,007 位訓練玩家建庫（每人至多 40 盤 HD≤1）計算各測試盤對手方之前十名候選，再使用 40 盤參考棋擬合之 27 維玩家條件似然比（LLR，μ=2）對前十名重排，建立 f′＝f8＋2·LLR。依 f′ 分數差之虛無分布分位數（q90／q99／q99.9／max）分五桶，以軟連結權重 w_i＝softmax((f′_i−f′_1)/3.5) 查表由 Valid-S 20,000 盤估計之相對級距核函數 P_Δ(r_j−r | 分段, HD, 執色)，得單盤對數證據 ℓ_g(r)。",
            "最終融合將先驗對齊後之 B2 對數機率與對手證據結合：log P(r|題)＝log P_B2(r)＋0.55·Σ_g ℓ_g(r)，亦把經風格向量高信心比對出之 9 組（21 題）同玩家測試題特徵合併計分，最後依官方「精確 1 分、差 ±1 級得 1/e、其餘 0」指標採用 Bayes 最佳解碼 score_rank＝argmax_r [p[r]＋(p[r−1]＋p[r+1])/e]，並以高信心訓練玩家比對表 L10 覆寫 7 道已確認級別題。",
        ],
        "參、創新性": [
            "本系統針對競賽二核心挑戰——「評分指標非線性（差兩級即零分）與 8,007 位玩家標籤雜訊」——提出四項關鍵創新設計。首先，推導並實作對應官方指標之 Bayes 最佳解碼器（metrics.score_rank）：一般分類採用機率 argmax 或序數期望值，於兩級誤差全零分下皆非最佳解；本隊精確最大化期望官方得分 p[r]＋(p[r−1]＋p[r+1])/e，自動壓抑無雙向鄰居分數保護之邊界段位（D、6D），實證穩定優於 argmax 與期望值解碼。",
            "其次，提出「自學幽玄十級條件向量（C v2）與讓先貼目校正（P3）」：由於 KataGo 原始 HumanSL 使用 KGS 段位體系且預設互先貼 6.5 目，與日本幽玄十級定義及讓先無貼目（HD1）存在域落差，本隊凍結網路權重僅學習 20 組 384 維條件向量重映射級別空間，並於 HD1 棋局將 KataGo 貼目調為 0.5 目以吻合實際規則。",
            "第三，首創「跨任務對手身分連結與 LLR 重排（opp7）」：我們打破單獨從查詢玩家棋譜推論棋力的框架，實證測試集約 60%–77% 的對手即為訓練集已知段位之 8,007 位玩家（官方測試 10,000 盤中高達 700 盤突破 q99 高信心門檻），且幽玄配對規則保證 HD0 幾乎同級（|Δ|≤1 達 94%）、HD1 精準反映一級強弱差。我們進一步將競賽一的 27 維 LLR 模型跨賽遷移至此對 f8 前十候選進行重排（將參考盤單盤 top-1 辨識率由 0.3367 提升至 0.3809），並以軟連結 DELTA 核函數平滑吸收不確定性，取代硬門檻挑選，使 Public LB 躍升 +0.0182（0.7563→0.7744）。",
            "第四，建構「風格同人測試題合併（qmerge）」與「玩家層級非線性 H 迴歸特徵（yH）」，從多個正交維度榨取棋局結構訊號。",
        ],
        "肆、資料處理": [
            "訓練集共 1,000,000 盤、8,007 位玩家，稽核確認「每位玩家僅隸屬單一棋力段位（跨級玩家數為 0）」，且各級玩家數分布懸殊（1D 僅 433 人、A 級高達 1,582 人）。若採傳統棋局層級隨機切分（game-disjoint），深度模型極易記憶玩家個人習慣而非棋力（實驗證實此類模型在未見玩家表現崩潰），故本隊全專案統一由 cv.py 強制實作「玩家互斥（player-disjoint）GroupKFold」切分，劃分完全未見玩家群 Valid-U（評估泛化）、訓練玩家留出盤 Valid-S（擬合 B2 與對手核）及每級補充 20 盤 train20。",
            "針對官方 v1.1 公布之讓子欄位 HD[n]，稽核發現測試集 Public／Private 共 10,000 盤全數為 HD0 或 HD1，不含 HD≥2；而訓練集 36,421 盤 HD≥2 棋譜不僅缺讓子星位擺子，其 SGF B/W 標記亦與實際落子相反。因此所有訓練、特徵抽取與驗證題組一律嚴格過濾為 HD≤1，確保資料分布與官方測試集完全一致；同時將 HD1 旗標納入模型並針對 HD1 棋局以貼目 0.5 重算 H 與 C v2 特徵，避免「讓先無貼目」被誤判為棋風異常。",
            "另外完成關鍵洩漏稽核：檢查測試集 10,000 盤前 40 手及其全部 8 種 D4 對稱變換與兩兩互比，確認測試棋局與訓練集零重疊、測試題之間亦零共用棋局、題號與段位無序號相關（Public Spearman p=0.12），排除任何資料洩漏疑慮。盤面處理統一將白棋對調通道轉成「我方／對方」視角，確保黑白雙色輸入分布一致。",
        ],
        "伍、訓練方式": [
            "本系統採取模組化分階段訓練、玩家分組交叉驗證與嚴格門檻控管流程，所有隨機種子、切分身分與硬體配置均寫入設定檔留存，主要步驟如下：",
            "一、盤面主幹與 C v2 級別條件向量：先於官方 HD≤1 棋局訓練 20-block 盤面 ResNet 骨幹，接著凍結 HumanSL 網路 b18c384nbt-humanv0.ckpt 主幹參數，僅最佳化幽玄 10 個級別各自對應的本人與對手 384 維條件向量（共 20 組向量），以 10 級交叉熵損失訓練並嚴格保留玩家切分邊界；此舉在維持 HumanSL 棋感之前提下，將 KGS 段位空間精準對齊至幽玄十級。",
            "二、非線性 H 特徵 yH 與 B2 題級分類器：於完整擬合池（包含 Valid-S 全部、train20 每級 20 盤與 Valid-U HD≤1 棋局）上，以玩家分組 5 折 out-of-fold 方式訓練 HistGradientBoostingRegressor（設定 400 輪迭代、學習率 0.05、31 葉、每葉至少 50 樣本、L2 正則化 1.0），將 30 維 H 特徵映射為單盤連續棋力估計（擬合池 OOF 預測與真實級別相關係數達 0.904）；正式推論階段則使用全擬合池重訓之模型產生特徵。隨後針對題目層級分類器 B2，對每個 k∈{5,10,15,20} 各從同一玩家同色棋局抽取 k 盤組成 20,000 道合成題，以 scikit-learn 多類 logistic regression 擬合 54 維平均特徵，並執行訓練至測試分布的先驗對齊；所有特徵消融與超參數驗證皆須通過五組獨立隨機種子（2027–2031）在未見玩家題組 E1／E2 之檢定。",
            "三、對手辨識重排與 DELTA 核函數擬合：沿用競賽一預訓之開局與三組風格編碼器（B／A1／A2）建立 8,007 位訓練玩家至多 40 盤之標準庫；再於 7,969 名合格訓練玩家（113,028 盤、每人至多 40 盤 HD≤1）抽取凍結 HumanSL＋C v2 基底前 32 名 policy，以 GPU 批次阻尼牛頓法求取 27 維對數線性殘差參數 θ（高斯先驗 σ=3）。針對每盤前十名對手候選重算重排分數 f′＝f8＋2·LLR（權重 μ=2 選定自獨立參考玩家半數校準集），再於 Valid-S 20,000 盤擁有者已知盤上依信心分段、讓子與執色統計相對段位差分布（加平滑常數 0.5）建立 DELTA 核函數，最後以本地 E1／E2 五種子交叉驗證選定軟連結溫度 τ=3.5 與證據權重 α=0.55。",
        ],
        "陸、分析與結論": [
            "競賽二最終版本 849173-opp7-l10.csv 於 Private LB 取得 0.7423199（排名第 19），Public LB 為 0.7744441（排名第 15，較官方 baseline 0.4631 高出 +0.3113）。演進軌跡如圖一所示：純手工特徵（0.412）未過門檻；盤面 V2 ResNet＋C2 跨過門檻（0.5553）；導入凍結 KataGo HumanSL 特徵大幅推升至 0.6833（+0.130）；經過 HD1 貼目校正、C v2 級別向量、先驗對齊與測試同人題合併（0.7531）、yH 非線性特徵（0.7563），最終透過「對手身分連結與 LLR 重排」將 Public LB 推進至 0.7726（opp2）與 0.7744（opp7），證明跨賽整合對手棋風特徵與讓子段位規則為本賽最大槓桿。",
            "從圖二(a) 可見盤數 k 對準確率的關鍵影響：隨每題棋局數由 k=5 增至 k=20，cvec2 B2 模型得分由 0.7022 穩步上升至 0.7740，且引入 opp7 對手連結後在五個隨機種子下對 opp2 全面正增益（E1 +0.0137、E2 +0.0113）。圖二(b) 直接驗證對手連結機制並非巧合：在高信心 q99.9 門檻下，HD0 連結之對手與本人同級或差一級比例高達 94%（完全同級 51%），遠高於隨機基準 29%；且 HD1 讓先時，連到之對手平均強於執黑者 +0.38 級、弱於執白者 −0.51 級，完全符合圍棋讓子棋理。成功案例包括同玩家跨 Public／Private 之四題互連群組（pub_q_0229 等）透過合併穩定判級，以及原本 k=5 盤少雜訊大之題目因識別出訓練集對手而修正一級誤差。",
            "同時我們系統性記錄多項失敗與拒絕方案以確保不對 400 題 Public LB 過擬合：一、純監督與自回歸深層骨幹（V1、SE、SE3）若未嚴格玩家切分，於 game-disjoint 驗證雖達 0.6239，在未見玩家隨即嚴重退化；二、KataGo 強網路著手品質特徵（S-select −0.0024）與 KataRank Lite／Full 29 級特徵（E2 −0.0088／−0.0080）均呈現負增益，顯示在 HumanSL 多段位剖面前，單純強弱勝負損失難以分辨接近的業餘段位；三、OGS 外部 25,777 盤段級迴歸特徵（E1 −0.0011／E2 −0.0009）未能超越 H 特徵，表示 HumanSL 已充分涵蓋外部段位分布；四、全池神經殘差頭（840005）在嚴格玩家切分下殘差收斂趨近於零，且 50 維 LLR 重排於 E1／E2 出現分歧（−0.0010 vs +0.0024），故依事前準則保留較穩健且不依賴 KGS 之 opp7 為最終版本（亦附帶觀察含 KGS A3 之 opp6 探針 Public 0.7758，僅差 0.0013）。",
            "總結而言，結合人類棋力概率模型、日本讓子段位結構與玩家身分再辨識，有效克服了業餘棋譜雜訊與玩家記憶過擬合；未來可進一步研究基於注意力機制之逐手對手／本人聯合編碼，以及更精細的時間段位漂移校準。",
        ],
        "柒、程式碼": [
            f"Github連結：{TEAM_CONFIG['github_url']}",
            "程式碼目錄結構與主要模組輸入／輸出說明如下（詳細安裝與重現指令見專案根目錄 README.md）：",
            "一、資料處理與共用基礎（src/go_aicup/data/ 與 features/）：sgf.py 與 preprocess.py 負責官方 SGF 稽核與 GameStore 建置；board_cache.py 實作盤面打包與歷史通道；cv.py 強制產生玩家互斥 GroupKFold 切分（Valid-U／Valid-S／train20）；humansl.py 封裝 KataGo v1.18.2 批次抽取 15 個段位 profile 之每手對數機率；humansl_rankvec.py 負責凍結 HumanSL 與學習 C v2 條件向量；metrics.py 實作官方分數與 score_rank 解碼。",
            "二、競賽二核心、對手連結與推論（src/go_aicup/tasks/strength/ 與 scripts/strength/）：rank_prod_a0_fusion.py 與 rank_ogs.py 實作 54 維特徵、yH 玩家分組 OOF 迴歸與 B2 分 k 擬合；rank_qmerge.py 執行同人測試題合併；rank_opp_link2.py 建置 f8 混合偵測器對 8,007 訓練玩家比對；rank_opp_llr.py 與 rank_opp_mu2_diag.py 執行 40 盤 27 維 LLR 重排（μ=2）、軟連結 DELTA 核融合（τ=3.5、α=0.55）、score_rank 解碼及 L10 覆寫匯出。",
            "三、環境建置與重現步驟：執行 pip install -r requirements.txt 與 pip install -e . 安裝鎖定套件，配置 KataGo v1.18.2 與 HumanSL 權重後，依序執行 rank_opp_link2、rank_opp_llr 與 rank_opp_mu2_diag.py 即可重現最終推論與報告數據。外部預訓權重 b18c384nbt-humanv0 可直接由 KataGo 官方 GitHub Release（v1.15.0）下載；本隊自訓權重與驗證參數可依 README.md 與設定檔重訓產生，完整凍結權重亦已留存備查，如需審核可立即提供。",
        ],
        "捌、使用的外部資源與參考文獻": [
            "一、外部資料集與預訓模型／開源工具：1. KataGo HumanSL 網路 b18c384nbt-humanv0.bin.gz 與 .ckpt（KataGo v1.15.0 release，KataGo Neural Network License）：作為本系統棋力特徵 H（Valid-U 由 0.5532 提升至 0.6833，+0.130）與 C v2 條件向量／對手 LLR 重排之核心基底。2. KataGo v1.18.2 引擎（MIT）與 NVIDIA CUDA 12.8／cuDNN 9.8 執行期函式庫：供批次抽取 HumanSL 逐手機率。3. OGS 公開棋譜合集 2013–2025-05（za3k.com/ogs，經 OGS 授權重散布與研究使用）：僅透過預訓之競賽一風格編碼器 A1／A2 用於計算對手身分餘弦相似度，協助 opp7 推升 Public LB 至 0.7744。4. scikit-learn 1.7.2（BSD-3）、PyTorch 2.6.0（BSD）、Numba 0.65.0（BSD）。",
            "二、生成式 AI 工具使用揭露：本專案開發與報告撰寫過程如實使用生成式 AI 輔助，包括 Anthropic Claude Code（Claude Opus 5.5）與 OpenAI Codex 協助撰寫實驗腳本、資料稽核、重現測試與自動化管線，以及 Google Antigravity（Gemini）協助彙整實驗數據、繪製分析圖表與編排本報告初稿；所有演算法發想、數學推導、超參數驗證、程式碼審閱與結果核對皆由隊員親自完成並負責。",
            "三、參考文獻（APA 格式）：",
            "Chen, Y.-C., Shih, C.-H., & Wu, I.-C. (2025). Strength estimation and human-like strength adjustment in games. International Conference on Learning Representations (ICLR 2025).",
            "Kuboki, R., et al. (2025). Policies of multiple skill levels for better strength estimation in games. arXiv preprint arXiv:2505.00279.",
            "McIlroy-Young, R., Wang, R., Sen, S., Kleinberg, J., & Anderson, A. (2021). Detecting individual decision-making style: Exploring behavioral stylometry in chess. Advances in Neural Information Processing Systems (NeurIPS 2021), 34.",
            "Wan, L., Wang, Q., Papir, A., & Moreno, I. L. (2018). Generalized end-to-end loss for speaker verification. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2018), 4879–4883.",
            "Wu, D. J. (2019). Accelerating self-play learning in Go. arXiv preprint arXiv:1902.10565. (KataGo / HumanSL framework).",
        ],
    }


# ---------------------------------------------------------------------------
# Word Formatting Helpers
# ---------------------------------------------------------------------------
def _set_run_font(run, size_pt: float, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")
    rfonts.set(qn("w:cs"), "Times New Roman")
    rfonts.set(qn("w:eastAsia"), "標楷體")


def _add_paragraph(doc: docx.Document, text: str, *, size_pt: float = 12.0,
                   bold: bool = False, align=WD_ALIGN_PARAGRAPH.LEFT,
                   space_before: float = 0.0, space_after: float = 4.0) -> docx.text.paragraph.Paragraph:
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    run = p.add_run(text)
    _set_run_font(run, size_pt=size_pt, bold=bold)
    return p


def _set_cell_borders(cell) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        "</w:tcBorders>"
    )
    tcPr.append(borders)


def _fill_cell(cell, text: str, *, size_pt: float = 10.5, bold: bool = False,
               align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    _set_cell_borders(cell)
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    _set_run_font(run, size_pt=size_pt, bold=bold)


def build_report_docx(comp_key: str, sections: dict[str, list[str]], figures: list[tuple[Path, str]]) -> Path:
    cfg = TEAM_CONFIG[comp_key]
    doc = docx.Document()

    # Configure A4 portrait and exact official margins
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.17)
    sec.right_margin = Cm(3.0)

    # Configure Normal style fonts
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    nrpr = normal._element.get_or_add_rPr()
    nrfonts = OxmlElement("w:rFonts")
    nrfonts.set(qn("w:ascii"), "Times New Roman")
    nrfonts.set(qn("w:hAnsi"), "Times New Roman")
    nrfonts.set(qn("w:cs"), "Times New Roman")
    nrfonts.set(qn("w:eastAsia"), "標楷體")
    nrpr.insert(0, nrfonts)

    # Title block (20pt Bold)
    _add_paragraph(doc, "AI CUP 2026 秋季賽", size_pt=20.0, bold=True,
                   align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=2)
    _add_paragraph(doc, f"{cfg['title']}報告", size_pt=20.0, bold=True,
                   align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=10)

    # Metadata lines (12pt)
    member_line = "、".join(
        f"{m['zh_name']}（{m['role']}）" if m["role"] == "隊長" else m["zh_name"]
        for m in TEAM_CONFIG["members"]
    )
    _add_paragraph(doc, f"隊伍：{cfg['team_id']}", size_pt=12.0, space_after=2)
    _add_paragraph(doc, f"隊員：{member_line}", size_pt=12.0, space_after=2)
    _add_paragraph(doc, f"指導教授：{TEAM_CONFIG['advisor']}", size_pt=12.0, space_after=2)
    _add_paragraph(
        doc,
        f"Private leaderboard：{cfg['private_score']} / Rank {cfg['private_rank']}"
        f"（參考 Public LB：{cfg['public_score']} / Rank {cfg['public_rank']}，最終提交檔：{cfg['final_submission']}）",
        size_pt=12.0, space_after=8
    )

    # 8 Official Sections in exact order (20pt Bold headings per official template)
    print(f"\n=== Word-Count Audit: {cfg['team_id']} ({comp_key.upper()}) ===")
    for heading, paras in sections.items():
        _add_paragraph(doc, heading, size_pt=20.0, bold=True, space_before=10, space_after=4)
        joined = "\n".join(paras)
        w_ms = count_msword_words(joined)
        w_cjk = count_cjk_chars(joined)

        if heading in SECTION_LIMITS:
            lo, hi = SECTION_LIMITS[heading]
            status = "OK" if (lo <= w_ms <= hi and lo <= w_cjk <= hi) else "OUT_OF_RANGE"
            print(f"  {heading:<18} | MS-Word≈{w_ms:4d} | CJK={w_cjk:4d} | Limit=[{lo},{hi}] -> {status}")
            assert lo <= w_cjk <= hi, f"{cfg['team_id']} {heading} CJK={w_cjk} outside [{lo},{hi}]"
            assert lo <= w_ms <= hi, f"{cfg['team_id']} {heading} MS-Word={w_ms} outside [{lo},{hi}]"
        else:
            print(f"  {heading:<18} | MS-Word≈{w_ms:4d} | CJK={w_cjk:4d} | (No strict word limit)")

        for ptxt in paras:
            _add_paragraph(doc, ptxt, size_pt=12.0, space_after=4)

        if heading == "陸、分析與結論":
            for fig_path, caption in figures:
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(4)
                p_img.paragraph_format.space_after = Pt(2)
                p_img.paragraph_format.line_spacing = 1.0
                run_img = p_img.add_run()
                run_img.add_picture(str(fig_path), width=Cm(14.5))
                _add_paragraph(doc, caption, size_pt=10.5, bold=False,
                               align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=6)

    # Page break + Author Contact Table
    doc.add_page_break()
    _add_paragraph(doc, "作者聯絡資料表", size_pt=20.0, bold=True,
                   align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=8)

    table = doc.add_table(rows=7, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_widths = [Cm(1.25), Cm(2.55), Cm(2.55), Cm(2.73), Cm(2.35), Cm(3.40)]
    for row in table.rows:
        for idx, w in enumerate(col_widths):
            row.cells[idx].width = w

    # Row 0: Team summary
    _fill_cell(table.cell(0, 0), "隊伍名稱", size_pt=9.5, bold=True)
    _fill_cell(table.cell(0, 1), cfg["team_id"], size_pt=9.5)
    _fill_cell(table.cell(0, 2), "Private\nLeaderboard\n成績", size_pt=9.0, bold=True)
    _fill_cell(table.cell(0, 3), str(cfg["private_score"]), size_pt=9.5)
    _fill_cell(table.cell(0, 4), "Private\nLeaderboard\n名次", size_pt=9.0, bold=True)
    _fill_cell(table.cell(0, 5), str(cfg["private_rank"]), size_pt=9.5)

    headers = [
        "身分\n（隊長/隊員）",
        "姓名\n(中英皆需填寫；英文為名, 姓，例：Xiao-Ming, Wu)",
        "學校＋系所中文全稱\n(請填寫完整全名，勿縮寫)",
        "學校＋系所英文全稱\n(請填寫完整全名，勿縮寫)",
        "電話",
        "E-mail",
    ]
    for ci, htxt in enumerate(headers):
        _fill_cell(table.cell(1, ci), htxt, size_pt=9.0, bold=True)

    for r_offset in range(5):
        row_idx = 2 + r_offset
        if r_offset < len(TEAM_CONFIG["members"]):
            m = TEAM_CONFIG["members"][r_offset]
            _fill_cell(table.cell(row_idx, 0), m["role"], size_pt=9.0, bold=True)
            _fill_cell(table.cell(row_idx, 1), f"{m['zh_name']}\n{m['en_name']}", size_pt=9.0)
            _fill_cell(table.cell(row_idx, 2), m["zh_org"], size_pt=9.0)
            _fill_cell(table.cell(row_idx, 3), m["en_org"], size_pt=9.0)
            _fill_cell(table.cell(row_idx, 4), m["phone"], size_pt=9.0)
            _fill_cell(table.cell(row_idx, 5), m["email"], size_pt=7.5)
        else:
            _fill_cell(table.cell(row_idx, 0), f"隊員{r_offset}", size_pt=9.5)
            for ci in range(1, 6):
                _fill_cell(table.cell(row_idx, ci), "", size_pt=9.5)

    _add_paragraph(
        doc,
        "★註1：請確認上述資料與 AI CUP 報名系統中填寫之內容相同。自 2023 年起，獎狀製作將依據報名系統中填寫內容為準，有特殊狀況需修正者，請主動於報告繳交期限內來信 moe.ai.ncu@gmail.com。報告繳交截止時間後將不予修改。",
        size_pt=12.0, space_before=8, space_after=2
    )
    _add_paragraph(
        doc,
        f"★註2：報告 word 檔名請統一格式為：「{cfg['filename'].removesuffix('.docx')}」",
        size_pt=12.0, space_before=0, space_after=0
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / cfg["filename"]
    doc.save(out_path)
    return out_path


def main() -> None:
    c1_figs = generate_c1_figures()
    c2_figs = generate_c2_figures()
    p_c1 = build_report_docx("c1", build_c1_sections(), c1_figs)
    p_c2 = build_report_docx("c2", build_c2_sections(), c2_figs)
    print("\nGenerated reports:")
    print(f"  C1: {p_c1} ({p_c1.stat().st_size / 1024:.1f} KiB)")
    print(f"  C2: {p_c2} ({p_c2.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
