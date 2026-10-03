"""W1 audit of the training set.

Answers the questions in STRATEGY.md section 2 that are answerable from the
training data alone (the candidate pools and test files are not released yet):

  * which SGF properties exist at all -- decides whether HA/KM/DT/AP are even
    candidates once the test files land
  * whether setup stones (AB/AW) betray a handicap even when HA is absent
  * player_id structure: how many players, games each, and whether a player
    ever spans more than one rank (this decides how GroupKFold must be built)
  * whether game_id encodes rank -- a leak we must ban in feature_policy
  * first-move distribution, i.e. whether the opening fingerprint (L7) has
    enough spread to identify players
  * duplicate SGF content within train

Usage:
    python scripts/audit_train.py --limit 2000       # fast smoke pass
    python scripts/audit_train.py                    # full pass (reads 1.3 GB)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import pathlib
import time

import numpy as np
import pandas as pd

from go_aicup.data import sgf as sgflib
from go_aicup.paths import ROOT

TRAIN_DIR = ROOT / "data" / "training"
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
CHUNK = 20_000


def _train_files() -> list[tuple[str, pathlib.Path]]:
    files = []
    for rank in RANKS:
        path = TRAIN_DIR / f"train_{rank}.csv"
        if path.exists():
            files.append((rank, path))
        else:
            print(f"  ! missing {path.name}")
    return files


def scan_metadata(files) -> dict:
    """Pass 1: everything except sgf_content."""
    frames = []
    for rank, path in files:
        frame = pd.read_csv(path, usecols=["player_id", "game_id", "rank", "color"])
        frames.append(frame)
        print(f"  {path.name}: {len(frame):,} rows")
    meta = pd.concat(frames, ignore_index=True)

    meta["game_num"] = (
        meta["game_id"].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    )

    per_player = meta.groupby("player_id", observed=True)
    ranks_per_player = per_player["rank"].nunique()
    games_per_player = per_player.size()

    game_id_by_rank = (
        meta.groupby("rank", observed=True)["game_num"].agg(["min", "max", "count"])
    )

    return {
        "total_rows": len(meta),
        "n_players": int(meta["player_id"].nunique()),
        "n_game_ids": int(meta["game_id"].nunique()),
        "ranks_per_player": ranks_per_player,
        "games_per_player": games_per_player,
        "impure_players": int((ranks_per_player > 1).sum()),
        "color_by_rank": pd.crosstab(meta["rank"], meta["color"]),
        "players_by_rank": meta.groupby("rank", observed=True)["player_id"].nunique(),
        "game_id_by_rank": game_id_by_rank,
        "player_id_len": meta["player_id"].astype(str).str.len().value_counts(),
    }


def scan_sgf(files, limit: int | None) -> dict:
    """Pass 2: parse sgf_content."""
    prop_counts: collections.Counter = collections.Counter()
    prop_by_rank: dict[str, collections.Counter] = {}
    move_counts: list[int] = []
    error_counts: collections.Counter = collections.Counter()
    first_black: collections.Counter = collections.Counter()
    setup_games = 0
    board_sizes: collections.Counter = collections.Counter()
    seen_hashes: dict[bytes, str] = {}
    duplicates: list[tuple[str, str]] = []
    parsed = 0

    for rank, path in files:
        rank_props: collections.Counter = collections.Counter()
        taken = 0
        reader = pd.read_csv(
            path, usecols=["game_id", "color", "sgf_content"], chunksize=CHUNK
        )
        for chunk in reader:
            if limit is not None and taken >= limit:
                break
            if limit is not None:
                chunk = chunk.iloc[: limit - taken]
            for game_id, color, content in zip(
                chunk["game_id"], chunk["color"], chunk["sgf_content"]
            ):
                game = sgflib.parse(content)
                parsed += 1
                taken += 1

                for key in game.props:
                    prop_counts[key] += 1
                    rank_props[key] += 1
                if game.moves.size:
                    prop_counts["B/W"] += 1
                    rank_props["B/W"] += 1

                move_counts.append(game.n_moves)
                board_sizes[game.board_size] += 1
                if game.setup_black or game.setup_white:
                    setup_games += 1
                for error in game.errors:
                    error_counts[error.split()[0]] += 1

                black = game.moves[game.colors == sgflib.BLACK]
                if black.size:
                    first_black[int(black[0])] += 1

                digest = hashlib.blake2b(
                    content.encode("utf-8"), digest_size=8
                ).digest()
                previous = seen_hashes.get(digest)
                if previous is not None:
                    duplicates.append((previous, str(game_id)))
                else:
                    seen_hashes[digest] = str(game_id)

        prop_by_rank[rank] = rank_props
        print(f"  {path.name}: parsed {taken:,}")

    return {
        "parsed": parsed,
        "prop_counts": prop_counts,
        "prop_by_rank": prop_by_rank,
        "move_counts": np.asarray(move_counts),
        "error_counts": error_counts,
        "first_black": first_black,
        "setup_games": setup_games,
        "board_sizes": board_sizes,
        "duplicates": duplicates,
    }


def verify_roundtrip(files, n: int = 500) -> tuple[int, int]:
    """Re-encode parsed moves back to SGF coordinates and compare."""
    checked = ok = 0
    _, path = files[0]
    frame = pd.read_csv(path, usecols=["sgf_content"], nrows=n)
    for content in frame["sgf_content"]:
        game = sgflib.parse(content)
        if not game.n_moves:
            continue
        rebuilt = [
            f"{'B' if c == sgflib.BLACK else 'W'}[{sgflib.index_to_point(int(m))}]"
            for m, c in zip(game.moves, game.colors)
        ]
        original = [
            token
            for token in content.replace("(", "").replace(")", "").split(";")
            if token[:1] in ("B", "W") and token[1:2] == "["
        ]
        checked += 1
        ok += int(rebuilt == original)
    return ok, checked


def render(meta: dict, sgf: dict, roundtrip: tuple[int, int], limit) -> str:
    games_per_player = meta["games_per_player"]
    moves = sgf["move_counts"]
    total_first = max(1, sum(sgf["first_black"].values()))

    lines: list[str] = []
    add = lines.append

    add("# W1 稽核報告 — 訓練集\n")
    add(f"產生時間：{time.strftime('%Y-%m-%d %H:%M')}　")
    add(f"SGF 解析樣本：{'全量' if limit is None else f'每檔前 {limit:,} 列'}"
        f"（共 {sgf['parsed']:,} 盤）\n")
    add("> 候選庫與測試檔尚未發布，因此 train vs test 的 property 比對、"
        "測試題重複稽核、一對一結構確認都無法在本次完成。\n")

    add("## 1. 結論摘要\n")
    ha = sgf["prop_counts"].get("HA", 0)
    km = sgf["prop_counts"].get("KM", 0)
    dt = sgf["prop_counts"].get("DT", 0)
    add(f"- **L3（讓子/貼目）**：`HA` 出現 {ha} 次、`KM` {km} 次、"
        f"帶 setup 石（AB/AW）的棋局 {sgf['setup_games']} 盤 → "
        f"{'可用，待 test 發布後確認' if (ha or km or sgf['setup_games']) else '**train 側就沒有，L3 作廢**'}")
    add(f"- **`DT` 日期**：{dt} 次 → "
        f"{'存活，時間聚簇可用' if dt else '已被移除，時間訊號不可用'}")
    add(f"- **玩家純度**：跨越多個 rank 的玩家 {meta['impure_players']:,} 位 → "
        f"{'rank 是盤級屬性，GroupKFold 仍以 player 分組但 rank 標籤需逐盤取' if meta['impure_players'] else 'rank 是玩家級屬性，可安全地用 player 當群組鍵'}")
    add(f"- **解析健康度**：round-trip 還原一致 {roundtrip[0]}/{roundtrip[1]}；"
        f"解析錯誤 {sum(sgf['error_counts'].values())} 件")
    add("")

    add("## 2. 規模與 player_id 結構\n")
    add(f"- 總列數：**{meta['total_rows']:,}**")
    add(f"- distinct `player_id`：**{meta['n_players']:,}**")
    add(f"- distinct `game_id`：{meta['n_game_ids']:,} "
        f"（{'唯一' if meta['n_game_ids'] == meta['total_rows'] else '**有重複**'}）")
    add(f"- 每位玩家的盤數：min={games_per_player.min()}, "
        f"中位數={games_per_player.median():.0f}, "
        f"平均={games_per_player.mean():.1f}, max={games_per_player.max()}")
    add(f"- `player_id` 長度分布：{dict(meta['player_id_len'])}")
    add("")
    add("每個 rank 的玩家數與黑白平衡：\n")
    add("| rank | 盤數 | 玩家數 | 每人平均盤數 | B | W |")
    add("|------|------|--------|------------|---|---|")
    colors = meta["color_by_rank"]
    for rank in RANKS:
        if rank not in colors.index:
            continue
        n_players = int(meta["players_by_rank"].get(rank, 0))
        total = int(colors.loc[rank].sum())
        add(f"| {rank} | {total:,} | {n_players:,} | "
            f"{total / max(1, n_players):.1f} | "
            f"{int(colors.loc[rank].get('B', 0)):,} | "
            f"{int(colors.loc[rank].get('W', 0)):,} |")
    add("")

    add("## 3. `game_id` 是否洩漏 rank（禁用檢查）\n")
    add("| rank | game_num 最小 | 最大 | 筆數 |")
    add("|------|-------------|------|------|")
    ranges = []
    for rank in RANKS:
        if rank not in meta["game_id_by_rank"].index:
            continue
        row = meta["game_id_by_rank"].loc[rank]
        add(f"| {rank} | {int(row['min']):,} | {int(row['max']):,} | {int(row['count']):,} |")
        ranges.append((int(row["min"]), int(row["max"]), rank))
    ranges.sort()
    disjoint = all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))
    add("")
    add(f"**區間{'互斥 → `game_id` 直接編碼 rank' if disjoint else '重疊'}。**"
        " 無論如何 `game_id` / `player_id` / 檔名衍生特徵一律禁用"
        "（測試題只有 `question_id`）。")
    add("")

    add("## 4. SGF property 清單\n")
    add("| property | 出現盤數 | 佔比 |")
    add("|----------|---------|------|")
    for key, count in sgf["prop_counts"].most_common():
        add(f"| `{key}` | {count:,} | {count / max(1, sgf['parsed']):.1%} |")
    add("")
    add(f"- 盤面大小分布：{dict(sgf['board_sizes'])}")
    add(f"- 帶 setup 石（AB/AW）的棋局：{sgf['setup_games']:,}")
    add(f"- 解析錯誤：{dict(sgf['error_counts']) or '無'}")
    add("")

    add("## 5. 棋局長度\n")
    if moves.size:
        qs = np.percentile(moves, [1, 25, 50, 75, 99])
        add(f"- min={moves.min()}, p1={qs[0]:.0f}, p25={qs[1]:.0f}, "
            f"中位數={qs[2]:.0f}, p75={qs[3]:.0f}, p99={qs[4]:.0f}, max={moves.max()}")
        add(f"- 平均={moves.mean():.1f}，序列模型的位置上限建議取 p99 附近："
            f"**{int(qs[4])}**")
        add(f"- 異常短局（<30 手）：{int((moves < 30).sum()):,} 盤")
    add("")

    add("## 6. 開局指紋可行性（L7）\n")
    add(f"- 黑方第一手出現過 **{len(sgf['first_black'])}** 個不同點位")
    add("- 前 10 熱門第一手：\n")
    add("| 座標 | 盤數 | 佔比 |")
    add("|------|------|------|")
    for index, count in sgf["first_black"].most_common(10):
        add(f"| `{sgflib.index_to_point(index)}` | {count:,} | {count / total_first:.1%} |")
    top1 = sgf["first_black"].most_common(1)
    if top1:
        share = top1[0][1] / total_first
        add("")
        add(f"最熱門第一手佔 {share:.1%} → "
            f"{'集中度高，單看第一手辨識力有限，需要前 10-20 手的 n-gram' if share > 0.3 else '分散度足夠，開局指紋有辨識力'}")
    add("")

    add("## 7. 重複棋譜\n")
    add(f"- 樣本內完全重複的 `sgf_content`：**{len(sgf['duplicates']):,}** 對")
    for previous, current in sgf["duplicates"][:5]:
        add(f"  - {previous} ≡ {current}")
    add("")

    add("## 8. 待測試檔發布後補做\n")
    add("- [ ] train / candidates / test 三邊的 property 存在率比對（決定 L3）")
    add("- [ ] 測試題 SGF 是否逐字出現在候選庫或訓練集")
    add("- [ ] `player_id` 空間在 train 與兩個候選庫之間是否重疊（決定 L9 成本）")
    add("- [ ] 400 題 vs 400 候選是否一對一（決定 L6 用 Hungarian 還是 Sinkhorn）")
    add("- [ ] 測試題的 `num_games` 是否真的是 5/10/15/20 各 100")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="每檔解析的列數上限（預設全量）")
    parser.add_argument("--out", default=str(ROOT / "notebooks" / "01_audit.md"))
    args = parser.parse_args()

    files = _train_files()
    if not files:
        print(f"no training files under {TRAIN_DIR}")
        return 1

    start = time.time()
    print("[1/3] metadata ...")
    meta = scan_metadata(files)
    print("[2/3] sgf ...")
    sgf = scan_sgf(files, args.limit)
    print("[3/3] round-trip ...")
    roundtrip = verify_roundtrip(files)

    report = render(meta, sgf, roundtrip, args.limit)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print(f"\nwrote {out}  ({time.time() - start:.0f}s)")
    print("-" * 60)
    print(report[: report.index("## 2.")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
