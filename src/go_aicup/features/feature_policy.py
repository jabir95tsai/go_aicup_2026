"""Feature policy: what a model is allowed to see (CLAUDE.md section 4).

Feature and augmentation rules for the current pipelines:

1. Identifier / bookkeeping columns never become features. Test questions
   carry only `question_id`, so any CV score that touched `game_id`,
   `player_id`, the source file or the row position is fake.
2. The only SGF content that exists on the training side is `SZ` plus moves
   (A-001). A feature derived from any other property is banned until the
   release audit shows it on both sides.

Plus the task-specific augmentation policy: D4 symmetry is on for rank
prediction (strength is orientation-free) and off for player identification
(the preferred corner *is* the fingerprint).
"""

from __future__ import annotations

from collections.abc import Iterable

# Columns that exist in our processed metadata but must never reach a model.
BANNED_EXACT = frozenset({
    "player_id", "game_id", "question_id", "rank", "rank_id", "source_row",
    "offset", "sgf_hash", "slot", "file", "source", "idx", "index", "fold",
    "n_errors", "first_color", "alternating",
})
# Any feature whose name contains one of these is derived from an identifier.
BANNED_SUBSTRINGS = ("player", "game_id", "question", "source", "hash", "file",
                     "offset", "fold")

# SGF properties seen on the training side (A-001). Anything else is unknown.
ALLOWED_SGF_PROPERTIES = frozenset({"SZ"})

D4_AUGMENT = {"rank": True, "pid": False}


class FeaturePolicyError(ValueError):
    pass


def banned_columns(columns: Iterable[str]) -> list[str]:
    offenders = []
    for column in columns:
        name = str(column).lower()
        if name in BANNED_EXACT or any(s in name for s in BANNED_SUBSTRINGS):
            offenders.append(str(column))
    return offenders


def check_feature_columns(columns: Iterable[str]) -> None:
    """Raise if any feature name is an identifier, the target or a derivative."""
    offenders = banned_columns(columns)
    if offenders:
        raise FeaturePolicyError(f"banned feature columns: {offenders}")


def check_sgf_properties(train_props: Iterable[str], test_props: Iterable[str]) -> list[str]:
    """Properties usable as features: present on both sides and whitelisted.

    Returns the usable set; raises if test carries a property train lacks,
    because that means the release differs from what we audited.
    """
    train_props, test_props = set(train_props), set(test_props)
    missing_in_train = test_props - train_props
    if missing_in_train:
        raise FeaturePolicyError(
            f"test has SGF properties absent from train: {sorted(missing_in_train)}"
        )
    return sorted((train_props & test_props) & ALLOWED_SGF_PROPERTIES)


def check_augment(task: str, d4: bool) -> None:
    if task not in D4_AUGMENT:
        raise FeaturePolicyError(f"unknown task {task!r}")
    if d4 and not D4_AUGMENT[task]:
        raise FeaturePolicyError(f"D4 augmentation is forbidden for task {task!r}")
