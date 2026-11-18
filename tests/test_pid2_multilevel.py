import numpy as np

from go_aicup.tasks.identity import multilevel
from go_aicup.tasks.identity.data import d4_tables


def _game(seed: int, plies: int = 70) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(361, size=plies, replace=False).astype(np.int64)


def test_keys_invariant_to_board_symmetries():
    tables = d4_tables().astype(np.int64)
    for seed in range(5):
        moves = _game(seed)
        for own in (0, 1):
            ref = multilevel.game_keys(moves, own, 15)
            for t in tables:
                assert np.array_equal(multilevel.game_keys(t[moves], own, 15), ref)


def test_keys_depend_on_the_move():
    moves = _game(0)
    other = moves.copy()
    other[2] = next(p for p in range(361) if p not in set(moves.tolist()))
    a = multilevel.game_keys(moves, 0, 15)
    b = multilevel.game_keys(other, 0, 15)
    assert not np.array_equal(a[1], b[1])


def test_corner_keys_same_habit_other_corner():
    # a 3-4 point answered the same way in two different corners, other corners differ
    rel = np.zeros((19, 19), np.int8)
    rel[3, 2] = 2                                            # opponent at (3,2) top-left
    a = multilevel.corner_keys(rel, 2 * 19 + 5)
    rel2 = np.zeros((19, 19), np.int8)
    rel2[15, 16] = 2                                         # mirrored into bottom-right
    rel2[3, 15] = 1                                          # unrelated stone elsewhere
    b = multilevel.corner_keys(rel2, 16 * 19 + 13)
    assert a == b


def test_capture_removes_stones():
    board = np.zeros((19, 19), np.int8)
    multilevel.play(board, 1 * 19 + 1, 2)                    # white stone at (1,1)
    for p in (0 * 19 + 1, 1 * 19 + 0, 1 * 19 + 2):
        multilevel.play(board, p, 1)
    multilevel.play(board, 2 * 19 + 1, 1)                    # last liberty
    assert board[1, 1] == 0


def test_stage2_keys_invariant_and_nonempty():
    tables = d4_tables().astype(np.int64)
    for seed in range(5):
        moves = _game(seed)
        for own in (0, 1):
            ref = multilevel.game_keys2(moves, own, 30)
            assert (ref[1:] != 0).all() and (ref[0, 4:] == 0).all()
            for t in tables:
                assert np.array_equal(multilevel.game_keys2(t[moves], own, 30), ref)


def test_stage1_keys_unchanged_by_window_parameter():
    rel = np.zeros((19, 19), np.int8)
    rel[3, 3], rel[4, 5] = 1, 2
    assert multilevel.reply_keys(rel, 4 * 19 + 5, 4 * 19 + 7) == multilevel.reply_keys(rel, 4 * 19 + 5, 4 * 19 + 7, 3, "r")


def test_offset_keys_fold_and_far():
    a = multilevel.offset_keys(10 * 19 + 10, 11 * 19 + 12, "o")
    b = multilevel.offset_keys(10 * 19 + 10, 8 * 19 + 9, "o")     # (2,1) vs (1,2): same shape
    assert a == b
    far1 = multilevel.offset_keys(0, 18 * 19 + 18, "o")
    far2 = multilevel.offset_keys(0, 10 * 19 + 0, "o")
    assert far1 == far2 and far1[1] != a[1]
