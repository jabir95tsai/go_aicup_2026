"""E2E-scratch (2026-11-17): activation checkpointing, encoder + B head module, ply batches, DDP."""
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from go_aicup.data.board_cache import write_store_cache
from go_aicup.tasks.identity.ge2e import IdentityEncoder, ge2e_loss, own_plies
from go_aicup.tasks.identity.seqagg import Head
from go_aicup.tasks.identity.train_e2e import PlyBatches, joint_games
from go_aicup.tasks.identity.train_ge2e import EncoderHead, head_payload

CONFIG = {"precision": "fp32"}
SHAPE = (3, 2, 4)


def _encoder(segments=0, seed=0):
    torch.manual_seed(seed)
    return IdentityEncoder(channels=8, blocks=4, embedding=16, checkpoint_segments=segments)


def test_checkpointed_encoder_matches_plain_outputs_and_gradients():
    plain = _encoder(0)
    ckpt = copy.deepcopy(plain)
    ckpt.checkpoint_segments = 2
    x = torch.randn(6, 18, 19, 19)
    out_a, out_b = plain(x), ckpt(x)
    torch.testing.assert_close(out_a, out_b)
    out_a.square().sum().backward()
    out_b.square().sum().backward()
    for (name, a), (_, b) in zip(plain.named_parameters(), ckpt.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
            continue
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6, msg=name)
    plain.eval(); ckpt.eval()
    with torch.no_grad():                         # eval never checkpoints
        assert ckpt.checkpoint_segments == 2 and ckpt(x).shape == (6, 16)


def _head(dim=16, seed=1):
    torch.manual_seed(seed)
    from go_aicup.tasks.identity import seqagg
    head = Head(attention=False, dim=dim, hidden=32)
    return head


def test_encoder_head_equals_joint_games_and_trains_both():
    encoder, head = _encoder(2), _head()
    encoder.w.requires_grad_(False); encoder.b.requires_grad_(False)
    net = EncoderHead(encoder, head, CONFIG, torch.device("cpu"), SHAPE)
    n, m, s = SHAPE
    x = torch.randn(n * m * s, 18, 19, 19)
    plies = torch.randint(0, 300, (n * m * s,))
    games = net(x, plies)
    ref = joint_games(head, encoder(x), plies, SHAPE)
    torch.testing.assert_close(games, ref)
    loss, _ = ge2e_loss(games, head.w, head.b)
    loss.backward()
    assert encoder.stem[0].weight.grad is not None and head.blocks[0]["ff"][0].weight.grad is not None
    assert head.w.grad is not None and encoder.w.grad is None
    state = head_payload(head, 7, 2026)
    assert state["arm"] == "B" and state["position"] is True and state["step"] == 7


def _toy_cache(tmp_path, n_games=12, length=110):
    rng = np.random.default_rng(0)
    games = [rng.choice(361, size=length, replace=False) for _ in range(n_games)]
    moves = np.concatenate(games).astype(np.int16)
    colors = np.concatenate([np.arange(length) % 2 for _ in games]).astype(np.int8)
    meta = pd.DataFrame({"offset": np.arange(n_games) * length, "length": length, "color": "B"})
    write_store_cache(moves, colors, meta, tmp_path / "cache")


def test_ply_batches_48_moves_own_colour_no_repeats(tmp_path):
    _toy_cache(tmp_path)
    colour = np.array([0, 1] * 6, np.int8)
    players = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9, 10, 11])]
    batches = PlyBatches(tmp_path / "cache", players, np.full(12, 110), players=3, games=2, moves=48,
                         steps=4, seed=2, rank=0, player_color=colour, min_ply=0, player_d4=True)
    planes, plies = batches[1]
    assert planes.shape == (3 * 2 * 48, 18, 19, 19) and plies.shape == (3 * 2 * 48,)
    games, again = batches.sample(1)
    np.testing.assert_array_equal(again, plies.numpy())
    for g in range(6):
        block = slice(g * 48, (g + 1) * 48)
        game = int(games[block][0])
        assert (games[block] == game).all()
        p = plies.numpy()[block]
        assert len(set(p.tolist())) == 48                        # 55 own plies >= 48: no repeats
        assert set(p.tolist()) <= set(own_plies(110, int(colour[game]), 0).tolist())


def _ddp_worker(rank, world, port, xs, plies, encoder_state, head_state, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    encoder, head = _encoder(0), _head()
    encoder.load_state_dict(encoder_state); head.load_state_dict(head_state)
    encoder.w.requires_grad_(False); encoder.b.requires_grad_(False)
    net = EncoderHead(encoder, head, CONFIG, torch.device("cpu"), SHAPE)
    model = torch.nn.parallel.DistributedDataParallel(net)
    loss, _ = ge2e_loss(model(xs[rank], plies[rank]), head.w, head.b)
    loss.backward()
    if rank == 0:
        queue.put({n: p.grad.detach().numpy().copy() for n, p in net.named_parameters() if p.requires_grad})
    dist.barrier()
    dist.destroy_process_group()


def test_two_process_ddp_gradient_is_mean_of_local_gradients():
    n, m, s = SHAPE
    torch.manual_seed(5)
    xs = [torch.randn(n * m * s, 18, 19, 19) for _ in range(2)]
    plies = [torch.randint(0, 300, (n * m * s,)) for _ in range(2)]
    encoder, head = _encoder(0), _head()
    # BN in train mode uses batch statistics per rank, so the local reference is per-rank batches
    local = []
    for r in range(2):
        e, h = copy.deepcopy(encoder), copy.deepcopy(head)
        e.w.requires_grad_(False); e.b.requires_grad_(False)
        net = EncoderHead(e, h, CONFIG, torch.device("cpu"), SHAPE)
        loss, _ = ge2e_loss(net(xs[r], plies[r]), h.w, h.b)
        loss.backward()
        local.append({k: p.grad.clone() for k, p in net.named_parameters() if p.requires_grad})
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, xs, plies, encoder.state_dict(),
                                                   head.state_dict(), queue)) for r in range(2)]
    for p in procs:
        p.start()
    grads = queue.get(timeout=120)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    for k, g in grads.items():
        torch.testing.assert_close(torch.from_numpy(g), (local[0][k] + local[1][k]) / 2, rtol=1e-4, atol=1e-6, msg=k)


def _lrs(total, steps, warmup=0.05, decay=0.2):
    from go_aicup.tasks.identity.train_ge2e import wsd_scheduler
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched, start = wsd_scheduler(opt, total, warmup, decay)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    return np.array(out), start, opt, sched


def test_wsd_warmup_constant_then_linear_decay_to_zero():
    lr, start, _, _ = _lrs(100, 101)
    assert start == 80
    assert lr[0] == 0.2 and lr[4] == 1.0 and (lr[5:80] == 1.0).all()
    assert np.isclose(lr[90], 0.5) and lr[100] == 0.0 and np.all(np.diff(lr[80:]) < 0)


def test_wsd_extension_keeps_the_stable_phase_and_resumes_the_scheduler():
    short, start, opt, sched = _lrs(9600, 7680)        # stop right at the pre-decay step
    longer, start2, _, _ = _lrs(12000, 12001)
    assert start == 7680 and start2 == 9600
    np.testing.assert_array_equal(short[600:], longer[600:7680])    # identical after both warmups
    from go_aicup.tasks.identity.train_ge2e import wsd_scheduler
    opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched2, _ = wsd_scheduler(opt2, 12000, 0.05, 0.2)
    opt2.load_state_dict(opt.state_dict()); sched2.load_state_dict(sched.state_dict())
    assert sched2.last_epoch == 7680 and opt2.param_groups[0]["lr"] == 1.0
    for _ in range(9600 - 7680):
        opt2.step(); sched2.step()
    assert opt2.param_groups[0]["lr"] == 1.0                            # decay starts at the new 9,600
    opt2.step(); sched2.step()
    assert opt2.param_groups[0]["lr"] < 1.0
