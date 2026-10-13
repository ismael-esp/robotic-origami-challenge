"""v9 (2026-08-19): replay-mixing sampler for finetuning on a second corpus.

Finetuning FoldingACT on the competition-paper-set corpus while replaying a slice of the
ORIGINAL post-cutoff corpus (catastrophic-forgetting guard). The two corpora live in two
separate FastFoldingDataset instances -- BOTH normalised with the SAME (union) norm stats
so one physical action has one normalised target (see policy/data/merge_norm_stats.py);
each keeps its own train/val split via `split_path` -- concatenated with torch's
ConcatDataset: indices [0, n_main) are the finetune corpus, [n_main, n_main + n_replay)
are the replay corpus.

MixedReplaySampler yields, per epoch, `epoch_len` = n_main indices in which a fraction
`replay_frac` come from the replay part (uniform, without replacement while possible) and
the rest from the main part (uniform, without replacement). It shards across DDP ranks the
same way DistributedSampler does (rank-strided after a seeded shuffle; per-epoch reseed via
set_epoch), so every rank sees a disjoint slice with the same replay ratio in expectation.

Usage:
    ds = ConcatDataset([main_ds, replay_ds])
    sampler = MixedReplaySampler(len(main_ds), len(replay_ds), replay_frac=0.15,
                                 rank=grank, world=world, seed=0)
    dl = DataLoader(ds, batch_size=bs, sampler=sampler, ...)
    ...  sampler.set_epoch(epoch)  each epoch (train_full.py already calls set_epoch on its sampler)
"""
import torch
from torch.utils.data import Sampler


class MixedReplaySampler(Sampler):
    def __init__(self, n_main, n_replay, replay_frac, rank=0, world=1, seed=0):
        assert n_main > 0 and n_replay > 0, "both corpora must be non-empty"
        assert 0.0 < replay_frac < 1.0, "replay_frac must be in (0,1) -- use no sampler for 0"
        self.n_main, self.n_replay = int(n_main), int(n_replay)
        self.replay_frac = float(replay_frac)
        self.rank, self.world, self.seed = int(rank), int(world), int(seed)
        self.epoch = 0
        # one "epoch" = one pass-worth of main, trimmed so every rank gets the SAME length
        # (unequal shards + drop_last would desync ranks' epoch boundaries / reseeds)
        self.epoch_len = self.n_main - (self.n_main % self.world)
        self.k_replay = int(round(self.epoch_len * self.replay_frac))
        self.k_main = self.epoch_len - self.k_replay

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _draw(self, g, n_pool, k, offset):
        """k indices from [offset, offset+n_pool): without replacement if k <= n_pool,
        else full permutations tiled (every element seen ~equally often)."""
        if k <= n_pool:
            return torch.randperm(n_pool, generator=g)[:k] + offset
        reps = -(-k // n_pool)
        return torch.cat([torch.randperm(n_pool, generator=g) for _ in range(reps)])[:k] + offset

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed * 100_003 + self.epoch)
        main = self._draw(g, self.n_main, self.k_main, 0)
        rep = self._draw(g, self.n_replay, self.k_replay, self.n_main)
        idx = torch.cat([main, rep])
        idx = idx[torch.randperm(len(idx), generator=g)]     # interleave main/replay
        # rank-strided shard (same recipe as DistributedSampler with drop_last=False, minus padding)
        return iter(idx[self.rank::self.world].tolist())

    def __len__(self):
        return self.epoch_len // self.world


if __name__ == "__main__":       # self-check: ratio, disjointness across ranks, determinism
    n_main, n_rep, f = 10_000, 3_000, 0.15
    world = 2
    seen = []
    for r in range(world):
        s = MixedReplaySampler(n_main, n_rep, f, rank=r, world=world, seed=1); s.set_epoch(3)
        idx = list(iter(s)); assert len(idx) == len(s)
        seen.append(set(idx))
        n_r = sum(i >= n_main for i in idx)
        ratio = n_r / len(idx)
        assert abs(ratio - f) < 0.02, ratio
        assert all(0 <= i < n_main + n_rep for i in idx)
        print(f"rank {r}: {len(idx)} idx, replay ratio {ratio:.3f}")
    assert not (seen[0] & seen[1]), "ranks overlap"
    for w in (2, 3, 4):   # equal shard length for every rank, any world size
        ls = {len(list(iter(MixedReplaySampler(10_001, 500, 0.1, rank=r, world=w)))) for r in range(w)}
        assert len(ls) == 1 and ls.pop() == (10_001 - 10_001 % w) // w, (w, ls)
    a = list(iter(MixedReplaySampler(n_main, n_rep, f, seed=1)))
    b = list(iter(MixedReplaySampler(n_main, n_rep, f, seed=1)))
    assert a == b, "not deterministic for equal seed/epoch"
    s2 = MixedReplaySampler(n_main, n_rep, f, seed=1); s2.set_epoch(1)
    assert list(iter(s2)) != a, "epoch reseed had no effect"
    # replay pool smaller than the requested draw -> tiled permutations, still in-range
    s3 = MixedReplaySampler(1000, 10, 0.5); idx3 = list(iter(s3))
    assert sum(i >= 1000 for i in idx3) == 500 and max(idx3) < 1010
    print("replay_sampler self-check OK")
