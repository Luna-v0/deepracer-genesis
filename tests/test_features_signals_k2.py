"""K.2: features are ordered selections of signals, read via the bus.

Pure-torch/CPU, no Genesis sim. A tiny fake env carries exactly the attributes
the feature blocks / ClassicFeatures read, plus a real ``SignalBus`` over the
production ``SIGNALS`` registry. These tests pin the two K.2 guarantees:

1. **Zero behavioral cost** — SelectFeatures over every block in order is
   byte-for-byte identical to ClassicFeatures.compute (the classic vector), and
   both are identical whether or not the bus is present (bus vs env-attr read).
2. **The bus is non-dead** — patching a signal in the bus changes the dependent
   feature block's output, proving blocks actually read ``env.signals``.
"""

import torch

from deepracer_genesis.envs.features import (
    ClassicFeatures, FEATURE_BLOCKS, PerceptionFeatures, SelectFeatures,
)
from deepracer_genesis.envs.signals import SIGNALS, SignalBus

FULL = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading",
        "last_action", "lookahead_xy")


class _FakeTrack:
    """Deterministic lookahead + curvature geometry with no waypoints DB."""

    def lookahead(self, wp_idx, k, stride, dir_sign):
        n = wp_idx.shape[0]
        return torch.arange(k).float().expand(n, k)

    def lookahead_points(self, la_idx):
        n, k = la_idx.shape
        # (N, K, 2): x = index, y = 0.5*index — arbitrary but fixed
        x = la_idx
        return torch.stack([x, 0.5 * x], dim=-1)

    def curvature_ahead(self, progress_m, distances, dir_sign):
        n = progress_m.shape[0]
        return torch.linspace(0.1, 0.2, len(distances)).expand(n, len(distances))


class _FakeEnv:
    """Minimal env exposing what ClassicFeatures + every block reads."""

    def __init__(self, *, with_bus=True, lookahead_k=3):
        n = 4
        self.num_envs = n
        self.device = torch.device("cpu")
        self.lookahead_k = lookahead_k
        self.v_forward = torch.tensor([0.5, 1.0, 1.5, 2.0])
        self.v_lateral = torch.tensor([0.0, -0.1, 0.2, -0.3])
        self.yaw_rate = torch.tensor([0.0, 0.5, -0.5, 1.0])
        self.up_z = torch.ones(n)
        self.lateral = torch.tensor([0.0, 0.2, -0.4, 0.5])
        self.half_width = torch.full((n,), 0.6)
        self.heading_err = torch.tensor([0.0, 0.3, -0.3, 0.1])
        self.d_progress = torch.tensor([0.01, 0.02, 0.03, 0.04])
        self.actions = torch.tensor([[0.1, 0.2], [-0.1, 0.3], [0.0, -0.2], [0.2, 0.1]])
        self.last_actions = torch.zeros(n, 2)
        self.dir_sign = torch.tensor([1.0, 1.0, -1.0, -1.0])
        self.yaw = torch.tensor([0.0, 0.1, -0.1, 0.2])
        self.wp_idx = torch.zeros(n, dtype=torch.long)
        self.base_pos = torch.zeros(n, 3)
        self.progress_m = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.track = _FakeTrack()
        self.cfg = {
            "action": {"min_speed": 0.1, "max_speed": 4.0},
            "obs": {"lookahead_stride": 1, "lookahead_scale": 2.0},
            "termination": {"wheel_margin": 0.0},
        }
        self.signals = SignalBus(self) if with_bus else None


def test_select_all_blocks_equals_classic_vector():
    env = _FakeEnv()
    classic = ClassicFeatures(env, {})
    select = SelectFeatures(env, {"features": FULL})
    assert torch.equal(select.compute(), classic.compute())


def test_select_block_order_covers_full_vocabulary():
    # a byte-for-byte guarantee is only meaningful if FULL is *every* block
    assert set(FULL) == set(FEATURE_BLOCKS)


def test_bus_and_no_bus_produce_identical_vectors():
    """Reading a signal via the bus must equal reading the raw env attr."""
    with_bus = ClassicFeatures(_FakeEnv(with_bus=True), {}).compute()
    no_bus = ClassicFeatures(_FakeEnv(with_bus=False), {}).compute()
    assert torch.equal(with_bus, no_bus)


def test_patching_a_bus_signal_changes_the_dependent_block():
    """The bus is non-dead: overriding a signal flows into the feature."""
    env = _FakeEnv()
    before = FEATURE_BLOCKS["v_forward"].compute(env)
    # patch the cached signal value directly on the bus
    env.signals._cache["v_forward"] = env.v_forward + 10.0
    after = FEATURE_BLOCKS["v_forward"].compute(env)
    assert not torch.equal(before, after)
    # exactly the +10/max_speed shift, proving the block used the bus value
    assert torch.allclose(after - before, torch.full_like(after, 10.0 / 4.0))


def test_patching_lateral_signal_flows_through_dir_sign_transform():
    """lateral is raw; the block applies dir_sign. Patching the raw signal
    must change the block, and the driving-frame transform must still apply."""
    env = _FakeEnv()
    env.signals._cache["lateral"] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out = FEATURE_BLOCKS["lateral"].compute(env).squeeze(1)
    expected = torch.tensor([1.0, 1.0, 1.0, 1.0]) * env.dir_sign / 0.6
    assert torch.allclose(out, expected)


def test_block_reads_are_registered_signals():
    """Every non-empty block read must be a real signal (a valid trace)."""
    for name, block in FEATURE_BLOCKS.items():
        for sig in block.reads:
            assert sig in SIGNALS, f"block {name!r} reads unknown signal {sig!r}"


def test_reads_for_traces_features_to_signals():
    # SelectFeatures reads = order-preserving dedup union of block reads
    assert SelectFeatures.reads_for(params={"features": FULL}) == (
        "v_forward", "v_lateral", "yaw_rate", "lateral", "half_width",
        "heading_err", "actions")
    # a subset selection traces only its blocks' signals
    assert SelectFeatures.reads_for(
        params={"features": ("lateral", "heading")}) == (
        "lateral", "half_width", "heading_err")
    # ClassicFeatures preset traces the same signal set as the full selection
    assert (set(ClassicFeatures.reads_for(params={}))
            == set(SelectFeatures.reads_for(params={"features": FULL})))


def test_perception_reads_via_bus_matches_env_attr():
    """PerceptionFeatures must also be bus/no-bus identical (byte-for-byte)."""
    with_bus = PerceptionFeatures(_FakeEnv(with_bus=True), {}).compute()
    no_bus = PerceptionFeatures(_FakeEnv(with_bus=False), {}).compute()
    assert torch.equal(with_bus, no_bus)


def test_perception_reads_are_registered_signals():
    for sig in PerceptionFeatures.reads_for(params={}):
        assert sig in SIGNALS


def test_lateral_signal_frame_is_raw():
    """The frame convention is now consistent: lateral exposes the RAW value,
    heading_err/d_progress the driving-frame value."""
    assert SIGNALS["lateral"].frame == "raw"
    assert SIGNALS["heading_err"].frame == "driving"
    assert SIGNALS["d_progress"].frame == "driving"
