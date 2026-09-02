"""The perception package: it imports, it does not run on import, it splits cleanly.

None of this touches ``data/``. The dataset cache is 11 GB and only exists on a
machine that has collected rollouts, so everything here works on synthetic
arrays or on the package's own structure.
"""

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:          # perception/ is not an installed package
    sys.path.insert(0, str(REPO_ROOT))

import perception  # noqa: E402

MODULES = ["perception"] + [m.name for m in
                            pkgutil.walk_packages(perception.__path__, "perception.")]
SOURCES = sorted((REPO_ROOT / "perception").rglob("*.py"))


def test_the_package_is_not_empty():
    assert len(MODULES) > 20 and len(SOURCES) > 20


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    """Catches a rename that left a dangling import behind."""
    importlib.import_module(name)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_nothing_runs_at_import_time(path):
    """A module read sys.argv at import once; importing it then needed arguments.

    Anything a script does belongs in main(), so that importing it -- to reuse a
    constant, or to collect these very tests -- stays free of side effects.
    """
    allowed = {"filterwarnings", "use", "insert", "getLogger"}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "?")
            assert name in allowed, f"{path.name} calls {name}() at import time"
        if isinstance(node, (ast.For, ast.While)):
            pytest.fail(f"{path.name} loops at import time")
        if isinstance(node, ast.If):
            assert ast.unparse(node.test).startswith("__name__"), \
                f"{path.name} branches at import time"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_a_script_can_be_run(path):
    """Every module defining main() must be runnable with python -m."""
    src = path.read_text()
    if "def main(" in src:
        assert '__name__ == "__main__"' in src, f"{path.name} has main() but no guard"


def test_the_holdout_split_is_disjoint_and_complete():
    from perception.dataset import DATASET_TRACKS, HOLDOUT_TRACKS, TRAINING_TRACKS

    assert set(HOLDOUT_TRACKS) & set(TRAINING_TRACKS) == set()
    assert set(HOLDOUT_TRACKS) | set(TRAINING_TRACKS) == set(DATASET_TRACKS)
    assert len(HOLDOUT_TRACKS) >= 5


def test_the_policy_and_the_cnn_hold_out_the_same_tracks():
    """The whole pipeline is measured on tracks neither half ever trained on."""
    from perception.dataset import HOLDOUT_TRACKS, track_names
    from perception.train_policy_with_noise import TEST_TRACKS, TRAIN_TRACKS

    assert set(TEST_TRACKS) == set(track_names(HOLDOUT_TRACKS))
    assert set(TRAIN_TRACKS) & set(TEST_TRACKS) == set()


def test_the_model_maps_a_frame_stack_to_the_seven_channels():
    import torch

    from perception.model import PerceptionCNN
    from perception.noisy_features import CHANNEL_NAMES, SIGMA

    net = PerceptionCNN()
    y = net(torch.zeros(2, 12, 120, 160))
    assert y.shape == (2, len(CHANNEL_NAMES)) == (2, len(SIGMA))


def test_augmenting_keeps_a_frame_a_frame():
    """Jitter may change the pixels; it may not change their type or range."""
    from perception import augment

    rng = np.random.default_rng(0)
    frame = rng.random((120, 160, 3), dtype=np.float32)
    out = augment.apply(frame.copy(), augment.sample_camera(rng), rng)

    assert out.shape == frame.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, frame)


def test_one_camera_state_treats_two_frames_alike():
    """The stack shares a camera state, so exposure cannot flicker inside it.

    Only the per-frame sensor noise may differ, and it is bounded by NOISE[1].
    """
    from perception import augment

    frame = np.full((60, 80, 3), 0.5, dtype=np.float32)
    camera = augment.sample_camera(np.random.default_rng(0))
    a = augment.apply(frame.copy(), camera, np.random.default_rng(1))
    b = augment.apply(frame.copy(), camera, np.random.default_rng(2))
    assert np.abs(a - b).max() <= 6 * augment.NOISE[1]
