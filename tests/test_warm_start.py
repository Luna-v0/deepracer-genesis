"""``spec.resume`` warm-starts training from a checkpoint (P4).

Smoke: builds real CPU sims (~20 s); deselect with ``-m "not smoke"``.
"""

import glob

import pytest
import torch

from deepracer_genesis.experiment import PPO, FeatureEnvironment, VectorPolicy, run


def _mlp_weights(run_dir_glob):
    path = glob.glob(run_dir_glob + "/model.pt")[0]
    sd = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    # normalizer buffers (step counts, running stats) drift by construction;
    # the warm-start contract is about the network weights
    return {k: t for k, t in sd.items() if "mlp" in k}


@pytest.mark.smoke
def test_spec_resume_warm_starts_from_checkpoint(tmp_path):
    """A resumed run trains onward from the parent's weights, not from scratch."""

    def one(variant, seed, resume=None):
        return run(
            FeatureEnvironment(tracks=("reinvent_base",), num_envs=4,
                               backend="cpu")
            >> VectorPolicy() >> PPO(),
            root=str(tmp_path), total_env_steps=96, seed=seed,
            resume=resume, group="p4", variant=variant)

    parent = one("parent", 7)
    ckpt = parent.train["checkpoint"]
    assert ckpt, "parent run saved no checkpoint"
    one("resumed", 8, resume=ckpt)
    one("scratch", 8)

    p = _mlp_weights(f"{tmp_path}/p4/parent-*")
    r = _mlp_weights(f"{tmp_path}/p4/resumed-*")
    s = _mlp_weights(f"{tmp_path}/p4/scratch-*")
    d_resumed = max((p[k] - r[k]).abs().max().item() for k in p)
    d_scratch = max((p[k] - s[k]).abs().max().item() for k in p)
    # after the same tiny budget, the resumed run must sit close to its parent
    # while an identically-seeded scratch run sits far away (~0.004 vs ~0.37)
    assert d_resumed < d_scratch / 5, (
        f"resume did not warm-start: |parent-resumed|={d_resumed:.4f}, "
        f"|parent-scratch|={d_scratch:.4f}")
