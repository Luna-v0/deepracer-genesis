"""What a CNN in the env loop sees: this step's frames, and its own estimates.

B-001 (camera-derived features lagged a step) and the true-state error-channel leak.
"""

import pytest
import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import (
    PPO,
    CameraEnvironment,
    DomainRandomizationCamera,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.perception.features import (
    CNNPerceptionFeatures,
    NoisyPerceptionFeatures,
)
from deepracer_genesis.perception.model import PerceptionCNN, save_checkpoint
from deepracer_genesis.physics.limits import (
    A_LAT_NORM,
    MAX_SPEED,
    MAX_STEER_RAD,
    WHEELBASE_M,
    YAW_RATE_NORM,
)

RES = (64, 48)          # (W, H): small, so the CPU rasterizer stays quick
STACK = 3               # deep enough to tell a single push from a double one
N_TARGETS = 7           # PerceptionFeatures defaults: 5 + two curvature horizons
SPEED, YAW = 2, 3       # CHANNEL_NAMES positions of speed and yaw rate
# newest speed / steer / yaw error channels under the default k_* params
I_SPEED_ERR, I_STEER_ERR, I_YAW_ERR = 11, 13, 21
DRIVE = (0.3, 0.6)      # a fixed [steer, speed] command, so the car moves


class StackProbe(PerceptionFeatures):
    """PerceptionFeatures that records the frame stack each time it computes."""

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        self.seen: list[torch.Tensor] = []

    def compute(self) -> torch.Tensor:
        self.seen.append(self.env.camera_stack.clone())
        return super().compute()


class CNNProbe(CNNPerceptionFeatures):
    """CNNPerceptionFeatures that records the frame stack its CNN reads."""

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        self.seen: list[torch.Tensor] = []

    def compute(self) -> torch.Tensor:
        self.seen.append(self.env.camera_stack.clone())
        return super().compute()


@pytest.fixture
def tiny_cnn(tmp_path):
    """A small random CNN checkpoint sized for RES and STACK."""
    torch.manual_seed(0)
    net = PerceptionCNN(in_channels=3 * STACK, n_targets=N_TARGETS,
                        input_hw=(RES[1], RES[0]), channels=(4,), kernels=(5,),
                        strides=(4,), head=8)
    path = tmp_path / "tiny_cnn.pt"
    save_checkpoint(net, path)
    return path


def _camera_sim(feature_set, feature_params=None, latency=0):
    """Build a 2-env CPU camera sim with fixed spawns."""
    torch.manual_seed(0)
    pipeline = CameraEnvironment(
        backend="cpu", resolution=RES, num_envs=2, tracks=("reinvent_base",),
        feature_set=feature_set, feature_params=feature_params,
        random_start=False, frame_stack=STACK)
    if latency:
        pipeline = pipeline >> DomainRandomizationCamera(latency_steps=latency)
    spec = (pipeline >> VectorPolicy(keys=("state",)) >> PPO()).build(seed=0)
    return Builder(spec).sim()


def _drive(sim):
    """Step the sim once with the fixed command; return the observations."""
    obs, _, _, _ = sim.step(torch.tensor([DRIVE] * sim.num_envs))
    return obs


def _expire(sim, env: int) -> None:
    """Make ``env`` time out on the next step, so the step auto-resets it."""
    sim.episode_length_buf[env] = sim.max_episode_length - 1


# ------------------------------------------------- B-001: no hidden frame lag
@pytest.mark.parametrize("latency", [0, 1, 2])
def test_features_lag_the_render_by_exactly_latency_steps(latency):
    """latency_steps is the only camera delay: 0 means this step's own frame."""
    sim = _camera_sim(StackProbe, latency=latency)
    frames = [sim.image_buf.clone()]          # pre-latency render of each step
    for _ in range(latency + 3):
        _drive(sim)
        frames.append(sim.image_buf.clone())
    seen = sim.feature_set.seen
    assert len(seen) == len(frames)           # one feature pass per step

    for t in range(1, len(frames)):           # the test needs a moving camera
        assert not torch.equal(frames[t], frames[t - 1])
    for t in range(latency, len(frames)):
        assert torch.equal(seen[t][:, -3:], frames[t - latency]), (
            f"step {t}: features did not see the frame from {latency} step(s) ago")


def test_the_cnn_reads_the_stack_the_policy_receives(tiny_cnn):
    """The CNN channels of the state are the CNN applied to the camera group."""
    sim = _camera_sim(CNNPerceptionFeatures, {"checkpoint": tiny_cnn})
    net = sim.feature_set.net
    previous = None
    for _ in range(3):
        obs = _drive(sim)
        with torch.inference_mode():
            expected = net(obs["camera"])
        assert torch.allclose(obs["state"][:, :N_TARGETS], expected, atol=1e-6)
        if previous is not None:              # a stale read would be detectable
            assert not torch.allclose(expected, previous, atol=1e-6)
        previous = expected


def test_an_auto_reset_pushes_the_other_envs_stack_once():
    """Env 0 respawns; env 1's stack advances one frame, read the same both passes."""
    sim = _camera_sim(StackProbe)
    for _ in range(2):
        _drive(sim)
    before = sim.camera_stack.clone()
    passes = len(sim.feature_set.seen)
    _expire(sim, 0)
    _drive(sim)
    assert sim.step_info["time_out"].tolist() == [True, False]

    after = sim.camera_stack
    assert torch.equal(after[1, :-3], before[1, 3:])        # shifted exactly once
    assert not torch.equal(after[1, -3:], before[1, -3:])
    seen = sim.feature_set.seen[passes:]
    assert len(seen) == 2                     # the step's pass + the respawn pass
    assert all(torch.equal(s[1], after[1]) for s in seen)


def test_terminal_state_carries_features_of_the_terminal_frame(tiny_cnn):
    """The pre-reset snapshot holds CNN output for the stack ending at that step."""
    sim = _camera_sim(CNNProbe, {"checkpoint": tiny_cnn})
    for _ in range(2):
        _drive(sim)
    before = sim.camera_stack.clone()
    passes = len(sim.feature_set.seen)
    _expire(sim, 0)
    _drive(sim)

    terminal_stack = sim.feature_set.seen[passes]
    assert torch.equal(terminal_stack[0, :-3], before[0, 3:])
    assert not torch.equal(terminal_stack[0, -3:], before[0, -3:])
    with torch.inference_mode():
        expected = sim.feature_set.net(terminal_stack)
    terminal = sim.step_info["terminal_state"]
    assert torch.allclose(terminal[:, :N_TARGETS], expected, atol=1e-6)
    assert not torch.equal(terminal[0], sim.state_buf[0])   # pre-reset, not respawn


@pytest.mark.parametrize("latency", [0, 1])
def test_a_respawned_env_reads_no_frame_from_its_previous_episode(tiny_cnn, latency):
    """Its first state comes from its fresh frame, repeated: the priming contract."""
    sim = _camera_sim(CNNProbe, {"checkpoint": tiny_cnn}, latency=latency)
    for _ in range(3):
        _drive(sim)
    _expire(sim, 0)
    obs = _drive(sim)

    fresh = sim.image_buf[0]                  # the respawn pose's render
    stack = obs["camera"][0]
    for k in range(STACK):
        assert torch.equal(stack[3 * k:3 * k + 3], fresh)
    assert torch.equal(sim.feature_set.seen[-1][0], stack)
    with torch.inference_mode():
        expected = sim.feature_set.net(obs["camera"])
    assert torch.allclose(obs["state"][0, :N_TARGETS], expected[0], atol=1e-6)


def test_an_auto_reset_rolls_the_other_envs_histories_once():
    """Env 0 respawns; env 1's action history must keep its previous action."""
    sim = _feature_sim(PerceptionFeatures)
    first, second = torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor(
        [[0.5, 0.6], [0.7, 0.8]])
    sim.step(first)
    _expire(sim, 0)
    obs, _, _, _ = sim.step(second)
    assert sim.step_info["time_out"].tolist() == [True, False]
    prev_actions = obs["state"][1, N_TARGETS:N_TARGETS + 4].view(2, 2)
    assert torch.equal(prev_actions, torch.stack([second[1], first[1]]))


# -------------------------------------- leak: errors come from the estimates
def _errors(sim, v_forward, yaw_rate):
    """Newest speed, steer and yaw errors for the given speed and yaw rate."""
    act, a = sim.cfg["action"], sim.actions
    steer = a[:, 0] * MAX_STEER_RAD
    speed_cmd = act["min_speed"] + (a[:, 1] + 1) * 0.5 * (
        act["max_speed"] - act["min_speed"])
    yaw_expected = v_forward * torch.tan(steer) / WHEELBASE_M
    speed_err = (speed_cmd - v_forward) / MAX_SPEED
    steer_err = ((v_forward * yaw_expected - v_forward * yaw_rate)
                 / A_LAT_NORM).clamp(-2, 2)
    yaw_err = ((yaw_expected - yaw_rate) / YAW_RATE_NORM).clamp(-2, 2)
    return torch.stack([speed_err, steer_err, yaw_err], dim=1)


def _error_channels(state):
    return state[:, [I_SPEED_ERR, I_STEER_ERR, I_YAW_ERR]]


def _assert_errors_follow_the_estimates(sim, state):
    """Errors match the state's own speed/yaw-rate channels, not the truth."""
    estimated = _errors(sim, state[:, SPEED] * MAX_SPEED,
                        state[:, YAW] * YAW_RATE_NORM)
    true = _errors(sim, sim.v_forward, sim.yaw_rate)
    assert torch.allclose(_error_channels(state), estimated, atol=1e-5)
    assert not torch.allclose(_error_channels(state), true, atol=1e-3)


def _feature_sim(feature_set, feature_params=None):
    """Build a 2-env CPU feature sim with fixed spawns."""
    torch.manual_seed(0)
    spec = (FeatureEnvironment(backend="cpu", num_envs=2, tracks=("reinvent_base",),
                               feature_set=feature_set,
                               feature_params=feature_params, random_start=False)
            >> VectorPolicy(keys=("state",)) >> PPO()).build(seed=0)
    return Builder(spec).sim()


def test_the_privileged_set_builds_its_errors_from_the_true_state():
    sim = _feature_sim(PerceptionFeatures)
    for _ in range(3):
        obs = _drive(sim)
    expected = _errors(sim, sim.v_forward, sim.yaw_rate)
    assert torch.allclose(_error_channels(obs["state"]), expected, atol=1e-6)


def test_noisy_error_channels_follow_the_noised_estimates():
    sim = _feature_sim(NoisyPerceptionFeatures, {"noise": 1.0})
    for _ in range(3):
        obs = _drive(sim)
    _assert_errors_follow_the_estimates(sim, obs["state"])


def test_cnn_error_channels_follow_the_cnn_estimates(tiny_cnn):
    sim = _camera_sim(CNNPerceptionFeatures, {"checkpoint": tiny_cnn})
    for _ in range(3):
        obs = _drive(sim)
    _assert_errors_follow_the_estimates(sim, obs["state"])
