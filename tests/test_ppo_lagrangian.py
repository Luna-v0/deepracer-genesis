"""Unit tests for the PID-Lagrangian controller (Phase 4)."""

from deepracer_genesis.algorithms import PIDLagrangian


def test_lambda_stays_zero_when_satisfied():
    pid = PIDLagrangian(budget=25.0, kp=0.05, ki=0.0005, kd=0.1)
    for _ in range(50):
        lam = pid.update(j_cost=10.0)          # well under budget
    assert lam == 0.0


def test_lambda_grows_under_sustained_violation():
    pid = PIDLagrangian(budget=25.0, kp=0.05, ki=0.0005, kd=0.1)
    lams = [pid.update(j_cost=50.0) for _ in range(100)]
    assert lams[0] > 0.0
    assert lams[-1] > lams[10]                  # integral term keeps pushing
    assert all(l >= 0.0 for l in lams)


def test_lambda_relaxes_after_violation_ends():
    pid = PIDLagrangian(budget=25.0, kp=0.05, ki=0.0005, kd=0.1)
    for _ in range(100):
        pid.update(j_cost=50.0)
    high = pid.value
    for _ in range(400):
        lam = pid.update(j_cost=5.0)            # persistent satisfaction
    assert lam < high                            # integral unwinds
    assert pid.integral >= 0.0                   # anti-windup floor


def test_derivative_kicks_only_on_worsening():
    pid = PIDLagrangian(budget=10.0, kp=0.0, ki=0.0, kd=1.0)
    assert pid.update(20.0) == 10.0             # error jumped 0 -> 10
    assert pid.update(20.0) == 0.0              # error flat -> no D action
    assert pid.update(15.0) == 0.0              # improving -> clamped at 0


def test_state_roundtrip():
    pid = PIDLagrangian(budget=25.0)
    for _ in range(10):
        pid.update(40.0)
    clone = PIDLagrangian(budget=25.0)
    clone.load_state_dict(pid.state_dict())
    assert clone.update(40.0) == pid.update(40.0)
