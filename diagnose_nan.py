"""Deterministic NaN-poisoning probe for the DR training collapse.

Reproduces the observed failure mode without waiting ~1M frames: inject a
non-finite value into one env (through the action path, then directly into the
dof velocities), then feed clean actions and forced resets, and report exactly
which quantities stay poisoned and whether the env ever heals.

Scenarios:
    A: one NaN ACTION step for env 0     -> clean actions, watch + reset
    B: NaN dof VELOCITY injected, env 0  -> clean actions, watch + reset
    C: all envs poisoned (storm replica) -> reset all, watch

Run: uv run python diagnose_nan.py
"""


import torch


def finite_report(env, tag):
    """One-line finiteness summary of every observable quantity."""
    pos, quat, vel, ang = env.car.kinematics()
    dv = env.car.entity.get_dofs_velocity()
    qp = env.car.entity.get_qpos()
    state = env.state_buf
    bad_dims = (~torch.isfinite(state)).any(dim=0).nonzero().flatten().tolist()
    n_bad = int((~torch.isfinite(state).all(dim=-1)).sum())

    def ok(t):
        return "OK " if torch.isfinite(t).all() else "NAN"

    print(f"[{tag:22s}] state:{n_bad:2d}/{env.num_envs} envs bad "
          f"(dims {bad_dims if len(bad_dims) < 10 else str(len(bad_dims)) + ' dims'}) "
          f"pos:{ok(pos)} quat:{ok(quat)} vel:{ok(vel)} ang:{ok(ang)} "
          f"qpos:{ok(qp)} dofs_vel:{ok(dv)}", flush=True)
    return n_bad


def clean_steps(env, n):
    acts = torch.zeros(env.num_envs, 2, device=env.device)
    acts[:, 1] = -0.5   # gentle forward
    for _ in range(n):
        env.step(acts)


def scenario(env, name, poison_fn, reset_ids):
    print(f"\n=== Scenario {name}")
    clean_steps(env, 20)
    finite_report(env, "before poison")
    poison_fn()
    finite_report(env, "after poison")
    clean_steps(env, 3)
    n = finite_report(env, "3 clean steps later")
    clean_steps(env, 10)
    n = finite_report(env, "13 clean steps later")
    env.reset_idx(reset_ids)
    env._post_physics(reset_ids)
    finite_report(env, "right after reset")
    clean_steps(env, 3)
    n = finite_report(env, "reset +3 steps")
    clean_steps(env, 20)
    n = finite_report(env, "reset +23 steps")
    print(f"=== Scenario {name}: {'HEALED' if n == 0 else 'STILL POISONED'}")
    return n == 0


def main():
    torch.manual_seed(0)
    from deepracer_genesis._gs import ensure_init
    ensure_init("gpu")
    from deepracer_genesis.configs.cfgs import get_env_cfg
    from deepracer_genesis.envs.base_env import DeepRacerEnv

    cfg = get_env_cfg(vision=False, track="reinvent_base", randomize=True,
                      backend="gpu", view="none")
    env = DeepRacerEnv(num_envs=16, env_cfg=cfg)
    dev = env.device
    ids0 = torch.tensor([0], device=dev)
    all_ids = torch.arange(env.num_envs, device=dev)

    # --- A: NaN through the ACTION path (the policy-loop route) ---
    def poison_action():
        acts = torch.zeros(env.num_envs, 2, device=dev)
        acts[0] = float("nan")
        env.step(acts)

    a = scenario(env, "A: NaN action env0", poison_action, ids0)

    # --- B: NaN directly into dof velocities (physics-blowup route) ---
    def poison_vel():
        dv = env.car.entity.get_dofs_velocity()
        dv = torch.nan_to_num(dv) if not torch.isfinite(dv).all() else dv
        dv[0] = float("nan")
        env.car.entity.set_dofs_velocity(dv)
        env.step(torch.zeros(env.num_envs, 2, device=dev))

    b = scenario(env, "B: NaN dofs_vel env0", poison_vel, ids0)

    # --- C: storm replica — poison EVERY env's velocities ---
    def poison_all():
        dv = env.car.entity.get_dofs_velocity()
        dv = torch.nan_to_num(dv)
        dv[:] = float("nan")
        env.car.entity.set_dofs_velocity(dv)
        env.step(torch.zeros(env.num_envs, 2, device=dev))

    c = scenario(env, "C: NaN all envs", poison_all, all_ids)

    print(f"\nRESULT: A(action)={'healed' if a else 'poisoned'} "
          f"B(velocity)={'healed' if b else 'poisoned'} "
          f"C(all)={'healed' if c else 'poisoned'}", flush=True)


if __name__ == "__main__":
    main()
