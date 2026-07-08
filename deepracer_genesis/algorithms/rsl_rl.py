"""Legacy rsl-rl-lib integration (the pre-TorchRL training path).

`python -m deepracer_genesis.train` uses this: DeepRacerEnv speaks the
rsl-rl 5.x VecEnv contract natively (TensorDict obs groups, no reset() from
the runner, extras["time_outs"]), so the glue is just runner construction.
The TorchRL experiment framework (deepracer_genesis.experiment) is the
maintained path; this stays for reproducing the early baselines.
"""

from __future__ import annotations

import copy
import os
import pickle


def build_runner(env, *, vision: bool, log_dir: str, device: str,
                 num_envs: int):
    """OnPolicyRunner over a DeepRacerEnv, with the cfgs pickled next to
    the logs (the eval CLI reloads them)."""
    from rsl_rl.runners import OnPolicyRunner

    from ..configs.cfgs import get_train_cfg

    train_cfg = get_train_cfg(vision=vision)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump({"env_cfg": env.cfg, "train_cfg": copy.deepcopy(train_cfg),
                     "num_envs": num_envs}, f)
    return OnPolicyRunner(env, train_cfg, log_dir, device=device)
