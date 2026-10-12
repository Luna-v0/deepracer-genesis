"""The rsl-rl algorithm interface a custom ``Algo(cls=...)`` must satisfy.

rsl-rl's ``OnPolicyRunner`` is **not** hard-wired to PPO: it resolves the
algorithm class from ``cfg["algorithm"]["class_name"]`` (a class object or an
import path, via ``rsl_rl.utils.resolve_callable``) and drives whatever it gets
through a fixed rollout/update loop::

    alg_class = resolve_callable(cfg["algorithm"]["class_name"])   # PPO by default
    self.alg  = alg_class.construct_algorithm(obs, env, cfg, device)
    ...  self.alg.act(obs) / process_env_step(...) / compute_returns(obs) / update()

deepracer-genesis passes the class you give to ``Algo(cls=...)`` straight into
that slot (``spec_to_train_cfg`` — ``resolve_callable`` accepts a class object
directly, so even a notebook-defined class works), so a custom algorithm plugs
into the **same** runner PPO uses. It only has to speak this interface.

Easiest path — subclass ``rsl_rl.algorithms.PPO`` and override the learning rule
(``compute_returns`` / ``update``); you inherit ``construct_algorithm`` (which
builds the actor/critic/storage), ``act``, ``get_policy``, ``save``/``load`` and
the rest for free::

    from rsl_rl.algorithms import PPO

    class Reinforce(PPO):
        def compute_returns(self, obs): ...      # Monte-Carlo returns, no GAE
        def update(self): ...                    # policy-gradient loss

    run(MyExperiment, ...)   # with a stage:  >> Algo(cls=Reinforce)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from tensordict import TensorDict


@runtime_checkable
class RslAlgorithm(Protocol):
    """What ``OnPolicyRunner`` requires of its algorithm (``self.alg``).

    Attributes:
        learning_rate: the current learning rate (the runner logs it each
            iteration and may adapt it).
    """

    learning_rate: float

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: Any, cfg: dict,
                            device: str) -> RslAlgorithm:
        """Build the algorithm from the first obs, the env, the resolved cfg, and device."""
        ...

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Return the actions to step the env with for this transition."""
        ...

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor,
                         dones: torch.Tensor, extras: dict) -> None:
        """Record the post-step transition (rewards, dones, bootstrapping)."""
        ...

    def compute_returns(self, obs: TensorDict) -> None:
        """Finalize returns/advantages over the collected rollout."""
        ...

    def update(self) -> dict[str, float]:
        """Run the learning update; return a name -> scalar loss dict for logging."""
        ...

    def train_mode(self) -> None: ...
    def eval_mode(self) -> None: ...
    def broadcast_parameters(self) -> None: ...

    def get_policy(self) -> Any:
        """Return the policy module (used for inference/eval/export)."""
        ...

    def save(self) -> dict: ...
    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool: ...


#: The methods ``OnPolicyRunner`` calls on the algorithm. Used for a build-time
#: conformance check (``learning_rate`` is a runtime attribute, not listed here).
ALGO_INTERFACE: tuple[str, ...] = (
    "construct_algorithm", "act", "process_env_step", "compute_returns",
    "update", "train_mode", "eval_mode", "broadcast_parameters",
    "get_policy", "save", "load",
)


def missing_algorithm_methods(cls: type) -> list[str]:
    """Return the :data:`ALGO_INTERFACE` methods a candidate algorithm class lacks.

    Args:
        cls: a class passed to ``Algo(cls=...)``.

    Returns:
        The interface method names ``cls`` neither defines nor inherits (empty
        list means it structurally conforms). A ``rsl_rl.algorithms.PPO``
        subclass conforms by inheritance.
    """
    return [m for m in ALGO_INTERFACE if not callable(getattr(cls, m, None))]
