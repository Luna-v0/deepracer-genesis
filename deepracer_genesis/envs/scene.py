from genesis import Scene
from genesis.options import SimOptions, RigidOptions, VisOptions
from genesis.constraint_solver import Newton


def define_scene(config: dict, render: str, show_viewer: bool = False) -> Scene:
    """
    Defines a scene for the deepracer car.

    Args:
        config: A configuration for the scene
        render: The type of render used
        show_viewer: If there is a user view.
    """

    so = SimOptions(dt=config["dt"], substep=1)

    ro = RigidOptions(
        dt=config["dt"],
        constraint_solver=Newton,
        enable_collision=True,
        enable_joint_limit=True,
        batch_dofs_info=bool(config.get("randomizer", False)),
        batch_links_info=bool(config.get("randomizer", False)),
    )

    vo = VisOptions(
        shadow=False,
        ambient_light=(0.35, 0.35, 0.35),
        background_color=tuple(config.get("background_color", (0.55, 0.72, 0.9))),
    )

    return Scene(
        sim_options=so,
        rigid_options=ro,
        vis_options=vo,
        renderer=render,
        show_viewer=show_viewer,
    )
