from genesis.morphs import Plane
from genesis.surfaces import Rough
from genesis import Scene


def build_field(config: dict, scene: Scene) -> Plane:
    """
    Creates the base field for the track

    Args:
        config: A configuration for the Plane
        scene: The scene where the plane will be constructed
    """

    fc = config.get("field_color", (0.30, 0.48, 0.32))
    return scene.add_entity(
        Plane(pos=(0, 0, -0.001)),
        surface=Rough(color=(*fc, 1.0)),
    )
