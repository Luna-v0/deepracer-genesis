"""Reference examples — copy these to author your own experiments.

Each example is an ``Experiment`` subclass showing the full structure end to
end: build the Environment, choose camera vs feature-vector observations, choose
domain randomization, pick the policy and algorithm, and set eval — plus an HPO
study (``hpo.py``). They blend the axes so you can see each combination:

    FeatureCpu          feature vector, CPU backend, no DR                (Part M)
    CameraCpu           camera on the CPU rasterizer, no GPU at all       (Part M)
    FeatureGpuDr        feature vector, GPU, physics DR via Space types   (Part H)
    CameraMadronaDr     camera, Madrona renderer, full DR, asymmetric
    CameraNyx           camera, Nyx path tracer, light DR
    WatchLiveFeature    feature vector with the interactive viewer + eval (Parts M/N)
    WatchLiveCamera     camera(render="madrona") with the interactive viewer + eval

There is no name registry: run an experiment by its class —
``FeatureCpu().run()`` or ``run(FeatureCpu)`` — or by its ``module:ClassName``
path from the CLI. ``examples/hpo.py`` is a study, run as a script.
"""

from .camera import CameraCpu, CameraMadronaDr, CameraNyx
from .feature_vector import FeatureCpu, FeatureGpuDr
from .watch_live import WatchLive, WatchLiveCamera, WatchLiveFeature

#: the reference experiment classes (for convenience / iteration)
EXAMPLES = (
    FeatureCpu, FeatureGpuDr, CameraCpu, CameraMadronaDr, CameraNyx,
    WatchLiveFeature, WatchLiveCamera,
)

__all__ = [
    "FeatureCpu", "FeatureGpuDr", "CameraCpu", "CameraMadronaDr", "CameraNyx",
    "WatchLive", "WatchLiveFeature", "WatchLiveCamera", "EXAMPLES",
]
