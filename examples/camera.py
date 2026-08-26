"""Camera (vision) examples: Madrona batch renderer and the Nyx path tracer.

Same structure as the feature examples, but the env renders an onboard RGB
camera and the policy is asymmetric — the actor sees pixels, the critic also
sees privileged state (the vision-mode privileged critic).

Run one::

    python examples/camera.py                              # runs CameraMadronaDr
    python -m deepracer_genesis.experiment examples.camera:CameraNyx
"""

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    Evaluation,
    Experiment,
)


class CameraMadronaDr(Experiment):
    """End-to-end vision on the Madrona batch renderer with the FULL DR stack.

    Blends: GPU + camera(render="madrona") + appearance/camera/physics/action
    DR + an asymmetric camera policy (actor=pixels, critic=pixels+state).
    """

    total_env_steps = 10_000_000
    eval_every_steps = 2_000_000
    ablation_group = "examples"
    variant = "camera_madrona_dr"

    def pipeline(self):
        return (
            # 64 envs + 8 minibatches: the default 4-frame stack quadruples
            # rollout-storage obs (24 x N x 12 x 120 x 160 f32) and each PPO
            # minibatch indexes a dense copy — 128 envs / 4 minibatches OOMs
            # an 8 GB GPU next to Genesis. Also set
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (see __main__).
            CameraEnvironment(render="madrona", resolution=(160, 120), num_envs=64)
            >> DomainRandomizationTrackAppearance(strength=0.6)     # world-color remap
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                         camera_jitter=True)         # image + mount DR
            >> DomainRandomizationPhysics()                         # dynamics DR
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                          delay_steps=1)             # actuation DR
            >> PPO(minibatches=8)
        )


class CameraNyx(Experiment):
    """End-to-end vision on the Nyx path tracer (photorealistic, slower).

    Blends: GPU + camera(render="nyx", single track) + light camera DR + charts.
    Nyx is single-track per process, so no heterogeneous track list here.
    """

    total_env_steps = 5_000_000
    eval_every_steps = 1_000_000
    ablation_group = "examples"
    variant = "camera_nyx"

    def pipeline(self):
        return (
            CameraEnvironment(render="nyx", resolution=(160, 120), num_envs=64)
            >> DomainRandomizationCamera(brightness=(0.8, 1.2), hue=0.03)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> Evaluation(charts=True)                              # Part N charts
        )


class CameraZoo(Experiment):
    """CameraMadronaDr's DR stack on the track zoo, with a train/test split.

    The zoo is the scene-level half of domain randomization: pre-compiled
    track variants (shapes, widths, palettes, fields, walls), one per world
    tile, each env living on its own instance. ``compile_zoo`` only BAKES —
    it registers the variants as ordinary tracks, so from here on everything
    is plain track names: ``TrackDataset`` splits them deterministically,
    training sees ``split.train``, and ``Evaluation(real_tracks=...)`` runs
    per-track holdout eval on the UNSEEN variants plus the printed track.
    (Already-installed tracks need no compile at all —
    ``deepracer_genesis.tracks.names()/generated()`` lists them directly.)
    """

    total_env_steps = 10_000_000
    eval_every_steps = 2_000_000
    ablation_group = "examples"
    variant = "camera_zoo"

    def pipeline(self):
        from deepracer_genesis.datasets.splits import TrackDataset
        from deepracer_genesis.tools.zoo import compile_zoo, demo_zoo

        # demo_zoo: 6 local reinvent variants (widths/palettes/fields/walls —
        # no network). For the real library swap in a manifest, e.g.:
        #   from examples.zoos import full_dr; names = compile_zoo(full_dr)
        names = compile_zoo(demo_zoo())
        # deterministic split: reinvent_base (the printed track) is held out
        # of BOTH train and test — final-eval only; test = unseen variants
        split = TrackDataset(names=names, holdout=("reinvent_base",),
                             test_fraction=0.2, seed=0)
        return (
            CameraEnvironment(render="madrona", resolution=(160, 120),
                              num_envs=64, tracks=split.train)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05,
                                         blur=0.3, camera_jitter=True)
            >> DomainRandomizationPhysics()
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                          delay_steps=1)
            >> PPO(minibatches=8)
            >> Evaluation(real_tracks=tuple(split.test) + tuple(split.holdout),
                          charts=True)
        )


class CameraMaxDr(Experiment):
    """EVERY camera-compatible DR knob, maxed to its catalog-suggested range.

    The world half comes from the zoo (per-env tracks incl. width variants —
    which IS camera-mode track-width DR; the ``track_width`` knob itself is
    feature-only and ``validate()`` would refuse it here). The obs half is
    every ``randomization/catalog.py`` image/visual/actuation knob at its
    suggested space. Deliberately absent: ``env_map`` (Nyx-only — the
    compatibility matrix refuses it under Madrona rather than letting it
    silently do nothing).
    """

    total_env_steps = 10_000_000
    eval_every_steps = 2_000_000
    ablation_group = "examples"
    variant = "camera_max_dr"

    def pipeline(self):
        from deepracer_genesis.tools.zoo import compile_zoo, demo_zoo

        tracks = compile_zoo(demo_zoo())
        return (
            # random_direction: coin-flip the driving direction each episode —
            # the per-episode geometry DR the tracks themselves can't provide
            CameraEnvironment(render="madrona", resolution=(160, 120),
                              num_envs=64, tracks=tracks,
                              random_direction=True)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(
                # photometric (catalog-suggested ranges)
                brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                saturation=(0.7, 1.3), hue=0.1, gamma=(0.7, 1.5),
                white_balance=0.1, vignette=0.4,
                # geometric / lens
                distortion=0.15, crop=0.2, blur=0.5,
                # sensor noise + occlusion
                shot_noise=0.05, noise=0.05, cutout=0.5,
                # temporal (stateful): pipeline delay + dropped frames
                latency_steps=2, frame_drop=0.1,
                # render-path noise + per-env mount jitter (per run)
                pixel_noise=0.05,
                camera_jitter={"pitch_deg": 2.0, "pos_m": 0.01})
            >> DomainRandomizationPhysics()      # defaults ARE the full stack
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.05, speed_noise=0.05,
                                          delay_steps=3)
            >> PPO(minibatches=8)
        )


if __name__ == "__main__":
    import os
    # Must be set before CUDA init: the stacked-obs copies fragment the
    # allocator on 8 GB GPUs without expandable segments.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    CameraMadronaDr().run()
