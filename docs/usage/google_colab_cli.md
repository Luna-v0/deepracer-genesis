# Google Colab (CLI)

Colab is good for **feature-vector (state) training** out of the box. The
Madrona **camera renderer also runs** once the VM gets a Vulkan ICD file (see
[Camera / Madrona on Colab](#camera-madrona-on-colab)); verified on the G4
runtime (RTX PRO 6000 Blackwell, 96 GB). Nyx is untested there. The reference
notebook is `notebooks/deepracer_genesis_colab.ipynb`.

> Mental model in one sentence: install from git with `uv`, define an `Experiment`
> class in a cell, `run()` it on `/content/runs`, then render a software-Mesa video
> and save the run to Drive.

---

## 1. Runtime + install

A T4 runtime is enough. First uninstall TensorFlow (its LLVM clashes), then install:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!pip uninstall -q -y tensorflow tensorflow-cpu tf-keras
!pip install -q uv
!uv pip install --system -q --no-cache \
    "deepracer-genesis @ git+https://github.com/Luna-v0/deepracer-genesis.git@main"
```

## 2. Define and run

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class ColabRacer(Experiment):
    total_env_steps = 5_000_000        # ~8 min on a T4
    eval_every_steps = 1_000_000
    num_envs = 512                     # T4-sized
    def pipeline(self):
        return FeatureEnvironment(lookahead_k=10, num_envs=self.num_envs) >> VectorPolicy(keys=("state",))

record = run(ColabRacer, root="/content/runs")
```

## 3. Watch a rollout

Videos render through the software (Mesa) rasterizer — the spectator view works
without the camera renderer:

```python
from deepracer_genesis.experiment.visualize import rollout_video
from IPython.display import Video
mp4 = rollout_video(ColabRacer, root="/content/runs", steps=300, num_envs=8,
                    spectator_res=(640, 480))
Video(mp4, embed=True, width=720)
```

## 4. Save the run

```python
from google.colab import drive; import os, glob, shutil
drive.mount("/content/drive")
run_dir = sorted(glob.glob("/content/runs/*/*"))[-1]
shutil.copytree(run_dir, "/content/drive/MyDrive/deepracer_runs/" + os.path.basename(run_dir),
                dirs_exist_ok=True)
```

## Camera / Madrona on Colab

The VM ships the NVIDIA driver's GL/Vulkan libraries and `libvulkan.so.1`, but
no ICD manifest, so Vulkan finds no driver. Point one at the **EGL** library
(the GLX one fails headless with `Could not get 'vkCreateInstance'`):

```python
import json, os
os.makedirs("/etc/vulkan/icd.d", exist_ok=True)
json.dump({"file_format_version": "1.0.0",
           "ICD": {"library_path": "libEGL_nvidia.so.0", "api_version": "1.3.0"}},
          open("/etc/vulkan/icd.d/nvidia_icd.json", "w"))
```

The VM's system Python is 3.13; genesis needs <3.13, so install into a uv
3.12 env (`uv python install 3.12`, then `uv sync -p 3.12` from a project that
pins genesis with the `vision` extra) and run scripts with that env's python.

With the [Colab CLI](https://github.com/googlecolab/colab-cli):
`colab new -s cam --gpu G4`, `colab upload` your `pyproject.toml`/`uv.lock`/
script, `colab exec -f setup.py`, launch long jobs with `nohup ... &` from an
`exec` (an `exec` that runs past the CLI's reply timeout errors locally while
the VM keeps working), poll, then `colab stop -s cam`.

For local, camera-capable training see [Local install & run](local.md).
