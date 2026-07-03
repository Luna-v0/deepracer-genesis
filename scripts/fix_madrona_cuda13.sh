#!/usr/bin/env bash
# Make gs-madrona (bundled nvJitLink 12.4) work on systems with a CUDA 13
# toolkit: give it a matching NVRTC 12.4 and stop it from dlopening the
# system libnvrtc.so.13. See README "CUDA 13 toolkit note".
set -euo pipefail
VENV="${1:-.venv}"
SP="$VENV/lib/python3.12/site-packages"

"$VENV/bin/python" -m pip install "nvidia-cuda-nvrtc-cu12==12.4.127" 2>/dev/null \
  || uv pip install --python "$VENV/bin/python" "nvidia-cuda-nvrtc-cu12==12.4.127"

ln -sf "$(realpath $SP)/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" "$SP/gs_madrona/libnvrtc.so.12"
ln -sf "$(realpath $SP)/nvidia/cuda_nvrtc/lib/libnvrtc.so.12" "$SP/gs_madrona/libnvrtc.so"
ln -sf "$(realpath $SP)/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.4" "$SP/gs_madrona/libnvrtc-builtins.so.12.4"

"$VENV/bin/python" - <<'EOF'
import pathlib
d = None
for p in pathlib.Path(".").glob(".venv/lib/python*/site-packages/gs_madrona"):
    d = p
assert d, "gs_madrona not found"
f = d / "libmadgs_mgr.so"
b = f.read_bytes()
if b"libnvrtc.so.13" in b:
    f.write_bytes(b.replace(b"libnvrtc.so.13", b"libnvrtc.so.12"))
    print("patched", f)
else:
    print("already patched")
EOF
echo "done"
