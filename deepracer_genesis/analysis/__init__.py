"""Run analysis: per-step telemetry and trajectory-on-track plotting.

The port of the community ``deepracer-utils`` analysis stack to this
simulator (see ``PLAN_TELEMETRY.md``): everything hangs off the per-step
trajectory log — the SIM_TRACE equivalent — which :mod:`.telemetry` records
during eval rollouts and :mod:`.trackplots` turns into the classic
trajectory/heatmap/hotspot views drawn over the track geometry.

Import weight: :mod:`.telemetry` needs torch + pyarrow (pandas lazily);
:mod:`.trackplots` needs matplotlib only when a plot is drawn (the
``charts.py`` convention). Nothing here imports genesis.
"""
