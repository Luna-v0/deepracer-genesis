"""Part N: eval config stage, per-track holdout eval loop, and charts.

All no-sim: the Evaluation DSL stage, the evaluate_on_tracks loop (with a stub
evaluate_policy), and chart rendering from a synthetic EvalRecord.
"""

import os

from deepracer_genesis.experiment import (
    Evaluation,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment import evaluator as ev
from deepracer_genesis.experiment.charts import render_charts
from deepracer_genesis.experiment.evaluator import EvalRecord, evaluate_on_tracks


# --------------------------------------------------------------- N.1 config
def test_evaluation_stage_sets_eval_config():
    s = (FeatureEnvironment(num_envs=8)
         >> VectorPolicy()
         >> Evaluation(real_tracks=("reinvent_base", "Oval_track"),
                       eval_num_envs=32, charts=False)).build()
    assert s.eval.real_tracks == ("reinvent_base", "Oval_track")
    assert s.eval.eval_num_envs == 32 and s.eval.charts is False


def test_eval_config_defaults_are_opt_in():
    s = (FeatureEnvironment(num_envs=8) >> VectorPolicy()).build()
    assert s.eval.real_tracks == ()      # holdout eval off by default
    assert s.eval.charts is True


# --------------------------------------------------------------- N.3 loop
def test_evaluate_on_tracks_returns_per_track(monkeypatch):
    calls = []

    def fake_eval(sim, actor, **kw):
        calls.append(sim)
        return {"completion_rate": 1.0 if sim == "t_good" else 0.0, "track": sim}

    monkeypatch.setattr(ev, "evaluate_policy", fake_eval)
    out = evaluate_on_tracks(actor=object(), tracks=["t_good", "t_bad"],
                             sim_factory=lambda t: t)
    assert set(out) == {"t_good", "t_bad"}
    assert out["t_good"]["completion_rate"] == 1.0
    assert out["t_bad"]["completion_rate"] == 0.0
    assert calls == ["t_good", "t_bad"]     # one fresh sim per track, in order


def test_eval_record_has_holdout_field():
    r = EvalRecord(spec_id="x", spec={}, seed=0, group=None, variant=None,
                   holdout={"reinvent_base": {"completion_rate": 0.5}})
    assert r.holdout["reinvent_base"]["completion_rate"] == 0.5


# --------------------------------------------------------------- N.4 charts
def test_render_charts_writes_pngs(tmp_path):
    record = EvalRecord(
        spec_id="x", spec={}, seed=0, group=None, variant=None,
        eval_history=[{"frames": 1000, "completion_rate": 0.1, "lap_time_s": 20.0},
                      {"frames": 2000, "completion_rate": 0.4, "lap_time_s": 18.0}],
        holdout={"reinvent_base": {"completion_rate": 0.5, "offtrack_rate": 0.2},
                 "Oval_track": {"completion_rate": 0.3, "offtrack_rate": 0.4}})
    paths = render_charts(record, str(tmp_path))
    assert paths, "no charts written"
    assert all(os.path.exists(p) and p.endswith(".png") for p in paths)
    # both a learning curve and a per-track holdout bar were produced
    names = [os.path.basename(p) for p in paths]
    assert any(n.startswith("learning_") for n in names)
    assert any(n.startswith("holdout_") for n in names)


def test_charts_empty_record_writes_nothing(tmp_path):
    record = EvalRecord(spec_id="x", spec={}, seed=0, group=None, variant=None)
    assert render_charts(record, str(tmp_path)) == []
