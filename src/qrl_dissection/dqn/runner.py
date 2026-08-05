"""
[NEW-04] Resumable run orchestration.

Colab sessions die. Long PQC runs die with them. This module makes a grid
restartable: every finished run drops a manifest, and re-running the grid skips
what is already on disk. Episode CSVs are written by upstream with a flush per
episode, so even an interrupted run leaves a usable partial learning curve.

Convention: results live OUTSIDE the repository (Drive, or ./results locally).
Code is versioned in git; run artefacts are not. Only the small summary tables
in docs/RESULTS-LOG.md get committed.
"""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..core.configs import build_arm_config
from .safe import GreedyEvalConfig, SafeDQN


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class RunSpec:
    """One cell of an experiment grid."""
    arm: str
    seed: int
    fix_autoreset: bool
    total_timesteps: int = 60_000
    dqn_kwargs: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""

    @property
    def run_name(self) -> str:
        fix = "fix01on" if self.fix_autoreset else "fix01off"
        base = f"{self.arm}__{fix}__s{self.seed}"
        return f"{base}__{self.tag}" if self.tag else base



# Fields that make two runs the same run. If any of these differ, an existing
# manifest is NOT an answer to the question being asked, and reusing it silently
# would report a 1,500-step smoke run as a 100,000-step result - which is exactly
# what happened before this guard existed.
REUSE_KEYS = ("arm", "seed", "fix_autoreset", "total_timesteps", "dqn_kwargs", "tag")


def reuse_or_none(manifest_path, expected, env_id=None, on_mismatch="error"):
    """Return an existing manifest only if it answers the same question.

    `run_name` deliberately encodes only arm, FIX-01 state, seed and tag - it is
    meant to be readable. That makes it an incomplete key: two runs with the same
    name can differ in step budget, batch size, buffer size or environment. This
    function closes that gap by comparing the stored spec against the requested
    one before agreeing to skip.

    on_mismatch:
        "error"  raise, naming the fields that differ (default - loud beats wrong)
        "rerun"  ignore the stale manifest and recompute
        "legacy" accept manifests that simply predate a field, but still refuse
                 ones that actively contradict the request
        "skip"   reuse anyway; only for deliberate archaeology
    """
    manifest_path = pathlib.Path(manifest_path)
    if not manifest_path.exists():
        return None
    stored = json.loads(manifest_path.read_text())
    stored_spec = stored.get("spec", stored)

    outcome = stored.get("outcome", {})
    diffs, unverifiable = [], []
    for key in REUSE_KEYS:
        if key not in expected:
            continue
        if key in stored_spec:
            found = stored_spec[key]
        elif key == "total_timesteps" and "total_timesteps" in outcome:
            # Manifests written before the spec was recorded in full still carry
            # the step budget inside the outcome, so this one is recoverable.
            found = outcome["total_timesteps"]
        else:
            unverifiable.append(key)
            continue
        if found != expected[key]:
            diffs.append(f"    {key}: on disk {found!r} != requested {expected[key]!r}")
    if env_id is not None and stored.get("env_id") not in (None, env_id):
        diffs.append(f"    env_id: on disk {stored.get('env_id')!r} != requested {env_id!r}")

    if diffs:
        if on_mismatch == "skip":
            return stored
        if on_mismatch == "rerun":
            return None
        raise RuntimeError(
            f"{manifest_path.name} exists but was produced by a different run:\n"
            + "\n".join(diffs)
            + "\n  Reusing it would report one experiment's numbers as another's."
            "\n  Fix by one of: delete that manifest, write to a different --outdir,"
            "\n  give the run a distinct tag, or pass force=True to recompute."
        )

    if unverifiable:
        # A LEGACY manifest: written before these fields were recorded. Absence is
        # not evidence of a mismatch, so refusing outright would throw away real
        # compute - but accepting silently is how FIX-08 happened. Say so, once.
        if on_mismatch in ("skip", "legacy"):
            return stored
        if on_mismatch == "rerun":
            return None
        raise RuntimeError(
            f"{manifest_path.name} is a LEGACY manifest: it does not record "
            f"{', '.join(unverifiable)}, so it cannot be checked against this "
            "request.\n  The run may be perfectly good - older experiment scripts "
            "simply did not write those fields.\n  Back-fill it once and this stops:"
            "\n      python scripts/migrate_manifests.py <results-dir> --dqn-kwargs '{...}'"
            "\n  Or pass on_mismatch='legacy' to accept legacy manifests as-is."
        )

    return stored


def run_arm(
    spec: RunSpec,
    outdir: pathlib.Path,
    env_id: str = "CartPole-v1",
    eval_cfg: Optional[GreedyEvalConfig] = None,
    progress_bar: bool = False,
    force: bool = False,
    on_mismatch: str = "error",
) -> Dict[str, Any]:
    """Execute one cell, or return the existing manifest if it answers the same
    question. A manifest with the same name but a different spec is a mismatch,
    not a hit - see `reuse_or_none`."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / f"{spec.run_name}.manifest.json"

    if not force:
        hit = reuse_or_none(manifest_path, asdict(spec), env_id=env_id,
                            on_mismatch=on_mismatch)
        if hit is not None:
            return hit

    agent_type, agent_config = build_arm_config(spec.arm, env_id=env_id)

    runner = SafeDQN(
        agent_type=agent_type,
        agent_config=agent_config,
        run_name=spec.run_name,
        seed=spec.seed,
        env_id=env_id,
        fix_autoreset=spec.fix_autoreset,
        eval_cfg=eval_cfg,
        outdir=outdir,
        **spec.dqn_kwargs,
    )
    outcome = runner.train(spec.total_timesteps, progress_bar=progress_bar)

    manifest = {
        "spec": asdict(spec),
        "run_name": spec.run_name,
        "outcome": asdict(outcome),
        "agent_type": agent_type,
        "agent_config": {k: str(v) for k, v in agent_config.items()},
        "env_id": env_id,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def run_grid(
    specs: Iterable[RunSpec],
    outdir: pathlib.Path,
    env_id: str = "CartPole-v1",
    eval_cfg: Optional[GreedyEvalConfig] = None,
    progress_bar: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    results = []
    specs = list(specs)
    for i, spec in enumerate(specs, 1):
        if verbose:
            print(f"[{i}/{len(specs)}] {spec.run_name}", flush=True)
        try:
            m = run_arm(spec, outdir, env_id=env_id, eval_cfg=eval_cfg,
                        progress_bar=progress_bar)
            results.append(m)
            if verbose:
                p = m["outcome"]["probe"]
                print(f"    ok  {m['outcome']['wall_seconds']}s  "
                      f"phantoms {100 * p['frac_poison']:.2f}% of steps  "
                      f"FIX-01 {'on' if p['fix01_active'] else 'off'}",
                      flush=True)
        except Exception as exc:  # keep the grid alive; report at the end
            if verbose:
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"run_name": spec.run_name, "error": repr(exc)})
    return results
