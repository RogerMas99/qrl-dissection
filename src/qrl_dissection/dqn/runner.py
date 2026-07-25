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


def run_arm(
    spec: RunSpec,
    outdir: pathlib.Path,
    env_id: str = "CartPole-v1",
    eval_cfg: Optional[GreedyEvalConfig] = None,
    progress_bar: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute one cell, or return the existing manifest if already done."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / f"{spec.run_name}.manifest.json"

    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

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
