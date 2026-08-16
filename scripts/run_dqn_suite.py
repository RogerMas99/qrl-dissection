"""
Run the whole DQN suite. One entry point, resumable, budgeted.

    python scripts/run_dqn_suite.py --plan                    # what would run, and where
    python scripts/run_dqn_suite.py --pass coverage           # 3 seeds, everything
    python scripts/run_dqn_suite.py --pass robustness         # 10 seeds, everything
    python scripts/run_dqn_suite.py --only exp03 exp03b       # just the pair that matters
    python scripts/run_dqn_suite.py --pass robustness --budget-minutes 300

DESIGN NOTES, because they are the difference between "runs" and "runs many times"
----------------------------------------------------------------------------------
**Two ways to bound a session, and the difference matters.**

`--budget-minutes` is a wall clock checked BETWEEN cells. When the deadline has
passed the suite lets the running cell finish, writes its manifest and stops.
It cannot preempt a cell in progress: killing one halfway loses its compute
entirely, since no manifest is written and the next session restarts it from
zero. So a session can overshoot by up to the length of one cell.

That makes it the wrong tool when a single cell is long. An 8-qubit hybrid at
100k steps can run for hours, and `--budget-minutes 180` on a 10-hour cell will
simply run for 10 hours - or, more likely, get killed by the runtime disconnect
and lose all of it.

`--max-cells N` is the control for that case: stop after N cells complete,
deterministically. Combine it with `--plan`, which reports the median wall time
per cell measured from the manifests already on this machine, so you can size a
session honestly instead of guessing. `--seeds 4 5` does the same thing from the
other direction: fewer cells per invocation.

**Several sessions at once.** Either partition by hand (`--seeds 1 2 3` in one,
`--seeds 4 5 6` in another - disjoint slices cannot collide), or pass `--claim`
and run the identical command everywhere: each session locks a cell before
starting it and skips cells another session holds. Locks go stale after 12h so a
disconnected session cannot strand work. See docs/REUSE.md.

**Resumability is at cell granularity.** Every experiment writes one manifest per
cell and skips finished cells. Interrupt at any point, rerun the same command,
and only what is missing is computed. That is what makes a budget flag safe: the
suite stops mid-grid without losing anything, and the existing coverage seeds
count as the first seeds of a robustness pass rather than being discarded.

**Seeds are the axis that scales, so they are a flag rather than a constant.**
`--pass coverage` is 3 seeds and answers "is there an effect worth measuring".
`--pass robustness` is 10 and is what a conclusion needs. The paper's own spreads
(docs/PAPER-BASELINES.md) show several published contrasts sitting inside one
standard deviation at 10 seeds; nothing here should be written up on 3.

**exp03 and exp03b are a pair.** exp03b differs from exp03 by one boolean and
exists to decide whether exp03's result is about reuploading or about
entanglement (CORRECTIONS.md#fix-07). Running one without the other leaves the
repo's only clean positive confounded, so `--only exp03` warns.

**Preflight is not optional.** FIX-05 fails silently in the replay path: a
FrozenLake run can complete and produce a plausible flat curve trained on one
observation repeated across the batch. The gate runs before any cell.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List

REPO = pathlib.Path(__file__).resolve().parents[1]

# Each entry: the script, its output subdirectory, how a seed list is passed, and
# how many cells one seed contributes (for the progress table).
SUITE: Dict[str, Dict] = {
    "exp01": {
        "script": "exp01_dqn_cartpole_capacity.py",
        "outdir": "exp01_dqn_cartpole_capacity",
        "cells_per_seed": 6,     # 3 CLASSICAL arms x FIX-01 on/off (script default)
        "default_steps": 60_000,
        "env": "CartPole-v1",
        "what": "block 3, ansatz/capacity, with the fair matched control",
    },
    "exp01h": {
        # The hybrid arm. Kept as a SEPARATE entry rather than folded into
        # exp01, because it is the expensive one (real PQC simulation) while
        # exp01's default is the three cheap classical arms - conflating them
        # would make --only exp01 silently expensive for anyone who forgot the
        # split exists. Passes --arms hybrid_fig4 explicitly, which the plain
        # script accepts but this suite previously had no entry point for.
        "script": "exp01_dqn_cartpole_capacity.py",
        "outdir": "exp01_dqn_cartpole_capacity",     # SAME dir: one experiment
        "cells_per_seed": 2,     # 1 arm x FIX-01 on/off
        "default_steps": 60_000,
        "env": "CartPole-v1",
        "what": "block 3, the hybrid arm (PQC) - expensive, run separately",
        "extra": ["--arms", "hybrid_fig4"],
    },
    "exp02": {
        "script": "exp02_dqn_cartpole_output_reuse.py",
        "outdir": "exp02_dqn_cartpole_output_reuse",
        "cells_per_seed": 8,     # R in {4,8,16,32} x {hybrid, classical control}
        "default_steps": 100_000,
        "env": "CartPole-v1",
        "what": "block 1, Output Reuse",
    },
    "exp03": {
        "script": "exp03_dqn_cartpole_data_reuploading.py",
        "outdir": "exp03_dqn_cartpole_data_reuploading",
        "cells_per_seed": 3,     # L in {1, 2, 5}
        "default_steps": 100_000,
        "env": "CartPole-v1",
        "what": "block 2, Data Reuploading - the repo's clean positive",
    },
    "exp03b": {
        "script": "exp03b_dqn_cartpole_dr_unentangled.py",
        "outdir": "exp03b_dqn_cartpole_dr_unentangled",
        "cells_per_seed": 3,
        "default_steps": 100_000,
        "env": "CartPole-v1",
        "what": "exp03 with ent=False - decides what exp03 was measuring",
    },
    "exp04": {
        "script": "exp04_dqn_frozenlake_embeddings.py",
        "outdir": "exp04_dqn_frozenlake_embeddings",
        "cells_per_seed": 6,     # stage 1: 3 arms x FIX-01 on/off
        "default_steps": 100_000,
        "env": "FrozenLake4x4Scalar-v0",
        "what": "second environment, embedding/DR, and FIX-01 where it is measurable",
        "extra": ["--stage", "1"],
    },
    "exp04b": {
        "script": "exp04_dqn_frozenlake_embeddings.py",
        "outdir": "exp04_dqn_frozenlake_embeddings",
        "cells_per_seed": 6,     # stage 2: Config A x4 + Config B x2
        "default_steps": 100_000,
        "env": "FrozenLake4x4Scalar-v0",
        "what": "exp04 stage 2, the hybrid sweeps",
        "extra": ["--stage", "2"],
        "requires": "exp04",
    },
}

PASSES = {"smoke": [1], "coverage": [1, 2, 3], "robustness": list(range(1, 11))}

# Note for --only: exp01's three classical arms and exp01h's hybrid arm are ONE
# experiment split into two suite entries, purely so the expensive PQC arm is
# never run by accident. Pass both if you want the full comparison:
#     --only exp01 exp01h


def preflight() -> bool:
    """Environment, vendored integrity and the FIX-05 gate. All or nothing."""
    print("=" * 68)
    print("PREFLIGHT")
    print("=" * 68)
    ok = True

    try:
        sys.path.insert(0, str(REPO / "src"))
        import qrl_dissection
        report = qrl_dissection.upstream_report()
        print(json.dumps(report, indent=2))
    except Exception as exc:
        print(f"  FAIL importing qrl_dissection: {type(exc).__name__}: {exc}")
        return False

    r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        str(REPO / "tests")], capture_output=True, text=True,
                       cwd=str(REPO))
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"
    print(f"\n  tests: {tail}")
    if r.returncode != 0:
        print("  FAIL - do not run the suite against a failing test set.")
        print(r.stdout[-1500:])
        ok = False
    return ok


def count_cells(outdir: pathlib.Path, steps: int | None = None):
    """(usable, unusable) cell counts.

    Counting manifests alone is misleading and was: a directory can be "full"
    of cells run at a different step budget, which the reuse guard will refuse
    one by one at run time. Better to say so in the plan, before a session is
    committed to work that will not happen.
    """
    if not outdir.exists():
        return 0, 0
    usable = unusable = 0
    for mp in outdir.glob("*.manifest.json"):
        if steps is None:
            usable += 1
            continue
        try:
            m = json.loads(mp.read_text())
            got = m.get("spec", {}).get("total_timesteps") or \
                m.get("outcome", {}).get("total_timesteps")
        except Exception:
            got = None
        if got is None or int(got) == int(steps):
            usable += 1
        else:
            unusable += 1
    return usable, unusable


def cell_seconds(outdir: pathlib.Path):
    """Median wall time of the completed cells here, or None.

    Measured on the machine that will run the rest, which is the only estimate
    worth having: PQC throughput varies several-fold between a Colab CPU, a
    Colab GPU runtime and a laptop.
    """
    times = []
    for mp in (outdir.glob("*.manifest.json") if outdir.exists() else []):
        try:
            w = json.loads(mp.read_text()).get("outcome", {}).get("wall_seconds")
            if w:
                times.append(float(w))
        except Exception:
            pass
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def _hms(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


def plan(outroot: pathlib.Path, seeds: List[int], only: List[str],
         steps: int | None = None) -> None:
    print(f"\n{'experiment':10s} {'done':>6s} {'target':>7s} {'per cell':>9s} "
          f"{'remaining':>10s}  what")
    print("-" * 100)
    total_left = 0.0
    unknown, stale_note = [], []
    for key in only:
        spec = SUITE[key]
        d = outroot / spec["outdir"]
        done, stale = count_cells(d, steps)
        target = spec["cells_per_seed"] * len(seeds)
        if stale:
            stale_note.append((key, stale))
        per = cell_seconds(d)
        left = max(0, target - done)
        if per:
            total_left += per * left
            est = _hms(per * left)
        else:
            unknown.append(key)
            est = "?"
        print(f"{key:10s} {done:6d} {target:7d} {(_hms(per) if per else '?'):>9s} "
              f"{est:>10s}  {spec['what']}")

    print(f"\nseeds: {seeds}    outroot: {outroot}")
    if total_left:
        print(f"estimated time remaining (measured on THIS machine): {_hms(total_left)}")
    if unknown:
        print(f"no timing yet for: {', '.join(unknown)} - run one cell to calibrate")
    if stale_note:
        print("\n!! cells present but at a DIFFERENT step budget than requested "
              f"({steps}):")
        for key, n in stale_note:
            print(f"     {key}: {n} cell(s). The reuse guard will refuse each of "
                  "them, so they count as pending.")
        print("   Decide deliberately: keep the old budget (drop --steps), or move")
        print("   those cells aside so the new budget starts from a clean directory.")
    print("Counts are per-experiment manifests; exp04 and exp04b share a directory.")
    print("\nIf a single cell is longer than a session you can hold open, size the"
          "\nsession in CELLS rather than minutes: --max-cells 2, or --seeds 4 5.")


def run(key: str, outroot: pathlib.Path, seeds: List[int], steps: int | None,
        deadline: float | None, budget, claim: bool = False,
        progress_every: float = 60.0) -> bool:
    """Returns False if a budget ran out before the experiment finished.

    `budget` is a one-element list holding the remaining cell allowance, mutated
    as cells complete so the count carries across experiments.
    """
    s = SUITE[key]
    cmd = [sys.executable, str(REPO / "experiments" / s["script"]),
           "--outdir", str(outroot / s["outdir"]),
           "--seeds", *[str(x) for x in seeds]]
    cmd += s.get("extra", [])
    if steps:
        cmd += ["--steps", str(steps)]
    if claim:
        cmd += ["--claim"]

    print("\n" + "=" * 68)
    print(f"{key}  -  {s['what']}")
    print("=" * 68, flush=True)

    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**__import__("os").environ,
                                 "PYTHONPATH": str(REPO / "src")})
    # The budget is checked at CELL BOUNDARIES, not mid-training. Killing a cell
    # halfway through loses its compute entirely - no manifest is written, so the
    # next session starts it again from zero. Waiting for the running cell to
    # finish costs at most one cell of overshoot and loses nothing.
    # PROGRESS. Upstream already prints `global_step=N` once per episode; the
    # suite used to drop those lines entirely, which left a multi-hour cell
    # looking indistinguishable from a hung one. They are throttled into a single
    # status line instead: where we are in the cell, where we are in the grid,
    # and two ETAs. With cells that run for hours, knowing whether to wait or to
    # go to bed is the difference between using a session and babysitting it.
    total_cells = s["cells_per_seed"] * len(seeds)
    budget_steps = steps or s.get("default_steps")
    cell_i = 0
    cell_t0 = time.time()
    last_report = 0.0
    step_re = re.compile(r"global_step=(\d+)")

    over_budget = False
    try:
        for line in proc.stdout:
            if line.startswith("global_step"):
                m = step_re.search(line)
                now = time.time()
                if m and budget_steps and now - last_report >= progress_every:
                    last_report = now
                    step = int(m.group(1))
                    frac = min(1.0, step / budget_steps)
                    el = now - cell_t0
                    cell_eta = (el / frac - el) if frac > 0.01 else None
                    grid_eta = None
                    if cell_eta is not None and cell_i:
                        per_cell = el / frac
                        grid_eta = cell_eta + per_cell * max(0, total_cells - cell_i)
                    bar = "#" * int(frac * 24) + "." * (24 - int(frac * 24))
                    print(f"    [{bar}] {frac*100:5.1f}%  cell {cell_i}/{total_cells}"
                          f"  elapsed {_hms(el)}"
                          f"  cell ETA {_hms(cell_eta) if cell_eta else '?'}"
                          f"  grid ETA {_hms(grid_eta) if grid_eta else '?'}",
                          flush=True)
                continue
            if line.startswith("[run "):
                cell_i += 1
                cell_t0 = time.time()
                last_report = 0.0
            print(line, end="")

            stripped = line.strip()
            cell_finished = (stripped.startswith("ok ") or stripped.startswith("[skip]")
                             or stripped.startswith("busy "))

            if deadline and not over_budget and time.time() > deadline:
                over_budget = True
                print("[budget] deadline passed - will stop once this cell finishes.",
                      flush=True)
            if cell_finished and budget[0] is not None and not stripped.startswith("[skip]"):
                budget[0] -= 1
                if budget[0] <= 0:
                    print("[budget] cell allowance used up. Rerun to continue.",
                          flush=True)
                    proc.terminate()
                    return False
            if over_budget and cell_finished:
                print("[budget] stopping here. Rerun the same command to continue.",
                      flush=True)
                proc.terminate()
                return False
    finally:
        proc.wait()
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outroot", default="results")
    p.add_argument("--pass", dest="which", choices=list(PASSES), default="coverage")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="overrides --pass")
    p.add_argument("--only", nargs="+", choices=list(SUITE), default=list(SUITE))
    p.add_argument("--steps", type=int, default=None,
                   help="override the per-experiment default budget")
    p.add_argument("--progress-every", type=float, default=60.0,
                   help="seconds between progress lines. 0 disables.")
    p.add_argument("--claim", action="store_true",
                   help="cooperative locking, for running the SAME command from "
                        "several Colab sessions at once. Each session takes "
                        "whatever is free instead of you partitioning by hand.")
    p.add_argument("--max-cells", type=int, default=None,
                   help="stop after N cells complete. Deterministic, unlike "
                        "--budget-minutes, and the right control when one cell "
                        "is longer than a session.")
    p.add_argument("--budget-minutes", type=float, default=None,
                   help="wall-clock bound for THIS invocation. Checked between "
                        "cells, so a session can overshoot by up to one cell "
                        "rather than throwing away a half-trained one.")
    p.add_argument("--plan", action="store_true", help="show status and exit")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    seeds = args.seeds or PASSES[args.which]
    outroot = pathlib.Path(args.outroot)
    order = [k for k in SUITE if k in args.only]     # keep declaration order

    if "exp03" in order and "exp03b" not in order:
        print("! exp03 without exp03b leaves the DR result confounded with")
        print("  entanglement (CORRECTIONS.md#fix-07). Run both, or know why not.\n")
    if "exp04b" in order and "exp04" not in order:
        print("! exp04 stage 2 before stage 1: the liveness gate has not run.\n")

    if args.plan:
        plan(outroot, seeds, order, args.steps)
        return 0

    if not args.skip_preflight and not preflight():
        return 1

    deadline = time.time() + args.budget_minutes * 60 if args.budget_minutes else None
    budget = [args.max_cells]
    t0 = time.time()
    for key in order:
        if not run(key, outroot, seeds, args.steps, deadline, budget, args.claim,
                   args.progress_every):
            break

    print("\n" + "=" * 68)
    print(f"elapsed {(time.time()-t0)/60:.1f} min")
    plan(outroot, seeds, order, args.steps)
    print("\nRerun the same command to continue; finished cells are skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
