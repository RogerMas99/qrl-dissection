"""
[FIX-04] Environment verification.

Two dependency problems make the published artefact non-executable, and a third
is specific to Colab. This script fails loudly and early with an actionable
message instead of letting an eleven-frame traceback do the explaining.

    autoray 0.8.0  removed autoray.autoray.NumpyMimic, which PennyLane 0.41.1
                   requires at import. The requirements.txt published in
                   qrl-dissection pins exactly 0.8.0, so `import pennylane`
                   fails before any project code runs. The upstream SimplyQRL
                   lock from June 2025 - the environment that produced the
                   published Fig. 4 - pins 0.7.1, which works.

    pennylane-lightning  the publication export drifted to 0.43.0 against
                   pennylane 0.41.1; the June lock has them matched at 0.41.1.

    jax            not present in EITHER lock, so not part of the authors'
                   environment - but Colab preinstalls it, PennyLane imports it
                   opportunistically, and jax >= 0.6.0 removed jax.core.Primitive.
                   Removing it is fidelity to the original environment, not a
                   workaround.

Note that gymnasium is >= 1.0 in BOTH locks (1.1.1 in June, 1.2.1 at
publication). The NEXT_STEP autoreset behaviour that FIX-01 corrects was
therefore present in the environment that produced the published results.

    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib.util
import sys

EXPECTED = {
    "gymnasium": ("1.1.1", "1.2.1"),
    "pennylane": ("0.41.1",),
    "autoray": ("0.7.1",),
}


def main() -> int:
    problems, notes = [], []

    if importlib.util.find_spec("jax") is not None:
        problems.append(
            "jax is installed. PennyLane 0.41.1 will import it and fail on "
            "jax.core.Primitive (removed in jax 0.6.0).\n"
            "    Fix: pip uninstall -y jax jaxlib   AND RESTART the runtime."
        )

    try:
        import autoray.autoray as aa
        if not hasattr(aa, "NumpyMimic"):
            problems.append(
                "autoray is too new: autoray.autoray.NumpyMimic is missing, so "
                "PennyLane 0.41.1 cannot import.\n"
                "    Fix: pip install autoray==0.7.1  AND RESTART the runtime."
            )
    except ImportError:
        notes.append("autoray not installed (fine if you are not using the hybrid arms)")

    for name, allowed in EXPECTED.items():
        try:
            mod = __import__(name)
        except Exception as exc:
            notes.append(f"{name}: not importable ({type(exc).__name__})")
            continue
        version = getattr(mod, "__version__", "?")
        mark = "ok " if version in allowed else "!! "
        notes.append(f"{mark}{name} {version} (expected one of {', '.join(allowed)})")

    try:
        import gymnasium as gym
        from gymnasium.vector import SyncVectorEnv
        env = SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        mode = env.metadata.get("autoreset_mode", "<undeclared>")
        env.close()
        notes.append(f"   autoreset mode: {mode}")
        if "NEXT_STEP" not in str(mode):
            notes.append("   NOTE: not NEXT_STEP - FIX-01 may be a no-op here. "
                         "Check docs/CORRECTIONS.md#fix-01 before interpreting results.")
    except Exception as exc:
        notes.append(f"   could not probe autoreset mode: {exc}")

    print("\n".join(notes))
    if problems:
        print("\n" + "=" * 70)
        for p in problems:
            print("PROBLEM: " + p)
        print("=" * 70)
        return 1
    print("\nenvironment ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
