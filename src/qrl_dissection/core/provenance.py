"""
Provenance of a result: was it produced by the REAL circuit or by a simulator?

POLICY (docs/SIMULATED-RESULTS-POLICY.md)
-----------------------------------------
Anything produced by `SU2SkolikEmulator` (agent_type ``su2``) or by the batched
statevector simulator (agent_type ``sv``) is an EXPLORATORY result and **must not enter
the final work - not in a table, a figure, a number in the text or an argument - unless
the author has approved it explicitly, for that specific use.** The real PennyLane
circuit (agent_type ``hybrid``) and the classical arms are not affected.

This module is the single place that knows which agent types are simulators. It is used
three ways:

* `run_arm` stamps a ``provenance`` block into every NEW manifest produced by a simulator;
* `analysis.summarise_run` / `to_dataframe` carry a ``simulated`` column (derived from the
  manifest's ``agent_type``, so it also covers manifests written before the stamp existed)
  and warn loudly when a table contains simulated rows;
* `exclude_simulated` is the guard to put in front of any table that is meant for the final
  work: it drops simulated rows unless ``approved=True`` is passed on purpose.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional

SIMULATED_AGENT_TYPES = frozenset({"su2", "sv"})

STATUS = "EXPLORATORY - NOT FOR THE FINAL WORK WITHOUT THE AUTHOR'S EXPLICIT APPROVAL"
POLICY_DOC = "docs/SIMULATED-RESULTS-POLICY.md"
NOTICE = (
    "SIMULATOR RESULTS PRESENT (agent_type su2/sv). " + STATUS + ". "
    "See " + POLICY_DOC + "."
)


def is_simulated_agent_type(agent_type: Optional[str]) -> bool:
    return str(agent_type).lower() in SIMULATED_AGENT_TYPES


def is_simulated_manifest(manifest: Dict[str, Any]) -> bool:
    """True for a manifest produced by a simulator, stamped or not."""
    if (manifest.get("provenance") or {}).get("simulated"):
        return True
    return is_simulated_agent_type(manifest.get("agent_type"))


def is_simulated_arm(arm: str) -> bool:
    """True if the registered arm runs on a simulator (looked up in `ARMS`)."""
    from .configs import ARMS

    entry = ARMS.get(arm)
    return bool(entry) and is_simulated_agent_type(entry[0])


def provenance_block(agent_type: str) -> Optional[Dict[str, Any]]:
    """The block `run_arm` writes into a manifest; None for non-simulator agents."""
    if not is_simulated_agent_type(agent_type):
        return None
    return {
        "simulated": True,
        "simulator": str(agent_type).lower(),
        "status": STATUS,
        "policy": POLICY_DOC,
    }


def warn_if_simulated(df) -> None:
    """Emit one loud warning if a DataFrame carries simulated rows."""
    if "simulated" in getattr(df, "columns", []) and bool(df["simulated"].any()):
        warnings.warn(NOTICE, stacklevel=3)


def exclude_simulated(df, approved: bool = False):
    """Guard for tables meant for the final work.

    Drops the rows produced by a simulator unless `approved=True`. Passing `approved=True`
    is a statement that the author has explicitly approved that use; it is never a default.
    """
    if approved or "simulated" not in getattr(df, "columns", []):
        return df
    return df[~df["simulated"]].copy()


def simulated_arms(arms: Iterable[str]):
    return [a for a in arms if is_simulated_arm(a)]
