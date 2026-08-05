# Vendored SimplyQRL

**Source:** https://github.com/javier-lazaro/SimplyQRL
**Revision:** `b534cc9` (June 2025 lock — the state that produced the published results)
**Modified:** no. Byte-identical to upstream, and byte-identical to the copy the
paper's own companion repository vendors at `src/simplyqrl/`.

Verify at any time:

```bash
git clone https://github.com/javier-lazaro/SimplyQRL.git /tmp/sq
git -C /tmp/sq checkout b534cc9
diff -r src/simplyqrl /tmp/sq/src/simplyqrl   # exits 0, ignoring this file
```

## Why vendored

The paper's companion repository states its reason plainly: *"For reproducibility,
this repository includes a self-contained copy of the SimplyQRL library (see
`src/simplyqrl/`), allowing all experiments to be executed out of the box without
external dependencies on the upstream version."* We match that choice so the two
repositories are comparable artefact-for-artefact, and so a reader can diff them
directly.

It also removes a class of failure this project has already hit once: a git
dependency that resolves differently over time is exactly how FIX-04 became
possible.

## What this does NOT mean

**This is not a fork.** Nothing here is edited. Every correction lives outside
this directory, in `qrl_dissection/`, and is applied at runtime by
`core/compat.py` with guards that fail loudly if upstream changes shape under
them. That separation is the point of the repository: the audit trail must show
what upstream does and what we do to it, as two separable things.

If you ever need to edit a file in here, don't. Add a guarded patch in
`core/compat.py`, or an adapter in `core/`, and record it in
`docs/CORRECTIONS.md`.
