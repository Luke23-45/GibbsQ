# Unknown Items — Gaps Without Sources

## Resolved Items
None needed.

## Permanently Unresolved
- UNKNOWN #1
  - Experiment   : cross-cutting (all 12)
  - Question     : Do end-to-end runs or hardened test suites surface runtime-only smoking guns not visible in static source review?
  - Tried        : `PYTHONPATH=src pytest -q tests/test_config_hardened.py tests/test_policies_hardened.py tests/test_simulator_hardened.py` (fails at collection: missing `hypothesis`); `python -m pip install hypothesis -q` (blocked by package index/proxy access).
  - Blocked by   : environment cannot install missing dependency from configured package index.
  - Action       : rerun in an environment with dependency resolution enabled, then execute hardened tests and at least one full reproduction pipeline pass.
