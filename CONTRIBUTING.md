# Contributing

## Dev setup

```bash
git clone https://github.com/adadaabhay/Fault-Prediction-in-Vehicles.git
cd Fault-Prediction-in-Vehicles
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev,bench]"
# Headless browser for the HUD smoke test:
pip install playwright && python -m playwright install --with-deps chromium
```

## Running tests

```bash
make verify           # ruff + mypy + fast tests
make test             # full pytest — see VERIFICATION.md for current counts
make data             # regenerate per-subsystem CSVs and manifests
python -m tools.check_artifacts    # model.json / config.json consistent
python -m tools.check_links        # docs links not 404
```

The current pass / skip / fail counts are recorded in
[`VERIFICATION.md`](VERIFICATION.md); the counts are not duplicated
here because they drift every time the suite grows.  The skipped
tests are dataset-backed and dataset-gated; they show as SKIP, not
ERROR, when the procurement copies under `datasets/` are absent.  See
`docs/PROVENANCE.md` for the procurement step.

## Pull requests

1. **One concern per PR.** A bug fix and a refactor are two PRs.
2. **Tests first.** A new feature ships with a test that fails without
   the feature and passes with it.  A bug fix ships with a test that
   reproduces the bug and then passes.
3. **No silent skip.** A `pytest.skip()` is a maintenance debt; the
   suite must grow the capability, not skip past it.  If the only way
   to get the test to pass is to skip, file an issue and link it.
4. **No silent fallback.** A `try: ... except: return None` is a bug
   unless the return is the documented no-op.  The `silent-failure-hunter`
   review agent looks for these.
5. **No hardcoded paths outside the repo.** A `Path(__file__).parent.parent`
   is fine (it self-resolves).  A `Path("C:/Users/...")` is not.
6. **Document the public surface.** Every module, class, and public
   function has a docstring.  The docstring is the contract; the
   implementation is the proof.  The `comment-analyzer` review agent
   verifies the docstring still matches the code.
7. **Update the changelog.** One bullet per PR under `[Unreleased]`.
   Use the categories `Added`, `Changed`, `Deprecated`, `Removed`,
   `Fixed`, `Security`.

## Code style

* `ruff` for formatting and import ordering (enforced in CI).
* `mypy --strict` on `telemetry_gateway/`, `ml/`, `c_engine/binding.py`,
  and the public entry points.  We do not require strict on the
  physics simulator or the per-subsystem generators -- they are
  numeric and benefit from the natural-typed style.
* Comments in English.  Doctype / class / function docstrings in
  English.  Issue and PR text in English.

## Reviewing a PR

The `pr-review-toolkit` agents we use for our own reviews:

* `code-reviewer` -- bugs, dead code, type issues
* `silent-failure-hunter` -- swallowed exceptions
* `type-design-analyzer` -- public types
* `comment-analyzer` -- docstring drift
* `pr-test-analyzer` -- test coverage

A PR that addresses an audit finding should reference the audit ID
(`AUDIT.md` line range) in the commit message so the audit history
stays navigable.

## Security

See `SECURITY.md`.  Do not file security issues in the public tracker.

## License

By contributing, you agree that your contributions are licensed
under the Apache-2.0 license (see `LICENSE`).
