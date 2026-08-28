# Security Policy

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :white_check_mark: (security fixes only) |
| < 0.1   | :x:                |

The project follows [semantic versioning](https://semver.org/).  Security
patches land on the latest minor of the previous major and the latest
minor of the current major.  Older minors receive best-effort fixes if
the regression is reproducible from the shipped artefacts.

## Reporting a vulnerability

**Please do not file a public issue.**  Use GitHub's
[Security Advisories](https://github.com/adadaabhay/Fault-Prediction-in-Vehicles/security/advisories)
private-vulnerability-reporting form, or email
`security@phm-vehicle.local` (the `.local` suffix is mDNS-resolved on
the maintainer's own network; if the address is undeliverable, fall
back to the GitHub form).  Include:

1. A short, reproducible description of the issue
2. The smallest code path that triggers it
3. The version (git tag or commit SHA) you observed it on
4. Whether you intend to disclose publicly and on what timeline

We will acknowledge within **3 business days** and aim to ship a fix
within **30 days** of acknowledgement, faster for issues with a CVSS
score >= 7.0.

## Threat model

The project's attack surface is:

* **Mutating ingest path** — `/api/telemetry/push` (REST), guarded by
  the `TELEMETRY_API_KEY` policy in `telemetry_gateway/server.py`.
  A 401 from this path bumps
  `telemetry_push_rejected_total{reason="auth"}` so brute-force
  attempts are visible in `/metrics`.
* **WebSocket stream** — `/ws/telemetry`, gated by the same
  `TELEMETRY_ALLOWED_ORIGINS` policy.
* **Observability endpoints** — `/healthz`, `/readyz`, `/metrics`.
  These are **intentionally unauthenticated** so a Kubernetes kubelet
  or a Prometheus scraper can hit them without provisioning a
  credential.  The only sensitive datum they expose is the
  `telemetry_websocket_clients` gauge (an integer count of connected
  WebSockets), which is low-value operational metadata.  If you
  reverse-proxy the gateway behind a network that does not allow
  probe traffic, restrict these routes at the proxy.
* **C edge runtime** — `c_engine/tank_pdm_infer.c`, the binary that
  runs on the vehicle bus.

Out of scope: physical tampering with deployed hardware, side-channel
on the C runtime, and denial-of-service from a host that already has
LAN access to the gateway.

## Trust boundary

By default the gateway is **fail-closed**: with no
`TELEMETRY_API_KEY` configured, only loopback callers may push.  Set
`TELEMETRY_API_KEY` (and `TELEMETRY_ALLOWED_ORIGINS` for browser clients)
to open the boundary explicitly.  See `telemetry_gateway/server.py`
for the full set of environment toggles.

## C engine

The C edge runtime (`c_engine/`) is **MISRA-C99-informed, not
MISRA-compliant**.  See `c_engine/CODING_STANDARD.md` for the four
documented deviations and the path to upgrade.  Treat any deployment
that relies on the C runtime for a safety-relevant decision as
advisory until the compliance claim is upgraded.

## What ships

* Synthetic data only.  No real fleet telemetry, no operator data, no
  internal measurements.  Public corpora are procured copies of
  publicly-available research data and are not redistributed by this
  project; see `docs/PROVENANCE.md`.
* No API keys, passwords, or credentials are committed.  CI secrets
  are read from the GitHub Actions environment only.

## Compliance

If you are reviewing this project for export-control, ITAR, or
internal classification, the **synthetic data is the only thing that
ships**; the public corpora are procured copies of publicly-available
research data.  The source tree is verified clean of `TODO`/`FIXME`
classification markers by `python -m tools.check_artifacts` at
build time; the `archive/` subdirectory contains pre-audit
development scripts and is excluded from the build artefacts.
