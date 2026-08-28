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

**Please do not file a public issue.**  Email
`security@phm-vehicle.local` with:

1. A short, reproducible description of the issue
2. The smallest code path that triggers it
3. The version (git tag or commit SHA) you observed it on
4. Whether you intend to disclose publicly and on what timeline

We will acknowledge within **3 business days** and aim to ship a fix
within **30 days** of acknowledgement, faster for issues with a CVSS
score >= 7.0.

## Threat model

The project's attack surface is the FastAPI ingest path
(`/api/telemetry/push`, `/ws/telemetry`) and the C edge runtime that
runs on a vehicle bus.  Out of scope: physical tampering with deployed
hardware, side-channel on the C runtime, and denial-of-service from a
host that already has LAN access to the gateway.

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
research data.  No `TODO`, `FIXME`, or classification marker exists in
the source tree.
