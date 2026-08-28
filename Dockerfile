# Multi-stage build for the MBT Tactical Telemetry Gateway.
#
# Stage 1 (builder) compiles the C edge runtime and resolves all Python
# wheels into a single prefix.  Stage 2 (runtime) is a distroless image
# with only the resolved prefix and the dashboard assets.  The final
# image runs as a non-root user, exposes the gateway on :8000 and the
# dashboard on :8080, and uses /healthz as the HEALTHCHECK endpoint.
#
# Build:    docker build -t phm-vehicle:dev .
# Run:      docker run --rm -p 8000:8000 -p 8080:8080 phm-vehicle:dev
# Verify:   curl -s http://localhost:8000/healthz   # → {"status":"ok"}
#           curl -s http://localhost:8000/readyz    # → {"status":"ready",...}
#           curl -s http://localhost:8000/metrics   # → Prometheus exposition

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
# A full Python image with gcc / cmake for the C engine, then a
# requirements-only "wheels" image that copies them into a slim prefix.
FROM python:3.12-slim AS builder

# System packages the C engine and the ingest path need at build time.
# libusb, gcc, and cmake are dropped from the runtime stage.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libusb-1.0-0-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the package metadata first so the wheel resolution is cached
# unless pyproject.toml / setup files actually change.
COPY pyproject.toml ./
COPY telemetry_gateway ./telemetry_gateway
COPY ml ./ml
COPY sim ./sim
COPY c_engine ./c_engine
COPY pipelines ./pipelines
COPY tools ./tools
COPY docs ./docs
COPY data ./data
COPY results ./results
COPY tests ./tests

# Resolve every wheel we need into /wheels, then install them into
# /install.  --no-deps would be faster but we need transitive deps, so
# we let pip resolve normally and rely on --no-cache-dir for size.
# ``.[bench]`` is the only extra the runtime needs — observability uses
# the hand-rolled ``telemetry_gateway/metrics.py`` (no third-party
# prometheus_client dep), and dev/test extras are not shipped in the
# image.  The previous version requested a non-existent ``observability``
# extra; the ``|| pip wheel ... [bench]`` fallback silently masked the
# failure and produced a build that the maintainer could not reproduce.
RUN pip wheel --wheel-dir=/wheels --no-cache-dir -e ".[bench]"

FROM python:3.12-slim AS installer
COPY --from=builder /wheels /wheels
COPY --from=builder /build /build
WORKDIR /build
RUN pip install --no-cache-dir --no-index --find-links=/wheels ".[bench]"

# ---------------------------------------------------------------------------
# Stage 2 — distroless runtime
# ---------------------------------------------------------------------------
# gcr.io/distroless/python3-debian12 is a Debian 12 base with Python 3
# and the libpython shared object; it has no shell, no apt, no package
# manager.  The nonroot UID (65532) is the distroless default.
FROM gcr.io/distroless/python3-debian12:nonroot

# Copy the resolved Python environment and the application code.
COPY --from=installer /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=installer /usr/local/bin /usr/local/bin
COPY --from=installer /build/telemetry_gateway /app/telemetry_gateway
COPY --from=installer /build/docs /app/docs
COPY --from=installer /build/ml /app/ml
COPY --from=installer /build/sim /app/sim
COPY --from=installer /build/c_engine /app/c_engine
COPY --from=installer /build/pipelines /app/pipelines
COPY --from=installer /build/tools /app/tools
COPY --from=installer /build/data /app/data
COPY --from=installer /build/results /app/results

WORKDIR /app

# Distroless runs as UID 65532 by default; PYTHONPATH lets the gateway
# find its sibling packages without installing them site-wide.
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT_GATEWAY=8000 \
    PORT_DASHBOARD=8080

# Probe /readyz, not /healthz.  Docker's HEALTHCHECK semantic is
# "container is doing useful work, restart it if not" — closer to a
# readiness/restart probe than a liveness probe.  /healthz is by
# design a flat 200 if the worker is alive; that means a container
# with a broken broker or pipeline would pass /healthz forever and
# Docker would never restart it.  /readyz is 503 in that case, so
# Docker restarts the container after ``--retries`` consecutive
# failures.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"]
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]

EXPOSE 8000 8080

USER nonroot:nonroot

# `python -m telemetry_gateway.server` binds to 0.0.0.0; the
# TELEMETRY_DASHBOARD_DIR env var tells the gateway to mount the static
# dashboard at /dashboard so a single container serves both surfaces.
ENTRYPOINT ["python", "-m", "telemetry_gateway.server"]
