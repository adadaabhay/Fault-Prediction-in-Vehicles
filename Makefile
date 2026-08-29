# ---------------------------------------------------------------------------
# phm-vehicle Makefile
#
# The package is also installable via `pip install -e .`; the targets below
# are the developer-friendly entry points and are intentionally dependency-
# free (no `make` plugins required).
# ---------------------------------------------------------------------------

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY      ?= python
PIP     ?= $(PY) -m pip
PYTEST  ?= $(PY) -m pytest
RUFF    ?= $(PY) -m ruff
ACT     ?= act

# ---- Help -------------------------------------------------------------------
.PHONY: help
help: ## show this help message
	@awk 'BEGIN {FS = ":.*?## "; printf "Targets:\n"} \
	/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' \
	$(MAKEFILE_LIST)

# ---- Install ----------------------------------------------------------------
.PHONY: install
install: ## install the package in editable mode with dev + bench extras
	$(PIP) install -e ".[dev,bench]"

.PHONY: install-prod
install-prod: ## production install (runtime deps only)
	$(PIP) install -e .

.PHONY: install-requirements
install-requirements: ## install from the pinned requirements files
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# ---- Test / Lint ------------------------------------------------------------
.PHONY: test
test: ## run the test suite
	$(PYTEST) -q

.PHONY: test-cov
test-cov: ## run the test suite with coverage
	$(PYTEST) \
		--cov=telemetry_gateway \
		--cov=ml \
		--cov=sim \
		--cov=c_engine \
		--cov=tools \
		--cov-report=term-missing

.PHONY: test-hud
test-hud: ## browser-driven HUD smoke tests (requires playwright browsers)
	$(PYTEST) -m hud -q

.PHONY: test-fast
test-fast: ## skip slow, hud, and hil tests
	$(PYTEST) -q -m "not slow and not hud and not hil"

.PHONY: lint
lint: ## ruff lint
	$(RUFF) check .

.PHONY: format
format: ## ruff format
	$(RUFF) format .

.PHONY: typecheck
typecheck: ## mypy (advisory)
	$(PY) -m mypy .

.PHONY: verify
verify: lint test-fast ## CI-style local gate (lint + fast tests)
	@echo "verify: PASS"

# ---- Sim / data -------------------------------------------------------------
.PHONY: data
data: ## generate one CSV per subsystem into data/simulated/
	$(PY) -m tools.generate_sensor_data --subsystem all --steps 500

.PHONY: data-cvrde
data-cvrde: ## generate the Thar Desert 60s mission stream
	$(PY) -m sim.cvrde.cvrde_generator

.PHONY: data-clean
data-clean: ## remove generated datasets
	rm -rf data/simulated

# ---- C edge runtime ---------------------------------------------------------
.PHONY: c-build
c-build: ## build the MISRA-C99 LSTM edge runtime (CMake)
	cmake -S c_engine -B c_engine/build -DCMAKE_BUILD_TYPE=Release
	cmake --build c_engine/build --config Release

.PHONY: c-test
c-test: c-build ## run the C ↔ Python parity suite
	$(PYTEST) -m parity -q

# ---- Server -----------------------------------------------------------------
.PHONY: serve
serve: ## run the FastAPI telemetry gateway on :8000
	$(PY) -m telemetry_gateway.server

.PHONY: serve-hud
serve-hud: ## serve docs/ on :8080 so the dashboard can be opened in a browser
	$(PY) -m http.server 8080 --directory docs

# ---- CI simulation ---------------------------------------------------------
.PHONY: act
act: ## run the GitHub Actions pipeline locally with `act` (https://nektosact.com)
	$(ACT) -j verify

# ---- Housekeeping -----------------------------------------------------------
.PHONY: clean
clean: ## remove caches and build outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf c_engine/build dist build *.egg-info

.PHONY: distclean
distclean: clean data-clean ## everything `clean` does, plus generated data
