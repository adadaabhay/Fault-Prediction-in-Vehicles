# Coding standard status — `c_engine/`

## Claim

**MISRA C:2012 *informed*. NOT MISRA C:2012 compliant.**

This file exists because the source previously carried the string
"MISRA-compatible C99" in its header comment, and `PROJECT.md` described the
edge runtime as a "MISRA-C99 embedded inference engine". Neither statement was
supportable.

## Why the previous claim was not valid

Per [MISRA Compliance:2020](https://misra.org.uk/app/uploads/2021/06/MISRA-Compliance-2020.pdf),
a claim of compliance requires all of:

1. A **Guideline Enforcement Plan** — how each guideline is checked.
2. A **Guideline Compliance Summary (GCS)** — a per-guideline table stating
   Compliant / Deviations / Disapplied / Violations.
3. A **deviation record** for every violated *Required* guideline, with
   justification and impact analysis.
4. Tool evidence from a checker configured against the ruleset.

This project has none of the four, and no MISRA checker runs in CI. Absence of
static-analysis findings is explicitly *not* compliance — many guidelines are
not statically decidable and require review or compiler evidence.

## Known deviations (undocumented at the time of the claim)

| Guideline | Category | Where | Notes |
|---|---|---|---|
| Rule 15.5 | Advisory | `tank_infer_step`, `fast_sigmoid`, `fast_tanh`, `tank_infer_invalidate` | Multiple points of exit. Early return on invalid input is deliberate — it is what makes the engine fail safe (see below) — but it is still a deviation and must be recorded as one. |
| Dir 4.6 | Advisory | all loop counters | Plain `int` rather than `int32_t`. |
| Dir 4.11 | Required | `expf`, `tanhf`, `logf` | Arguments are bounded by saturation clamps before the call, but the precondition is not formally validated or documented per-call-site. **A Required deviation — this alone invalidates a compliance claim until recorded.** |
| Rule 21.6 | Required | — | Not applicable; `stdio.h` is used only in `main_test.c`, which is test scaffolding and not part of the deployed unit. |

## What would be needed to make the claim

1. Run a MISRA-configured checker (Cppcheck `--addon=misra`, PC-lint Plus, or
   Polyspace) over `tank_pdm_infer.[ch]` and `tank_pdm_weights.[ch]` only —
   `main_test.c` is out of scope and must be excluded explicitly.
2. Add that run to `.github/workflows/tests.yml` as a blocking step.
3. Produce the GCS and a deviation record per Required violation.
4. Only then restore compliance language, and cite the GCS from the header.

Until step 4, the header says "informed, not compliant", and so does every
other document in this repository.

## Fail-safe behaviour (relevant to any future safety argument)

`tank_infer_step` returns `TankInferStatus`. It previously returned `void`, and
a single non-finite input propagated NaN through the softmax; because every
`fault_probs[c] > top_prob` comparison is false against NaN, the engine
returned `top_fault_id = 0` — class `"healthy"` — with `top_fault_prob = NaN`.
A corrupted sensor frame produced a confident all-clear.

The runtime now:

- validates every input element for finiteness before doing any work;
- re-checks the outputs for finiteness before committing them;
- on any failure, drives all outputs to zero, sets
  `top_fault_id = TANK_INFER_FAULT_UNKNOWN` (`0xFFFFFFFF`, never a valid class
  index), and returns a non-`TANK_INFER_OK` status.

**Callers must check the return value.** A result whose `status` is not
`TANK_INFER_OK` carries no information and must not be acted on.
