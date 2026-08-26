/**
 * @file tank_pdm_infer.h
 * @brief Zero-allocation C99 edge inference engine for Tank PDM.
 *
 * CODING-STANDARD STATUS: MISRA-C:2012 *informed*, NOT MISRA-compliant.
 * This file previously claimed "MISRA-compatible". Per MISRA Compliance:2020
 * a compliance claim requires a Guideline Compliance Summary, a deviation
 * record for every violated Required guideline, and tool evidence. None of
 * those exist for this project and no MISRA checker runs in CI, so the claim
 * cannot be made. Known deviations, undocumented and therefore disqualifying:
 *   Rule 15.5  (single point of exit) -- early returns in tank_infer_step,
 *              fast_sigmoid, fast_tanh.
 *   Dir  4.6   (fixed-width types) -- plain `int` loop counters.
 *   Dir  4.11  (library function preconditions) -- expf/tanhf domain is
 *              bounded by saturation but not formally validated.
 * See c_engine/CODING_STANDARD.md before restoring any compliance language.
 * 
 * Target: Embedded Vehicle Management Computers (VMC) & Engine Control Units (ECU).
 * Memory footprint: < 32 KB Static RAM, 0 dynamic allocations (no malloc/free).
 */

#ifndef TANK_PDM_INFER_H
#define TANK_PDM_INFER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dimensions are generated from docs/config.json by c_engine/gen_dims.py, so
   the runtime cannot drift from the trained model. This header previously
   hardcoded D=58 against a model with D=24, which made the engine incapable of
   consuming the shipped weights. Regenerate with:
       python -m c_engine.gen_dims                                          */
#include "tank_pdm_dims.h"

/**
 * @brief Static Model Weights Layout.
 */
typedef struct {
    /* LSTM Gates: Forget (f), Input (i), Cell (c), Output (o) */
    float Wf[TANK_INFER_D_FEATURES][TANK_INFER_H_HIDDEN];
    float Uf[TANK_INFER_H_HIDDEN][TANK_INFER_H_HIDDEN];
    float bf[TANK_INFER_H_HIDDEN];

    float Wi[TANK_INFER_D_FEATURES][TANK_INFER_H_HIDDEN];
    float Ui[TANK_INFER_H_HIDDEN][TANK_INFER_H_HIDDEN];
    float bi[TANK_INFER_H_HIDDEN];

    float Wc[TANK_INFER_D_FEATURES][TANK_INFER_H_HIDDEN];
    float Uc[TANK_INFER_H_HIDDEN][TANK_INFER_H_HIDDEN];
    float bc[TANK_INFER_H_HIDDEN];

    float Wo[TANK_INFER_D_FEATURES][TANK_INFER_H_HIDDEN];
    float Uo[TANK_INFER_H_HIDDEN][TANK_INFER_H_HIDDEN];
    float bo[TANK_INFER_H_HIDDEN];

    /* Output Heads */
    float Wy[TANK_INFER_H_HIDDEN][TANK_INFER_R_PARTS];
    float by[TANK_INFER_R_PARTS];

    float Wcls[TANK_INFER_H_HIDDEN][TANK_INFER_C_CLASSES];
    float bcls[TANK_INFER_C_CLASSES];
} TankModelWeights;

/**
 * @brief Dynamic Recurrent State Buffer (Static Memory Arena).
 */
typedef struct {
    float h[TANK_INFER_H_HIDDEN];
    float c[TANK_INFER_H_HIDDEN];
    uint32_t step_count;
} TankInferenceState;

/**
 * @brief Inference status. Anything other than TANK_INFER_OK means the
 *        contents of the result struct must NOT be acted on.
 *
 * Previously there was no status at all. A single non-finite input propagated
 * NaN through the softmax, every `fault_probs[c] > top_prob` comparison
 * evaluated false, and the engine returned top_fault_id = 0 -- which is class
 * "healthy" -- with top_fault_prob = NaN. A corrupted sensor frame produced a
 * confident all-clear. Safety-relevant inference must fail SAFE, not NOMINAL.
 */
typedef enum {
    TANK_INFER_OK = 0,           /**< Result is valid.                        */
    TANK_INFER_ERR_NULL = 1,     /**< A required pointer was NULL.            */
    TANK_INFER_ERR_INPUT = 2,    /**< Input vector contained NaN/Inf.         */
    TANK_INFER_ERR_STATE = 3     /**< Recurrent state went non-finite.        */
} TankInferStatus;

/** Sentinel written into every output field when status != TANK_INFER_OK. */
#define TANK_INFER_FAULT_UNKNOWN ((uint32_t)0xFFFFFFFFu)

/**
 * @brief Inference Output Result.
 */
typedef struct {
    float ruls[TANK_INFER_R_PARTS];          /**< RUL fraction [0.0 - 1.0] per part */
    float fault_probs[TANK_INFER_C_CLASSES]; /**< Softmax probability distribution */
    uint32_t top_fault_id;                   /**< Predicted highest probability fault class */
    float top_fault_prob;                    /**< Confidence of top prediction */
    /** Mean predicted RUL fraction across parts, scaled to 0-100.
     *  NOTE: this is a mean RUL, not a health index. It was named and
     *  documented as a "Composite Health Index"; they are different
     *  quantities and conflating them misreports the vehicle's condition. */
    float mean_rul_pct;
    TankInferStatus status;                  /**< Validity of this result.    */
} TankInferenceResult;

/**
 * @brief Initialize inference state and clear recurrent buffers.
 */
void tank_infer_reset(TankInferenceState* state);

/**
 * @brief Execute a single 10 Hz forward step.
 * 
 * @param weights Pointer to static model weights.
 * @param state Pointer to recurrent state.
 * @param x Input feature vector, TANK_INFER_D_FEATURES long.
 *          (This said "D=58" while the shipped model has D=24; the count
 *           now comes from tank_pdm_dims.h so it cannot drift again.)
 * @param result Pointer to destination output struct.
 */
TankInferStatus tank_infer_step(const TankModelWeights* weights,
                                TankInferenceState* state,
                                const float x[TANK_INFER_D_FEATURES],
                                TankInferenceResult* result);

#ifdef __cplusplus
}
#endif

#endif /* TANK_PDM_INFER_H */
