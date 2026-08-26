/**
 * @file tank_pdm_infer.c
 * @brief Zero-allocation C99 edge inference engine implementation.
 *
 * MISRA-C:2012 informed, not MISRA-compliant. See tank_pdm_infer.h.
 */

#include "tank_pdm_infer.h"
#include <math.h>
#include <string.h>

/* Activation Functions */
static inline float fast_sigmoid(float z) {
    if (z > 40.0f) return 1.0f;
    if (z < -40.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z));
}

static inline float fast_tanh(float z) {
    if (z > 20.0f) return 1.0f;
    if (z < -20.0f) return -1.0f;
    return tanhf(z);
}

static void fast_softmax(const float* logits, float* probs, int len) {
    if (len <= 0) {
        return;  /* nothing to compute; caller must not rely on probs contents */
    }
    float max_val = logits[0];
    for (int i = 1; i < len; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }

    float inv_sum = (sum > 1e-12f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < len; i++) {
        probs[i] *= inv_sum;
    }
}

void tank_infer_reset(TankInferenceState* state) {
    if (state != NULL) {
        memset(state->h, 0, sizeof(state->h));
        memset(state->c, 0, sizeof(state->c));
        state->step_count = 0;
    }
}

/* Drive every output field to an unambiguous "do not use" state. */
static void tank_infer_invalidate(TankInferenceResult* res, TankInferStatus st) {
    int k;
    for (k = 0; k < TANK_INFER_R_PARTS; k++) {
        res->ruls[k] = 0.0f;
    }
    for (k = 0; k < TANK_INFER_C_CLASSES; k++) {
        res->fault_probs[k] = 0.0f;
    }
    res->top_fault_id = TANK_INFER_FAULT_UNKNOWN;
    res->top_fault_prob = 0.0f;
    res->mean_rul_pct = 0.0f;
    res->status = st;
}

static int tank_is_finite(float v) {
    /* NaN fails both comparisons; +/-Inf fails the magnitude bound. */
    return (v == v) && (v < 3.0e38f) && (v > -3.0e38f);
}

TankInferStatus tank_infer_step(const TankModelWeights* w,
                                TankInferenceState* s,
                                const float x[TANK_INFER_D_FEATURES],
                                TankInferenceResult* res) {
    if (w == NULL || s == NULL || x == NULL || res == NULL) {
        if (res != NULL) {
            tank_infer_invalidate(res, TANK_INFER_ERR_NULL);
        }
        return TANK_INFER_ERR_NULL;
    }

    /* Input plausibility. The upstream FDIR gate should already have removed
       non-finite samples, but the edge runtime cannot assume it ran. */
    {
        int k;
        for (k = 0; k < TANK_INFER_D_FEATURES; k++) {
            if (!tank_is_finite(x[k])) {
                tank_infer_invalidate(res, TANK_INFER_ERR_INPUT);
                return TANK_INFER_ERR_INPUT;
            }
        }
    }

    const int D = TANK_INFER_D_FEATURES;
    const int H = TANK_INFER_H_HIDDEN;
    const int R = TANK_INFER_R_PARTS;
    const int C = TANK_INFER_C_CLASSES;

    float f[TANK_INFER_H_HIDDEN];
    float i_gate[TANK_INFER_H_HIDDEN];
    float c_t[TANK_INFER_H_HIDDEN];
    float o[TANK_INFER_H_HIDDEN];
    float h_new[TANK_INFER_H_HIDDEN];
    float c_new[TANK_INFER_H_HIDDEN];

    /* Gate Computations */
    for (int j = 0; j < H; j++) {
        float act_f = w->bf[j];
        float act_i = w->bi[j];
        float act_c = w->bc[j];
        float act_o = w->bo[j];

        /* Input projection: x @ W */
        for (int k = 0; k < D; k++) {
            float x_val = x[k];
            act_f += x_val * w->Wf[k][j];
            act_i += x_val * w->Wi[k][j];
            act_c += x_val * w->Wc[k][j];
            act_o += x_val * w->Wo[k][j];
        }

        /* Recurrent projection: h_prev @ U */
        for (int k = 0; k < H; k++) {
            float h_val = s->h[k];
            act_f += h_val * w->Uf[k][j];
            act_i += h_val * w->Ui[k][j];
            act_c += h_val * w->Uc[k][j];
            act_o += h_val * w->Uo[k][j];
        }

        f[j] = fast_sigmoid(act_f);
        i_gate[j] = fast_sigmoid(act_i);
        c_t[j] = fast_tanh(act_c);
        o[j] = fast_sigmoid(act_o);

        /* Cell and hidden state are staged, NOT written here.
           Writing s->h[j] inside this loop aliased the recurrent input: the
           k-loop above reads s->h[k] as h_{t-1}, so once h[0] had been
           overwritten every subsequent unit j summed a mixture of the current
           and previous timesteps. The error appeared from the very first step
           (where h_{t-1} is all zeros) and drove the RUL head 9.0e-4 away from
           the reference implementation. */
        c_new[j] = f[j] * s->c[j] + i_gate[j] * c_t[j];
        h_new[j] = o[j] * fast_tanh(c_new[j]);
    }

    /* Commit the timestep only once every unit has been computed. */
    for (int j = 0; j < H; j++) {
        s->c[j] = c_new[j];
        s->h[j] = h_new[j];
    }

    s->step_count++;

    /* RUL Regression Head: ruls = sigmoid(h @ Wy + by) */
    float sum_chi = 0.0f;
    for (int r = 0; r < R; r++) {
        float act_y = w->by[r];
        for (int j = 0; j < H; j++) {
            act_y += s->h[j] * w->Wy[j][r];
        }
        res->ruls[r] = fast_sigmoid(act_y);
        sum_chi += res->ruls[r];
    }
    res->mean_rul_pct = (sum_chi / (float)R) * 100.0f;

    /* Fault Classification Head: Softmax(h @ Wcls + bcls) */
    float logits[TANK_INFER_C_CLASSES];
    for (int ci = 0; ci < C; ci++) {
        logits[ci] = w->bcls[ci];
        for (int j = 0; j < H; j++) {
            logits[ci] += s->h[j] * w->Wcls[j][ci];
        }
    }
    fast_softmax(logits, res->fault_probs, C);

    /* Find Top Predicted Fault */
    uint32_t top_idx = 0;
    float top_prob = res->fault_probs[0];
    for (int ci = 1; ci < C; ci++) {
        if (res->fault_probs[ci] > top_prob) {
            top_prob = res->fault_probs[ci];
            top_idx = (uint32_t)ci;
        }
    }
    /* If the recurrent state diverged, top_prob is NaN and top_idx is still 0
       ("healthy") because every comparison above was false. Catch it here
       rather than reporting a confident all-clear. */
    if (!tank_is_finite(top_prob) || !tank_is_finite(res->mean_rul_pct)) {
        tank_infer_invalidate(res, TANK_INFER_ERR_STATE);
        return TANK_INFER_ERR_STATE;
    }

    res->top_fault_id = top_idx;
    res->top_fault_prob = top_prob;
    res->status = TANK_INFER_OK;
    return TANK_INFER_OK;
}
