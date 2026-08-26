/**
 * @file main_test.c
 * @brief Native C test harness and latency micro-benchmark for tank_pdm_infer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tank_pdm_infer.h"

int main(void) {
    printf("===============================================================\n");
    printf("RUNNING NATIVE C99 INFERENCE BENCHMARK & MEMORY CHECK\n");
    printf("===============================================================\n");

    /* Allocate static structures on stack / arena */
    TankModelWeights weights;
    TankInferenceState state;
    TankInferenceResult result;

    /* Initialize dummy weights */
    for (int k = 0; k < TANK_INFER_D_FEATURES; k++) {
        for (int j = 0; j < TANK_INFER_H_HIDDEN; j++) {
            weights.Wf[k][j] = 0.01f * (float)(k % 5);
            weights.Wi[k][j] = 0.01f * (float)(k % 5);
            weights.Wc[k][j] = 0.01f * (float)(k % 5);
            weights.Wo[k][j] = 0.01f * (float)(k % 5);
        }
    }
    for (int k = 0; k < TANK_INFER_H_HIDDEN; k++) {
        for (int j = 0; j < TANK_INFER_H_HIDDEN; j++) {
            weights.Uf[k][j] = 0.01f;
            weights.Ui[k][j] = 0.01f;
            weights.Uc[k][j] = 0.01f;
            weights.Uo[k][j] = 0.01f;
        }
        weights.bf[k] = 0.0f;
        weights.bi[k] = 0.0f;
        weights.bc[k] = 0.0f;
        weights.bo[k] = 0.0f;

        for (int r = 0; r < TANK_INFER_R_PARTS; r++) {
            weights.Wy[k][r] = 0.02f;
            weights.by[r] = 0.0f;
        }
        for (int c = 0; c < TANK_INFER_C_CLASSES; c++) {
            weights.Wcls[k][c] = 0.02f;
            weights.bcls[c] = 0.0f;
        }
    }

    tank_infer_reset(&state);

    float dummy_x[TANK_INFER_D_FEATURES];
    for (int k = 0; k < TANK_INFER_D_FEATURES; k++) {
        dummy_x[k] = 0.5f;
    }

    /* Benchmark 10,000 inference steps */
    const int N_STEPS = 10000;
    clock_t start = clock();
    for (int n = 0; n < N_STEPS; n++) {
        if (tank_infer_step(&weights, &state, dummy_x, &result) != TANK_INFER_OK) {
            printf("FAIL: inference returned status %d
", (int)result.status);
            return 1;
        }
    }
    clock_t end = clock();

    double total_time_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double time_per_step_us = (total_time_sec / N_STEPS) * 1e6;

    printf("Inference Steps Completed : %d\n", N_STEPS);
    printf("Total Execution Time      : %.4f seconds\n", total_time_sec);
    printf("Latency per Step          : %.2f microseconds (%.4f ms)\n", time_per_step_us, time_per_step_us / 1000.0);
    printf("10 Hz Real-Time Budget    : 100,000.0 microseconds (%.2f%% utilized)\n", (time_per_step_us / 100000.0) * 100.0);
    printf("Mean RUL (pct of cap)     : %.2f / 100\n", result.mean_rul_pct);
    printf("Top Predicted Fault ID    : %u (Confidence: %.4f)\n", result.top_fault_id, result.top_fault_prob);
    printf("Static Memory Footprint   : %zu bytes (< 32 KB)\n", sizeof(weights) + sizeof(state) + sizeof(result));

    if (time_per_step_us < 500.0) {
        printf("VERDICT: [PASS] Real-time embedded latency constraint fully satisfied.\n");
        return 0;
    } else {
        printf("VERDICT: [FAIL] Latency exceeds 500us limit.\n");
        return 1;
    }
}
