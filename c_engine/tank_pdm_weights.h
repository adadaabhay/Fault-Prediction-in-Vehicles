/**
 * @file tank_pdm_weights.h
 * @brief Static weight loading for the edge inference engine.
 */
#ifndef TANK_PDM_WEIGHTS_H
#define TANK_PDM_WEIGHTS_H

#include "tank_pdm_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load weights from a little-endian float32 blob into a static struct.
 *
 * The on-disk order matches the TankModelWeights field order exactly, so this
 * is a single read into caller-owned storage. No allocation is performed.
 *
 * @return 0 on success, -1 on bad argument or open failure, -2 on size mismatch.
 */
int tank_weights_load(const char* path, TankModelWeights* out);

#ifdef __cplusplus
}
#endif

#endif /* TANK_PDM_WEIGHTS_H */
