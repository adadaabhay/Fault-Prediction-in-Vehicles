#include "tank_pdm_weights.h"
#include <stdio.h>

int tank_weights_load(const char* path, TankModelWeights* out)
{
    FILE* fh;
    size_t want;
    size_t got;

    if ((path == NULL) || (out == NULL)) {
        return -1;
    }

    fh = fopen(path, "rb");
    if (fh == NULL) {
        return -1;
    }

    want = sizeof(TankModelWeights) / sizeof(float);
    got = fread(out, sizeof(float), want, fh);
    (void)fclose(fh);

    return (got == want) ? 0 : -2;
}
