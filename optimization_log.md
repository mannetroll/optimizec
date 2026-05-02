1 | fuse STEP2A zero-high-kx and z-reshuffle kernels | 7276.349 -> 7351.273 | kept
2 | use 32x8 x-major blocks for hot pointwise kernels | 7464.894 -> 7907.284 | kept
3 | remove redundant OM2PHYS synchronize before blocking copy | 7907.284 -> 7921.674 | kept
4 | use one 256-thread block for STEP2B middle-row zero | 7921.674 -> 8014.892 | kept
5 | read STEP2B middle row as zero inside STEP3 | 8013.856 -> 7872.452 | reverted
6 | remove dead DEN zero guard in STEP3 | 8014.892 -> 8069.238 | kept
7 | compute display sigma with persistent GPU reductions | 8069.238 -> 8441.688 | kept
8 | replace CFLM atomic reduction with per-block max reduction | 8441.688 -> 8887.644 | kept
9 | fuse OM2PHYS clear/copy/reshuffle setup | 8887.644 -> 8615.068 | reverted
10 | use 32x8 blocks for OM2PHYS setup kernels | 8887.644 -> 8660.987 | reverted
11 | cap display sigma reductions at 1024 blocks | 8887.644 -> 8895.321 | kept
