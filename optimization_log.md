1 | fuse STEP2A zero-high-kx and z-reshuffle kernels | 7276.349 -> 7351.273 | kept
2 | fuse STEP2B middle-row zero into STEP3 | 7464.894 -> 7955.925 | reverted (baseline drift)
3 | use 32x8 x-major blocks for hot pointwise kernels | 7464.894 -> 7907.284 | kept
