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
12 | cap display sigma reductions at 512 blocks | 8895.321 -> 8732.070 | reverted
13 | reuse host vectors for diagnostic reductions | 8895.321 -> 8609.558 | reverted
14 | use 64x4 blocks for STEP2A/STEP2B/STEP3 | 8895.321 -> 8663.334 | reverted
15 | move STEP2B middle-row zero into STEP3 with state writeback | 8895.321 -> 8960.900 | reverted (output changed)
16 | precompute STEP3 time coefficients on host | 8895.321 -> 8881.905 | reverted
17 | write STEP3 high-z outputs directly in post-reshuffle layout | 8396.441 -> output changed | reverted
18 | use 32x16 blocks for STEP2A prepare | 8513.859 -> 8769.802 | reverted (profile slower; below prior best)
19 | compute STEP3 wavenumbers from indices | 8513.859 -> 8552.225 | kept
20 | vectorize STEP2B uiuj build across x pairs | 8552.225 -> 8732.283 | kept
21 | vectorize STEP2B uiuj build across x quads | 8732.283 -> 8685.888 | reverted
22 | use 64x4 blocks for STEP3 only | 8732.283 -> 8757.972 | kept
23 | add restrict qualifiers to STEP3 arrays | 8757.972 -> 8717.399 | reverted
24 | use 64x4 blocks for STEP2B vec2 build | 8757.972 -> 8541.963 | reverted
25 | use 64x4 blocks for STEP2A prepare only | 8757.972 -> 8759.579 | kept
26 | add GPU final reduction for CFLM | 8759.579 -> 8702.644 | reverted
27 | use 128x2 blocks for STEP2A prepare | 8759.579 -> 8469.071 | reverted
28 | use 16x16 blocks for STEP2B vec2 build | 8759.579 -> 8850.884 | kept
29 | use 16x16 blocks for STEP3 | 8850.884 -> 8794.379 | reverted
30 | precompute STEP3 time coefficients after wavenumber change | 8850.884 -> 8617.305 | reverted
31 | use 16x16 blocks for STEP2A prepare | 8850.884 -> 8469.300 | reverted
32 | use direct pair-stride indexing in STEP2B vec2 | 8850.884 -> 8662.522 | reverted
33 | raise display sigma reduction cap to 2048 blocks | 8850.884 -> 8561.911 | reverted
34 | prefer L1 cache for STEP3 fused kernel | 8850.884 -> 8626.245 | reverted
35 | prefer L1 cache for STEP2B vec2 build | 8850.884 -> 8660.450 | reverted
36 | replace STEP2B middle-zero kernel with cudaMemsetAsync | 8850.884 -> 8477.093 | reverted
37 | use 64x8 blocks for STEP3 | 8850.884 -> 8646.326 | reverted
38 | use 32x16 blocks for STEP2B vec2 build | 8850.884 -> 8667.122 | reverted
39 | use persistent OM2PHYS spectral scratch | 8826.200 -> 8717.652 | reverted
40 | capture repeated non-diagnostic timesteps with CUDA graph | 8786.579 -> 7318.357 | reverted
41 | use 128x2 blocks for STEP3 | 8496.233 -> 8693.180 | reverted
42 | manually address STEP2A prepare component planes | 8497.446 -> 8456.015 | reverted
43 | use 512-thread CFLM reduction blocks | 8790.592 -> 8338.351 | reverted
44 | cap display sigma reductions at 768 blocks | 8788.668 -> 8662.362 | reverted
45 | add restrict qualifiers to STEP2B vec2 build | 8748.420 -> 8555.321 | reverted
46 | unroll STEP2B middle-row zero stores | 8170.191 -> 8670.561 | reverted
47 | use warp reduction for display sigma minmax | 8787.201 -> 8368.538 | reverted
48 | use 64x8 blocks for STEP2A prepare | 8776.260 -> 8534.619 | reverted
49 | split STEP2B forward R2C cuFFT into batch2+batch1 plans for N=4096 | 81.280 -> 80.823 | reverted
50 | enable cuFFT patient JIT for STEP2B forward R2C plan | 81.280 -> 89.150 | reverted (output changed)
51 | use three single-batch STEP2B forward R2C plans for N=16384 | 5.447 -> output changed | reverted
52 | split STEP2A prepare zero-high-kx and z-reshuffle domains for N=16384 | 5.447 -> 5.453 | kept
53 | use cudaMemset2DAsync for STEP2A high-kx zero band at N=16384 | 5.453 -> 5.452 | reverted
54 | use 256x1 block for split STEP2A high-kx zero kernel at N=16384 | 5.453 -> 5.452 | reverted
55 | use 128x2 block for split STEP2A z-reshuffle kernel at N=16384 | 5.453 -> 5.452 | reverted
