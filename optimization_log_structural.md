1 | explicit stream CUDA graph replay for stable timestep groups | 8831.690 -> 9644.179 | kept
2 | reuse CUDA graph stream across stable timestep groups | 9644.179 -> 9691.366 | kept
3 | update cached CUDA graph executable after recapture | 9691.366 -> 9670.917 | reverted
4 | keep hot solver on graph stream between groups | 9691.366 -> 9691.226 | reverted
5 | upload CUDA graph during instantiation | 9691.366 -> 9690.743 | reverted
6 | cache stable-step graph with STEP3 scalars in device memory | 9691.366 -> 9588.668 | reverted
7 | check phase timing flag directly in timing macro | 9691.366 -> 9656.558 | reverted
8 | use thread-local CUDA stream capture mode | 9691.366 -> 9664.715 | reverted
9 | use nonblocking graph stream with explicit default-stream wait | 9691.366 -> 9680.741 | reverted
10 | use current CUDA graph instantiate overload | 9691.366 -> 9669.772 | reverted
11 | avoid switching diagnostic cuFFT plan stream in graph path | 9691.366 -> 9629.544 | reverted
12 | use batch-3 inverse C2R cuFFT plan | 9691.366 -> 8503.376 | reverted
13 | disable stable-step CUDA graph replay for N>=4096 | 81.353 -> 81.354 | kept
14 | use CUFFT_WORKAREA_MINIMAL for STEP2B forward R2C plan | 81.354 -> unsupported | reverted
15 | enable cuFFT patient JIT for STEP2A inverse C2R plan | 81.354 -> 86.161 | reverted (output changed)
16 | use CUFFT_WORKAREA_MINIMAL for STEP2A inverse C2R plan | 81.354 -> unsupported | reverted
17 | split STEP2A inverse C2R into two single-component cuFFT launches | 81.354 -> 80.724 | reverted
18 | use dense null-embed layout for STEP2B forward R2C cuFFT plan | 81.354 -> 81.414 | kept
19 | use dense null-embed layout for STEP2A inverse C2R cuFFT plan | 81.414 -> 81.406 | reverted
20 | skip stable-step graph grouping block for N>=4096 | 81.414 -> 81.401 | reverted
21 | use cuFFT Xt typed interface for STEP2B forward R2C plan | 81.377 -> 81.372 | reverted
22 | prune STEP2B z FFT to kx<N/2 columns | 81.377 -> 89.880 | kept
23 | prune STEP2A inverse z FFT to kx<N/2 columns | 89.880 -> 96.762 | kept
24 | use cuFFT Xt typed interface for pruned C2C z plan | 96.762 -> 96.739 | reverted
25 | enable cuFFT patient JIT for pruned C2C z plan | 96.762 -> 96.675 | reverted
26 | enable cuFFT patient JIT for pruned R2C x plan | 96.762 -> output changed | reverted
27 | enable cuFFT patient JIT for pruned C2R x plan | 96.762 -> output changed | reverted
28 | use cufftMakePlanMany64 for pruned R2C x plan | 96.762 -> 96.694 | reverted
29 | use cufftMakePlanMany64 for pruned C2R x plan | 96.762 -> 96.670 | reverted
30 | use cufftMakePlanMany64 for pruned C2C z plan | 96.762 -> 96.791 | kept
31 | use contiguous scratch for STEP2B C2C z FFT | 96.791 -> output changed | reverted
32 | use 32x8 blocks for STEP2B uiuj build | 96.791 -> 96.765 | reverted
33 | pad UC_full spectral pitch to even complex stride | 96.791 -> 97.664 | kept
34 | pad UC_full spectral pitch to 32-byte row stride | 97.664 -> 98.299 | kept
35 | pad UC_full spectral pitch to 64-byte row stride | 98.299 -> 97.888 | reverted
36 | use 32x8 blocks for STEP3 fused | 98.299 -> 98.289 | reverted
37 | use 32x8 blocks for STEP2A prepare | 98.299 -> 98.354 | kept
38 | pad UC_full spectral pitch to alternate 32-byte stride | 98.354 -> 97.675 | reverted
39 | use 16x8 blocks for STEP2A prepare | 98.354 -> 98.086 | reverted
40 | use 32x4 blocks for STEP2A prepare | 98.354 -> 98.332 | reverted
41 | use 32x4 blocks for STEP3 fused | 98.354 -> 98.337 | reverted
42 | use 16x8 blocks for STEP2B uiuj build | 98.354 -> 98.311 | reverted
43 | use 8x16 blocks for STEP2B uiuj build | 98.354 -> 97.762 | reverted
44 | use 64x2 blocks for STEP3 fused | 98.354 -> 98.274 | reverted
45 | use 16x32 blocks for STEP2B uiuj build | 98.354 -> 98.213 | reverted
46 | use 96x2 blocks for STEP3 fused | 98.354 -> 98.315 | reverted
47 | use 64x2 blocks for STEP2A prepare | 98.354 -> 98.373 | kept
48 | use 96x2 blocks for STEP2A prepare | 98.373 -> 98.395 | kept
49 | use 128x1 blocks for STEP2A prepare | 98.395 -> 98.367 | reverted
50 | use 96x1 blocks for STEP2A prepare | 98.395 -> 98.353 | reverted
51 | retry pruned STEP2B forward z FFT to kx<N/2 for N=16384 | 5.443 -> 3.233 | reverted
52 | retry pruned STEP2A inverse z FFT to kx<N/2 for N=16384 | 5.443 -> 3.738 | reverted
53 | retry 32-byte UC_full spectral pitch for N=16384 | 5.443 -> 5.401 | reverted
54 | retry even UC_full spectral pitch for N=16384 | 5.443 -> 5.397 | reverted
55 | use cufftMakePlanMany64 for STEP2B full R2C plan at N=16384 | 5.443 -> 5.442 | reverted
56 | use cufftMakePlanMany64 for STEP2A full C2R plan at N=16384 | 5.443 -> 5.441 | reverted
57 | retry 64x2 STEP2A prepare blocks for N=16384 | 5.443 -> 5.445 | kept
58 | retry 96x2 STEP2A prepare blocks for N=16384 | 5.445 -> 5.446 | kept
