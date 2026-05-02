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
