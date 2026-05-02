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
