# cudaturbo

CUDA DNS benchmark/simulation driver using cuFFT.

## Build

For H100:

```bash
make clean
make ARCH=sm_90
```

For RTX 3090:

```bash
make clean
make ARCH=sm_86
```

Confirm the binary target:

```bash
strings ./mem_cuda | grep -Eo 'sm_[0-9]+|compute_[0-9]+' | sort -u
```

For H100, `sm_90` should be present. A stray `sm_30` string can appear from linked CUDA libraries and is not a problem.

## Run Location

Large runs should be started from NVMe, not `/tmp` or the small root volume:

```bash
cd /opt/dlami/nvme/cudaturbo-runs
```

If using the NVMe checkout:

```bash
cd /opt/dlami/nvme/cudaturbo-runs/cudaturbo
make clean
make ARCH=sm_90
```

Then either run from that checkout, or copy/use the binary from it.

## Command Format

```bash
./mem_cuda N Re K0 STEPS CFL UPDATE ADAPT_VISC
```

Example:

```bash
./mem_cuda 29400 1E7 4 1000000 0.1 100 0
```

Arguments:

```text
N           base grid size
Re          Reynolds number
K0          initial spectrum parameter
STEPS       max number of timesteps
CFL         CFL target
UPDATE      diagnostic/update interval
ADAPT_VISC  0 disabled, 1 enabled
```

For normal large H100 runs, use `ADAPT_VISC=0`.

## Background Run

From the run directory:

```bash
cd /opt/dlami/nvme/cudaturbo-runs

nohup stdbuf -oL /opt/dlami/nvme/cudaturbo-runs/cudaturbo/mem_cuda 29400 1E7 4 1000000 0.1 100 0 > mem_cuda.txt 2>&1 &
echo $!
```

Monitor:

```bash
tail -f mem_cuda.txt
```

## Timed Run

Run for about 1 hour, then save and exit:

```bash
cd /opt/dlami/nvme/cudaturbo-runs

nohup timeout -s HUP 1h stdbuf -oL /opt/dlami/nvme/cudaturbo-runs/cudaturbo/mem_cuda 29400 1E7 4 1000000 0.1 100 0 > mem_cuda.txt 2>&1 &
echo $!
```

`timeout` sends `SIGHUP` after 1 hour. `mem_cuda` treats `SIGHUP` as graceful save-and-exit.

## Signals

Use the PID printed by `echo $!`, or find the process:

```bash
pgrep -af mem_cuda
```

Save a snapshot and keep running:

```bash
kill -USR1 <pid>
```

Save images plus restart binary and keep running:

```bash
kill -RTMIN <pid>
```

This writes `restart.bin` into the same `output_...` folder as the PGM/CSV files. Linux has `SIGUSR1` and `SIGUSR2`, but no `SIGUSR3`; `SIGRTMIN` is used as the extra save signal.

Restart from that file by passing it as the final argument:

```bash
./mem_cuda 29400 1E7 4 1000000 0.1 100 0 output_.../restart.bin
```

Or restart from the output folder:

```bash
./mem_cuda RESTART output_...
```

`RESTART` reads `meta.csv` and `restart.bin` from the folder. It defaults to target iteration `1000000`; override with `RESTART_STEPS=200000`.

Pause after the current CUDA iteration:

```bash
kill -USR2 <pid>
```

Resume a paused run:

```bash
kill -CONT <pid>
```

Save and exit:

```bash
kill -HUP <pid>
```

Kill immediately without saving:

```bash
kill -INT <pid>
```

For a foreground run, Ctrl-C sends the same `SIGINT` and terminates immediately without saving.
For a background job, use `kill -INT <pid>` or bring it foreground with `fg` before pressing Ctrl-C.

## Output

Saves create folders like:

```text
output_29400_4_1E07_0.10_123400
```

The suffix includes:

```text
N_K0_Re_CFL_iteration
```

Each output folder contains PGM fields, energy spectrum CSV, metrics CSV, and metadata.

Large `N` output folders are large. For example, `N=29400` writes `44100 x 44100` PGM files and can produce multi-GB output folders. Keep these on NVMe.

## Known Good H100 Examples

Short verification:

```bash
cd /opt/dlami/nvme/cudaturbo-runs
timeout -s INT 120 stdbuf -oL /opt/dlami/nvme/cudaturbo-runs/cudaturbo/mem_cuda 29400 1E7 4 41 0.1 20 0
```

Long run:

```bash
cd /opt/dlami/nvme/cudaturbo-runs
nohup stdbuf -oL /opt/dlami/nvme/cudaturbo-runs/cudaturbo/mem_cuda 29400 1E7 4 1000000 0.1 100 0 > mem_cuda.txt 2>&1 &
echo $!
```

## CUDA Setup Checks

```bash
nvidia-smi
nvcc --version
ls /usr/local/cuda/lib64 | grep cufft
```

If `nvcc` is not on `PATH`:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
