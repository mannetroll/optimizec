#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/opt/dlami/nvme/cudaturboruns}
BIN=${BIN:-/home/ubuntu/cudaturbo/mem_cuda}
LOG=${LOG:-mem_cuda.txt}
TIMER_LOG=${TIMER_LOG:-snapshot_timer.txt}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-1h}
MOV=${MOV:-0}
MOV_DIR=${MOV_DIR:-/opt/dlami/nvme/cudaturboruns}
RESTART=${RESTART:-}

N=${N:-29400}
RE=${RE:-1E8}
K0=${K0:-2}
STEPS=${STEPS:-1000000}
CFL=${CFL:-0.1}
UPDATE=${UPDATE:-100}
ADAPT_VISC=${ADAPT_VISC:-0}

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

exec > "$LOG"

cmd=(stdbuf -oL "$BIN")
if [[ -n "$RESTART" ]]; then
    if [[ ! -f "$RESTART" ]]; then
        echo "restart file not found: $RESTART" >&2
        exit 1
    fi
    cmd+=("$N" "$RE" "$K0" "$STEPS" "$CFL" "$UPDATE" "$ADAPT_VISC" "$RESTART")
else
    cmd+=("$N" "$RE" "$K0" "$STEPS" "$CFL" "$UPDATE" "$ADAPT_VISC")
fi
if [[ "$MOV" == "1" ]]; then
    export MOV=1
    export MOV_DIR
    cmd+=(MOV)
fi

echo "command: ${cmd[*]}"
if [[ -n "$RESTART" ]]; then
    echo "restart source: $RESTART"
fi
if [[ "$MOV" == "1" ]]; then
    echo "movie root: $MOV_DIR"
fi
echo "snapshot interval: $SNAPSHOT_INTERVAL"
echo "snapshot signal: SIGUSR1"

setsid "${cmd[@]}" 2>&1 &
SIM_PID=$!
echo "mem_cuda PID: $SIM_PID"
echo "$SIM_PID" > mem_cuda.pid

setsid bash -c '
set -euo pipefail
pid="$1"
interval="$2"
while kill -0 "$pid" 2>/dev/null; do
    sleep "$interval"
    kill -USR1 "$pid" 2>/dev/null || exit 0
done
' bash "$SIM_PID" "$SNAPSHOT_INTERVAL" > "$TIMER_LOG" 2>&1 &
TIMER_PID=$!
echo "snapshot timer PID: $TIMER_PID"
echo "$TIMER_PID" > snapshot_timer.pid

echo "log: $RUN_DIR/$LOG"
echo "timer log: $RUN_DIR/$TIMER_LOG"
echo
echo "Monitor:"
echo "  tail -f $RUN_DIR/$LOG"
echo
echo "Restart examples:"
echo "  RESTART=/path/to/restart.bin $0"
echo
echo "Save now and continue:"
echo "  kill -USR1 $SIM_PID"
echo
echo "Save images + restart.bin and continue:"
echo "  kill -RTMIN $SIM_PID"
echo
echo "Pause:"
echo "  kill -USR2 $SIM_PID"
echo
echo "Resume:"
echo "  kill -CONT $SIM_PID"
echo
echo "Save and exit:"
echo "  kill -HUP $SIM_PID"
echo
echo "Kill immediately without saving:"
echo "  kill -INT $SIM_PID"
echo "  (Ctrl-C only reaches mem_cuda when it is the foreground job.)"
echo
echo "Cancel hourly snapshots only:"
echo "  kill $TIMER_PID"
