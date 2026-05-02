#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/tobbe/sim_16384_10}

if [[ ! -d "$RUN_DIR" ]]; then
    echo "RUN_DIR does not exist: $RUN_DIR" >&2
    exit 1
fi

mapfile -d '' folders < <(
    find "$RUN_DIR" -maxdepth 1 -type d -name 'output_*' \
        ! -exec test -f '{}/restart.bin' \; \
        -print0 | sort -z
)

if (( ${#folders[@]} == 0 )); then
    echo "No output_* folders without restart.bin found in $RUN_DIR"
    exit 0
fi

echo "Folders to remove:"
for folder in "${folders[@]}"; do
    echo "  $folder"
done
echo
printf "Remove these %d folder(s)? [y/N] " "${#folders[@]}"
read -r answer

case "$answer" in
    y|Y|yes|YES)
        rm -rf -- "${folders[@]}"
        echo "Removed ${#folders[@]} folder(s)."
        ;;
    *)
        echo "Cancelled."
        ;;
esac
