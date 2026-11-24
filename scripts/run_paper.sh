#!/usr/bin/env bash
# Reproduce every result of the paper: all sweeps, then the tables and Fig. 2.
#
#   bash scripts/run_paper.sh            # all sweeps (resumable: finished rows are skipped)
#   bash scripts/run_paper.sh --quick    # smoke test: 2 records, 3 horizons, 2/3 seeds (a few minutes)
#
# Outputs: results/<sweep>_{grid,best}.csv, results/tables/*.md, results/figures/rmse.pdf
#          (smoke test: results/quick/, same layout)
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=()
OUT=results
MODE="full reproduction (9 records, 21 horizons, 20 validation / 50 test seeds)"
if [[ "${1:-}" == "--quick" ]]; then
  ARGS=(--seqs 2,5 --horizons 0.1s,0.6s,2.0s --seeds-cv 2 --seeds-test 3)
  OUT=results/quick
  MODE="smoke test (records 2 and 5, horizons 0.1 / 0.6 / 2.0 s, 2 validation / 3 test seeds)"
fi

SWEEPS=(
  persistence
  lms_if lms_df_abs lms_df_disp
  snap1_if snap1_df_abs snap1_df_disp
  lsnn_analog_if lsnn_analog_df_abs lsnn_analog_df_disp
  lsnn_sod_if lsnn_sod_df_abs lsnn_sod_df_disp
  hcrsnn_analog_if hcrsnn_analog_df_abs hcrsnn_analog_df_disp
  hcrsnn_sod_if hcrsnn_sod_df_abs hcrsnn_sod_df_disp
  hcrsnn_sod_if_thr0.01 hcrsnn_sod_if_thr0.02 hcrsnn_sod_if_thr0.05
  hcrsnn_sod_df_disp_thr0.01 hcrsnn_sod_df_disp_thr0.02 hcrsnn_sod_df_disp_thr0.05
  snap1_df_disp_recompute lsnn_analog_df_disp_recompute hcrsnn_sod_df_disp_recompute
)
N=${#SWEEPS[@]}

fmt() { local s=$1; printf "%dh%02dm%02ds" $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }

echo "################################################################################"
echo "# rsnn-resp: $MODE"
echo "# $N sweeps -> $OUT/    started $(date '+%F %T')"
echo "################################################################################"

T0=$SECONDS
TIMES=()
for i in "${!SWEEPS[@]}"; do
  s=${SWEEPS[$i]}
  t=$SECONDS
  uv run python scripts/sweep.py "configs/sweeps/$s.json" --out "$OUT/$s" --tag "[$((i + 1))/$N]" "${ARGS[@]}"
  TIMES+=("$(printf '%-34s %s' "$s" "$(fmt $((SECONDS - t)))")")
  echo "  total elapsed $(fmt $((SECONDS - T0)))"
done

echo
echo "################################################################################"
echo "# all $N sweeps done in $(fmt $((SECONDS - T0)))"
printf '#   %s\n' "${TIMES[@]}"
echo "################################################################################"

if [[ "$OUT" == "results" ]]; then
  uv run python scripts/make_tables.py
  uv run python scripts/make_figures.py
else
  echo "smoke test: tables and figure are only built for the full run (they need all records and horizons)"
fi
