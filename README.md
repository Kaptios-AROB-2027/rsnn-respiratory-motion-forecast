# Feasibility of RSNNs for real-time respiratory motion forecasting

Code for the AROB 2027 paper *Feasibility of Recurrent Spiking Neural Networks for Real-Time Event-Based
Respiratory Motion Tracking in Lung Radiotherapy*.

| Model | Config prefix | Description |
| --- | --- | --- |
| Persistence | `persistence` | forecast the last observed sample |
| LMS filter | `lms` | Pohl et al.'s `LMS_predict.m` |
| SnAp-1 RNN | `snap1` | tanh RNN trained with SnAp-1, Pohl et al.'s `rnn_SnAp1.m` |
| LSNN | `lsnn_analog`, `lsnn_sod` | LIF + ALIF neurons, symmetric feedback, full eligibility traces, float32 weights |
| HC-RSNN | `hcrsnn_analog`, `hcrsnn_sod` | hardware-constrained (ReckOn-like): LIF only, fixed random feedback, local traces, 8-bit weights |

## Setup

```bash
uv sync
uv run python -c "import jax; print(jax.devices())"
uv run python scripts/fetch_data.py
```

`fetch_data.py` clones `pohl-michel/2D-MR-image-prediction` (`Time_series_forecasting/` only) into
`external/` and converts the recordings to `data/extmarker.npz`. It checks the 10 Hz record lengths and exits
non-zero on a mismatch.

## Reproducing the paper

```bash
bash scripts/run_paper.sh --quick     # smoke test of every sweep (2 records, 3 horizons, few seeds), minutes
bash scripts/run_paper.sh             # all 28 sweeps, then the tables and Fig. 2 (about a day on one RTX 5090)
```

`run_paper.sh` runs every config in `configs/sweeps/` and then

* `scripts/make_tables.py` -> `results/tables/`: `accuracy.md` (Table 1), `latency.md` (Table 2), `cost.md`
  (Table 3), `tests.md` (Wilcoxon tests quoted in the text), `threshold.md` (SOD threshold ablation) and
  `recompute.md` (two-network check);
* `scripts/make_figures.py` -> `results/figures/rmse.pdf` (Fig. 2).

Sweeps are resumable: rerunning skips the rows already in `results/<sweep>_grid.csv`. A single sweep runs with

```bash
uv run python scripts/sweep.py configs/sweeps/hcrsnn_sod_df_disp.json [--seqs 2,5 --horizons 0.3s,1.2s]
```

### Sweeps

| Config | Role in the paper |
| --- | --- |
| `persistence` | reference |
| `<model>_if` | IFeed, absolute outputs, stable learning-rate rule (Table 1, Fig. 2a) |
| `<model>_df_abs`, `<model>_df_disp` | DFeed with absolute or displacement outputs; the output is chosen per model on validation (Table 1, Tables 2 and 3, Fig. 2b) |
| `hcrsnn_sod_if_thr*`, `hcrsnn_sod_df_disp_thr*` | SOD threshold ablation |
| `*_df_disp_recompute` | supplementary check: DFeed recomputed with a second network copy (coupled loop) |

A config sets the method, the loop (`coupled`: IFeed plus the two-network DFeed stream; `stored`: DFeed), the
model parameters, the static grid (signal history length, SnAp-1 hidden size) and the learning rates. Shared RSNN
parameters are in `configs/rsnn_base.json`; the structural switches of the two networks are in
`NETWORK_PRESETS` (`src/models/rsnn.py`).

### Protocol

* Each recording is normalised per channel with the mean and standard deviation of its first 30 s. The model sees
  the last k samples (signal history length k / 10 Hz) and forecasts the position h samples ahead.
* Every model learns online over the whole recording. Forecasts whose target falls in [30 s, 60 s) form the
  validation range, those after 60 s the test range.
* Selection per (record, horizon): among configurations with the fewest diverged seeds, the lowest validation
  RMSE. Under IFeed a learning rate is also discarded for a record if its validation RMSE exceeds twice that of
  persistence at any horizon and signal history length.
* 20 seeds on validation, 50 fresh seeds on test for each selected configuration. RMSE is the 3D distance between
  forecast and measured marker positions (mm), averaged over markers, test range and seeds, then over the nine
  records and 21 horizons.
* Costs on ReckOn are estimates: synaptic operations x 5.3 pJ, and a cycle model of the ReckOn design.

## Data, licences, citations

Data: CC BY 4.0, from Krilavicius et al. (arXiv:1508.00749), as distributed by Pohl et al. Cite Pohl et al.,
Computer Methods and Programs in Biomedicine 222 (2022) 106908, and Pohl et al., arXiv:2403.01607 (the
benchmark and online RNN learners reproduced here). Pohl et al.'s code is BSD-3. Methods: Bellec et al., Nature Communications 11 (2020) 3625 (e-prop); Frenkel & Indiveri, ISSCC 2022
(ReckOn).
