from __future__ import annotations

PJ_PER_SOP_RECKON = 5.3
PJ_PER_MAC_FP32 = 4.6


def lms_costs(k: int) -> dict:
    m = 9 * k + 1
    fwd = 9 * m
    learn = 2 * 9 * m
    return {
        "fwd_macs": fwd,
        "learn_ops": learn,
        "learn_state": 0,
        "n_params": 9 * m,
        "weight_bits": 32 * 9 * m,
        "energy_fwd_pj": fwd * PJ_PER_MAC_FP32,
        "energy_total_pj": (fwd + learn) * PJ_PER_MAC_FP32,
        "ops_axis": fwd,
    }


def rnn_costs(q: int, k: int) -> dict:
    m = 9 * k + 1
    fwd = q * (q + m) + 9 * q
    learn = 3 * q * (q + m)
    n_params = q * q + q * m + 9 * q
    return {
        "fwd_macs": fwd,
        "learn_ops": learn,
        "learn_state": q * (q + m),
        "n_params": n_params,
        "weight_bits": 32 * n_params,
        "energy_fwd_pj": fwd * PJ_PER_MAC_FP32,
        "energy_total_pj": (fwd + learn) * PJ_PER_MAC_FP32,
        "ops_axis": fwd,
    }


def snn_costs(cfg, k: int, rec_spikes: float, in_spikes: float) -> dict:
    """cfg: SNNConfig. rec_spikes / in_spikes: measured per marker sample (predictor, test range)."""
    return _snn_costs_one(cfg, k, rec_spikes, in_spikes, 9)


def _snn_costs_one(cfg, k: int, rec_spikes: float, in_spikes: float, n_ch: int) -> dict:
    N, K = cfg.N, cfg.K
    sod = cfg.encoding == "sod"
    n_in = (2 * n_ch if sod else n_ch) * k + 1
    n_ad = int(round(cfg.frac_adaptive * N))
    n_rec = cfg.p_rec * N * (N - 1)  # expected number of recurrent synapses
    fan_out = cfg.p_rec * (N - 1) + n_ch  # recurrent + readout targets per spike
    if sod:
        input_ops = in_spikes * N + N * K  # spikes x fan-out, plus the bias input every substep
        dense_in = K * N * n_in
    else:
        input_ops = N * n_in  # analog input: dense MACs once per sample (zero-order hold)
        dense_in = N * n_in
    neuron_updates = N * K
    dense_macs = dense_in + K * (N * N + n_ch * N)
    syn_ops = input_ops + rec_spikes * fan_out + neuron_updates
    n_syn = N * n_in + n_rec
    if cfg.traces == "full":
        # per substep: e (2 ops) + ebar (2 ops) per synapse, eps_a (3 ops) on adaptive rows
        trace_ops = K * (4 * n_syn + 3 * n_ad * (n_in + cfg.p_rec * (N - 1)))
        learn_state = N * n_in + N * N + n_ad * (n_in + N) + n_in + 2 * N
    else:
        trace_ops = K * (2 * n_in + 2 * N + 2 * N)  # bar_in, bar_rec, psibar
        learn_state = n_in + 2 * N
    update_ops = 2 * (n_syn + n_ch * N + n_ch)
    learn_ops = trace_ops + update_ops
    n_params = N * n_in + n_rec + n_ch * N + n_ch
    bits = cfg.quant_bits if cfg.quant_bits else 32
    weight_bits = bits * (N * n_in + n_rec + n_ch * N) + 32 * n_ch
    e_fwd = syn_ops * PJ_PER_SOP_RECKON
    e_tot = (syn_ops + learn_ops) * PJ_PER_SOP_RECKON
    ops_axis = syn_ops
    return {
        "rec_spikes": rec_spikes,
        "in_spikes": in_spikes,
        "syn_ops": syn_ops,
        "dense_macs": dense_macs,
        "fwd_macs": dense_macs,
        "learn_trace_ops": trace_ops,
        "learn_update_ops": update_ops,
        "learn_ops": learn_ops,
        "learn_state": learn_state,
        "n_params": n_params,
        "weight_bits": weight_bits,
        "energy_fwd_pj": e_fwd,
        "energy_total_pj": e_tot,
        "ops_axis": ops_axis,
    }
