import numpy as np
import numba as nb
from numba import njit, prange

# Assumes Pauli encoding 0=I, 1=X, 2=Y, 3=Z.
# If you paste these into Pauli_path_Heis_full_layer_restricted.py, remove this
# obs_value if the file already defines the same function.
@njit(cache=True)
def obs_value_qpm(P):
    val = 1
    for i in range(0, P.shape[0], 2):
        a = P[i]
        b = P[i + 1]
        if a != b:
            return 0
        if a != 0:
            val = -val
    return val


@njit(cache=True)
def count_pm_qpm(P_out, s_out):
    n_p = 0
    n_m = 0
    n_0 = 0
    for j in range(P_out.shape[0]):
        v = obs_value_qpm(P_out[j]) * s_out[j]
        if v == 1:
            n_p += 1
        elif v == -1:
            n_m += 1
        else:
            n_0 += 1
    return n_p, n_m, n_0


@njit(cache=True)
def split_log_amp_lambda_pm(P_out, s_out, amp_out, wi_out, damp_out):
    """
    Split samples into +/- observable sectors.

    Returns
    -------
    loga_p, lam_p, loga_m, lam_m
        loga = log(amp_out * wi_out)
        lam  = -log(damp_out)

    This matches Q_pm(s) = <exp(loga - s*lam)>_pm / <exp(loga)>_pm.
    """
    n_p, n_m, _ = count_pm_qpm(P_out, s_out)

    loga_p = np.empty(n_p, dtype=np.float64)
    lam_p = np.empty(n_p, dtype=np.float64)
    loga_m = np.empty(n_m, dtype=np.float64)
    lam_m = np.empty(n_m, dtype=np.float64)

    ip = 0
    im = 0
    for j in range(P_out.shape[0]):
        v = obs_value_qpm(P_out[j]) * s_out[j]
        if v == 1:
            loga_p[ip] = np.log(amp_out[j]) + np.log(wi_out[j])
            lam_p[ip] = -np.log(damp_out[j])
            ip += 1
        elif v == -1:
            loga_m[im] = np.log(amp_out[j]) + np.log(wi_out[j])
            lam_m[im] = -np.log(damp_out[j])
            im += 1

    return loga_p, lam_p, loga_m, lam_m


@njit(cache=True)
def split_log_amp_lambda_pm_no_iw(P_out, s_out, amp_out, damp_out):
    """
    Split samples into +/- observable sectors for full-layer mixture outputs.

    Here amp_out already contains the full importance-corrected amplitude,
    including any Z_free * wi_ratio factors from the mixture sampler.
    """
    n_p, n_m, _ = count_pm_qpm(P_out, s_out)

    loga_p = np.empty(n_p, dtype=np.float64)
    lam_p = np.empty(n_p, dtype=np.float64)
    loga_m = np.empty(n_m, dtype=np.float64)
    lam_m = np.empty(n_m, dtype=np.float64)

    ip = 0
    im = 0
    for j in range(P_out.shape[0]):
        v = obs_value_qpm(P_out[j]) * s_out[j]
        if v == 1:
            if amp_out[j] > 0.0 and damp_out[j] > 0.0:
                loga_p[ip] = np.log(amp_out[j])
                lam_p[ip] = -np.log(damp_out[j])
            else:
                loga_p[ip] = -np.inf
                lam_p[ip] = 0.0
            ip += 1
        elif v == -1:
            if amp_out[j] > 0.0 and damp_out[j] > 0.0:
                loga_m[im] = np.log(amp_out[j])
                lam_m[im] = -np.log(damp_out[j])
            else:
                loga_m[im] = -np.inf
                lam_m[im] = 0.0
            im += 1

    return loga_p, lam_p, loga_m, lam_m


@njit(cache=True)
def _q_grid_from_logs_one_sector(loga, lam, s_grid):
    """
    Stable Q(s) and delta-method standard error for one sector.

    Q(s) = sum_i exp(loga_i - s*lam_i) / sum_i exp(loga_i).

    The reported stderr is for the finite-sample ratio estimator within this
    sector:
        SE[Q] ~= sqrt( Var[X - QY] / (n * mean(Y)^2) )
    with X=exp(loga-s*lam), Y=exp(loga), evaluated in a common shifted scale.
    """
    ns = s_grid.shape[0]
    n = loga.shape[0]

    Q = np.empty(ns, dtype=np.float64)
    logQ = np.empty(ns, dtype=np.float64)
    se = np.empty(ns, dtype=np.float64)
    log_num = np.empty(ns, dtype=np.float64)
    log_den = np.empty(ns, dtype=np.float64)

    if n == 0:
        for k in range(ns):
            Q[k] = np.nan
            logQ[k] = np.nan
            se[k] = np.nan
            log_num[k] = -np.inf
            log_den[k] = -np.inf
        return Q, logQ, se, log_num, log_den

    # denominator max/logsum once
    max_y = loga[0]
    for i in range(1, n):
        if loga[i] > max_y:
            max_y = loga[i]

    sum_y_den = 0.0
    for i in range(n):
        sum_y_den += np.exp(loga[i] - max_y)
    ld = max_y + np.log(sum_y_den)

    for k in range(ns):
        ss = s_grid[k]

        # numerator max
        max_x = loga[0] - ss * lam[0]
        for i in range(1, n):
            z = loga[i] - ss * lam[i]
            if z > max_x:
                max_x = z

        sum_x_num = 0.0
        for i in range(n):
            sum_x_num += np.exp(loga[i] - ss * lam[i] - max_x)

        ln = max_x + np.log(sum_x_num)
        lq = ln - ld
        log_num[k] = ln
        log_den[k] = ld
        logQ[k] = lq
        Q[k] = np.exp(lq)

        # Common scale for ratio and variance.
        m = max_x
        if max_y > m:
            m = max_y

        sum_y = 0.0
        sum_r2 = 0.0
        q = Q[k]
        for i in range(n):
            x = np.exp(loga[i] - ss * lam[i] - m)
            y = np.exp(loga[i] - m)
            sum_y += y
            r = x - q * y
            sum_r2 += r * r

        mean_y = sum_y / n
        if n > 1 and mean_y > 0.0:
            # sample variance of r divided by n*mean_y^2
            var_r = sum_r2 / (n - 1)
            se[k] = np.sqrt(var_r / n) / mean_y
        else:
            se[k] = 0.0

    return Q, logQ, se, log_num, log_den


@njit(cache=True)
def qpm_grid_from_split_logs(loga_p, lam_p, loga_m, lam_m, s_grid):
    Qp, logQp, SEp, logNp, logDp = _q_grid_from_logs_one_sector(loga_p, lam_p, s_grid)
    Qm, logQm, SEm, logNm, logDm = _q_grid_from_logs_one_sector(loga_m, lam_m, s_grid)
    return Qp, Qm, logQp, logQm, SEp, SEm, logNp, logNm, logDp, logDm


@njit(cache=True)
def qpm_grid_from_outputs(P_out, s_out, amp_out, wi_out, damp_out, s_grid):
    """
    One-call wrapper for outputs of evolve_many_*_mixture_restricted-like routines.

    Expected arrays:
      P_out    : (n_samples, n_qubits) int8 final Paulis
      s_out    : (n_samples,) int8 path signs
      amp_out  : (n_samples,) float64 amplitude 1-norm product
      wi_out   : (n_samples,) float64 importance weight product
      damp_out : (n_samples,) float64 damping product
      s_grid   : (n_s,) float64 values of extrapolation/noise parameter s

    Returns:
      Qp, Qm, logQp, logQm, SEp, SEm, counts
    where counts = [n_plus, n_minus, n_zero].
    """
    n_p, n_m, n_0 = count_pm_qpm(P_out, s_out)
    counts = np.empty(3, dtype=np.int64)
    counts[0] = n_p
    counts[1] = n_m
    counts[2] = n_0

    loga_p, lam_p, loga_m, lam_m = split_log_amp_lambda_pm(
        P_out, s_out, amp_out, wi_out, damp_out
    )
    Qp, Qm, logQp, logQm, SEp, SEm, _, _, _, _ = qpm_grid_from_split_logs(
        loga_p, lam_p, loga_m, lam_m, s_grid
    )
    return Qp, Qm, logQp, logQm, SEp, SEm, counts


@njit(cache=True)
def qpm_grid_from_outputs_no_iw(P_out, s_out, amp_out, damp_out, s_grid):
    """
    One-call wrapper when there is no separate wi_out.

    Use this for evolve_many_full_layer_sampling_mixture_restricted outputs:
        P_out, P_even_last, s_out, amp_out, damp_out, triggered_out

    The mixture importance factor is already included in amp_out by the sampler.
    P_even_last and triggered_out are diagnostics and are not needed for Q_pm(s).
    """
    n_p, n_m, n_0 = count_pm_qpm(P_out, s_out)
    counts = np.empty(3, dtype=np.int64)
    counts[0] = n_p
    counts[1] = n_m
    counts[2] = n_0

    loga_p, lam_p, loga_m, lam_m = split_log_amp_lambda_pm_no_iw(
        P_out, s_out, amp_out, damp_out
    )
    Qp, Qm, logQp, logQm, SEp, SEm, _, _, _, _ = qpm_grid_from_split_logs(
        loga_p, lam_p, loga_m, lam_m, s_grid
    )
    return Qp, Qm, logQp, logQm, SEp, SEm, counts


@njit(cache=True)
def qpm_grid_from_full_layer_mixture_outputs(P_out, P_even_last, s_out, amp_out, damp_out, triggered_out, s_grid):
    """
    Explicit wrapper matching evolve_many_full_layer_sampling_mixture_restricted.

    P_even_last and triggered_out are accepted to match the returned tuple, but
    Q_pm only depends on final P_out, s_out, amp_out, and damp_out.
    """
    return qpm_grid_from_outputs_no_iw(P_out, s_out, amp_out, damp_out, s_grid)
