"""Validated JSON configuration; all paths resolve relative to the config file."""

import copy
import json
from pathlib import Path
import numpy as np
from .circuit import CHANNELS

DEFAULT = {
    "circuit": {"N": 6, "L": 2, "phi": float(np.pi / 16), "O": "IIZIZI"},
    "noise": {
        "placement": "before_cnot",
        "single": [0.001, 0.001, 0.0001],
        "pair": 0.0,
        "rates_file": None,
        "channels": {},
    },
    "s_grid": [1.0, 2.0, 4.0, 8.0],
    "mps": {
        "bond_dims": [16, 32, 64],
        "cutoff": 1e-12,
        "svd": "randomized",
        "power_iterations": 2,
        "oversample": 24,
        "seed": 7,
    },
    "reference": {
        "statevector_max_n": 20,
        "density_max_n": 8,
        "memory_mb": 512,
        "bond_dims": [128, 256, 512],
        "rtol": 1e-4,
        "atol": 1e-8,
        "consecutive": 2,
    },
    "fit": {
        "rate_bounds": [0.001, 5.0],
        "amplitude_bound": 1.0,
        "trials": 1000,
        "seed": 7,
        "shot_scenarios": {
            "100k_each": 100000,
            "1m_each": 1000000,
            "10m_each": 10000000,
        },
    },
}


def merge(base, updates, path=""):
    if not isinstance(updates, dict):
        raise ValueError(f'{path or "config"} must be an object')
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if k not in base:
            raise ValueError(f"Unknown configuration key: {path+k}")
        if isinstance(base[k], dict) and k not in ("channels", "shot_scenarios"):
            out[k] = merge(base[k], v, path + k + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def integer(v, name, minimum=0):
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return v


def expand(value, shape, name):
    try:
        a = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must broadcast to {shape}") from e
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{name} rates must be finite and nonnegative")
    return a


def load_config(path=None, overrides=None):
    path = Path(path).resolve() if path else None
    c = merge(DEFAULT, json.loads(path.read_text()) if path else {})
    if overrides:
        c = merge(c, overrides)
    n = integer(c["circuit"]["N"], "N", 2)
    l = integer(c["circuit"]["L"], "L")
    if n % 2:
        raise ValueError("N must be even for the initial product of singlets")
    o = c["circuit"]["O"]
    if not isinstance(o, str) or len(o) != n or any(x not in "IXYZ" for x in o):
        raise ValueError(
            "O must be an N-character Pauli string using I,X,Y,Z, qubit 0 first"
        )
    if not np.isfinite(c["circuit"]["phi"]):
        raise ValueError("phi must be finite (radians)")
    grid = np.array(c["s_grid"], float)
    if (
        grid.ndim != 1
        or len(grid) < 1
        or not np.isfinite(grid).all()
        or (grid <= 0).any()
        or len(np.unique(grid)) != len(grid)
    ):
        raise ValueError(
            "s_grid must contain distinct positive finite scales; s=0 is computed automatically, never fitted"
        )
    c["s_grid"] = sorted(grid.tolist())
    if c["noise"]["placement"] not in ("before_cnot", "after_cnot"):
        raise ValueError("noise placement must be before_cnot or after_cnot")
    for key in ("mps", "reference"):
        caps = c[key]["bond_dims"]
        if not isinstance(caps, list) or not caps:
            raise ValueError(f"{key}.bond_dims must be a nonempty list")
        for k in caps:
            integer(k, f"{key}.bond_dims", 1)
        if caps != sorted(set(caps)):
            raise ValueError(f"{key}.bond_dims must be strictly increasing")
    if c["mps"]["svd"] not in ("full", "randomized"):
        raise ValueError("mps.svd must be full or randomized")
    for key in ("power_iterations", "oversample", "seed"):
        integer(c["mps"][key], f"mps.{key}")
    for key in ("statevector_max_n", "density_max_n"):
        integer(c["reference"][key], f"reference.{key}")
    integer(c["reference"]["consecutive"], "reference.consecutive", 1)
    for section, key in [
        ("mps", "cutoff"),
        ("reference", "atol"),
        ("reference", "rtol"),
        ("reference", "memory_mb"),
    ]:
        v = c[section][key]
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"{section}.{key} must be finite and nonnegative")
    bounds = np.array(c["fit"]["rate_bounds"], float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[0] < 0
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("rate_bounds must be [nonnegative minimum, larger maximum]")
    if not np.isfinite(c["fit"]["amplitude_bound"]) or c["fit"]["amplitude_bound"] <= 0:
        raise ValueError("amplitude_bound must be positive")
    integer(c["fit"]["trials"], "fit.trials", 2)
    integer(c["fit"]["seed"], "fit.seed")
    shot_vectors(c)
    return c, path.parent if path else Path.cwd()


def schedules(config, base):
    n = config["circuit"]["N"]
    l = config["circuit"]["L"]
    noise = config["noise"]
    single = expand(noise["single"], (l, 4, n, 3), "noise.single")
    pair = expand(noise["pair"], (l, 4, n - 1, 3, 3), "noise.pair")
    # A full NPZ replaces the shared defaults; named overrides apply afterward.
    if noise["rates_file"] is not None:
        with np.load(base / noise["rates_file"], allow_pickle=False) as f:
            if set(f.files) != {"single_rates", "pair_rates"}:
                raise ValueError("Noise NPZ needs single_rates and pair_rates")
            single = expand(f["single_rates"], (l, 4, n, 3), "single_rates")
            pair = expand(f["pair_rates"], (l, 4, n - 1, 3, 3), "pair_rates")
    if not isinstance(noise["channels"], dict):
        raise ValueError("noise.channels must be an object")
    for name, values in noise["channels"].items():
        if name not in CHANNELS:
            raise ValueError(f"Unknown channel {name}; expected {CHANNELS}")
        if not isinstance(values, dict) or set(values) - {"single", "pair", "file"}:
            raise ValueError(f"Invalid channel configuration for {name}")
        i = CHANNELS.index(name)
        if "file" in values:
            if set(values) != {"file"}:
                raise ValueError("A channel file cannot be combined with inline rates")
            with np.load(base / values["file"], allow_pickle=False) as f:
                if set(f.files) != {"single_rates", "pair_rates"}:
                    raise ValueError("Channel NPZ needs single_rates and pair_rates")
                single[:, i] = expand(f["single_rates"], (l, n, 3), name + ".single")
                pair[:, i] = expand(f["pair_rates"], (l, n - 1, 3, 3), name + ".pair")
        else:
            if "single" in values:
                single[:, i] = expand(values["single"], (l, n, 3), name + ".single")
            if "pair" in values:
                pair[:, i] = expand(values["pair"], (l, n - 1, 3, 3), name + ".pair")
    return single, pair


def shot_vectors(config):
    out = {}
    grid = config["s_grid"]
    scenarios = config["fit"]["shot_scenarios"]
    if not isinstance(scenarios, dict):
        raise ValueError("shot_scenarios must be a name -> shots mapping")
    for name, spec in scenarios.items():
        if not isinstance(name, str) or name == "noiseless":
            raise ValueError("shot scenario needs a string name other than noiseless")
        if isinstance(spec, dict):
            try:
                by_s = {float(k): v for k, v in spec.items()}
            except (ValueError, TypeError) as e:
                raise ValueError("shot-map keys must be s values") from e
            if len(by_s) != len(spec) or set(by_s) != set(grid):
                raise ValueError("per-s shot map must cover every s exactly once")
            vec = [by_s[s] for s in grid]
        elif isinstance(spec, list):
            if len(spec) != len(grid):
                raise ValueError(
                    "shot vector length must equal s_grid length (sorted order)"
                )
            vec = spec
        else:
            vec = [spec] * len(grid)
        for v in vec:
            if v is not None:
                integer(v, "shot count (or null for exact)", 1)
        out[name] = vec
    return out
