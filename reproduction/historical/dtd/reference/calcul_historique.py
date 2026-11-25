"""Normalisation et bootstrap repris tels quels du contrôle HSMT de centrage. Formules figées."""
import numpy as np

SEED = 18112026


def compact(raw, training=None, params=None):
    if (training is None) == (params is None):
        raise ValueError("Choisir exactement une source de normalisation.")
    blocks, fitted = [], {}
    for name, dimension in (("F7", 9), ("F8", 3)):
        x = raw[name].astype(np.float64)
        ii, jj = np.triu_indices(dimension)
        x[..., np.flatnonzero(ii != jj)] *= np.sqrt(2.)
        if params is None:
            xt = x[training]
            mean = xt.mean(axis=(0, 2), keepdims=True)
            sd = np.sqrt(np.maximum(1e-8, xt.var(axis=(0, 2), keepdims=True)))
        else:
            mean, sd = params[name + "_mean"], params[name + "_std"]
        fitted[name + "_mean"], fitted[name + "_std"] = mean, sd
        x = np.clip((x - mean) / sd, -12., 12.).reshape(len(x), -1)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        blocks.append(np.divide(x, norm, out=np.zeros_like(x), where=norm > 0))
    return np.concatenate(blocks, axis=1) / np.sqrt(2.), fitted


def paired_bootstrap(ci, delta, groups, nc, nboot=10000):
    rng = np.random.default_rng(SEED)
    if groups is None:
        # Oxford val has equal numbers per class. Sample within each class.
        sim = np.zeros((nboot, delta.shape[1]))
        for k in range(nc):
            values = delta[ci == k]
            n = len(values)
            w = rng.multinomial(n, np.full(n, 1 / n), size=nboot)
            sim += (w @ values) / n
        return np.quantile(100 * sim / nc, [.025, .975], axis=0)
    keys, idx = np.unique(groups, return_inverse=True)
    ng, ncomp = len(keys), delta.shape[1]
    n_by = np.zeros((ng, nc), dtype=int)
    d_by = np.zeros((ng, nc, ncomp), dtype=int)
    np.add.at(n_by, (idx, ci), 1)
    np.add.at(d_by, (idx, ci), delta)
    matrix = np.column_stack((n_by, d_by.reshape(ng, -1))).astype(float)
    resampled, count = [], 0
    while count < nboot:
        weights = rng.multinomial(ng, np.full(ng, 1 / ng), size=min(250, nboot - count))
        totals = weights.astype(float) @ matrix
        denominator = totals[:, :nc]
        valid = np.all(denominator > 0, axis=1)
        diff = totals[valid, nc:].reshape(-1, nc, ncomp)
        resampled.append(100 * (diff / denominator[valid, :, None]).mean(1))
        count += int(valid.sum())
    return np.quantile(np.vstack(resampled), [.025, .975], axis=0)
