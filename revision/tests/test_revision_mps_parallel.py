"""Executable integration check, separate from lightweight unit tests."""
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pytest


def _mps_probe(seed):
    from repro_baselines import predict_mps_direct_sampling
    zz, yy, xx = np.indices((8, 8, 8))
    image = (((zz - 3.5) ** 2 + (yy - 3.5) ** 2 + (xx - 3.5) ** 2) < 8).astype(np.float32)
    obs = np.zeros((2, 8, 8, 8), dtype=np.float32)
    obs[1, :, 2, 2] = 1
    obs[1, :, 4, 4] = 1
    obs[0] = image * obs[1]
    return predict_mps_direct_sampling(obs, image, seed, realizations=2,
                                       n_cond=5, max_iterations=100)


@pytest.mark.skipif(os.environ.get("GEOVAE_MPS_INTEGRATION") != "1",
                    reason="Opt-in executable integration check")
def test_isolated_parallel_mps_matches_serial():
    serial = _mps_probe(173)
    with ProcessPoolExecutor(max_workers=2,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        parallel = list(pool.map(_mps_probe, [173, 173]))
    assert np.array_equal(serial, parallel[0])
    assert np.array_equal(serial, parallel[1])
    assert np.isfinite(serial).all()
