"""A module for testing concurrency with Openseize producer instances.

These tests should be considered a work in progress as there may be unforseen
difficulties in multiprocessing with producers within complex DSP pipelines
This file test if a basic downsample, notch filter and data reducer pipeline
multiprocess correctly.

Typical usage example:
    # -rA flag shows extra summary see pytest -h
    !pytest -rA test_concurrency::<TEST_NAME>
"""

import multiprocessing as mp
import pickle
import time

import numpy as np
import pytest

from openseize import producer
from openseize.demos import paths
from openseize.file_io import edf
from openseize.filtering.iir import Notch
from openseize.resampling.resampling import downsample


@pytest.fixture(scope="module")
def rng():
    """Returns a numpy default_rng object for generating reproducible but
    random ndarrays."""

    seed = 0
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def random2D(rng):
    """Returns a random 2D array."""

    return rng.random((5, 17834000))


def test_pickling(demo_data):
    """Assert that a producer from a reader is picklable."""

    reader = edf.Reader(demo_data)

    pro = producer(reader, chunksize=30e5, axis=-1)

    sbytes = pickle.dumps(pro)
    assert isinstance(sbytes, bytes)

def pipeline(pro):
    """An examplar pipeline that downsamples, notch filters and computes the
    mean of arrays produced by a producer."""

    dpro = downsample(pro, M=10, fs=5000, chunksize=1e5)
    notch = Notch(fstop=60, fs=500, width=8)
    notch_pro = notch(dpro, chunksize=1e5, axis=-1, dephase=True)

    return np.array([np.mean(arr, axis=-1) for arr in notch_pro])


def test_pipelines(random2D, demo_data):
    """Verifies that a pipeline run concurrently on multiple producers yields
    the same result as calling the pipeline sequentially on the same producers.
    """

    # build a producer from the demo data
    reader = edf.Reader(demo_data)
    rpro = producer(reader, chunksize=20e5, axis=-1)

    # build a producer from a random 2D array
    apro = producer(random2D, chunksize=20e5, axis=-1)

    def multiprocessor(pros, ncores=2):
        """A 2-core multiprocessor that executes pipeline on each core."""

        print('starting multiprocessor')
        t_0 = time.perf_counter()
        with mp.Pool(processes=ncores) as pool:
            results = []
            for res in pool.imap(pipeline, pros):
                results.append(res)
        t_1 = time.perf_counter()
        print(f'Multiprocessor completed in {t_1-t_0} s')

        return results

    # construct the multiprocessed results
    multi_process_results = multiprocessor([rpro, apro])

    # call the pipeline sequentially on each file
    print('Starting sequential processing')
    t_0 = time.perf_counter()
    sequential_results = [pipeline(pro) for pro in [rpro, apro]]
    t_1 = time.perf_counter()
    print(f'Sequential processing completed in {t_1-t_0} s')

    # compare the arrays in each result
    for mp_res, seq_result in zip(multi_process_results, sequential_results):
        assert np.allclose(mp_res, seq_result)
