"""A module of tools for measuring memory and compute resources."""

import pickle
from typing import Any, Optional, Tuple, Type

import numpy as np
import psutil


def assignable(
    shape: Tuple,
    dtype: Optional[Type[float]] = float,
    limit: Optional[int] = None,
    msg=True,
) -> bool:
    """Estimates if an object with shape number of items of dtypes is
    assignable to virtual memory.

    Args:
        shape:
            The shape of the items to assign whose product is count.
        dtype:
            A python or numpy data type.
        limit:
            The maximum memory usage for assigning count dtype items, If None,
            the limit will be all available memory.
        msg:
            If count items of dtype are not assignable, this boolean will
            display the required memory and the available memory.

    Returns:
        True if count items of dtype can be assigned and False otherwise
    """

    limit = psutil.virtual_memory().available if not limit else limit
    required = np.prod(shape) * np.dtype(dtype).itemsize

    tolerance = int(50e6)  # ensure required is at least 50MB below tolerance
    if required < limit - tolerance:
        return True

    info = (
        f"{shape} type '{dtype.__name__}' requires"
        f" {required / 1e9 :.2f} GB which exceeds the"
        f" {(limit)/1e9:.1f} GB available"
    )
    if msg:
        print(info)
    return False


def allocate(jobs: int, requesting: Optional[int] = None) -> int:
    """Allocates requested number of physical cores to run jobs.

    This function assumes jobs are CPU bound and disregards hyperthreaded CPU's.
    The number of allocated cores will be the minimum of jobs, ncores and the
    system available cores.

    Args:
        jobs:
            The number of CPU bound tasks to be executed in parallel.
        requesting:
            The number of physical cores being requested to operate on jobs in
            parallel. If None, requesting will equal the number of jobs.

    Returns:
        The minimum of jobs, requesting and available physical cores.
    """

    requesting = jobs if requesting is None else requesting
    # get hyperthread count to get available physical cores
    hyperthreads = psutil.cpu_count() // psutil.cpu_count(False)
    available = len(psutil.Process().cpu_affinity())  # type: ignore
    available //= hyperthreads

    return min(jobs, requesting, available)


def pickleable(obj: Any) -> bool:
    """Returns True if obj is pickleable and False otherwise.

    Args:
        obj:
            An object to attempt to pickle.

    Returns:
        A boolean indicating of object is pickleable.
    """

    try:
        pickle.dumps(obj)

    # pylint: disable-next=broad-except
    except Exception:
        # for any error return False
        return False

    return True
