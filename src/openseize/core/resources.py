"""A module of tools for measuring memory and compute resources."""

import pickle
from typing import Any, Optional, Tuple, Type

import numpy as np
import psutil


def assignable(shape: Tuple,
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
    required = np.product(shape) * np.dtype(dtype).itemsize

    tolerance = int(50e6) # ensure required is at least 50MB below tolerance
    if required < limit - tolerance:
        return True

    info = (f"{shape} type '{dtype.__name__}' requires"
            f" {required / 1e9 :.2f} GB which exceeds the"
            f" {(limit)/1e9:.1f} GB available")
    if msg:
        print(info)
    return False

def assign_cores(ntasks: int, requested: Optional[int]) -> int:
    """Assigns requested number of physical cores to run ntasks.

    The actual number of cores assigned will be the minimum of ntasks, requested
    & available physical cores. This may differ from the number of cores defined
    by the OS because of hyperthreading. Hyperthreading CPUs can concurrently
    run multiple logical processes per physical core. Openseize runs CPU bound
    processes on just the physical cores for speed.

    Args:
        ntasks:
            The number of tasks to be executed.
        requested:
            The number of physical cores to execute ntasks on. This value will
            differ from the assigned core count if requested > ntasks or if
            requested > available cores.

    Returns:
        The minimum of ntasks, requested and available cores.
    """

    # get number of hyperthreads per core
    hyperthreads = psutil.cpu_count() // psutil.cpu_count(False)
    # get all available cores including logical (hyperthreads)
    available = len(psutil.Process().cpu_affinity()) #type: ignore
    available //= hyperthreads
    return min(ntasks, requested, available)

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

    #pylint: disable-next=broad-except
    except Exception:
        # for any error return False
        return False

    return True
