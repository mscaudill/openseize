import psutil
import numpy as np

def assignable_array(shape, dtype=float, allowable=None):
    """Estimates if an array of proposed shape with dtype elements 
    can be assigned into virtual memory.

    Args:
        shape: tuple of ints
            The proposed shape of array to store.
        dtype: numpy datatype 
            The datatype of each element to be stored.
        allowable: int
            The maximum allowed memory usage in bytes. If None, query and
            use the available system memory. Default is None.

    Returns:
        Boolean indicating if proposed array can be assigned, the available
        memory and the required memory of the proposed array.
    """

    if allowable is None:
        allowable = psutil.virtual_memory().available
        
    required = np.product(shape) * np.dtype(dtype).itemsize

    assignable = False if required >= allowable else True
    
    return assignable, allowable, required


def is_assignable(pro, dtype=float, allowable=None):
    """Validates if a producer can be assigned to an in-memory array.

    Args:
        pro: producer of ndarrays.
            The producer to convert to an in-memory array.
        dtype: numpy datatype.
            The data type of each element of each array in the producer.
        allowable: int
            The maximum allowed memory usage in bytes. If None, query and
            use the available system memory. Default is None.
    
    Returns: True if the producer is assignable and raises a memory error if
             not.
    """

    resource_result  = assignable_array(pro.shape, dtype)
    assignable, allowable, required = resource_result
        
    if not assignable:
        a, b = np.round(np.array([required, allowable]) / 1e9, 1)
        msg = 'Producer will consume {} GB but only {} GB are available'
        raise MemoryError(msg.format(a, b))
    
    else:
        return True

    

    


