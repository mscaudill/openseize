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

    

    


