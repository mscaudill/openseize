"""A module for testing openseize pipeline callable.

Typical usage example:
    !pytest test_pipelines.py::<TEST NAME>
"""

import pytest
import numpy as np
import scipy.signal as sps

from openseize import producer
from openseize.tools.pipeline import Pipeline

# the goal here is to build several pipelines and compare results with scipy
# then I need to test that pipelines support concurrency -- currently this will
# fail due to as_producer decorator in core that should be replaced with
# partials just like in spectra module.
