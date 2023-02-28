"""A module for testing Openseize's path utilities,

Typical usage example:
    !pytest test_path_utils.py::<TEST_NAME>
"""

import re
import pytest
from openseize.file_io import path_utils

from pathlib import Path


@pytest.fixture(scope="module")
def matched_paths():
    """Returns 2 lists of path instances that can be matched with a regex."""

    # these will need to be matched upto cohort
    paths = ['MSC_mouse_07_11-2-2023_cohort_a_trial_9.edf',
             'DBD_mouse_14_11-09-2023_cohort_b_trial_0.edf',
             'MSC_mouse_07_10-10-2019_cohort_a_trial_6.edf']

    others = ['MSC_mouse_07_11-2-2023_cohort_a.txt',
             'DBD_mouse_14_11-09-2023_cohort_b.txt',
             'MSC_mouse_07_10-10-2019_cohort_a.txt']

    paths = [Path(x) for x in paths]
    others = [Path(y) for y in others]

    return paths, others


@pytest.fixture(scope="module")
def unequal_lengths(matched_paths):
    """Returns 2 lists of path instances with different numbers of paths."""

    paths, others = matched_paths
    return paths, others[1:]


@pytest.fixture(scope="module")
def unmatched_paths(matched_paths):
    """Returns 2 list with 1 pair of unmatched paths."""

    paths, others = matched_paths
    paths[-1] = Path('MSC_mouse_16_10-10-2019_cohort_a_trial_6.edf')

    return paths, others


@pytest.fixture(scope="module")
def matched_stems():
    """Returns paths and others with stems that match"""

    paths = ['MSC_mouse_07_11-2-2023_cohort_a_trial_9.edf',
             'DBD_mouse_14_11-09-2023_cohort_b_trial_0.edf',
             'MSC_mouse_07_10-10-2019_cohort_a_trial_6.edf']

    others = ['MSC_mouse_07_11-2-2023_cohort_a_trial_9.txt',
             'DBD_mouse_14_11-09-2023_cohort_b_trial_0.txt',
             'MSC_mouse_07_10-10-2019_cohort_a_trial_6.txt']

    paths = [Path(x) for x in paths]
    others = [Path(y) for y in others]

    return paths, others


@pytest.fixture(scope="module")
def unmatched_stems(matched_stems):
    """Returns paths and others with one unmatched stem."""

    paths, others = matched_stems
    others[0] = Path('MSC_mouse_04_11-2-2023_cohort_a_trial_9.txt')

    return paths, others


def test_re_match(matched_paths):
    """Test if re_match returns the expected output for two list of paths that
    share a common regex pattern."""

    paths, others = matched_paths
    # pattern is to read all letters and - through the date
    result = path_utils.re_match(paths, others, pattern=r'[\w-]+-\d+')
    assert result == list(zip(paths, others))


def test_re_match_invalid_lengths(unequal_lengths):
    """Test that re_match raises a ValueError if the lengths of the supplied
    paths don't match"""

    paths, others = unequal_lengths

    with pytest.raises(ValueError):
        path_utils.re_match(paths, others, pattern=r'[\w-]+-\d+')


def test_re_match_bad_pattern(matched_paths):
    """Test that if a pattern is passed that does not match any paths or others
    re_match raises a ValueError."""

    paths, others = matched_paths

    with pytest.raises(ValueError):
        # set an impossible pattern starting with a number
        path_utils.re_match(paths, others, pattern=r'^\d+')


def test_re_match_unmatched(unmatched_paths):
    """Test that re_match raises a ValueError if match unsuccessful."""

    paths, others = unmatched_paths
    with pytest.raises(ValueError) as e:
        path_utils.re_match(paths, others, pattern=r'[\w-]+-\d+')

    print(e.value)


def test_re_match_multimatches(matched_paths):
    """Test that re_match raises ValueError when the pattern is not distinct
    enough to resolve multiple matches."""

    paths, others = matched_paths

    with pytest.raises(ValueError) as e:
        # go to first underscore from start -- not enough to resolve
        path_utils.re_match(paths, others, pattern=r'^\w+_')
    
    print(e.value)


def test_mismatched(unmatched_stems):
    """Test if mismatched detects the correct unmatched paths."""

    paths, others = unmatched_stems
    result = path_utils.mismatched(paths, others)
    # paths[0] and others[0] do not match
    assert result == set([paths[0].stem, others[0].stem])



