""""A collection of functions for working with path instances in Openseize.

This module includes the following utilities:

    - match: Regex matching of paths from two sequences of paths.
    - mismatched: Locates paths from two sequences with mismatched path stems.
    - rename: In-place path renaming by substring search & replacement.
    - metadata: Path instance metadata extraction to a dictionary.
"""

import os
from pathlib import Path
from pathlib import PosixPath
import re
from typing import List, Pattern, Sequence, Set, Tuple, Union


def match(files: Sequence[Path], others: Sequence[Path], pattern: Pattern 
)-> (List[Tuple[Path, Path]]):
    """Matches 2 equal lengthed sequences of Path instances using a regex
    pattern common to both.

    Args:
        files:
            A sequence of path instances.
        others:
            Another sequence of path instances.
        pattern:
            A regular expression pattern to match files with others by.

    Returns:
        A list of matched file tuples.

    Raises:
        ValueError if length of files does not match length of others or if
        pattern is missing in any of files or others.
        
    Examples:
        >>> files = [Path(x) for x in ['test_01_a.edf', 'test_02_b.edf']]
        >>> others = [Path(x) for x in ['test_01.txt', 'test_02.txt']]
        >>> # match from the start looking for letters then '_' then nums
        >>> match(files, others, '^\w+_\d+') # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('test_01_a.edf'), PosixPath('test_01.txt')), 
        (PosixPath('test_02_b.edf'), PosixPath('test_02.txt'))]
    """

    # validate the number of passed files match
    if len(files) != len(others):
        msg = 'Number of paths in files and others must match: {} != {}'
        raise ValueError(msg.format(len(files), len(others)))

    # validate that the pattern exist in all passed files
    if not all([re.search(pattern, str(x)) for x in files + others]):
        msg = 'Pattern {} is missing from at least one file in files or others.'
        raise ValueError(msg.format(pattern))

    result = []
    for fname in files:
        matched = re.search(pattern, str(fname))
        oname = [other for other in others if matched.group() in str(other)][0]
        result.append((fname, oname))

    return result


def mismatched(files: Sequence[Path], others: Sequence[Path]) -> Set[Path]:
    """Identifies mismatched Path stems between files and others.

    Args:
        files:
            A sequence of Path instances.
        others:
            Another sequence of Path instances.
    
    Returns:
        A set of stems from files that have no match in others.

    Examples:
        >>> files = [Path(x) for x in ['test_01.edf', 'test_02_b.edf']]
        >>> others = [Path(x) for x in ['test_01.txt', 'test_02.text']]
        >>> # find mismatches
        >>> sorted(list(mismatched(files, others)))
        ['test_02', 'test_02_b']
    """

    file_stems = set(fp.stem for fp in files)
    other_stems = set(op.stem for op in others)
    return file_stems.symmetric_difference(other_stems)


def rename(files: Sequence[Path], substring: str, replacement: str):
    """Renames each file in files by replacing substring in file name with
    replacement.

    Args:
        files: 
            A sequence of Path instances.
        substr:
            A substring in each file to be replaced.
        replacement:
            A string to substitute in-place of substr.

    Returns: None

    Raises:
        FileNotFoundError if any file in files does not exist.

    Examples:
        >>> import tempfile
        >>> # make a temp dir containing 2 files
        >>> tempdir = tempfile.mkdtemp()
        >>> paths = [Path(tempdir).joinpath(x) for x in ['ts_1.edf', 'ts_2.edf']]
        >>> x = [path.touch() for path in paths]
        >>> # rename the 'ts' substring to 'demo'
        >>> rename(paths, 'ts', 'demo')
        >>> # verify the filenames now start with demo in tmp dir
        >>> renamed = sorted(list(Path(tempdir).glob('*.edf')))
        >>> print([fp.name for fp in renamed])
        ['demo_1.edf', 'demo_2.edf']
    """

    for fp in files:
        if substring in str(fp):
            target = Path(str(fp).replace(substring, replacement))
            fp.rename(target)


def metadata(path: Union[Path, str], strict: bool = True, **patterns: Pattern):
    """Converts a path into a dictionary of metadata.

    Args:
        path: 
            A string or Path instance to extract metadata from.
        strict:
            If True, this function will raise a ValueError if the pattern search
            is None else it will silently ignore the pattern.
        **patterns:
            A collection of named regex expression patterns used to search path
            and store matches.

    Returns:
        A dictionary of metadata.

    Examples:
        >>> path = 'Group_A_cohort_1_m_3.edf'
        >>> print(metadata(path, group='Group_(\w)', cohort='cohort_(\d)',
        ... mouse='m_(\d)'))
        {'group': 'A', 'cohort': '1', 'mouse': '3'}
    """
    
    # TODO we need a note here that we only look for group 1 so regex must use
    # groups

    fname = str(path)

    result = dict()
    for name, pattern in patterns.items():

        match = re.search(pattern, fname)
        if not match:
            continue
        
        else:
            result[name] = match.group(1)
    
    return result




    





if __name__ == '__main__':

    files = [Path(x) for x in ['test_01_a.edf', 'test_02_b.edf']]
    others = [Path(x) for x in ['test_01.txt', 'test_02.text']]
    # match from the start looking for letters then '_' then nums
    results = match(files, others, pattern='^\w+_\d+')

    #results = mismatched(files, others)

    #rename(files, 'test', 'demo')

    #fp = Path('Group_A_cohort_v_m_3')
    #meta = as_metadata(fp, group='Group_(\w)', cohort='cohort_(\w)', mouse='m_(\d)')

