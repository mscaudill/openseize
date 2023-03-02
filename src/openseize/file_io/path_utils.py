""""A collection of functions for working with path instances in Openseize.

This module includes the following utilities:

    - re_match: Regex matching of paths from two sequences of paths.
    - mismatched: Locates paths from two sequences with mismatched path stems.
    - rename: In-place path renaming by substring search & replacement.
    - metadata: Path instance metadata extraction to a dictionary.
"""

from pathlib import Path
import re
from typing import Dict, List, Sequence, Set, Tuple, Union


def re_match(paths: List[Path], others: List[Path], pattern: str
) -> List[Tuple[Path, Path]]:
    r"""Matches 2 equal lengthed sequences of Path instances using a regex
    pattern string common to the Path stems of both.

    Args:
        paths:
            A list of path instances.
        others:
            Another list of path instances.
        pattern:
            A regex pattern to match path stems with other stems by.

    Returns:
        A list of matched path instance tuples.

    Raises:
        ValueError if length of paths does not equal length of others or if
        pattern is missing in any stem or if a match can not be found.

    Examples:
        >>> paths = [Path(x) for x in ['test_01_a.edf', 'test_02_b.edf']]
        >>> others = [Path(x) for x in ['test_01.txt', 'test_02.txt']]
        >>> # match from the start looking for letters then '_' then nums
        >>> re_match(paths, others, r'\w+_\d+') # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('test_01_a.edf'), PosixPath('test_01.txt')),
        (PosixPath('test_02_b.edf'), PosixPath('test_02.txt'))]
    """

    # validate the number of passed paths match
    if len(paths) != len(others):
        msg = 'Number of paths in paths and others must match: {} != {}'
        raise ValueError(msg.format(len(paths), len(others)))

    # validate that the pattern exist in all passed path stems
    missing = [fp.stem for fp in paths + others
               if not re.search(pattern, fp.stem)]
    if missing:
        msg = 'Pattern {} is missing in path stems: {}'
        raise ValueError(msg.format(pattern, missing))

    result = []
    for apath in paths:
        matched = re.search(pattern, apath.stem)
        # missing guard ensures matched is not None (i.e. has group) for mypy
        opath = [other for other in others
                if matched.group() in other.stem] #type: ignore[union-attr]

        if len(opath) != 1:
            msg = ('The matches for path {} using pattern {} are {}. '
                   'The number of matches must be exactly 1.')
            msg = msg.format(str(apath), pattern, [x.stem for x in opath])
            raise ValueError(msg)

        result.append((apath, opath[0]))

    return result


def mismatched(paths: Sequence[Path], others: Sequence[Path]) -> Set[str]:
    """Identifies mismatched Path stems between paths and others.

    Args:
        paths:
            A sequence of Path instances.
        others:
            Another sequence of Path instances.

    Returns:
        A set of stems that have no matches in either paths or others.

    Examples:
        >>> paths = [Path(x) for x in ['test_01.edf', 'test_02_b.edf']]
        >>> others = [Path(x) for x in ['test_01.txt', 'test_02.text']]
        >>> # find mismatches
        >>> sorted(list(mismatched(paths, others)))
        ['test_02', 'test_02_b']
    """

    stems = set(fp.stem for fp in paths)
    other_stems = set(op.stem for op in others)
    return stems.symmetric_difference(other_stems)


def rename(paths: Sequence[Path], substring: str, replacement: str):
    """Renames each filepath in paths by replacing substring in file name with
    replacement.

    Note: This function renames the file in-place (no copy).

    Args:
        paths:
            A sequence of Path instances.
        substr:
            A substring in each path name to be replaced.
        replacement:
            A string to substitute in-place of substr.

    Returns: None

    Raises:
        FileNotFoundError if any filepath in paths does not exist.

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

    for file_path in paths:
        if substring in str(file_path):
            target = Path(str(file_path).replace(substring, replacement))
            file_path.rename(target)


def metadata(path: Union[Path, str], **patterns: str,
) -> Dict:
    r"""Converts a path into a dictionary of metadata.

    Args:
        path:
            A string or Path instance to extract metadata from.
        **patterns:
            A collection of named regex expression patterns used to search path
            and store matches. This pattern's value must contain one and only one
            regex group within the pattern e.g. cohort = 'cohort_(\d+)' where
            (\d+) group is the value to extract for the cohort named pattern.

    Returns:
        A dictionary of metadata.

    Examples:
        >>> path = 'Group_A_cohort_1_m_3.edf'
        >>> print(metadata(path, group=r'Group_(\w)', cohort=r'cohort_(\d)',
        ... mouse=r'm_(\d)'))
        {'group': 'A', 'cohort': '1', 'mouse': '3'}

    Notes:
        Named patterns missing from path will NOT raise an error but be skipped.
    """

    fname = str(path)

    result = {}
    for name, pattern in patterns.items():

        match = re.search(pattern, fname)
        if not match:
            # skip missing patterns
            continue

        result[name] = match.group(1)

    return result
