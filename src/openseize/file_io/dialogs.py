"""A collection of dialogs and message boxes.

This module contains the following functions
```
    standard:
        A unified tkinter filedialog that gracefully destroys itself when
        a selection is complete or the window is closed.

    message:
        A unified tkinter messagebox that gracefully destroys itself when the
        window is closed.

    matching:
        A dialog that uses regex patterns to match file stems of two sets of
        file paths selected by path selection dialogs, or a directory dialog or
        by manually specifying a directory containing all paths to match.
```
"""

from collections import defaultdict
from pathlib import Path
import tkinter as tk
import tkinter.filedialog as tkdialogs
import tkinter.messagebox as tkmsgbox
from typing import List, Optional, Tuple, Union

from openseize.file_io.path_utils import re_match


def root_deco(dialog):
    """Decorates a dialog with a toplevel that is destroyed on dialog exit."""

    def decorated(*args, **kwargs):
        #create root and withdraw from screen
        root = tk.Tk()
        root.withdraw()
        #open dialog returning result and destroy root
        result = dialog(*args, parent=root, **kwargs)
        root.destroy()
        return result
    return decorated


@root_deco
def standard(kind: str, **options: str) -> Union[Path, List[Path]]:
    """Opens a tkinter modal file dialog & returns a Path instance or list of
    Path instances.

    Args:
        kind:
            Name of a tkinter file dialog.
        options:
            - title (str):            title of the dialog window
            - initialdir (str):       dir dialog starts in
            - initialfile (str):      file selected on dialog open
            - filetypes (seq):        sequence of (label, pattern tuples) '*'
                                    with wildcard allowed
            - defaultextension (str): default ext to append during save dialogs
            - multiple (bool):        when True multiple selection enabled

    Returns:
        A Path instance or list of Path instances.
    """

    # disable parent placement since root_deco automates this
    options.pop('parent', None)
    paths = getattr(tkdialogs, kind)(**options)
    return Path(paths) if isinstance(paths, str) else [Path(p) for p in paths]


@root_deco
def message(kind: str, **options: str):
    """Opens a tkinter modal message box & returns a string response.

    Args:
        kind:
            Name of a tkinter messagebox (eg. 'askyesno')
        options:
            Any valid option for the message box except for 'parent'.

    Returns:
        A subset of (True, False, OK, None, Yes, No).
    """

    #disable parent placement since root_deco automates this
    options.pop('parent', None)
    return getattr(tkmsgbox, kind)(**options)


def matching(pattern: str, kind: Optional[str] = None,
             dirpath: Optional[Union[str, Path]] = None, **options
) -> List[Tuple[Path, Path]]:
    r"""A dialog that regex pattern matches Path stems of two sets of files.

    This dialog can match two sets of files from two separate dialogs or
    match two sets of files contained in a single directory specified by
    a single dialog or a manually supplied directory path.

    Args:
        pattern:
            A regex pattern raw string used to perform match.
        kind:
            A string indicating if dialog should be of type 'askdirectory' or
            'askopenfilenames'. If 'askdirectory' all the files to match must be
            in a single dir. If None a dirpath must be supplied.
        dirpath:
            A optional path to a directory. If given, kind argument is ignored
            and no dialogs are opened. It is assumed that all paths to match
            are in the given dir.
        **options:
            - initialdir (str):     The dir to begin path selection. Default is
                                    current dir.
            - filetypes (list):     sequence of (label, pattern tuples). The '*'
                                    wildcard allowed. This option is only valid
                                    if kind is 'askopenfilenames'.
    Returns:
        A list of matched path instance tuples.

    Raises:
        A TypeError is raised if kind is not one of 'askdirectory' or
        'askopenfilenames' and the dirpath is None.

    Examples:
        >>> # This example matches by directory without dialoging.
        >>> import tempfile
        >>> # make a temp dir containing 2 .edf & 2 .txt files
        >>> tempdir = tempfile.mkdtemp()
        >>> paths = [Path(tempdir).joinpath(x)
        ... for x in ['eeg_1.edf', 'eeg_2.edf']]
        >>> others = [Path(tempdir).joinpath(x)
        ... for x in ['annotation_1.txt', 'annotation_2.txt']]
        >>> x = [path.touch() for path in paths+others]
        >>> # match the files stored in the dir
        >>> result = matching(r'\d+', dirpath=Path(tempdir))
        >>> # matching does not guarantee ordering so chk set equivalence
        >>> set(result) == set(zip(paths, others))
        True
    """

    if dirpath or kind == 'askdirectory':

        # dialog for dir if none given
        dirpath = standard(kind, **options) if not dirpath else Path(dirpath)
        # separate file paths in dirpat by suffix
        filepaths = dirpath.glob('*.*')
        sorted_paths = defaultdict(list)
        for path in filepaths:
            sorted_paths[path.suffix].append(path)
        paths, others = list(sorted_paths.values())

    elif kind == 'askopenfilenames':

        # open two dialogs to user select files to match
        paths = standard(kind, title='Select File Set 1', **options)
        others = standard(kind, title='Select File Set 2', **options)

    else:

        msg = ("matching dialog requires 'kind' argument to be one of '{}' "
                "or '{}' or a Path passed to the dirpath argument.")
        raise TypeError(msg.format('askdirectory', 'askopenfilenames'))

    return re_match(paths, others, pattern)
