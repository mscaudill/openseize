import tkinter as tk
import tkinter.filedialog as tkdialogs
import tkinter.messagebox as tkmsgbox
import re
from pathlib import Path
from collections import defaultdict
from openseize.file_io.path_utils import re_match
from typing import List, Tuple


def root_deco(dialog):
    """Decorates a dialog with a toplevel that is destroyed when the dialog
    is closed."""

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
def standard(kind, **options):
    """Opens a tkinter modal file dialog & returns a Path instance.

    Args:
        kind (str):             name of a tkinter dialog
    **options:
        parent (widget):        ignored
        title (str):            title of the dialog window
        initialdir (str):       dir dialog starts in 
        initialfile (str):      file selected on dialog open
        filetypes (seq):        sequence of (label, pattern tuples) '*'
                                wildcard allowed
        defaultextension (str): default ext to append during save dialogs
        multiple (bool):        when True multiple selection enabled
    """

    fp = getattr(tkdialogs, kind)(**options)
    return Path(fp) if isinstance(fp, str) else [Path(p) for p in fp]


@root_deco
def message(kind, **options):
    """Opens a tkinter modal message box & returns a string response.
    
    Args:
        kind (str):         name of tkinter message box
    **options:
        default:            default button, if unspecified then 1st button
        parent (widget):    ignored
    """

    return getattr(tkmsgbox, kind)(**options)


def matching(kind: str, pattern: str, **options) -> List[Tuple[Path, Path]]:
    r"""A dialog that regex pattern matches files with different suffixes
    residing in a single directory or are selected with 2 file dialogs.

    Args:
        kind:
            A tkinter dialog string name; must be 'askdirectory' or
            'askopenfilenames'. If askdirectory, the files to be matched are
            assumed to be in the selected directory. If askopenfilenames, the
            user will use two dialogs to select the files to match.
        pattern:
            A regular expression pattern raw string used to perform match.
        **options:
            - initialdir (str):     The dir to begin path selection. Default is
                                    current dir.
            - filetypes (list):     sequence of (label, pattern tuples). The '*'
                                    wildcard allowed. This option is only valid
                                    if kind is 'askopenfilenames'.

    Returns:
        A list of matched path instance tuples.

    Examples:
        >>> import tempfile
        >>> # make a temp dir containing 2 .edf & 2 .txt files
        >>> tempdir = tempfile.mkdtemp()
        >>> print(tempdir)
        >>> paths = [Path(tempdir).joinpath(x) 
        ... for x in ['eeg_1.edf', 'eeg_2.edf']]
        >>> others = [Path(tempdir).joinpath(x)
        ... for x in ['annotation_1.txt', 'annotation_2.txt']]
        >>> x = [path.touch() for path in paths+others]
        >>> # match the files stored in the dir
        >>> result = matching('askdirectory', r'\d+')
    """

    if kind == 'askdirectory':
        # separate file paths in dir by suffix
        dirpath = standard(kind, **options)
        filepaths = dirpath.glob('*.*')
        sorted_paths = defaultdict(list)
        for path in filepaths:
            sorted_paths[path.suffix].append(path)
        
        result = list(sorted_paths.values())
        
    elif kind == 'askopenfilenames':
        # open two dialogs to user select files to match
        paths = standard(kind, title='Select File Set 1', **options)
        others = standard(kind, title='Select File Set 2', **options)
        
        result = [paths, others]

    else:
        msg = "matching dialog requires a dialog with name '{}' or '{}'"
        raise TypeError(msg.format('askdirectory', 'askopenfilenames'))

    return re_match(*result, pattern)



    


def matching_dialog(titles=['', ''], regex=None, initialdir=None):
    """Opens a sequence of dialogs and matches each file selected in the
    first dialog with a single file selected in the subsequent dialogs.

    Args:
        titles: sequence
            A list of dialog titles one per dialog to be opened. Default is
            ['', ''] which opens two un-named dialogs.
        regex: regular expression pattern
            A string specifying a regular expression pattern. This pattern
            will be used to extract a character string (i.e. token) from
            each path in the first dialog to match against each path in
            subsequent dialogs. If None (default) the path stem (i.e.
            filename without ext) will be used as the token for matching
            paths across dialogs.
        initialdir: str
            The initialdir for the dialogs to query file selections.

    Typical Usage Example:

    If trying to match edf files [5872_animal0_xx3.edf, 81_animal9_yz3.edf]
    with text files [5872_animal0.txt, 81_animal9.txt]

    >> matching_dialog(['Select EDFs', 'Select Texts'], regex='\d+_w+')
    
    >> # returns the following 
    
    >> [(5872_animal0_xx3.edf, 5872_animal0.txt),
             (81_animal9_yz3.edf,81_animal9.txt)]

    Returns: A sequence of tuples where each tuple contains the matched
             paths on per dialog opened.
    """

    matcher = _Matcher(titles, initialdir=initialdir)
    matcher.select()
    return matcher.match(regex)


class _Matcher:
    """A sequence of dialogs for selecting and matching filepaths based on
    filename matching or regular expression pattern matching. 

    Each path in the first dialog will be matched with a single path
    returned from each of the subsequent dialogs. If multiple matches or no
    matches are found, this Matcher raises OSError.

    This class is not intended to be called externally.

    Attrs:
        titles: sequence of strings
            A sequence of dialog titles, one per dialog to be opened.
        options: dict
            Any valid kwarg for tkinter's askopenfilenames dialog.
            
    Computed Attrs:
        selected: sequence
            A sequence of path lists one per dialog opened by this
            matcher.
    """

    def __init__(self, titles, **options):
        """Intialize this matcher with the title of each dialog and the
        dialog box options.

        Args:
            titles: sequence
                A sequence of string titles for each dialog. Default is to
                open 2 dialogs with no string name. 
            options: dict
                Any valid kwargs for tkinter askopenfilenames.
        """

        self.titles = titles
        self.opts = options

    def validate_selection(self, selected):
        """Validates the number of paths returned from each dialog equal."""

        lengths = [len(paths) for paths in selected]

        if not all([length == lengths[0] for length in lengths]):
            
            msg = 'Number of files from each dialog do not match {} != {}'
            raise OSError(msg.format(min(lengths), max(lengths)))

    def select(self):
        """Opens dialogs one for each title in this Matcher's titles."""

        selected = []
        for idx, title in enumerate(self.titles):

            title = 'dialog {}'.format(idx) if not title else title
            paths = standard('askopenfilenames', title=title, **self.opts)
            selected.append(paths)

        # validate that lengths of paths from each dialog are equal
        self.validate_selection(selected)
        self.selected = selected

    def _tokenize(self, regex, path):
        """Converts a path to a token using a regex pattern or path stem.

        A token is a set of characters that will be used to match files
        returned from each dialog. If no regex pattern is provided the
        path's stem (i.e. filename without ext) is used to perform the
        matching.

        Returns: A token string to use to match files from each dialog.
        """

        # if no regex the token is the path's stem
        if not regex:
            return path.stem

        # if regex given -> compute token
        match = re.search(regex, str(path))
        if match:
            token = match.group()
        
        else:
            msg = 'regex pattern {} not found in path {}'
            raise OSError(msg.format(regex, path))
        
        return token

    def _match_by_token(self, token):
        """Locates a single path from each dialogs' returned paths that
        contains token.

        Returns: A sequence of length titles containing the path from each
        dialog that contains the token string.
        """

        if isinstance(token, Path):
            token = token.stem

        result = []
        for title, dialog in zip(self.titles[1:], self.selected[1:]):
            
            others = [path for path in dialog if token in str(path)]
            
            # if there are multiple matches from one of the dialogs
            if len(others) > 1:
                msg = "{} files from dialog '{}' contain the token '{}'"
                raise OSError(msg.format(len(others), title, token))

            # if there are no matches from one of the dialogs
            elif len(others) < 1:
                msg = "No files from dialog '{}' contain the token '{}'"
                raise OSError(msg.format(title, token))
            
            else:
                result.append(others[0])
        
        return result

    def match(self, regex):
        """Matches a token extracted from the paths of the first dialog to
        the subsequent dialogs opened by this Matcher.

        Args:
            regex: an re str
                A regular expression string used to build a searchable token
                for each path returned from the first dialog to match against
                filenames in the subsequent dialogs. If None, the token used
                to search subsequent dialogs will be the path stem of each
                path in the first dialog.

        Returns: a sequence of tuples of matched path instances.
        """
       
        results = []

        for path in self.selected[0]:
            
            # build token from path, match and store
            token = self._tokenize(regex, path)
            tup = tuple([path] + self._match_by_token(token))
            results.append(tup)

        return results


if __name__ == '__main__':

    #result = matching('askopenfilenames', r'\d+_\w+\s\w')
    result = matching('askdirectory', r'\d+_\w+\s\w')

    
