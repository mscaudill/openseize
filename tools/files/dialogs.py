import tkinter as tk
import tkinter.filedialog as tkdialogs
import tkinter.messagebox as tkmsgbox
import re
from pathlib import Path

from openseize.io.files import ops

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

    paths = getattr(tkdialogs, kind)(**options)
    return [Path(p) for p in paths] if len(paths) > 1 else Path(paths)


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

def _validate_lengths(paths, others):
    """Raises OSError if len of paths does not match len of other paths."""

    if len(paths) != len(others):
        msg = 'length of selected paths do not match {} != {}'
        raise OSError(msg.format(len(paths), len(others)))

def matched(titles=['', ''], **options):
    """Opens two standard dialogs & matches the path instances by Path stem.

    Assumes the filenames match exactly except for the extension. For a more
    general pattern match between files see regexmatched.
    
    Args:
        titles (seq):           2-el seq of string titles one per dialog
        **options:              passed to standard dialog
    
    Returns: list of matched tuples of Path instances
    """

    res = []
    t0, t1 = titles
    #dialog for paths and match by stems
    fpaths = standard('askopenfilenames', title=t0, **options)
    opaths = standard('askopenfilenames', title=t1, **options)
    _validate_lengths(fpaths, opaths)
    #match stems
    ostems = [op.stem for op in opaths]
    for fpath in fpaths:
        try:
            idx = ostems.index(fpath.stem)
        except IndexError:
            msg = 'file {} does not match any of {}'
            raise OSError(msg.format(fpath.stem, ostems))
        res.append((fpath, opaths[idx]))
    return res

def regexmatched(pattern, titles=['',''], **options):
    """Opens two dialogs and matches the path instances by regex pattern.

    Args:
        titles (seq):           2-el seq of string titles one per dialog
        **options:              passed to standard dialog

    Returns: list of matched tuples of Path instances
    """
    
    results = []
    t0, t1 = titles
    #dialog for paths and fetch stems
    fpaths = standard('askopenfilenames', title=t0, **options)
    opaths = standard('askopenfilenames', title=t1, **options)
    _validate_lengths(fpaths, opaths)
    ostems = [op.stem for op in opaths]
    fstems = [fp.stem for fp in fpaths]
    #make match by regex pattern
    for path in fpaths:
        match = re.search(pattern, path.stem)
        if match:
            other = [opath for opath, ostem in zip(opaths, ostems) 
                     if match.group() in ostem]
            if len(other) == 1:
                results.append(other)
            else:
                msg = ('zero or multiple matches found for path {}'
                      ' with pattern {}')
                raise OSError(msg.format(path, pattern))
        else:
            msg = 'pattern {} is not found in any of {}'
            raise OSError(msg.format(pattern, fstems))
    return results




if __name__ == '__main__':

    #path = standard('askopenfilename', title='hubbub')
    #path = standard('asksaveasfilename', defaultextension='.pkl')
    paths = regexmatched('\d+_\w+_\w+')
    #response = message('askyesno')
