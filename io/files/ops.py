import re
import os
from pathlib import Path

def replace(files, old_substr, new_substr):
    """Replaces a substr in a set of src files with a new substr."""

    for f in files:
        if old_substr in f:
            name = str(f).replace(old_substr, new_substr)
            os.rename(f, name)

def mismatches(files, others):
    """Compares the stems of files with others to find mismatches."""

    file_stems = [Path(f).stem for f in files]
    other_stems = [Path(o).stem for o in others]
    return set(file_stems).symmetric_difference(set(other_stems))
     

