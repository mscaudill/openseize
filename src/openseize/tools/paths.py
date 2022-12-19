import re
import os
from pathlib import Path
from collections.abc import Sequence

class MetaData(dict):
    """An extended dict of metadata extracted from a filepath.

    Args:
        path (Path or str):      Path obj or string to extract metadata from
        **searches (kwargs):     a string, a regex expression, or a sequence
                                 of strings or regexes to locate specific
                                 path parts to include in PathMeta. Please
                                 see example below for usage.
        warn (bbol):             warn if a search in searches fails (Default
                                 is True)
    Extensions:
    1. dot '.' notation access to the underlying dict data

    Usage: given a path,

            path = '/dir0/group_a/5872_mouse a_left_dbs.dat'
           
           where the group is one of ('a', 'w'), the mouse id (5872), the 
           mouse geno is one of ('a', 'b'), the side is one of 
           ('left', 'right') and the treatment is one of ('dbs','sham') 
           are to be extracted, the searches required to build a PathMeta:

           PathMeta(path, group=['group_(a)', 'group_(q)'], mouse='\d+', 
                    geno=['\s(a)', '\s(b)', side=['left, 'right'],
                    treatment=['dbs', 'sham'])

            If kwarg has a sequence value, we look for each element in
            value to match in path (like 'side'). If both match only the
            first occurrence is stored. If kwarg is a single regex 
            (like '\d+') we search for first occurrence of the regex. If 
            parantheses are used (like 'group'), this means search for the
            entire pattern but only keep what is inside. For example here 
            we used parenthesis to differentiate the group (a) from 
            genotype (a) by including the surrounding context 'group_(<>)' or
            \s(<>).

            The returned PathMeta will be {'group': 'a', 'mouse': '5872',
            'geno': 'a', 'side': 'left', 'treatment': 'dbs'}
    """

    def __init__(self, path, warn=True, **searches):
        """Initialize this PathMeta."""

        self.path = str(path)
        self.warn = warn
        dict.__init__(self)
        metadata = self._search(**searches)
        self.update(metadata)
    
    def __getattr__(self, name):
        """Provides '.' notation access to this PathMeta's values."""

        try:
            return self[name]
        except:
            #explicitly raise error, needed for getmembers
            raise AttributeError

    def _search(self, **kwargs):
        """Returns a dict of search pattern matches from the Metas path."""

        meta = dict()
        for name, substrs in kwargs.items():
            #kwarg substrs is a single str or regex
            if isinstance(substrs, str):
                pattern = r'{}'.format(substrs)
                match = [re.search(pattern, self.path)]
            #kwargs is a sequence of strs and/or regexes
            elif isinstance(substrs, Sequence):
                patterns = [r'{}'.format(substr) for substr in substrs]
                matches = [re.search(p, self.path) for p in patterns]
                match = [m for m in matches if m]
            #possibly warn if matches not found
            if not any(match):
                if self.warn:
                    msg = "no match found for search: '{}' = {}"
                    print(msg.format(name, substrs))
            else:
                #only get first match
                match = match[0]
                #handle if grouping (parenthesis) were passed
                meta.update({name: match.groups()[0] if match.groups()
                             else match.group()})
        return meta


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
