from pathlib import Path

from openseize.mixins import ViewInstance

class FileManager(ViewInstance):
    """A context manager for ensuring files are closed at the conclusion
    of reading or writing or in case of any raised exceptions.

    This class defines a partial interface and can not be instantiated.
    """

    def __init__(self, path, mode):
        """Initialize this FileManager inheritor with a path and mode."""

        if type(self) is FileManager:
            msg = '{} class cannot be instantiated'
            print('msg'.format(type(self).__name__))
        self.path = Path(path)
        self._fobj = open(path, mode)

    def __enter__(self):
        """Return instance as target variable of this context."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close this instances file object & propagate any error by 
        returning None."""

        self.close()

    def close(self):
        """Close this instance's opened file object."""

        self._fobj.close()
