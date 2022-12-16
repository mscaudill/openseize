"""A tool for locating and downloading demonstration data for openseize's
tutorials, docstrings and tests.

This module contains the following class:

    DataLocator

See DataLocator docs for examples.
"""

import inspect
from pathlib import Path
import reprlib

# wget missing stubs & request usage here is very limited
import requests # type: ignore
import wget # type: ignore

from openseize.file_io.dialogs import message


class DataLocator:
    """A tool for locating, downloading and returning local file paths for
    Openseize's demo data.

    Data files are not included with the openseize software. Data that is
    needed to run the demos are stored at Zenodo. When a client request
    a path for a demo data file, this locator will first look on the clients
    local machine for the data file. If it is found, this locator returns
    the local path. If this locator does not find the file, it will download
    the file from Zenodo and then return the downloaded file's local path.

    Attributes:
        data_dir (Path):
            A python path instance to Openseizes.demo.data dir.
        records_url (str):
            A string url to the Zenodo repository housing the
            demo data.

    Examples:

        >>> # determine what data is locally available
        >>> paths = DataLocator()
        >>> paths.available()
        >>> # fetch the file path of the recording_001.edf
        >>> # this will download the file if not local
        >>> fp = paths.locate('recording_001.edf')
    """

    def __init__(self):
        """Initialize an instance with the data_dir location & Zenodo url.

        It is important that the data dir always remain in the demo module
        of Openseize since we use a relative path to locate it here.
        """

        # module (demos) name of locator from call stack & make data dir
        current = Path(inspect.getabsfile(inspect.currentframe())).parent
        self.data_dir = current.joinpath('data')

        # define the zenodo records url containing openseize demo data
        self.records_url = 'https://zenodo.org/api/records/6799475'

    def _local(self):
        """Returns a dict of file paths in the data directory."""

        result = []
        for item in Path.iterdir(self.data_dir):
            if item.is_file() and item.suffix not in ('.py'):
                result.append((item.name, item))

        return dict(result)

    def _remote(self):
        """Returns a mapping of remote urls keyed on filename using
        requests."""

        response = requests.get(self.records_url)
        files = response.json()['files']

        # sizes are not needed if the file is local
        # pylint: disable-next=attribute-defined-outside-init
        self._sizes = {item['key'] : item['size'] for item in files}

        return {item['key']: item['links']['self'] for item in files}
        #return dict([(item['key'], item['links']['self']) for item in files])

    def _available(self):
        """Returns a dict of both remote and local files available."""

        result = self._remote()
        result.update(self._local())
        return result

    def locate(self, name: str, dialog: bool = True):
        """Locates a named file downloading the file from Zenodo if it is
        not in the data directory.

        Args:
            name:
                The filename including extension of the filepath to be
                located. (e.g. name = 'recording.edf')
            dialog:
                A boolean indicating if a dialog asking for confirmation
                should be opened prior to downloading.

        Returns:
            A local Path instance to a located & possibly downloaded file.
        """

        # local path lookup
        local = self._local()
        if name in local:
            return local[name]

        # remote path lookup & download
        repo = self._remote()

        if name in repo:
            url, size = repo[name], self._sizes[name]

            if dialog:
                msg = '{} will use {} MB of space. Continue?'
                msg = msg.format(name, round(size / 1e6, 1))

                if message('askquestion', message=msg) == 'no':
                    msg = '{} not downloaded - user cancelled.'
                    print(msg.format(name))
                    return None

            print(f"{'Downloading data from Zenodo...'}")
            out = self.data_dir.joinpath(name)
            filename = wget.download(url, out=str(out))
            print(f"File saved to {out}")
            return filename

        # client is asking for unknown name/path
        msg = '{} contains no path for data named {}'
        raise AttributeError(msg.format('Demos', name))

    @property
    def available(self):
        """Prints string representation of this locator displaying all
        available demo data files & their locations, both local & remote."""

        msg_start =  '---Available demo data files & location---'
        msg_header = '-' * len(msg_start)

        # build string of available file names and locs.
        fmt = reprlib.aRepr
        fmt.maxstring = 40
        result = self._available()
        msg = {name: fmt.repr(str(path)) for name, path in result.items()}
        # new lines are not as clear in fstrings
        # pylint: disable-next=consider-using-f-string
        msg_body = ['{:30} {}'.format(k, v) for k, v  in msg.items()]
        print('\n'.join([msg_start, msg_header, '\n'.join(msg_body)]))
