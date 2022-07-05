import inspect
import requests
import wget
import shutil
import reprlib
from pathlib import Path

from openseize.io.dialogs import message

class DataLocator:
    """The openseize demos data locator.

    Data files are not included with the openseize software. Data that is
    needed to run the demos are stored at Zenodo. When a client request
    a path for a demo data file, this locator will first look on the clients
    local machine for the data file. If it is found, this locator returns
    the local path. If this locator does not find the file, it will download
    the file from Zenodo and then return the downloaded file's local path.
    """

    def __init__(self):
        """ """
    
        # module (demos) name of locator from call stack & make data dir
        current = Path(inspect.getabsfile(inspect.currentframe())).parent
        self.data_dir = current.joinpath('data')
        
        # define the zenodo records url containing openseize demo data
        self.records_url = 'https://zenodo.org/api/records/5999292'

    def _local(self):
        """A dict of local file paths keyed on filename."""

        result = []
        for item in Path.iterdir(self.data_dir):
            if item.is_file() and item.suffix not in ('.py'):
                result.append((item.name, item))
    
        return dict(result)

    def _remote(self):
        """A dict of remote urls keyed on filename."""

        response = requests.get(self.records_url)
        files = response.json()['files']
        self._sizes = {item['key'] : item['size'] for item in files}
    
        return dict([(item['key'], item['links']['self']) for item in files])

    def _available(self):
        """A dict of both remote and local demo files available for use."""

        result = self._remote()
        result.update(self._local())
        return result

    def locate(self, name):
        """Locates a named file downloading the file from zenodo if it is
        not on this machine.

        Args:
            name: str
                The filename including extension of the filepath to be
                located. (E.g. name = 'recording.edf' will return the full
                local path to the file for recording.edf).

        Returns: local filepath Path instance to named file.
        """

        # local path lookup
        local = self._local()
        if name in local:
            return local[name]
        
        # remote path lookup & download
        repo = self._remote()
        if name in repo:

            url, size = repo[name], self._sizes[name]
            msg = '{} will use {} MB of space. Continue?'
            msg = msg.format(name, round(size / 1e6, 1))
            
            if message('askquestion', message=msg) == 'yes':

                print('Downloading data from Zenodo...')
                out = self.data_dir.joinpath(name)
                filename = wget.download(url, out=str(out))
                print('\n')
                print('File saved to {}'.format(out))
                return filename

        # client is asking for unknown name/path
        else:
            msg = '{} contains no path for data named {}'
            AttributeError(msg.format(type(self).__name__, name))

    @property
    def available(self):
        """Prints string representation of this locator displaying all 
        available demo data files & their locations, both local & remote."""

        msg_start =  '---Available demo data files & location---'
        msg_header = '-' * len(msg_start)

        # build string of available file names and locs.
        r  = reprlib.aRepr
        r.maxstring = 40
        result = self._available()
        msg = {name: r.repr(str(path)) for name, path in result.items()}
        msg_body = ['{:30} {}'.format(k, v) for k, v  in msg.items()]
        
        print('\n'.join([msg_start, msg_header, '\n'.join(msg_body)]))


# run on demo import
paths = DataLocator()
