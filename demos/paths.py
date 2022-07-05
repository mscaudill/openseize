import requests
import wget
import shutil
from pathlib import Path

class Datapaths:
    """ """

    records_url = 'https://zenodo.org/api/records/5999292'
    data_dir = Path.cwd()

    def __init__(self):
        """ """

        self.names = self._remote()
        self.names.update(self._local())

    def available(self):
        """Displays available local & remote files for openseize demos."""

        print('Available files...', end='\n')
        print('-' * 30)
        for name, loc in self.names.items():
            msg = '{} | located at {}'
            print(msg.format(name, loc))

    def _local(self):
        """List the data file paths in data directory of demos."""

        result = []
        for item in Path.iterdir(self.data_dir):
            if item.is_file() and item.suffix not in ('.py'):
                result.append((item.name, item))
    
        return dict(result)

    def _remote(self):
        """List the urls of files in the openseize Zenodo repository."""

        response = requests.get(self.records_url)
        files = response.json()['files']

        return dict([(item['key'], item['links']['self']) for item in files])

    def fetch(self, data_name):
        """ """

        # make sure it has a suffix
        lookup = data_name

        if lookup not in self.names:
            msg = '{} contains no path for data named {}'
            AttributeError(msg.format(type(self).__name__, data_name))
    
        elif isinstance(self.names[lookup], Path):
            # local filepath lookup
            return self.names[data_name]
        
        else:
            remote_url = self.names[data_name]
            print('Downloading data from Zenodo..')
            filename = wget.download(remote_url)
            print('File save to {}'.format(Path.cwd().joinpath(data_name)))
            return filename


if __name__ == '__main__':

    
    #url = 'https://zenodo.org/record/5999292/files/dredd_behavior.pkl?download=1'
    #record_url = 'https://zenodo.org/api/records/5999292'

    dpaths = Datapaths()
