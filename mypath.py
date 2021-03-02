import os

class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'thinobject5k':
            return os.path.join('data', 'ThinObject5K') 
        elif database == 'coift':
            return os.path.join('data', 'COIFT')
        elif database == 'hrsod':
            return os.path.join('data', 'HRSOD')
        elif database == 'thin_regions':
            return os.path.join('data', 'thin_regions')
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
