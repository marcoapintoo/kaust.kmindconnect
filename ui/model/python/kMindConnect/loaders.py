import os
import pandas as pd
import numpy as np
import scipy.io

class CSVReader:
    def __init__(self, **config):
        self.data = None
        self.config = config
        self.config.setdefault("sep", None)
        self.config.setdefault("header", 1)
    
    def read(self, path):
        self.data = pd.read_csv(path, **self.config).to_dict('records')

class ARFFReader:
    def __init__(self, **config):
        self.data = None
        self.config = config
    
    def read(self, path):
        self.data = scipy.io.arff.loadarff(path, **self.config).data

class MATReader:
    def __init__(self, **config):
        self.data = None
        self.config = config

    def read(self, path):
        self.data = scipy.io.loadmat(path, **self.config)

class AutofileReader:
    readers = {}
    def __init__(self, **config):
        self.data = None
        self.config = config
    
    def read(self, path, **config):
        config = self.config.copy()
        for k, w in config.items():
            config[k] = w
        target = config.pop("target", None)
        extension = os.path.splitext(path)[1].lower()
        if extension not in self.readers:
            raise ValueError("Data file extension not known: {0}".format(extension))
        reader = self.readers[extension](**config)
        reader.read(path)
        self.data = reader.data if target is None else reader.data[target]
        return self.data
    
    @staticmethod
    def get(path, **config):
        reader = AutofileReader()
        return reader.read(path, **config)

AutofileReader.readers[".csv"] = CSVReader
AutofileReader.readers[".arff"] = ARFFReader
AutofileReader.readers[".mat"] = MATReader

