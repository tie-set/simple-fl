from typing import Dict
import numpy as np

class LimitedDict(dict):
    def __init__(self, keys):
        self._keys = keys
        self.clear()

    def __setitem__(self, key, value):
        if key not in self._keys:
            raise KeyError
        dict.__setitem__(self, key, value)

    def clear(self):
        for key in self._keys:
            self[key] = list()

def convert_LDict_to_Dict(ld: LimitedDict) -> Dict[str,np.array]:
    d = dict()
    for key, val in ld.items():
        d[key] = val[0]
    return d
