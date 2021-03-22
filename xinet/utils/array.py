import numpy as np


class NumPyArrayLike:
    '''创建一个类 Numpy array
    '''

    def __init__(self, data, dtype=None):
        self._data = [*data] if data else []
        self._dtype = dtype

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    def append(self, values):
        self._data.append(values)

    def __iter__(self):
        for d in self.data:
            yield d

    def __len__(self):
        return len(self.data)

    def __array__(self, dtype=None):
        """Returns a NumPy ndarray.
        """
        return np.asarray(self.data, dtype)

    def __index__(self):
        return self.item()

    def cumsum(self):
        """返回累计数据"""
        return np.array(self.data).cumsum().tolist()
