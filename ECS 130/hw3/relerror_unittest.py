import numpy as np, unittest
def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

class RelerrorTestCase(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-8, msg=None):
        """
        Assert that numpy arrays or floating point values `a` and `b` are
        equal within a relative error of `tol`.
        """
        self.assertLessEqual(np.linalg.norm(a - b), tol * np.linalg.norm(b), msg=msg)

import pickle, lzma
class TestData:
    def __init__(self, name, notexist_ok=False):
        self.path = f'data/{name}_test_data.pkl.xz'
        try: self.data = pickle.load(lzma.open(self.path, 'rb'))
        except FileNotFoundError:
            if notexist_ok: self.data = []
            else: raise

    def add(self, d): self.data.append(d)
    def dump(self):
        print('Dumping test data to', self.path)
        pickle.dump(self.data, lzma.open(self.path, 'wb'))
