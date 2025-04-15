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
